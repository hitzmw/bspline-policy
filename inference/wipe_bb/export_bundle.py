"""Export unchanged EMA tensors and verify the small local inference adapter."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bspline_policy"), str(ROOT / "diffusion_policy")]

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import zarr

from wipe_bb_policy import WipeBBPolicy


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "ema_weights.pt").exists():
        raise FileExistsError("Output already contains weights; choose a new output directory")
    payload = torch.load(str(args.checkpoint), map_location="cpu", pickle_module=dill, mmap=True)
    cfg = payload["cfg"]
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    expected_target = "bspline_policy.policy.drifting_unet_franka_bspline_image_policy.DriftingUnetFrankaBSplineImagePolicy"
    if policy_cfg["_target_"] != expected_target or not cfg.training.use_ema:
        raise ValueError("Expected Franka EMA checkpoint")
    state = payload["state_dicts"]["ema_model"]
    torch.save(dict(state), str(args.output / "ema_weights.pt"))
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "robomimic", "numpy", "scipy", "Pillow", "omegaconf", "einops", "zarr")
    }
    metadata = {
        "format": "wipe_bb_ema_fp32_v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_checkpoint_bytes": args.checkpoint.stat().st_size,
        "source_epoch": int(dill.loads(payload["pickles"]["epoch"])),
        "selected_weights": "ema_model",
        "tensor_count": len(state),
        "weights_bytes": (args.output / "ema_weights.pt").stat().st_size,
        "weight_dtype": "float32 (no quantization)",
        "policy_config": policy_cfg,
        "action_layout": "7 absolute joint targets in radians; no gripper",
        "data_frequency_hz": 20,
        "tested_versions": versions,
        "python_version": sys.version.split()[0],
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    reference = hydra.utils.instantiate(cfg.policy)
    reference.load_state_dict(state, strict=True)
    reference.eval().requires_grad_(False)
    adapter = WipeBBPolicy(args.output, device="cpu")
    for key, value in adapter.network.state_dict().items():
        if not torch.equal(value.cpu(), state[key].cpu()):
            raise AssertionError("Export changed EMA tensor: " + key)
    print("All {} EMA tensors unchanged".format(len(state)), flush=True)
    data = zarr.open_group(str(ROOT / "diffusion_policy/data/wipe_bb/data_save.zarr"), mode="r")["data"]
    records = {key: [] for key in policy_cfg["shape_meta"]["obs"]}
    records.update(noise=[], expected_actions=[], expected_parameters=[])
    errors = []
    with torch.no_grad():
        for index, start in enumerate((0, 113, 1000)):
            obs = {}
            for key in policy_cfg["shape_meta"]["obs"]:
                data_key = {"sideview_image": "D435_color", "wrist_image": "D405_color"}.get(key, key)
                values = data[data_key][start:start + 2]
                if key.endswith("image"):
                    values = np.moveaxis(values, -1, 1).astype(np.float32) / 255.0
                obs[key] = torch.from_numpy(values.astype(np.float32)[None])
                records[key].append(obs[key].numpy())
            seed = 1700 + index
            original = reference.predict_action(obs, generator=torch.Generator().manual_seed(seed))
            noise = torch.randn((1, 16, 8), generator=torch.Generator().manual_seed(seed))
            result = adapter.predict_action(obs, noise=noise)
            for key in ("action", "bspline_action", "projected_bspline_action"):
                torch.testing.assert_close(result[key], original[key], rtol=0, atol=0)
            errors.append(float((result["action"] - original["action"]).abs().max()))
            records["noise"].append(noise.numpy())
            records["expected_actions"].append(original["action"].numpy())
            records["expected_parameters"].append(original["bspline_action"].numpy())
    np.savez_compressed(args.output / "reference_inputs.npz", **{key: np.stack(value) for key, value in records.items()})
    del payload, state, reference, adapter
    gc.collect()
    metadata["verification"] = {
        "all_ema_tensors_bitwise_equal": True,
        "real_observation_pairs": 3,
        "max_action_abs_error_cpu": max(errors),
        "original_policy_outputs_bitwise_equal": True,
        "cuda_verified": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    source = Path(__file__).resolve().parent
    for name in ("wipe_bb_policy.py", "self_test.py", "readme.txt"):
        shutil.copy2(source / name, args.output / name)
    shutil.copy2(ROOT / "LICENSE", args.output / "LICENSE")
    files = sorted(path for path in args.output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.output / "SHA256SUMS").write_text(
        "".join(sha256(path) + "  " + path.name + "\n" for path in files), encoding="utf-8"
    )
    print(json.dumps(metadata["verification"], indent=2), flush=True)
    print("Bundle:", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
