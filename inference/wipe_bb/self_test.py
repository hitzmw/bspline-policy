"""Offline deployment check. Never imports or commands a robot driver."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from wipe_bb_policy import WipeBBPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-checksum", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    folder = Path(__file__).resolve().parent
    if not args.skip_checksum:
        for line in (folder / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            digest = hashlib.sha256()
            with (folder / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError("Checksum mismatch: " + name)
        print("SHA256: all files OK")
    print("Python:", sys.version.split()[0], "PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    for name in ("torchvision", "robomimic", "numpy", "scipy", "Pillow"):
        print(name + ":", importlib.metadata.version(name))

    policy = WipeBBPolicy(folder, device=args.device)
    references = np.load(folder / "reference_inputs.npz", allow_pickle=False)
    errors = []
    for index in range(len(references["noise"])):
        obs = {key: references[key][index] for key in policy.obs_shape_meta}
        started = time.perf_counter()
        result = policy.predict_action(obs, noise=references["noise"][index])
        elapsed = time.perf_counter() - started
        parameters = result["bspline_action"].cpu().numpy()
        actions = result["action"].cpu().numpy()
        assert actions.shape == (1, 8, 7)
        assert parameters.shape == (1, 16, 8)
        np.testing.assert_allclose(
            parameters, references["expected_parameters"][index], rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            actions, references["expected_actions"][index], rtol=2e-4, atol=2e-4
        )
        error = float(np.max(np.abs(actions - references["expected_actions"][index])))
        errors.append(error)
        print("sample {}: action (1,8,7), max abs error {:.8g}, {:.3f}s".format(index, error, elapsed))

    # Exercise raw-camera aliases, channel ordering, and history handling.
    frames = []
    for timestep in range(2):
        frame = {}
        for key, meta in policy.obs_shape_meta.items():
            value = references[key][0, 0, timestep]
            if meta.get("type") == "rgb":
                value = np.rint(np.moveaxis(value, 0, -1) * 255).astype(np.uint8)
            raw_key = {"sideview_image": "D455_color", "wrist_image": "D405_color"}.get(key, key)
            frame[raw_key] = value
        prepared = policy.prepare_frame(frame)
        for key in prepared:
            np.testing.assert_allclose(prepared[key], references[key][0, 0, timestep], atol=1e-7)
        frames.append(frame)
    assert policy.step(frames[0]) is None
    assert policy.step(frames[1], infer=False) is None
    assert len(policy.history) == 2
    policy.reset()
    assert len(policy.history) == 0
    # The camera API must reproduce the tensor API, including the decoder.
    obs = {key: references[key][0] for key in policy.obs_shape_meta}
    seed = 3107
    tensor_action = policy.predict_action(
        obs, generator=torch.Generator(device=policy.device).manual_seed(seed)
    )["action"][0].cpu().numpy()
    camera_action = policy.predict(
        frames, generator=torch.Generator(device=policy.device).manual_seed(seed)
    )
    np.testing.assert_allclose(camera_action, tensor_action, rtol=0, atol=0)
    assert policy.step(frames[0]) is None
    history_action = policy.step(
        frames[1], generator=torch.Generator(device=policy.device).manual_seed(seed)
    )
    np.testing.assert_allclose(history_action, tensor_action, rtol=0, atol=0)
    policy.reset()
    assert not any(name.startswith("bspline_policy") for name in sys.modules)
    print(json.dumps({"status": "PASS", "max_action_error": max(errors), "robot_commands_sent": 0}))


if __name__ == "__main__":
    main()
