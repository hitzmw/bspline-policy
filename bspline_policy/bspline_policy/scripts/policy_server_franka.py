"""ZMQ inference server for the Franka Drifting-BSpline checkpoint.

This server returns the policy adapter's already-decoded physical control
sequence as well as the raw and projected B-spline parameters.  It does not
send commands to the robot; joint limits, rate limits, watchdogs, and the
Franka driver remain the responsibility of the robot-side client.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional


BSPLINE_POLICY_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BSPLINE_POLICY_DIR.parent
DIFFUSION_POLICY_DIR = REPO_ROOT / "diffusion_policy"
for source_dir in (BSPLINE_POLICY_DIR, DIFFUSION_POLICY_DIR):
    if source_dir.is_dir() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import cv2 as cv
import dill
import hydra
import numpy as np
import torch
import zmq

from diffusion_policy.common.pytorch_util import dict_apply


OBS_ALIASES = {
    "sideview_image": "D435_color",
    "wrist_image": "D405_color",
    "gripper_pos": "gripper_position",
}


def _select_policy_state(payload: dict, cfg) -> tuple[dict, str]:
    """Select EMA weights from either a full or inference-only checkpoint."""
    if "policy_state_dict" in payload:
        return payload["policy_state_dict"], str(
            payload.get("selected_policy", "inference_policy")
        )

    state_dicts = payload.get("state_dicts")
    if not isinstance(state_dicts, dict):
        raise KeyError(
            "Checkpoint must contain policy_state_dict or state_dicts"
        )
    use_ema = bool(cfg.training.get("use_ema", False))
    selected = "ema_model" if use_ema else "model"
    if selected not in state_dicts:
        raise KeyError(f"Checkpoint does not contain {selected!r}")
    return state_dicts[selected], selected


class FrankaDriftingBSplinePolicy:
    """Load a Franka checkpoint and expose decoded 8-step control chunks."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        warmup: bool = True,
        swap_rb: bool = False,
    ) -> None:
        self.ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        self.device = torch.device(device)
        self.swap_rb = bool(swap_rb)

        payload = torch.load(
            self.ckpt_path,
            map_location="cpu",
            pickle_module=dill,
        )
        cfg = payload["cfg"]
        expected_target = (
            "bspline_policy.policy.drifting_unet_franka_bspline_image_policy."
            "DriftingUnetFrankaBSplineImagePolicy"
        )
        if str(cfg.policy._target_) != expected_target:
            raise ValueError(
                "This server requires the Franka policy adapter, got "
                f"{cfg.policy._target_!s}"
            )

        policy = hydra.utils.instantiate(cfg.policy)
        policy_state, selected_policy = _select_policy_state(payload, cfg)
        policy.load_state_dict(policy_state)
        del policy_state, payload

        policy.eval().to(self.device)
        self.policy = policy
        self.cfg = cfg
        self.obs_shape_meta = cfg.shape_meta["obs"]
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.execution_action_steps = int(cfg.n_action_steps)
        self.physical_action_dim = int(cfg.shape_meta.action.shape[0])
        self.selected_policy = selected_policy

        print(f"Loaded Franka Drifting-BSpline policy: {self.ckpt_path}")
        print(f"  device: {self.device}")
        print(f"  selected weights: {self.selected_policy}")
        print(f"  observations: {list(self.obs_shape_meta.keys())}")
        print(
            "  decoded action shape: "
            f"[{self.execution_action_steps}, {self.physical_action_dim}]"
        )
        if warmup:
            self._warmup()

    def _dummy_observation(self) -> dict:
        result = {}
        for key, meta in self.obs_shape_meta.items():
            shape = tuple(int(value) for value in meta["shape"])
            if meta.get("type") == "rgb":
                _, height, width = shape
                result[key] = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                result[key] = np.zeros(shape, dtype=np.float32)
        return result

    def _warmup(self) -> None:
        observations = [self._dummy_observation()] * self.n_obs_steps
        obs_dict = self.convert_observations(observations)
        print("Warming up Franka policy...")
        with torch.no_grad():
            self.policy.predict_action(obs_dict)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        print("Franka policy warmup complete")

    def reset(self) -> None:
        self.policy.reset()

    def _lookup_value(self, observation: dict, key: str):
        if key in observation:
            return observation[key]
        alias = OBS_ALIASES.get(key)
        if alias is not None and alias in observation:
            return observation[alias]
        raise KeyError(
            f"Missing observation {key!r}; received {sorted(observation.keys())}"
        )

    def _prepare_image(self, value, key: str, meta: dict) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 1:
            image = cv.imdecode(image, cv.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode compressed image {key!r}")
        elif image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be HWC/CHW RGB image, got {image.shape}")
        if image.dtype in (np.float32, np.float64):
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        _, height, width = (int(value) for value in meta["shape"])
        if image.shape[:2] != (height, width):
            image = cv.resize(image, (width, height))
        if self.swap_rb:
            image = image[..., ::-1]
        return np.ascontiguousarray(np.moveaxis(image, -1, 0)).astype(
            np.float32
        ) / 255.0

    def convert_observations(self, observations: list[dict]) -> dict:
        if len(observations) != self.n_obs_steps:
            raise ValueError(
                f"Expected {self.n_obs_steps} observations, got {len(observations)}"
            )
        converted = {}
        for key, meta in self.obs_shape_meta.items():
            values = [self._lookup_value(obs, key) for obs in observations]
            if meta.get("type") == "rgb":
                converted[key] = np.stack(
                    [self._prepare_image(value, key, meta) for value in values]
                )
            else:
                expected_shape = tuple(int(value) for value in meta["shape"])
                prepared = []
                for value in values:
                    array_value = np.asarray(value, dtype=np.float32)
                    if array_value.size != int(np.prod(expected_shape)):
                        raise ValueError(
                            f"{key} value has {array_value.size} elements, "
                            f"expected {int(np.prod(expected_shape))}"
                        )
                    prepared.append(array_value.reshape(expected_shape))
                array = np.stack(prepared)
                expected = (self.n_obs_steps,) + expected_shape
                if array.shape != expected:
                    raise ValueError(
                        f"{key} shape mismatch: expected {expected}, got {array.shape}"
                    )
                converted[key] = array
        return dict_apply(
            converted,
            lambda value: torch.from_numpy(value).unsqueeze(0).to(self.device),
        )

    def predict(self, observations: list[dict]) -> dict:
        obs_dict = self.convert_observations(observations)
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)

        decoded_raw = result["action"][0].detach().cpu().numpy().astype(
            np.float32
        )
        bspline = result["bspline_action"][0].detach().cpu().numpy().astype(
            np.float32
        )
        projected = result["projected_bspline_action"][0].detach().cpu().numpy(
        ).astype(np.float32)

        expected_action_shape = (
            self.execution_action_steps,
            self.physical_action_dim,
        )
        expected_bspline_shape = (int(self.cfg.horizon), self.physical_action_dim + 1)
        if decoded_raw.shape != expected_action_shape:
            raise ValueError(
                f"Decoded action shape {decoded_raw.shape}, expected "
                f"{expected_action_shape}"
            )
        if bspline.shape != expected_bspline_shape:
            raise ValueError(
                f"B-spline shape {bspline.shape}, expected {expected_bspline_shape}"
            )
        if not all(
            np.isfinite(value).all()
            for value in (decoded_raw, bspline, projected)
        ):
            raise FloatingPointError("Policy output contains NaN or Inf")

        # The gripper is the only channel whose safe numerical bounds are known
        # from the dataset. Robot-specific joint and velocity limits must still
        # be applied by the Franka client before execution.
        decoded = decoded_raw.copy()
        decoded[:, -1] = np.clip(decoded[:, -1], -1.0, 1.0)
        return {
            "ready": True,
            "action": decoded,
            "raw_decoded_action": decoded_raw,
            "bspline": bspline,
            "projected_bspline": projected,
            "meta": {
                "action_layout": "7_abs_joint_targets_plus_gripper",
                "action_shape": list(expected_action_shape),
                "bspline_shape": list(expected_bspline_shape),
                "selected_policy": self.selected_policy,
                "gripper_clipped": True,
            },
        }


class FrankaPolicyServer:
    def __init__(
        self,
        policy: FrankaDriftingBSplinePolicy,
        port: int = 5555,
    ) -> None:
        self.policy = policy
        self.history = deque(maxlen=policy.n_obs_steps)
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{int(port)}")
        print(f"Franka Drifting-BSpline server listening on port {port}")

    def reset(self) -> None:
        self.history.clear()
        self.policy.reset()

    def process_observations(self, value) -> dict:
        sequence = value if isinstance(value, list) else [value]
        response = None
        for observation in sequence:
            self.history.append(observation)
            if len(self.history) == self.policy.n_obs_steps:
                response = self.policy.predict(list(self.history))
        if response is None:
            return {
                "ready": False,
                "history": len(self.history),
                "required_history": self.policy.n_obs_steps,
            }
        return response

    def run(self) -> None:
        while True:
            request = self.socket.recv_pyobj()
            try:
                if "reset" in request:
                    self.reset()
                    response = {"ready": False, "reset": True}
                elif "obs" in request:
                    response = self.process_observations(request["obs"])
                else:
                    response = {"ready": False, "error": "expected reset or obs"}
            except Exception as exc:  # keep the REP socket synchronized
                traceback.print_exc()
                response = {
                    "ready": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self.socket.send_pyobj(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Franka Drifting-BSpline ZMQ inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the startup inference warmup",
    )
    parser.add_argument(
        "--swap-rb",
        action="store_true",
        help="Swap input image red/blue channels before inference",
    )
    args = parser.parse_args()

    policy = FrankaDriftingBSplinePolicy(
        ckpt_path=args.ckpt_path,
        device=args.device,
        warmup=not args.no_warmup,
        swap_rb=args.swap_rb,
    )
    FrankaPolicyServer(policy, port=args.port).run()


if __name__ == "__main__":
    main()
