"""Sequential online evaluator for the RoboCasa v0.2 kitchen tasks.

RoboCasa is imported lazily so training and offline tests do not depend on the
simulator being installed.  The defaults mirror the public Cosmos Policy
RoboCasa evaluation protocol used with the Human-50 data mirror.
"""

from __future__ import annotations

from collections import deque
import copy
import json
import os
from pathlib import Path
import pickle
import random
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import torch

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


RGB_OBSERVATION_KEYS = {
    "robot0_agentview_left_rgb": "robot0_agentview_left_image",
    "robot0_agentview_right_rgb": "robot0_agentview_right_image",
    "robot0_eye_in_hand_rgb": "robot0_eye_in_hand_image",
}
MOBILE_BASE_ACTION = np.asarray([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
REFERENCE_OSC_POSE_CONTROLLER_CONFIG = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping_ratio": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "damping_ratio_limits": [0, 10],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}


def adapt_policy_action(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    """Map the Human-50 7D manipulation action to PandaMobile's 12D action."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 1:
        raise ValueError(f"Expected a 1D policy action, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Policy action contains NaN or infinity")
    if action.shape[0] == env_action_dim:
        return action
    if action.shape[0] == 7 and env_action_dim == 12:
        return np.concatenate([action, MOBILE_BASE_ACTION])
    raise ValueError(
        f"Cannot adapt policy action dim {action.shape[0]} to environment "
        f"action dim {env_action_dim}"
    )


def prepare_observation_frame(
    observation: Mapping[str, np.ndarray],
    *,
    flip_images: bool,
) -> dict[str, np.ndarray]:
    """Convert one robosuite observation to the Human-50 training schema."""
    missing = [key for key in RGB_OBSERVATION_KEYS.values() if key not in observation]
    missing += [
        key
        for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
        if key not in observation
    ]
    if missing:
        raise KeyError(f"RoboCasa observation is missing keys: {missing}")

    # Importing robosuite can initialize numba, so keep it behind the online
    # execution boundary. RoboCasa quaternions use robosuite's convention.
    from robosuite.utils import transform_utils as transform

    frame: dict[str, np.ndarray] = {}
    for target_key, source_key in RGB_OBSERVATION_KEYS.items():
        image = np.asarray(observation[source_key])
        if flip_images:
            image = np.flipud(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image for {source_key}, got {image.shape}")
        frame[target_key] = np.ascontiguousarray(image)

    frame["ee_pos"] = np.asarray(
        observation["robot0_eef_pos"], dtype=np.float32
    )
    frame["ee_ori"] = np.asarray(
        transform.quat2axisangle(observation["robot0_eef_quat"]),
        dtype=np.float32,
    )
    frame["gripper_states"] = np.asarray(
        observation["robot0_gripper_qpos"], dtype=np.float32
    )
    return frame


def stack_observation_history(
    history: Sequence[Mapping[str, np.ndarray]],
    *,
    n_obs_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Create a batched, oldest-to-newest policy observation dictionary."""
    if not history:
        raise ValueError("Observation history cannot be empty")
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")

    selected = list(history)[-n_obs_steps:]
    if len(selected) < n_obs_steps:
        selected = [selected[0]] * (n_obs_steps - len(selected)) + selected

    result: dict[str, torch.Tensor] = {}
    for key in selected[0]:
        values = np.stack([np.asarray(frame[key]) for frame in selected], axis=0)
        if key in RGB_OBSERVATION_KEYS:
            values = np.moveaxis(values, -1, 1).astype(np.float32) / 255.0
        else:
            values = values.astype(np.float32)
        result[key] = torch.from_numpy(np.ascontiguousarray(values))[None].to(
            device=device,
            dtype=dtype,
        )
    return result


class RoboCasaImageRunner(BaseImageRunner):
    """Run a B-spline image policy directly in RoboCasa, one episode at a time."""

    def __init__(
        self,
        output_dir: str,
        task_name: str = "TurnOffSinkFaucet",
        n_test: int = 1,
        n_test_vis: int = 1,
        max_steps: int = 500,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        image_size: int = 224,
        seed: int = 195,
        policy_seed: int = 195,
        deterministic: bool = True,
        settle_steps: int = 10,
        flip_images: bool = True,
        robots: str = "PandaMobile",
        controller: str = "OSC_POSE",
        controller_config_path: str | None = None,
        obj_instance_split: str = "B",
        randomize_cameras: bool = False,
        layout_and_style_ids: Sequence[Sequence[int]] = (
            (1, 1),
            (2, 2),
            (4, 4),
            (6, 9),
            (7, 10),
        ),
        fps: int = 20,
        clip_actions: bool = False,
    ):
        super().__init__(output_dir)
        if n_test <= 0:
            raise ValueError("n_test must be positive")
        if n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        self.task_name = task_name
        self.n_test = int(n_test)
        self.n_test_vis = int(n_test_vis)
        self.max_steps = int(max_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.policy_seed = int(policy_seed)
        self.deterministic = bool(deterministic)
        self.settle_steps = int(settle_steps)
        self.flip_images = bool(flip_images)
        self.robots = robots
        self.controller = controller
        self.controller_config_path = controller_config_path
        self.obj_instance_split = obj_instance_split
        self.randomize_cameras = bool(randomize_cameras)
        self.layout_and_style_ids = tuple(
            tuple(int(value) for value in pair) for pair in layout_and_style_ids
        )
        self.fps = int(fps)
        self.clip_actions = bool(clip_actions)

    def _controller_config(self):
        if self.controller_config_path:
            path = Path(self.controller_config_path).expanduser().resolve()
            with path.open("rb") as controller_file:
                return pickle.load(controller_file)
        if self.controller != "OSC_POSE":
            raise ValueError(
                "Only the Human-50 OSC_POSE controller is built in; pass "
                "controller_config_path for a different controller"
            )
        return copy.deepcopy(REFERENCE_OSC_POSE_CONTROLLER_CONFIG)

    def _make_env(self, episode_index: int):
        # This avoids a numba caching issue in cloned Conda environments and
        # only affects this evaluator process.
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import robocasa  # noqa: F401 - registers RoboCasa environments
        import robosuite

        if not self.layout_and_style_ids:
            selected_scenes = None
        else:
            scene_index = (episode_index // 10) % len(self.layout_and_style_ids)
            selected_scenes = (self.layout_and_style_ids[scene_index],)
        env_seed = self.seed * episode_index * 256 if self.deterministic else None
        return robosuite.make(
            env_name=self.task_name,
            robots=self.robots,
            controller_configs=self._controller_config(),
            camera_names=[
                key.removesuffix("_image")
                for key in RGB_OBSERVATION_KEYS.values()
            ],
            camera_widths=self.image_size,
            camera_heights=self.image_size,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
            seed=env_seed,
            obj_instance_split=self.obj_instance_split,
            generative_textures=None,
            randomize_cameras=self.randomize_cameras,
            layout_and_style_ids=selected_scenes,
            translucent_robot=False,
        )

    def _seed_episode(self, episode_index: int) -> None:
        episode_seed = self.policy_seed + episode_index
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)

    def _save_video(self, frames: Sequence[np.ndarray], episode: int, success: bool):
        if not frames or episode >= self.n_test_vis:
            return
        media_dir = Path(self.output_dir) / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        path = media_dir / f"episode={episode:03d}-success={int(success)}.mp4"
        with imageio.get_writer(path, fps=self.fps) as writer:
            for frame in frames:
                writer.append_data(frame)

    def _run_episode(self, policy, episode_index: int) -> dict[str, Any]:
        self._seed_episode(episode_index)
        policy.reset()
        env = self._make_env(episode_index)
        frames: list[np.ndarray] = []
        history: deque[dict[str, np.ndarray]] = deque(maxlen=self.n_obs_steps)
        action_queue: deque[np.ndarray] = deque()
        success = False
        episode_length = 0
        task_description = self.task_name

        try:
            observation = env.reset()
            task_description = env.get_ep_meta().get("lang", self.task_name)
            for _ in range(self.settle_steps):
                dummy_action = np.zeros(env.action_spec[0].shape, dtype=np.float32)
                observation, _, _, _ = env.step(dummy_action)

            for _ in range(self.max_steps):
                frame = prepare_observation_frame(
                    observation,
                    flip_images=self.flip_images,
                )
                history.append(frame)
                if episode_index < self.n_test_vis:
                    frames.append(
                        np.concatenate(
                            [frame[key] for key in RGB_OBSERVATION_KEYS],
                            axis=1,
                        )
                    )

                if not action_queue:
                    policy_observation = stack_observation_history(
                        history,
                        n_obs_steps=self.n_obs_steps,
                        device=policy.device,
                        dtype=policy.dtype,
                    )
                    with torch.no_grad():
                        prediction = policy.predict_action(policy_observation)["action"]
                    action_chunk = prediction.detach().cpu().numpy()
                    if action_chunk.ndim == 3 and action_chunk.shape[0] == 1:
                        action_chunk = action_chunk[0]
                    if action_chunk.ndim != 2:
                        raise ValueError(
                            "Expected policy action chunk [T,D] or [1,T,D], got "
                            f"{action_chunk.shape}"
                        )
                    action_queue.extend(action_chunk[: self.n_action_steps])

                action = adapt_policy_action(action_queue.popleft(), env.action_dim)
                if self.clip_actions:
                    action = np.clip(action, env.action_spec[0], env.action_spec[1])
                observation, _, _, _ = env.step(action)
                episode_length += 1
                if env._check_success():
                    success = True
                    break
        finally:
            env.close()

        self._save_video(frames, episode_index, success)
        return {
            "episode": episode_index,
            "success": success,
            "length": episode_length,
            "task_description": task_description,
        }

    def run(self, policy) -> dict[str, float]:
        results = []
        for episode_index in range(self.n_test):
            result = self._run_episode(policy, episode_index)
            results.append(result)
            print(
                f"RoboCasa episode {episode_index + 1}/{self.n_test}: "
                f"success={result['success']} length={result['length']}"
            )

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "robocasa_episodes.json").open("w") as result_file:
            json.dump(results, result_file, indent=2)

        successes = np.asarray([item["success"] for item in results], dtype=np.float32)
        lengths = np.asarray([item["length"] for item in results], dtype=np.float32)
        success_rate = float(successes.mean())
        return {
            "test/mean_score": success_rate,
            "test/success_rate": success_rate,
            "test/num_successes": float(successes.sum()),
            "test/num_episodes": float(len(results)),
            "test/mean_episode_length": float(lengths.mean()),
        }
