"""Franka real-robot Zarr data with B-spline action parameters.

  - D435_color: fixed third-person view [128,128,3] (D455 in wipe_bb)
  - D405_color: wrist (eye-in-hand) view [128,128,3]
  - ee_pose: [6] xyz + euler
  - joint_positions: optional [7] joint-state observation
  - gripper_position: optional scalar jaw width (0..0.08m)
  - control: [7] absolute joint targets, or [8] including gripper command

Cloned from PushTBSplineImageDataset with Franka keys. The action for
B-spline fitting is the raw ``control`` vector (absolute joint space), so
deployment replays decoded actions directly through the same controller.
"""

from __future__ import annotations

import copy
import os
from typing import Dict

import numpy as np
import torch
import zarr

from bspline_policy.common.bspline_action import (
    BSplineChunkSampler,
    make_bspline_sampler_cache_path,
)
from bspline_policy.common.knots import decode_relative_knots
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class FrankaBSplineImageDataset(BaseImageDataset):
    """Franka dual-camera observations paired with B-spline action parameters."""

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        n_obs_steps: int = 2,
        chunk_size: int = 10,
        bspline_degree: int = 3,
        max_error: float = 0.01,
        stride: int = 1,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: int | None = None,
        relative_knots: bool = False,
        cache_base_path: str | None = None,
        include_joint_positions: bool = False,
        raw_action_concat: bool = False,
        raw_action_sampling_mode: str = "current",
    ):
        super().__init__()
        zarr_path = os.path.expanduser(zarr_path)
        source = zarr.open_group(zarr_path, mode="r")
        self.has_gripper = "gripper_position" in source["data"]
        self.include_joint_positions = bool(include_joint_positions)
        self.replay_keys = ["D435_color", "D405_color", "ee_pose", "control"]
        if self.has_gripper:
            self.replay_keys.append("gripper_position")
        if self.include_joint_positions:
            if "joint_positions" not in source["data"]:
                raise ValueError("include_joint_positions requires data/joint_positions")
            self.replay_keys.append("joint_positions")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            store=zarr.MemoryStore(),
            keys=self.replay_keys,
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.zarr_path = zarr_path
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.chunk_size = int(chunk_size)
        self.bspline_degree = int(bspline_degree)
        self.max_error = float(max_error)
        self.stride = int(stride)
        self.relative_knots = bool(relative_knots)
        self.raw_action_concat = bool(raw_action_concat)
        self.raw_action_sampling_mode = str(raw_action_sampling_mode)
        if self.raw_action_sampling_mode not in {"current", "bspline_interval"}:
            raise ValueError(
                "raw_action_sampling_mode must be 'current' or 'bspline_interval'"
            )
        if not self.raw_action_concat and self.raw_action_sampling_mode != "current":
            raise ValueError(
                "raw_action_sampling_mode='bspline_interval' requires raw_action_concat=True"
            )
        self.cache_base_path = (
            os.path.expanduser(cache_base_path) if cache_base_path else None
        )
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.n_action_steps = self.chunk_size + 2 * self.bspline_degree
        self.regular_action_dim = int(self.replay_buffer["control"].shape[-1])
        self.n_bspline_action_channels = 1 + self.regular_action_dim
        self.n_action_channels = self.n_bspline_action_channels + (
            self.regular_action_dim if self.raw_action_concat else 0
        )
        self.key_first_k = {
            "D435_color": self.n_obs_steps,
            "D405_color": self.n_obs_steps,
            "ee_pose": self.n_obs_steps,
        }
        if self.has_gripper:
            self.key_first_k["gripper_position"] = self.n_obs_steps
        if self.include_joint_positions:
            self.key_first_k["joint_positions"] = self.n_obs_steps

        if self.n_action_steps != self.horizon:
            raise ValueError(
                "B-spline parameter length must match the policy horizon: "
                f"{self.n_action_steps} != {self.horizon}"
            )

        self.sampler = self._make_sampler(self.train_mask)
        if len(self.sampler) == 0:
            raise RuntimeError("Franka B-spline training split is empty")

        action_shape = self.sampler.sample_sequence(0)["control"].shape
        expected_shape = (self.n_action_steps, self.n_bspline_action_channels)
        if action_shape != expected_shape:
            raise AssertionError(
                f"Expected B-spline action shape {expected_shape}, got {action_shape}"
            )
        print(f"Franka B-spline action shape: {action_shape}")

    def _make_sampler(self, episode_mask: np.ndarray) -> BSplineChunkSampler:
        cache_path = None
        if self.cache_base_path:
            cache_path = make_bspline_sampler_cache_path(
                base_path=self.cache_base_path,
                episode_mask=episode_mask,
                key_first_k=self.key_first_k,
                chunk_size=self.chunk_size,
                degree=self.bspline_degree,
                max_error=self.max_error,
                stride=self.stride,
                n_action_steps=self.n_action_steps,
                n_action_channels=self.n_bspline_action_channels,
                relative_knots=self.relative_knots,
            )
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

        return BSplineChunkSampler(
            replay_buffer=self.replay_buffer,
            chunk_size=self.chunk_size,
            degree=self.bspline_degree,
            max_error=self.max_error,
            stride=self.stride,
            episode_mask=episode_mask,
            keys=self.replay_keys,
            key_first_k=self.key_first_k,
            action_key="control",
            n_action_steps=self.n_action_steps,
            n_action_channels=self.n_bspline_action_channels,
            relative_knots=self.relative_knots,
            cache_path=cache_path,
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask
        val_set.sampler = val_set._make_sampler(val_set.train_mask)
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        del kwargs
        normalizer = LinearNormalizer()

        def f32(stats):
            # zarr stores ee_pose/control as float64; keep all normalizer
            # params float32 so normalized tensors stay float32.
            return {
                key: np.asarray(value, dtype=np.float32)
                for key, value in stats.items()
            }

        action_stats = self.sampler.get_action_stats()
        bspline_channel_stats = {
            "min": np.min(action_stats["min"], axis=1, keepdims=True),
            "max": np.max(action_stats["max"], axis=1, keepdims=True),
            "mean": np.mean(action_stats["mean"], axis=1, keepdims=True),
            "std": np.mean(action_stats["std"], axis=1, keepdims=True),
        }
        if self.raw_action_concat:
            raw_stats = self._get_training_stats("control")
            channel_stats = {
                key: np.concatenate(
                    [bspline_channel_stats[key].reshape(-1), raw_stats[key].reshape(-1)]
                ).reshape(1, 1, -1)
                for key in ("min", "max", "mean", "std")
            }
        else:
            channel_stats = bspline_channel_stats
        stat = {
            key: np.broadcast_to(
                value,
                (1, self.n_action_steps, self.n_action_channels),
            ).reshape(-1)
            for key, value in channel_stats.items()
        }
        normalizer["action"] = get_range_normalizer_from_stat(f32(stat))

        ee_pose_stats = self._get_training_stats("ee_pose")
        normalizer["ee_pose"] = get_range_normalizer_from_stat(
            f32(ee_pose_stats)
        )
        if self.has_gripper:
            gripper_stats = self._get_training_stats("gripper_position")
            normalizer["gripper_pos"] = get_range_normalizer_from_stat(
                f32(gripper_stats)
            )
        if self.include_joint_positions:
            normalizer["joint_positions"] = get_range_normalizer_from_stat(
                f32(self._get_training_stats("joint_positions"))
            )
        normalizer["sideview_image"] = get_image_range_normalizer()
        normalizer["wrist_image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        if self.raw_action_concat:
            actions = [
                self._make_action_target(
                    idx, self.sampler.all_actions[self.sampler.timestep_to_chunk[timestep]]
                )
                for idx, timestep in enumerate(self.sampler.valid_timesteps)
            ]
            return torch.from_numpy(np.stack(actions))
        return torch.from_numpy(self.sampler.all_actions)

    def __len__(self) -> int:
        return len(self.sampler)

    def _get_training_stats(self, key: str) -> dict[str, np.ndarray]:
        """Fit low-dimensional normalizers on the selected episodes only."""
        ends = np.asarray(self.replay_buffer.episode_ends, dtype=np.int64)
        starts = np.r_[0, ends[:-1]]
        arrays = [
            np.asarray(self.replay_buffer[key][start:end], dtype=np.float32)
            for start, end, selected in zip(starts, ends, self.train_mask)
            if selected and end > start
        ]
        if not arrays:
            raise RuntimeError("Cannot normalize observations/actions from an empty split")
        values = np.concatenate(arrays, axis=0)
        if values.ndim == 1:
            values = values[:, None]
        return array_to_stats(values)

    def _get_raw_action_sequence(
        self,
        idx: int,
        bspline_action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample absolute joint targets without crossing episode boundaries.

        Interval mode matches the phase points used by the differentiable
        consistency decoder. Inference still uses Franka integer control ticks.
        """
        timestep = int(self.sampler.valid_timesteps[idx])
        episode_index = int(
            np.searchsorted(self.sampler.episode_ends, timestep, side="right")
        )
        episode_start = (
            0 if episode_index == 0 else int(self.sampler.episode_ends[episode_index - 1])
        )
        episode_end = int(self.sampler.episode_ends[episode_index])

        if self.raw_action_sampling_mode == "bspline_interval":
            if bspline_action is None:
                raise ValueError("bspline_action is required for interval-aligned raw actions")
            parameters = np.asarray(bspline_action, dtype=np.float32)
            if self.relative_knots:
                parameters = decode_relative_knots(parameters, degree=self.bspline_degree)
            knots = parameters[:, 0]
            t_min = float(knots[self.bspline_degree])
            t_max = float(knots[-(self.bspline_degree + 1)])
            if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
                raise ValueError(f"Invalid B-spline raw-action interval: [{t_min}, {t_max}]")

            positions = timestep + np.linspace(
                t_min, t_max, self.n_action_steps, dtype=np.float64
            )
            positions = np.clip(positions, episode_start, episode_end - 1)
            left_indices = np.floor(positions).astype(np.int64)
            right_indices = np.minimum(left_indices + 1, episode_end - 1)
            weights = (positions - left_indices).reshape(-1, 1)
            # Read only the covered action range; RGB is never needed here.
            start = int(left_indices.min())
            end = int(right_indices.max()) + 1
            controls = np.asarray(self.replay_buffer["control"][start:end], dtype=np.float32)
            left = controls[left_indices - start]
            right = controls[right_indices - start]
            actions = (left + weights * (right - left)).astype(np.float32)
        else:
            end = min(timestep + self.n_action_steps, episode_end)
            actions = np.asarray(self.replay_buffer["control"][timestep:end], dtype=np.float32)
            if len(actions) == 0:
                raise RuntimeError(f"No raw action is available at timestep {timestep}")
            missing = self.n_action_steps - len(actions)
            if missing > 0:
                actions = np.concatenate(
                    [actions, np.repeat(actions[-1:], missing, axis=0)], axis=0
                )

        expected_shape = (self.n_action_steps, self.regular_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"Expected raw action shape {expected_shape}, got {actions.shape}")
        return actions

    def _make_action_target(self, idx: int, bspline_action: np.ndarray) -> np.ndarray:
        bspline_action = np.asarray(bspline_action, dtype=np.float32)
        if not self.raw_action_concat:
            return bspline_action
        raw_action = self._get_raw_action_sequence(idx, bspline_action)
        return np.concatenate([bspline_action, raw_action], axis=-1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        sideview = np.moveaxis(sample["D435_color"], -1, 1).astype(np.float32)
        sideview *= 1.0 / 255.0
        wrist = np.moveaxis(sample["D405_color"], -1, 1).astype(np.float32)
        wrist *= 1.0 / 255.0
        data = {
            "obs": {
                "sideview_image": np.ascontiguousarray(sideview),
                "wrist_image": np.ascontiguousarray(wrist),
                "ee_pose": sample["ee_pose"].astype(np.float32),
            },
            "action": self._make_action_target(idx, sample["control"]),
        }
        if self.has_gripper:
            data["obs"]["gripper_pos"] = sample["gripper_position"].reshape(
                self.n_obs_steps, 1
            ).astype(np.float32)
        if self.include_joint_positions:
            data["obs"]["joint_positions"] = sample["joint_positions"].astype(np.float32)
        return dict_apply(data, torch.from_numpy)
