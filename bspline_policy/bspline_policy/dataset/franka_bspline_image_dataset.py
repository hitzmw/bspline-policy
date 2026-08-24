"""Franka real-robot pick-and-place (zarr) with B-spline action parameters.

Data: /root/autodl-tmp/data/franka_collect.zarr
  - 100 episodes, 20Hz (median dt 50ms)
  - D435_color: fixed third-person view [128,128,3]
  - D405_color: wrist (eye-in-hand) view [128,128,3]
  - ee_pose: [6] xyz + euler
  - gripper_position: scalar jaw width (0..0.08m)
  - control: [8] action = 7 absolute joint position targets + binary gripper
    command (+-1); control[t] tracks joint_positions[t+1] with corr ~0.99.

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
    ):
        super().__init__()
        zarr_path = os.path.expanduser(zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            store=zarr.MemoryStore(),
            keys=[
                "D435_color",
                "D405_color",
                "ee_pose",
                "gripper_position",
                "control",
            ],
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
        self.cache_base_path = (
            os.path.expanduser(cache_base_path) if cache_base_path else None
        )
        self.train_mask = train_mask
        self.n_action_steps = self.chunk_size + 2 * self.bspline_degree
        self.n_action_channels = 1 + int(self.replay_buffer["control"].shape[-1])
        self.key_first_k = {
            "D435_color": self.n_obs_steps,
            "D405_color": self.n_obs_steps,
            "ee_pose": self.n_obs_steps,
            "gripper_position": self.n_obs_steps,
        }

        if self.n_action_steps != self.horizon:
            raise ValueError(
                "B-spline parameter length must match the policy horizon: "
                f"{self.n_action_steps} != {self.horizon}"
            )

        self.sampler = self._make_sampler(self.train_mask)
        if len(self.sampler) == 0:
            raise RuntimeError("Franka B-spline training split is empty")

        action_shape = self.sampler.sample_sequence(0)["control"].shape
        expected_shape = (self.n_action_steps, self.n_action_channels)
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
                n_action_channels=self.n_action_channels,
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
            keys=[
                "D435_color",
                "D405_color",
                "ee_pose",
                "gripper_position",
                "control",
            ],
            key_first_k=self.key_first_k,
            action_key="control",
            n_action_steps=self.n_action_steps,
            n_action_channels=self.n_action_channels,
            relative_knots=self.relative_knots,
            cache_path=cache_path,
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
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
        channel_stats = {
            "min": np.min(action_stats["min"], axis=1, keepdims=True),
            "max": np.max(action_stats["max"], axis=1, keepdims=True),
            "mean": np.mean(action_stats["mean"], axis=1, keepdims=True),
            "std": np.mean(action_stats["std"], axis=1, keepdims=True),
        }
        stat = {
            key: np.broadcast_to(
                value,
                (1, self.n_action_steps, self.n_action_channels),
            ).reshape(-1)
            for key, value in channel_stats.items()
        }
        normalizer["action"] = get_range_normalizer_from_stat(f32(stat))

        ee_pose_stats = array_to_stats(self.replay_buffer["ee_pose"])
        normalizer["ee_pose"] = get_range_normalizer_from_stat(
            f32(ee_pose_stats)
        )
        gripper_stats = array_to_stats(
            np.asarray(self.replay_buffer["gripper_position"])[:, None]
        )
        normalizer["gripper_pos"] = get_range_normalizer_from_stat(
            f32(gripper_stats)
        )
        normalizer["sideview_image"] = get_image_range_normalizer()
        normalizer["wrist_image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.sampler.all_actions)

    def __len__(self) -> int:
        return len(self.sampler)

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
                "gripper_pos": sample["gripper_position"][..., None].astype(
                    np.float32
                ),
            },
            "action": sample["control"].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)
