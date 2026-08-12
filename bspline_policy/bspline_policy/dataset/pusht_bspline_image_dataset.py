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


class PushTBSplineImageDataset(BaseImageDataset):
    """Push-T image observations paired with B-spline action parameters."""

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        n_obs_steps: int = 2,
        chunk_size: int = 10,
        bspline_degree: int = 3,
        max_error: float = 1.0,
        stride: int = 1,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: int | None = None,
        relative_knots: bool = False,
        cache_base_path: str | None = None,
        raw_action_steps: int = 8,
    ):
        super().__init__()
        zarr_path = os.path.expanduser(zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            store=zarr.MemoryStore(),
            keys=["img", "state", "action"],
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
        self.raw_action_steps = int(raw_action_steps)
        self.cache_base_path = (
            os.path.expanduser(cache_base_path) if cache_base_path else None
        )
        self.train_mask = train_mask
        self.n_action_steps = self.chunk_size + 2 * self.bspline_degree
        self.n_action_channels = 1 + int(self.replay_buffer["action"].shape[-1])
        self.key_first_k = {
            "img": self.n_obs_steps,
            "state": self.n_obs_steps,
        }

        if self.n_action_steps != self.horizon:
            raise ValueError(
                "B-spline parameter length must match the policy horizon: "
                f"{self.n_action_steps} != {self.horizon}"
            )

        self.sampler = self._make_sampler(self.train_mask)
        if len(self.sampler) == 0:
            raise RuntimeError("Push-T B-spline training split is empty")

        action_shape = self.sampler.sample_sequence(0)["action"].shape
        expected_shape = (self.n_action_steps, self.n_action_channels)
        if action_shape != expected_shape:
            raise AssertionError(
                f"Expected B-spline action shape {expected_shape}, got {action_shape}"
            )
        print(f"Push-T B-spline action shape: {action_shape}")

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
            keys=["img", "state", "action"],
            key_first_k=self.key_first_k,
            action_key="action",
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
        normalizer["action"] = get_range_normalizer_from_stat(stat)

        raw_action_stats = array_to_stats(self.replay_buffer["action"])
        normalizer["raw_action"] = get_range_normalizer_from_stat(
            raw_action_stats
        )

        agent_pos_stats = array_to_stats(self.replay_buffer["state"][:, :2])
        normalizer["agent_pos"] = get_range_normalizer_from_stat(agent_pos_stats)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.sampler.all_actions)

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        reconstruction = self.sampler.sample_reconstruction_sequence(
            idx,
            num_actions=self.raw_action_steps,
            action_params=sample["action"],
        )
        image = np.moveaxis(sample["img"], -1, 1).astype(np.float32)
        image *= 1.0 / 255.0
        data = {
            "obs": {
                "image": np.ascontiguousarray(image),
                "agent_pos": sample["state"][:, :2].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
            "raw_action": reconstruction["raw_action"].astype(np.float32),
            "raw_action_time": reconstruction["raw_action_time"],
            "raw_action_mask": reconstruction["raw_action_mask"],
            "raw_action_episode_mask": reconstruction[
                "raw_action_episode_mask"
            ],
        }
        return dict_apply(data, torch.from_numpy)
