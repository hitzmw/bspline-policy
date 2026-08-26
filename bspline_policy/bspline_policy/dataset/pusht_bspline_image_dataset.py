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
        raw_action_concat: bool = False,
        raw_action_sampling_mode: str = "current",
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
        self.raw_action_concat = bool(raw_action_concat)
        self.raw_action_sampling_mode = str(raw_action_sampling_mode)
        if self.raw_action_sampling_mode not in {
            "current",
            "bspline_interval",
        }:
            raise ValueError(
                "raw_action_sampling_mode must be 'current' or "
                "'bspline_interval'"
            )
        if (
            not self.raw_action_concat
            and self.raw_action_sampling_mode != "current"
        ):
            raise ValueError(
                "raw_action_sampling_mode='bspline_interval' requires "
                "raw_action_concat=True"
            )
        self.cache_base_path = (
            os.path.expanduser(cache_base_path) if cache_base_path else None
        )
        self.train_mask = train_mask
        self.n_action_steps = self.chunk_size + 2 * self.bspline_degree
        self.regular_action_dim = int(self.replay_buffer["action"].shape[-1])
        self.n_bspline_action_channels = 1 + self.regular_action_dim
        self.n_action_channels = self.n_bspline_action_channels + (
            self.regular_action_dim if self.raw_action_concat else 0
        )
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
        expected_shape = (
            self.n_action_steps,
            self.n_bspline_action_channels,
        )
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
            keys=["img", "state", "action"],
            key_first_k=self.key_first_k,
            action_key="action",
            n_action_steps=self.n_action_steps,
            n_action_channels=self.n_bspline_action_channels,
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
        bspline_channel_stats = {
            "min": np.min(action_stats["min"], axis=1, keepdims=True),
            "max": np.max(action_stats["max"], axis=1, keepdims=True),
            "mean": np.mean(action_stats["mean"], axis=1, keepdims=True),
            "std": np.mean(action_stats["std"], axis=1, keepdims=True),
        }
        if self.raw_action_concat:
            raw_action_stats = self._get_training_raw_action_stats()
            channel_stats = {
                key: np.concatenate(
                    [
                        bspline_channel_stats[key].reshape(-1),
                        raw_action_stats[key].reshape(-1),
                    ]
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
        normalizer["action"] = get_range_normalizer_from_stat(stat)

        agent_pos_stats = array_to_stats(self.replay_buffer["state"][:, :2])
        normalizer["agent_pos"] = get_range_normalizer_from_stat(agent_pos_stats)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        if self.raw_action_concat:
            actions = [
                self._make_action_target(
                    idx,
                    self.sampler.sample_sequence(idx)["action"],
                )
                for idx in range(len(self))
            ]
            return torch.from_numpy(np.stack(actions))
        return torch.from_numpy(self.sampler.all_actions)

    def __len__(self) -> int:
        return len(self.sampler)

    def _get_raw_action_sequence(
        self,
        idx: int,
        bspline_action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return dense actions aligned with the B-spline decode interval."""
        timestep = int(self.sampler.valid_timesteps[idx])
        episode_index = int(
            np.searchsorted(self.sampler.episode_ends, timestep, side="right")
        )
        episode_start = (
            0
            if episode_index == 0
            else int(self.sampler.episode_ends[episode_index - 1])
        )
        episode_end = int(self.sampler.episode_ends[episode_index])

        if self.raw_action_sampling_mode == "bspline_interval":
            if bspline_action is None:
                raise ValueError(
                    "bspline_action is required for interval-aligned raw actions"
                )
            parameters = np.asarray(bspline_action, dtype=np.float32)
            if self.relative_knots:
                parameters = decode_relative_knots(
                    parameters,
                    degree=self.bspline_degree,
                )
            knots = parameters[:, 0]
            t_min = float(knots[self.bspline_degree])
            t_max = float(knots[-(self.bspline_degree + 1)])
            if not np.isfinite(t_min) or not np.isfinite(t_max):
                raise ValueError("B-spline knot interval must be finite")
            if t_max <= t_min:
                raise ValueError(
                    f"Invalid B-spline raw-action interval: [{t_min}, {t_max}]"
                )

            positions = timestep + np.linspace(
                t_min,
                t_max,
                self.n_action_steps,
                dtype=np.float32,
            )
            positions = np.clip(
                positions,
                float(episode_start),
                float(episode_end - 1),
            )
            left_indices = np.floor(positions).astype(np.int64)
            right_indices = np.minimum(left_indices + 1, episode_end - 1)
            interpolation_weight = (positions - left_indices).reshape(-1, 1)
            episode_actions = np.asarray(
                self.replay_buffer["action"][episode_start:episode_end],
                dtype=np.float32,
            )
            left_actions = episode_actions[left_indices - episode_start]
            right_actions = episode_actions[right_indices - episode_start]
            actions = (
                left_actions
                + interpolation_weight * (right_actions - left_actions)
            ).astype(np.float32)
        else:
            sequence_end = min(timestep + self.n_action_steps, episode_end)
            actions = np.asarray(
                self.replay_buffer["action"][timestep:sequence_end],
                dtype=np.float32,
            )
            if len(actions) == 0:
                raise RuntimeError(
                    f"No raw action is available at timestep {timestep}"
                )
            missing = self.n_action_steps - len(actions)
            if missing > 0:
                actions = np.concatenate(
                    [actions, np.repeat(actions[-1:], missing, axis=0)],
                    axis=0,
                )

        expected_shape = (self.n_action_steps, self.regular_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Expected raw action shape {expected_shape}, got {actions.shape}"
            )
        return actions

    def _make_action_target(
        self,
        idx: int,
        bspline_action: np.ndarray,
    ) -> np.ndarray:
        bspline_action = np.asarray(bspline_action, dtype=np.float32)
        if not self.raw_action_concat:
            return bspline_action
        raw_action = self._get_raw_action_sequence(idx, bspline_action)
        return np.concatenate([bspline_action, raw_action], axis=-1)

    def _get_training_raw_action_stats(self) -> dict[str, np.ndarray]:
        """Compute dense-action statistics from training episodes only."""
        episode_ends = np.asarray(self.sampler.episode_ends, dtype=np.int64)
        episode_mask = np.asarray(self.sampler.episode_mask, dtype=bool)
        episode_starts = np.concatenate(
            [np.zeros(1, dtype=np.int64), episode_ends[:-1]]
        )
        training_actions = [
            np.asarray(
                self.replay_buffer["action"][start:end],
                dtype=np.float32,
            )
            for start, end, selected in zip(
                episode_starts,
                episode_ends,
                episode_mask,
            )
            if selected and end > start
        ]
        if not training_actions:
            raise RuntimeError("Cannot normalize raw actions from an empty split")
        return array_to_stats(np.concatenate(training_actions, axis=0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        image = np.moveaxis(sample["img"], -1, 1).astype(np.float32)
        image *= 1.0 / 255.0
        data = {
            "obs": {
                "image": np.ascontiguousarray(image),
                "agent_pos": sample["state"][:, :2].astype(np.float32),
            },
            "action": self._make_action_target(idx, sample["action"]),
        }
        return dict_apply(data, torch.from_numpy)
