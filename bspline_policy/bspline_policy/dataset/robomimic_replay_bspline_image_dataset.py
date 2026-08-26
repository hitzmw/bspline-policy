from typing import Dict
import copy
import os
import shutil

import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits

from bspline_policy.common.bspline_action import (
    BSplineChunkSampler,
    make_bspline_sampler_cache_path,
)
from bspline_policy.common.knots import decode_relative_knots
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.preprocessed_sample_cache import (
    build_preprocessed_sample_cache,
    get_preprocessed_cached_item,
    make_empty_preprocessed_sample_cache,
    normalize_rgb_cache_dtype,
)
from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    _convert_robomimic_to_replay,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def _cache_base_path(dataset_path: str, cache_suffix: str = None) -> str:
    if not cache_suffix:
        return dataset_path
    if not cache_suffix.startswith("."):
        cache_suffix = "." + cache_suffix
    return dataset_path + cache_suffix


class RobomimicReplayBSplineImageDataset(BaseImageDataset):
    """Robomimic image dataset that represents actions as B-spline chunks.

    This keeps the target repo's existing real bimanual conversion logic
    unchanged, then replaces the regular sequence sampler with
    BSplineChunkSampler.  The opt-in ``raw_action_concat`` mode appends a
    horizon of dense replay actions to every B-spline parameter row:
    ``[knot, control..., raw_action...]``.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        chunk_size=10,
        bspline_degree=3,
        max_error=0.002,
        stride=1,
        abs_action=False,
        rotation_rep="rotation_6d",
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        relative_knots=False,
        cache_suffix=None,
        cache_decoded_replay=False,
        cache_preprocessed_samples=False,
        cache_preprocessed_device="cpu",
        cache_preprocessed_share_memory=False,
        cache_preprocessed_rgb_dtype="float32",
        observation_history=False,
        cache_base_path=None,
        action_indices=None,
        raw_action_concat=False,
        raw_action_sampling_mode="current",
    ):
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        cache_preprocessed_rgb_dtype = normalize_rgb_cache_dtype(
            cache_preprocessed_rgb_dtype
        )

        replay_buffer = None
        if use_cache:
            replay_cache_base = _cache_base_path(
                cache_base_path or dataset_path,
                cache_suffix,
            )
            cache_zarr_path = replay_cache_base + ".zarr.zip"
            cache_directory = os.path.dirname(cache_zarr_path)
            if cache_directory:
                os.makedirs(cache_directory, exist_ok=True)
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                            action_indices=action_indices,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                        if cache_decoded_replay:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=replay_buffer.root.store, store=None
                            )
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        if cache_decoded_replay:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=None
                            )
                        else:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=zarr.MemoryStore()
                            )
                    print("Loaded!")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                action_indices=action_indices,
            )
            if cache_decoded_replay:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=replay_buffer.root.store, store=None
                )

        rgb_keys = []
        lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        n_action_steps = int(chunk_size) + 2 * int(bspline_degree)
        n_control_dims = shape_meta["action"]["shape"][0]
        n_bspline_action_channels = 1 + n_control_dims

        sampler_cache_path = None
        if use_cache:
            sampler_cache_path = make_bspline_sampler_cache_path(
                base_path=_cache_base_path(
                    cache_base_path or dataset_path,
                    cache_suffix,
                ),
                episode_mask=train_mask,
                key_first_k=key_first_k,
                chunk_size=chunk_size,
                degree=bspline_degree,
                max_error=max_error,
                stride=stride,
                n_action_steps=n_action_steps,
                n_action_channels=n_bspline_action_channels,
                relative_knots=relative_knots,
            )

        sampler = BSplineChunkSampler(
            replay_buffer=replay_buffer,
            chunk_size=chunk_size,
            degree=bspline_degree,
            max_error=max_error,
            stride=stride,
            episode_mask=train_mask,
            key_first_k=key_first_k,
            action_key="action",
            n_action_steps=n_action_steps,
            n_action_channels=n_bspline_action_channels,
            relative_knots=relative_knots,
            cache_path=sampler_cache_path,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_path = dataset_path
        self.cache_base_path = cache_base_path
        self.cache_suffix = cache_suffix
        self.use_cache = use_cache
        self.chunk_size = chunk_size
        self.bspline_degree = bspline_degree
        self.max_error = max_error
        self.stride = stride
        self.relative_knots = bool(relative_knots)
        self.action_indices = (
            None
            if action_indices is None
            else tuple(int(index) for index in action_indices)
        )
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
        self.regular_action_dim = int(n_control_dims)
        self.n_action_steps = n_action_steps
        self.n_bspline_action_channels = n_bspline_action_channels
        self.n_action_channels = n_bspline_action_channels + (
            self.regular_action_dim if self.raw_action_concat else 0
        )
        self.cache_decoded_replay = cache_decoded_replay
        self.cache_preprocessed_samples = cache_preprocessed_samples
        self.cache_preprocessed_device = cache_preprocessed_device
        self.cache_preprocessed_share_memory = cache_preprocessed_share_memory
        self.cache_preprocessed_rgb_dtype = cache_preprocessed_rgb_dtype
        self.observation_history = bool(observation_history)
        self._preprocessed_cache = None
        self._length = len(sampler)

        if len(sampler) > 0:
            action_shape = sampler.sample_sequence(0)["action"].shape
            expected_shape = (n_action_steps, n_bspline_action_channels)
            print(f"Action shape from B-spline sampler: {action_shape}")
            print(f"Expected shape: {expected_shape}")
            if action_shape != expected_shape:
                raise AssertionError(
                    f"Action shape mismatch: expected {expected_shape}, got {action_shape}"
                )

        if cache_preprocessed_samples:
            self._build_preprocessed_cache(
                device=cache_preprocessed_device,
                share_memory=cache_preprocessed_share_memory,
            )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        key_first_k = {}
        if self.n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys:
                key_first_k[key] = self.n_obs_steps

        sampler_cache_path = None
        if self.use_cache:
            sampler_cache_path = make_bspline_sampler_cache_path(
                base_path=_cache_base_path(
                    self.cache_base_path or self.dataset_path,
                    self.cache_suffix,
                ),
                episode_mask=~self.train_mask,
                key_first_k=key_first_k,
                chunk_size=self.chunk_size,
                degree=self.bspline_degree,
                max_error=self.max_error,
                stride=self.stride,
                n_action_steps=self.n_action_steps,
                n_action_channels=self.n_bspline_action_channels,
                relative_knots=self.relative_knots,
            )

        val_set.sampler = BSplineChunkSampler(
            replay_buffer=self.replay_buffer,
            chunk_size=self.chunk_size,
            degree=self.bspline_degree,
            max_error=self.max_error,
            stride=self.stride,
            episode_mask=~self.train_mask,
            key_first_k=key_first_k,
            action_key="action",
            n_action_steps=self.n_action_steps,
            n_action_channels=self.n_bspline_action_channels,
            relative_knots=self.relative_knots,
            cache_path=sampler_cache_path,
        )
        val_set.train_mask = ~self.train_mask
        val_set._length = len(val_set.sampler)
        if self._preprocessed_cache is not None:
            val_set._preprocessed_cache = None
            val_set._build_preprocessed_cache(
                device=self.cache_preprocessed_device,
                share_memory=self.cache_preprocessed_share_memory,
            )
        return val_set

    def __copy__(self):
        result = self.__class__.__new__(self.__class__)
        result.__dict__.update(self.__dict__)
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("_preprocessed_cache") is not None:
            state["replay_buffer"] = None
            state["sampler"] = None
        return state

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        action_stats = self.sampler.get_action_stats()
        bspline_channel_stats = {
            "min": np.min(action_stats["min"], axis=1, keepdims=True),
            "max": np.max(action_stats["max"], axis=1, keepdims=True),
            "mean": np.mean(action_stats["mean"], axis=1, keepdims=True),
            "std": np.mean(action_stats["std"], axis=1, keepdims=True),
        }
        n_action_steps = action_stats["min"].shape[1]
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
        n_channels = channel_stats["min"].shape[2]
        stat = {}
        for key in ["min", "max", "mean", "std"]:
            stat[key] = np.broadcast_to(
                channel_stats[key], (1, n_action_steps, n_channels)
            ).reshape(-1)
        normalizer["action"] = get_range_normalizer_from_stat(stat)

        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if "pos" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif "quat" in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif "qpos" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == "base_pose":
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif "ori" in key or "gripper" in key or "state" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"unsupported lowdim key: {key}")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

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

    def __len__(self):
        return self._length

    def _get_observation_history(self, idx: int, key: str) -> np.ndarray:
        """Return oldest-to-newest observations ending at the action timestep.

        ``BSplineChunkSampler`` historically returns the first observations at
        and after the action timestep. That is useful for the original replay
        tools, but an online Robomimic runner supplies past/current
        observations. The opt-in history path aligns training with that
        rollout contract and repeats the first episode observation for initial
        padding, matching ``SequenceSampler``.
        """
        timestep = int(self.sampler.valid_timesteps[idx])
        episode_index = int(
            np.searchsorted(
                self.sampler.episode_ends,
                timestep,
                side="right",
            )
        )
        episode_start = (
            0
            if episode_index == 0
            else int(self.sampler.episode_ends[episode_index - 1])
        )
        history_start = max(
            episode_start,
            timestep - int(self.n_obs_steps) + 1,
        )
        history = self.replay_buffer[key][history_start : timestep + 1]
        missing = int(self.n_obs_steps) - len(history)
        if missing > 0:
            history = np.concatenate(
                [np.repeat(history[:1], missing, axis=0), history],
                axis=0,
            )
        return history

    def _get_raw_action_sequence(
        self,
        idx: int,
        bspline_action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return dense actions aligned to the configured temporal contract.

        ``current`` preserves the original current-to-future sequence for old
        checkpoints. ``bspline_interval`` samples the replay trajectory at the
        same uniformly spaced phase points used when the associated B-spline
        is decoded, so the auxiliary target is time-aligned with rollout.
        """
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
            # Zarr v2 supports basic indexing but not NumPy index arrays. Read
            # the small contiguous episode once, then interpolate in memory.
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

    def _sample_uncached_item(
        self,
        idx: int,
        rgb_dtype: str = "float32",
    ) -> Dict[str, torch.Tensor]:
        if self.sampler is None:
            raise RuntimeError(
                "RobomimicReplayBSplineImageDataset was serialized without "
                "replay/sampler because cache_preprocessed_samples is enabled, "
                "but no preprocessed cache is available for this item."
        )
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        if self.observation_history:
            for key in self.rgb_keys + self.lowdim_keys:
                data[key] = self._get_observation_history(idx, key)
        t_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            image = np.moveaxis(data[key][t_slice], -1, 1)
            if rgb_dtype == "uint8":
                obs_dict[key] = np.ascontiguousarray(image, dtype=np.uint8)
            else:
                obs_dict[key] = np.ascontiguousarray(image, dtype=np.float32)
                obs_dict[key] *= 1.0 / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][t_slice].astype(np.float32)
            del data[key]

        action = self._make_action_target(idx, data["action"])

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }

    def _build_preprocessed_cache(self, device: str = "cpu", share_memory: bool = False):
        print(
            f"Building B-spline preprocessed sample cache: samples={len(self)}, "
            f"device={device}, share_memory={share_memory}, "
            f"rgb_dtype={self.cache_preprocessed_rgb_dtype}"
        )
        obs_steps = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        empty_cache = make_empty_preprocessed_sample_cache(
            shape_meta=self.shape_meta,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.lowdim_keys,
            action_shape=(self.n_action_steps, self.n_action_channels),
            obs_steps=obs_steps,
            device=torch.device(device),
            share_memory=share_memory,
            rgb_dtype=self.cache_preprocessed_rgb_dtype,
        )
        self._preprocessed_cache = build_preprocessed_sample_cache(
            length=len(self),
            sample_fn=lambda idx: self._sample_uncached_item(
                idx,
                rgb_dtype=self.cache_preprocessed_rgb_dtype,
            ),
            device=device,
            share_memory=share_memory,
            empty_cache=empty_cache,
            desc="Precomputing B-spline samples",
        )
        print("B-spline preprocessed sample cache ready.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._preprocessed_cache is not None:
            return get_preprocessed_cached_item(
                self._preprocessed_cache,
                idx,
                rgb_keys=self.rgb_keys,
            )
        return self._sample_uncached_item(idx)
