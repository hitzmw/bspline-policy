"""Push-T dataset contracts for Drifting-BSpline raw-action consistency."""

from __future__ import annotations

import numpy as np
import torch

from bspline_policy.dataset.pusht_bspline_image_dataset import (
    PushTBSplineImageDataset,
)
from diffusion_policy.common.normalize_util import array_to_stats


def _stub_dataset() -> PushTBSplineImageDataset:
    dataset = PushTBSplineImageDataset.__new__(PushTBSplineImageDataset)
    dataset.raw_action_concat = True
    dataset.raw_action_sampling_mode = "current"
    dataset.n_action_steps = 4
    dataset.regular_action_dim = 2
    dataset.n_bspline_action_channels = 3
    dataset.n_action_channels = 5
    dataset.bspline_degree = 1
    dataset.relative_knots = False
    dataset.replay_buffer = {
        "action": np.arange(10, dtype=np.float32).reshape(5, 2),
        "state": np.arange(15, dtype=np.float32).reshape(5, 3),
    }

    class Sampler:
        valid_timesteps = np.asarray([1, 3])
        episode_ends = np.asarray([3, 5])
        episode_mask = np.asarray([True, False])

        @staticmethod
        def get_action_stats():
            shape = (1, 4, 3)
            return {
                "min": np.zeros(shape, dtype=np.float32),
                "max": np.ones(shape, dtype=np.float32),
                "mean": np.full(shape, 0.5, dtype=np.float32),
                "std": np.full(shape, 0.25, dtype=np.float32),
            }

    dataset.sampler = Sampler()
    return dataset


def test_current_raw_actions_are_padded_inside_each_episode():
    dataset = _stub_dataset()

    np.testing.assert_array_equal(
        dataset._get_raw_action_sequence(0),
        np.asarray(
            [[2, 3], [4, 5], [4, 5], [4, 5]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        dataset._get_raw_action_sequence(1),
        np.asarray(
            [[6, 7], [8, 9], [8, 9], [8, 9]],
            dtype=np.float32,
        ),
    )


def test_interval_raw_actions_form_a_five_channel_target():
    dataset = _stub_dataset()
    dataset.raw_action_sampling_mode = "bspline_interval"
    bspline = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    target = dataset._make_action_target(0, bspline)
    expected_raw = np.asarray(
        [
            [2.0, 3.0],
            [2.0 + 2.0 / 3.0, 3.0 + 2.0 / 3.0],
            [2.0 + 4.0 / 3.0, 3.0 + 4.0 / 3.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )

    assert target.shape == (4, 5)
    np.testing.assert_array_equal(target[:, :3], bspline)
    np.testing.assert_allclose(target[:, 3:], expected_raw)


def test_joint_normalizer_uses_training_raw_actions_only():
    dataset = _stub_dataset()
    actual_stats = dataset._get_training_raw_action_stats()
    expected_stats = array_to_stats(dataset.replay_buffer["action"][:3])
    for key in expected_stats:
        np.testing.assert_allclose(actual_stats[key], expected_stats[key])

    target = torch.from_numpy(
        dataset._make_action_target(
            0,
            np.arange(12, dtype=np.float32).reshape(4, 3),
        )
    )
    normalizer = dataset.get_normalizer()
    normalized = normalizer["action"].normalize(target)
    reconstructed = normalizer["action"].unnormalize(normalized)

    assert normalized.shape == (4, 5)
    torch.testing.assert_close(reconstructed, target)
