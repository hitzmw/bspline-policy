"""Configuration and observation-history tests for Drifting-BSpline Can."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from bspline_policy.dataset.robomimic_replay_bspline_image_dataset import (
    RobomimicReplayBSplineImageDataset,
)


CONFIG_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "bspline_policy" / "config"
)


def _compose_config():
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIRECTORY),
    ):
        config = compose(
            config_name=(
                "train_drifting_unet_can_image_bspline_workspace"
            )
        )
    OmegaConf.resolve(config)
    return config


def test_can_config_combines_reference_drifting_and_bspline_contracts():
    config = _compose_config()

    assert config.dataloader.batch_size == 64
    assert config.policy.gen_per_label == 8
    assert config.training.gradient_accumulate_every == 1
    assert config.training.num_epochs == 200
    assert config.training.lr_scheduler == "constant_with_warmup"

    assert list(config.shape_meta.action.shape) == [7]
    assert config.horizon == 16
    assert config.policy.n_action_steps == 16
    assert config.policy.execution_action_steps == 8
    assert config.task.dataset.chunk_size == 10
    assert config.task.dataset.bspline_degree == 3
    assert config.task.dataset.max_error == 0.002
    assert config.task.dataset.abs_action is False
    assert config.task.dataset.observation_history is True
    assert config.task.dataset.cache_base_path == (
        "data/cache/robomimic/can_ph_image"
    )
    assert config.task.env_runner.n_action_steps == 8


def test_can_hydra_targets_are_importable():
    config = _compose_config()

    workspace_class = hydra.utils.get_class(config._target_)
    policy_class = hydra.utils.get_class(config.policy._target_)
    dataset_class = hydra.utils.get_class(config.task.dataset._target_)

    assert workspace_class.__name__ == "TrainDriftingBSplineImageWorkspace"
    assert policy_class.__name__ == (
        "DriftingUnetRobomimicBSplineImagePolicy"
    )
    assert dataset_class.__name__ == "RobomimicReplayBSplineImageDataset"
    assert config.task.env_runner._target_.endswith(
        "RobomimicBSplineImageRunner"
    )


def test_observation_history_is_padded_and_does_not_cross_episodes():
    dataset = RobomimicReplayBSplineImageDataset.__new__(
        RobomimicReplayBSplineImageDataset
    )
    dataset.n_obs_steps = 2
    dataset.replay_buffer = {
        "state": np.arange(10, dtype=np.float32).reshape(5, 2)
    }

    class Sampler:
        valid_timesteps = np.asarray([0, 1, 3, 4])
        episode_ends = np.asarray([3, 5])

    dataset.sampler = Sampler()

    np.testing.assert_array_equal(
        dataset._get_observation_history(0, "state"),
        np.asarray([[0, 1], [0, 1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dataset._get_observation_history(1, "state"),
        np.asarray([[0, 1], [2, 3]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        dataset._get_observation_history(2, "state"),
        np.asarray([[6, 7], [6, 7]], dtype=np.float32),
    )
