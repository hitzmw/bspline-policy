"""Configuration contract for the pure Diffusion Policy RoboCasa baseline."""

from pathlib import Path

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    RobomimicReplayImageDataset,
)


CONFIG_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "bspline_policy" / "config"
)


def _compose(config_name: str):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIRECTORY),
    ):
        config = compose(config_name=config_name)
    OmegaConf.resolve(config)
    return config


def test_pure_diffusion_matches_original_bspline_control_settings():
    diffusion = _compose(
        "train_diffusion_unet_turn_off_sink_faucet_image_workspace"
    )
    bspline = _compose(
        "train_diffusion_unet_turn_off_sink_faucet_image_bspline_workspace"
    )

    assert diffusion.task.dataset.dataset_path == bspline.task.dataset.dataset_path
    assert diffusion.horizon == bspline.horizon == 16
    assert diffusion.n_obs_steps == bspline.n_obs_steps == 2
    assert diffusion.n_action_steps == bspline.n_action_steps == 8
    assert diffusion.policy.n_action_steps == 8
    assert list(diffusion.policy.crop_shape) == list(bspline.policy.crop_shape)
    assert list(diffusion.policy.down_dims) == list(bspline.policy.down_dims)
    assert diffusion.policy.num_inference_steps == bspline.policy.num_inference_steps == 16
    assert diffusion.dataloader.batch_size == bspline.dataloader.batch_size == 8
    assert diffusion.optimizer.lr == bspline.optimizer.lr
    assert diffusion.training.seed == bspline.training.seed == 42
    assert diffusion.training.num_epochs == bspline.training.num_epochs == 200
    assert diffusion.training.lr_scheduler == bspline.training.lr_scheduler
    assert diffusion.checkpoint.topk.monitor_key == "val_loss"


def test_pure_diffusion_robocasa_targets_are_importable():
    config = _compose(
        "train_diffusion_unet_turn_off_sink_faucet_image_workspace"
    )

    workspace_class = hydra.utils.get_class(config._target_)
    policy_class = hydra.utils.get_class(config.policy._target_)
    dataset_class = hydra.utils.get_class(config.task.dataset._target_)

    assert workspace_class.__name__ == "TrainDiffusionUnetHybridWorkspace"
    assert policy_class.__name__ == "DiffusionUnetHybridImagePolicy"
    assert dataset_class.__name__ == "RobomimicReplayImageDataset"
    assert list(config.task.dataset.action_indices) == list(range(7))
    assert config.task.dataset.pad_before == 1
    assert config.task.dataset.pad_after == 7


def test_standard_dataset_normalizes_robocasa_lowdim_keys():
    dataset = RobomimicReplayImageDataset.__new__(
        RobomimicReplayImageDataset
    )
    dataset.replay_buffer = {
        "action": np.zeros((4, 7), dtype=np.float32),
        "ee_pos": np.zeros((4, 3), dtype=np.float32),
        "ee_ori": np.zeros((4, 3), dtype=np.float32),
        "gripper_states": np.zeros((4, 2), dtype=np.float32),
    }
    dataset.abs_action = False
    dataset.use_legacy_normalizer = False
    dataset.lowdim_keys = ["ee_pos", "ee_ori", "gripper_states"]
    dataset.rgb_keys = []

    normalizer = dataset.get_normalizer()

    assert set(normalizer.params_dict.keys()) == {
        "action",
        "ee_pos",
        "ee_ori",
        "gripper_states",
    }
