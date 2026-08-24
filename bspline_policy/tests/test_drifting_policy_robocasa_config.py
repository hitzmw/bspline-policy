"""Contracts for the direct-action Drifting RoboCasa ablation."""

from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


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


def test_direct_drifting_has_no_bspline_targets():
    config = _compose(
        "train_drifting_unet_turn_off_sink_faucet_image_workspace"
    )

    workspace_class = hydra.utils.get_class(config._target_)
    policy_class = hydra.utils.get_class(config.policy._target_)
    dataset_class = hydra.utils.get_class(config.task.dataset._target_)

    assert workspace_class.__name__ == "TrainDriftingUnetHybridWorkspace"
    assert policy_class.__name__ == "DriftingUnetHybridImagePolicy"
    assert dataset_class.__name__ == "RobomimicReplayImageDataset"
    policy_module = config.policy._target_.split(".")[-2]
    dataset_module = config.task.dataset._target_.split(".")[-2]
    assert "bspline" not in policy_module.lower()
    assert "bspline" not in dataset_module.lower()
    assert list(config.shape_meta.action.shape) == [7]
    assert config.policy.horizon == 16
    assert config.policy.n_action_steps == 8
    assert list(config.task.dataset.action_indices) == list(range(7))
    assert config.task.dataset.pad_before == 1
    assert config.task.dataset.pad_after == 7


def test_direct_drifting_is_a_controlled_bspline_ablation():
    direct = _compose(
        "train_drifting_unet_turn_off_sink_faucet_image_workspace"
    )
    bspline = _compose(
        "train_drifting_unet_turn_off_sink_faucet_image_bspline_workspace"
    )

    assert direct.task.dataset.dataset_path == bspline.task.dataset.dataset_path
    assert direct.horizon == bspline.horizon == 16
    assert direct.n_obs_steps == bspline.n_obs_steps == 2
    assert direct.n_action_steps == bspline.n_action_steps == 8
    assert list(direct.policy.crop_shape) == list(bspline.policy.crop_shape)
    assert list(direct.policy.down_dims) == list(bspline.policy.down_dims)
    assert list(direct.policy.temperatures) == list(bspline.policy.temperatures)
    assert direct.policy.gen_per_label == bspline.policy.gen_per_label == 8
    assert direct.policy.per_timestep_loss is True
    assert bspline.policy.per_timestep_loss is True
    assert direct.dataloader.batch_size == 32
    assert direct.val_dataloader.batch_size == 32
    assert bspline.dataloader.batch_size == 8
    assert direct.training.gradient_accumulate_every == 1
    assert bspline.training.gradient_accumulate_every == 1
    assert direct.optimizer == bspline.optimizer
    assert direct.training.num_epochs == bspline.training.num_epochs == 200
    assert direct.training.seed == bspline.training.seed == 42
