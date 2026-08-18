"""Configuration contract for the original B-spline RoboCasa baseline."""

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


def test_original_bspline_matches_drifting_robocasa_control_settings():
    original = _compose(
        "train_diffusion_unet_turn_off_sink_faucet_image_bspline_workspace"
    )
    drifting = _compose(
        "train_drifting_unet_turn_off_sink_faucet_image_bspline_workspace"
    )

    assert original.task.name == drifting.task.name
    assert original.task.dataset.dataset_path == drifting.task.dataset.dataset_path
    assert original.horizon == drifting.horizon == 16
    assert original.n_obs_steps == drifting.n_obs_steps == 2
    assert original.n_action_steps == drifting.n_action_steps == 8
    assert original.policy.n_action_steps == drifting.policy.n_action_steps == 16
    assert original.policy.execution_action_steps == 8
    assert list(original.policy.crop_shape) == list(drifting.policy.crop_shape)
    assert list(original.policy.down_dims) == list(drifting.policy.down_dims)
    assert original.dataloader.batch_size == drifting.dataloader.batch_size == 8
    assert original.training.seed == drifting.training.seed == 42
    assert original.training.num_epochs == drifting.training.num_epochs == 200
    assert original.training.lr_scheduler == drifting.training.lr_scheduler
    assert original.task.dataset.bspline_degree == 3

    assert original.policy.noise_scheduler.num_train_timesteps == 100
    assert original.policy.num_inference_steps == 16
    assert original.policy.noise_scheduler.prediction_type == "epsilon"
    assert original.checkpoint.topk.monitor_key == "val_loss"


def test_original_bspline_robocasa_targets_are_importable():
    config = _compose(
        "train_diffusion_unet_turn_off_sink_faucet_image_bspline_workspace"
    )

    workspace_class = hydra.utils.get_class(config._target_)
    policy_class = hydra.utils.get_class(config.policy._target_)
    dataset_class = hydra.utils.get_class(config.task.dataset._target_)

    assert workspace_class.__name__ == "TrainDiffusionUnetHybridWorkspace"
    assert policy_class.__name__ == "DiffusionUnetPushTBSplineImagePolicy"
    assert dataset_class.__name__ == "RobomimicReplayBSplineImageDataset"
