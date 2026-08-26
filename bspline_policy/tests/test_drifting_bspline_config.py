"""Configuration contract for canonical Drifting-BSpline Push-T."""

from __future__ import annotations

import tempfile
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "bspline_policy" / "config"
)


def _compose_config(
    config_name="train_drifting_unet_pusht_image_bspline_workspace",
):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIRECTORY),
    ):
        config = compose(config_name=config_name)
    OmegaConf.resolve(config)
    return config


def test_config_combines_canonical_drifting_and_bspline_pusht():
    config = _compose_config()

    assert config.dataloader.batch_size == 64
    assert config.policy.gen_per_label == 8
    assert config.training.gradient_accumulate_every == 1
    assert config.training.num_epochs == 300
    assert config.training.lr_scheduler == "constant_with_warmup"
    assert list(config.policy.temperatures) == [0.02, 0.05, 0.2]
    assert config.policy.per_timestep_loss is True

    assert config.task.dataset._target_.endswith(
        "PushTBSplineImageDataset"
    )
    assert config.task.dataset.bspline_degree == 3
    assert config.task.dataset.chunk_size == 10
    assert config.task.dataset.max_error == 1.0
    assert config.horizon == 16
    assert config.policy.n_action_steps == 16
    assert config.policy.execution_action_steps == 8
    assert list(config.shape_meta.action.shape) == [2]


def test_hydra_targets_are_importable():
    config = _compose_config()
    workspace_class = hydra.utils.get_class(config._target_)
    policy_class = hydra.utils.get_class(config.policy._target_)

    assert workspace_class.__name__ == "TrainDriftingBSplineImageWorkspace"
    assert policy_class.__name__ == (
        "DriftingUnetPushTBSplineImagePolicy"
    )


def test_raw_consistency_config_preserves_pusht_baseline_settings():
    baseline = _compose_config()
    config = _compose_config(
        "train_drifting_unet_pusht_image_bspline_raw_concat_workspace"
    )

    assert config.policy.raw_action_concat is True
    assert config.policy.raw_action_training_mode == "decode_consistency"
    assert config.policy.raw_action_loss_weight == 0.1
    assert config.policy.raw_action_consistency_detach_knots is True
    assert config.task.dataset.raw_action_concat is True
    assert config.task.dataset.raw_action_sampling_mode == "bspline_interval"
    assert config.exp_name == "drifting_bspline_raw_consistency"
    assert list(config.shape_meta.action.shape) == [2]
    assert 1 + 2 * int(config.shape_meta.action.shape[0]) == 5

    for key in (
        "horizon",
        "n_obs_steps",
        "n_action_steps",
        "dataloader",
        "val_dataloader",
        "optimizer",
        "training",
        "ema",
    ):
        config_value = config[key]
        baseline_value = baseline[key]
        if OmegaConf.is_config(config_value):
            config_value = OmegaConf.to_container(config_value, resolve=True)
        if OmegaConf.is_config(baseline_value):
            baseline_value = OmegaConf.to_container(
                baseline_value,
                resolve=True,
            )
        assert config_value == baseline_value


def test_workspace_constructs_model_ema_and_optimizer():
    config = _compose_config()
    config.policy.down_dims = [16, 32]
    config.policy.diffusion_step_embed_dim = 16
    workspace_class = hydra.utils.get_class(config._target_)

    with tempfile.TemporaryDirectory() as output_directory:
        workspace = workspace_class(
            config,
            output_dir=output_directory,
        )

    assert type(workspace.model).__name__ == (
        "DriftingUnetPushTBSplineImagePolicy"
    )
    assert type(workspace.ema_model).__name__ == (
        "DriftingUnetPushTBSplineImagePolicy"
    )
    assert type(workspace.optimizer).__name__ == "AdamW"
    assert workspace.global_step == 0
    assert workspace.epoch == 0
