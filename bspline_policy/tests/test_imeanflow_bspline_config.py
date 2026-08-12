"""Hydra contracts for iMeanFlow B-spline image policies."""

from pathlib import Path
import tempfile

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "bspline_policy" / "config"


def _compose(name):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIRECTORY)):
        config = compose(config_name=name)
    OmegaConf.resolve(config)
    return config


def test_pusht_imeanflow_bspline_config_contract_and_targets():
    config = _compose("train_imeanflow_unet_pusht_image_bspline_workspace")
    assert config.policy.num_inference_steps == 1
    assert config.policy.clip_sample is True
    assert config.policy.reconstruction_loss_weight == 0.05
    assert config.policy.reconstruction_warmup_ratio == 0.1
    assert config.policy.rollout_decode_mode == "raw_time_clamp"
    assert config.policy.relative_knots is False
    assert config.policy.n_action_steps == 16
    assert config.policy.execution_action_steps == 8
    assert config.task.dataset.raw_action_steps == 8
    assert hydra.utils.get_class(config._target_).__name__ == (
        "TrainIMeanFlowBSplineImageWorkspace"
    )
    assert hydra.utils.get_class(config.policy._target_).__name__ == (
        "IMeanFlowUnetPushTBSplineImagePolicy"
    )


def test_square_imeanflow_bspline_config_contract_and_targets():
    config = _compose("train_imeanflow_unet_square_image_bspline_workspace")
    assert list(config.shape_meta.action.shape) == [7]
    assert config.task.dataset.raw_action_steps == 8
    assert config.task.dataset.relative_knots is False
    assert hydra.utils.get_class(config.policy._target_).__name__ == (
        "IMeanFlowUnetRobomimicBSplineImagePolicy"
    )


def test_pusht_workspace_constructs_with_small_network():
    config = _compose("train_imeanflow_unet_pusht_image_bspline_workspace")
    config.policy.down_dims = [16, 32]
    config.policy.diffusion_step_embed_dim = 16
    config.policy.n_groups = 4
    workspace_class = hydra.utils.get_class(config._target_)
    with tempfile.TemporaryDirectory() as output_directory:
        workspace = workspace_class(config, output_dir=output_directory)
    assert type(workspace.model).__name__ == "IMeanFlowUnetPushTBSplineImagePolicy"
    assert type(workspace.ema_model).__name__ == (
        "IMeanFlowUnetPushTBSplineImagePolicy"
    )
    assert type(workspace.optimizer).__name__ == "AdamW"
