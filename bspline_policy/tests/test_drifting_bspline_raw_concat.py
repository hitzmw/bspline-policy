"""Contracts for the RoboCasa Drifting-BSpline raw-action concat mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from bspline_policy.dataset.robomimic_replay_bspline_image_dataset import (
    RobomimicReplayBSplineImageDataset,
)
from bspline_policy.common.bspline_action import (
    decode_bspline_action,
    decode_bspline_action_torch,
    project_monotonic_knots_torch,
)
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer


CONFIG_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "bspline_policy" / "config"
)
ROBOCASA_TASKS = (
    "turn_off_sink_faucet",
    "turn_off_microwave",
    "close_single_door",
    "coffee_press_button",
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


@pytest.mark.parametrize("task", ROBOCASA_TASKS)
def test_raw_concat_configs_only_change_the_action_contract(task):
    baseline = _compose(
        f"train_drifting_unet_{task}_image_bspline_workspace"
    )
    concat = _compose(
        f"train_drifting_unet_{task}_image_bspline_raw_concat_workspace"
    )

    assert concat.policy.raw_action_concat is True
    assert concat.policy.raw_action_training_mode == "decode_consistency"
    assert concat.policy.raw_action_loss_weight == pytest.approx(0.1)
    assert concat.policy.raw_action_consistency_detach_knots is True
    assert concat.task.dataset.raw_action_concat is True
    assert concat.task.dataset.raw_action_sampling_mode == "bspline_interval"
    assert concat.exp_name == "drifting_bspline_raw_consistency"
    assert concat.task.dataset.cache_suffix.endswith("raw_consistency_v2")
    assert list(concat.shape_meta.action.shape) == [7]
    assert concat.horizon == 16
    assert 1 + 2 * int(concat.shape_meta.action.shape[0]) == 15

    expected_batch_size = 32 if task == "turn_off_sink_faucet" else (
        baseline.dataloader.batch_size
    )
    assert concat.dataloader.batch_size == expected_batch_size
    assert concat.val_dataloader.batch_size == expected_batch_size
    for loader_key in ("dataloader", "val_dataloader"):
        concat_loader = OmegaConf.to_container(
            concat[loader_key],
            resolve=True,
        )
        baseline_loader = OmegaConf.to_container(
            baseline[loader_key],
            resolve=True,
        )
        concat_loader.pop("batch_size")
        baseline_loader.pop("batch_size")
        assert concat_loader == baseline_loader

    for key in (
        "horizon",
        "n_obs_steps",
        "n_action_steps",
        "optimizer",
        "training",
        "ema",
    ):
        concat_value = concat[key]
        baseline_value = baseline[key]
        if OmegaConf.is_config(concat_value):
            concat_value = OmegaConf.to_container(concat_value, resolve=True)
        if OmegaConf.is_config(baseline_value):
            baseline_value = OmegaConf.to_container(
                baseline_value,
                resolve=True,
            )
        assert concat_value == baseline_value


def _stub_raw_concat_dataset():
    dataset = RobomimicReplayBSplineImageDataset.__new__(
        RobomimicReplayBSplineImageDataset
    )
    dataset.raw_action_concat = True
    dataset.raw_action_sampling_mode = "current"
    dataset.n_action_steps = 4
    dataset.regular_action_dim = 2
    dataset.bspline_degree = 1
    dataset.relative_knots = False
    dataset.replay_buffer = {
        "action": np.arange(10, dtype=np.float32).reshape(5, 2)
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
    dataset.lowdim_keys = []
    dataset.rgb_keys = []
    return dataset


def test_raw_actions_start_at_current_timestep_and_pad_inside_episode():
    dataset = _stub_raw_concat_dataset()

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

    bspline = np.arange(12, dtype=np.float32).reshape(4, 3)
    target = dataset._make_action_target(0, bspline)
    assert target.shape == (4, 5)
    np.testing.assert_array_equal(target[:, :3], bspline)
    np.testing.assert_array_equal(
        target[:, 3:],
        dataset._get_raw_action_sequence(0),
    )


def test_raw_actions_can_be_sampled_over_the_bspline_interval():
    dataset = _stub_raw_concat_dataset()
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

    actual = dataset._get_raw_action_sequence(0, bspline)
    expected = np.asarray(
        [
            [2.0, 3.0],
            [2.0 + 2.0 / 3.0, 3.0 + 2.0 / 3.0],
            [2.0 + 4.0 / 3.0, 3.0 + 4.0 / 3.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    target = dataset._make_action_target(0, bspline)
    assert target.shape == (4, 5)
    np.testing.assert_array_equal(target[:, :3], bspline)
    np.testing.assert_allclose(target[:, 3:], expected)


def test_raw_action_normalizer_stats_exclude_validation_episodes():
    dataset = _stub_raw_concat_dataset()
    actual = dataset._get_training_raw_action_stats()
    expected = array_to_stats(dataset.replay_buffer["action"][:3])
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])


def test_joint_action_normalizer_has_bspline_and_raw_channels():
    dataset = _stub_raw_concat_dataset()
    normalizer = dataset.get_normalizer()
    target = torch.from_numpy(
        dataset._make_action_target(
            0,
            np.arange(12, dtype=np.float32).reshape(4, 3),
        )
    )

    normalized = normalizer["action"].normalize(target)
    reconstructed = normalizer["action"].unnormalize(normalized)

    assert normalized.shape == (4, 5)
    torch.testing.assert_close(reconstructed, target)


def _raw_concat_normalizer() -> LinearNormalizer:
    normalizer = LinearNormalizer()
    per_timestep_min = np.asarray(
        [-4.0] + [-2.0] * 7 + [-1.0] * 7,
        dtype=np.float32,
    )
    per_timestep_max = np.asarray(
        [20.0] + [2.0] * 7 + [1.0] * 7,
        dtype=np.float32,
    )
    minimum = np.tile(per_timestep_min, 16)
    maximum = np.tile(per_timestep_max, 16)
    normalizer["action"] = get_range_normalizer_from_stat(
        {
            "min": minimum,
            "max": maximum,
            "mean": (minimum + maximum) / 2,
            "std": (maximum - minimum) / np.sqrt(12),
        }
    )
    normalizer["agent_pos"] = get_range_normalizer_from_stat(
        array_to_stats(
            np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        )
    )
    normalizer["image"] = get_image_range_normalizer()
    return normalizer


def test_differentiable_bspline_decoder_matches_scipy_and_backpropagates():
    parameters = torch.zeros(2, 16, 8, dtype=torch.float32)
    parameters[..., 0] = torch.linspace(-3.0, 12.0, 16)
    parameters[..., 1:] = torch.linspace(-1.0, 1.0, 16 * 7).reshape(
        1, 16, 7
    )
    parameters.requires_grad_(True)

    projected = project_monotonic_knots_torch(parameters)
    actual = decode_bspline_action_torch(
        projected,
        degree=3,
        num_actions=16,
    )
    expected = np.stack(
        [
            decode_bspline_action(
                item.detach().numpy(),
                degree=3,
                num_actions=16,
            )
            for item in projected
        ]
    )
    np.testing.assert_allclose(
        actual.detach().numpy(),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

    actual.square().mean().backward()
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()
    assert torch.count_nonzero(parameters.grad[..., 1:]) > 0


def test_raw_concat_policy_decode_consistency_and_bspline_only_rollout():
    from bspline_policy.policy.drifting_unet_robomimic_bspline_image_policy import (
        DriftingUnetRobomimicBSplineImagePolicy,
    )

    policy = DriftingUnetRobomimicBSplineImagePolicy(
        shape_meta={
            "obs": {
                "image": {"shape": [3, 96, 96], "type": "rgb"},
                "agent_pos": {"shape": [2], "type": "low_dim"},
            },
            "action": {"shape": [7]},
        },
        horizon=16,
        n_action_steps=16,
        execution_action_steps=8,
        n_obs_steps=2,
        crop_shape=[84, 84],
        diffusion_step_embed_dim=16,
        down_dims=[16, 32],
        n_groups=8,
        obs_encoder_group_norm=True,
        eval_fixed_crop=True,
        temperatures=[0.02, 0.05, 0.2],
        per_timestep_loss=True,
        gen_per_label=2,
        bspline_degree=3,
        raw_action_concat=True,
        raw_action_training_mode="decode_consistency",
        raw_action_loss_weight=0.1,
        raw_action_consistency_detach_knots=True,
    )
    policy.set_normalizer(_raw_concat_normalizer())

    assert policy.action_dim == 15
    assert policy.model_action_dim == 8
    assert policy.raw_action_head is None

    generator = torch.Generator().manual_seed(13)
    knots = torch.linspace(-3.0, 19.0, 16).reshape(1, 16, 1)
    controls = torch.rand(1, 16, 7, generator=generator) * 2 - 1
    raw_actions = torch.rand(1, 16, 7, generator=generator) * 2 - 1
    batch = {
        "obs": {
            "image": torch.rand(1, 2, 3, 96, 96, generator=generator),
            "agent_pos": torch.rand(1, 2, 2, generator=generator),
        },
        "action": torch.cat([knots, controls, raw_actions], dim=-1),
    }

    loss, diagnostics = policy.compute_loss(batch, return_info=True)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert diagnostics.keys() == {
        "scale",
        "loss_0.02",
        "loss_0.05",
        "loss_0.2",
        "drift_loss",
        "raw_action_consistency_loss",
        "raw_action_weighted_loss",
    }
    torch.testing.assert_close(
        loss.detach(),
        diagnostics["drift_loss"]
        + diagnostics["raw_action_weighted_loss"],
    )
    torch.testing.assert_close(
        diagnostics["raw_action_weighted_loss"],
        diagnostics["raw_action_consistency_loss"] * 0.1,
    )
    loss.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in policy.model.parameters()
    )
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(
            batch["obs"],
            generator=torch.Generator().manual_seed(17),
        )

    assert result["joint_action_pred"].shape == (1, 16, 15)
    assert result["raw_action_pred"].shape == (1, 16, 7)
    assert result["action_pred"].shape == (1, 16, 8)
    assert result["bspline_action"].shape == (1, 16, 8)
    assert result["projected_bspline_action"].shape == (1, 16, 8)
    assert result["action"].shape == (1, 8, 7)
