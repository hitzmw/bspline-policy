"""Real-robot data, normalization, and policy contracts for wipe_bb."""

from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from bspline_policy.common.knots import encode_relative_knots
from bspline_policy.dataset.franka_bspline_image_dataset import FrankaBSplineImageDataset


CONFIG_NAME = "train_drifting_unet_wipe_bb_image_bspline_raw_concat_workspace"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "bspline_policy" / "config"


def _config():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name=CONFIG_NAME)
    OmegaConf.resolve(config)
    return config


def _write_replay(path, with_gripper=False):
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    lengths = np.asarray([32, 36, 40, 44])
    ends = np.cumsum(lengths)
    steps = int(ends[-1])
    episode_ids = np.repeat(np.arange(4), lengths)
    time = np.concatenate([np.arange(n) for n in lengths])
    action_dim = 8 if with_gripper else 7
    # Linear trajectories make the oracle independent of fitting accuracy.
    control = episode_ids[:, None] * 10 + time[:, None] * 0.01 + np.arange(action_dim)
    data.create_dataset("control", data=control.astype(np.float64))
    data.create_dataset("joint_positions", data=control[:, :7] - 0.02)
    data.create_dataset("ee_pose", data=control[:, :6])
    if with_gripper:
        data.create_dataset("gripper_position", data=episode_ids * 0.01)
    for key in ("D435_color", "D405_color"):
        images = np.broadcast_to(
            episode_ids[:, None, None, None], (steps, 128, 128, 3)
        ).astype(np.uint8)
        data.create_dataset(key, data=images, chunks=(1, 128, 128, 3))
    root.create_group("meta").create_dataset("episode_ends", data=ends)


def _dataset(tmp_path, **kwargs):
    path = tmp_path / "replay.zarr"
    _write_replay(path, with_gripper=kwargs.pop("with_gripper", False))
    return FrankaBSplineImageDataset(
        zarr_path=str(path),
        val_ratio=0.25,
        cache_base_path=str(tmp_path / "cache"),
        **kwargs,
    )


@pytest.mark.parametrize("with_gripper", [False, True])
def test_optional_gripper_preserves_baseline_and_concat_contracts(tmp_path, with_gripper):
    dataset = _dataset(tmp_path, with_gripper=with_gripper)
    dim = 8 if with_gripper else 7
    sample = dataset[0]
    assert sample["action"].shape == (16, 1 + dim)
    assert ("gripper_pos" in sample["obs"]) == with_gripper
    assert "joint_positions" not in sample["obs"]
    if with_gripper:
        assert sample["obs"]["gripper_pos"].shape == (2, 1)

    concat = FrankaBSplineImageDataset(
        zarr_path=dataset.zarr_path,
        val_ratio=0.25,
        cache_base_path=dataset.cache_base_path,
        raw_action_concat=True,
        raw_action_sampling_mode="bspline_interval",
    )
    # The cached splines can be shared; dense actions are assembled on demand.
    assert concat.sampler.cache_path == dataset.sampler.cache_path
    torch.testing.assert_close(concat[0]["action"][:, : 1 + dim], sample["action"])
    assert concat[0]["action"].shape == (16, 1 + 2 * dim)
    for index in (0, len(concat) - 1):
        item = concat[index]
        normalizer = concat.get_normalizer()
        for key, value in item["obs"].items():
            assert value.dtype == torch.float32
            assert torch.isfinite(normalizer[key].normalize(value)).all()
        normalized = normalizer["action"].normalize(item["action"])
        assert normalized.dtype == torch.float32
        torch.testing.assert_close(
            normalizer["action"].unnormalize(normalized), item["action"], atol=2e-5, rtol=2e-5
        )


@pytest.mark.parametrize("relative_knots", [False, True])
def test_interval_alignment_and_episode_boundary_clipping(tmp_path, relative_knots):
    dataset = _dataset(
        tmp_path, raw_action_concat=True,
        raw_action_sampling_mode="bspline_interval", relative_knots=relative_knots,
    )
    idx = len(dataset) - 1
    timestep = int(dataset.sampler.valid_timesteps[idx])
    episode = np.searchsorted(dataset.sampler.episode_ends, timestep, side="right")
    end = int(dataset.sampler.episode_ends[episode])
    parameters = np.zeros((16, 8), dtype=np.float32)
    # Fractional phase points extending beyond the last frame must stay in episode.
    parameters[:, 0] = np.r_[[-3, -2, -1], np.linspace(0.5, 8, 10), [9, 10, 11]]
    if relative_knots:
        parameters = encode_relative_knots(parameters, degree=3)
    actual = dataset._get_raw_action_sequence(idx, parameters)
    positions = np.minimum(timestep + np.linspace(0.5, 8, 16), end - 1)
    controls = np.asarray(dataset.replay_buffer["control"])
    expected = np.stack([
        np.interp(positions, np.arange(len(controls)), controls[:, channel])
        for channel in range(7)
    ], axis=-1)
    np.testing.assert_allclose(actual, expected, atol=3e-6)

    dataset.raw_action_sampling_mode = "current"
    current = dataset._get_raw_action_sequence(idx)
    np.testing.assert_allclose(current[0], controls[timestep])
    np.testing.assert_allclose(current[-1], controls[end - 1])


def test_training_stats_and_validation_exclude_unused_episodes(tmp_path):
    dataset = _dataset(
        tmp_path, max_train_episodes=1, include_joint_positions=True,
        raw_action_concat=True, raw_action_sampling_mode="bspline_interval",
    )
    validation = dataset.get_validation_dataset()
    assert dataset.train_mask.sum() == validation.train_mask.sum() == 1
    assert not np.any(dataset.train_mask & validation.train_mask)
    ends = np.asarray(dataset.replay_buffer.episode_ends)
    selected = int(np.flatnonzero(dataset.train_mask)[0])
    start = 0 if selected == 0 else int(ends[selected - 1])
    end = int(ends[selected])
    normalizer = dataset.get_normalizer()
    for key in ("control", "ee_pose", "joint_positions"):
        stats = dataset._get_training_stats(key)
        values = np.asarray(dataset.replay_buffer[key][start:end], dtype=np.float32)
        np.testing.assert_array_equal(stats["min"], values.min(0))
        np.testing.assert_array_equal(stats["max"], values.max(0))
    action_stats = normalizer["action"].get_input_stats()
    np.testing.assert_allclose(
        action_stats["min"].detach().numpy().reshape(16, 15)[:, 8:],
        np.broadcast_to(dataset._get_training_stats("control")["min"], (16, 7)),
    )
    actions = dataset.get_all_actions()
    assert actions.shape == (len(dataset), 16, 15)
    torch.testing.assert_close(actions[-1], dataset[len(dataset) - 1]["action"])


def test_wipe_config_loss_backward_and_integer_tick_inference(tmp_path):
    config = _config()
    assert list(config.shape_meta.action.shape) == [7]
    assert "gripper_pos" not in config.shape_meta.obs
    assert config.policy.raw_action_training_mode == "decode_consistency"
    assert config.task.dataset.raw_action_sampling_mode == "bspline_interval"
    assert config.task.dataset.include_joint_positions is True
    assert config.horizon == 16 and config.n_action_steps == 8

    dataset = _dataset(
        tmp_path, include_joint_positions=True,
        raw_action_concat=True, raw_action_sampling_mode="bspline_interval",
    )
    # Keep both real image encoders; reduce only UNet width for the CPU check.
    config.policy.down_dims = [32, 64, 128]
    config.policy.diffusion_step_embed_dim = 32
    policy = hydra.utils.instantiate(config.policy)
    policy.set_normalizer(dataset.get_normalizer())
    sample = dataset[0]
    batch = {
        "obs": {key: value[None] for key, value in sample["obs"].items()},
        "action": sample["action"][None],
    }
    loss, metrics = policy.compute_loss(batch, return_info=True)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["raw_action_consistency_loss"])
    loss.backward()
    for module in (policy.model, policy.obs_encoder):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert any(torch.count_nonzero(grad) > 0 for grad in grads)
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(batch["obs"])
    assert result["joint_action_pred"].shape == (1, 16, 15)
    assert result["bspline_action"].shape == (1, 16, 8)
    assert result["action"].shape == (1, 8, 7)
    assert torch.isfinite(result["action"]).all()
