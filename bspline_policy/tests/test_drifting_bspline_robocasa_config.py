"""Configuration and action-selection tests for RoboCasa Drifting-BSpline."""

from pathlib import Path

import h5py
import hydra
import numpy as np
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    _convert_robomimic_to_replay,
)
from diffusion_policy.model.common.rotation_transformer import (
    RotationTransformer,
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
                "train_drifting_unet_turn_off_sink_faucet_"
                "image_bspline_workspace"
            )
        )
    OmegaConf.resolve(config)
    return config


def test_robocasa_config_contract():
    config = _compose_config()

    assert list(config.shape_meta.action.shape) == [7]
    assert list(config.task.dataset.action_indices) == list(range(7))
    assert config.task.dataset.observation_history is True
    assert config.policy.gen_per_label == 8
    assert config.dataloader.batch_size == 8
    assert config.checkpoint.topk.monitor_key == "val_loss"
    assert config.task.env_runner._target_.endswith("OfflineImageRunner")

    workspace_class = hydra.utils.get_class(config._target_)
    policy_class = hydra.utils.get_class(config.policy._target_)
    dataset_class = hydra.utils.get_class(config.task.dataset._target_)
    runner_class = hydra.utils.get_class(config.task.env_runner._target_)
    assert workspace_class.__name__ == "TrainDriftingBSplineImageWorkspace"
    assert policy_class.__name__ == "DriftingUnetRobomimicBSplineImagePolicy"
    assert dataset_class.__name__ == "RobomimicReplayBSplineImageDataset"
    assert runner_class.__name__ == "OfflineImageRunner"


def test_robomimic_conversion_selects_action_channels(tmp_path):
    dataset_path = tmp_path / "actions.hdf5"
    raw_actions = np.arange(48, dtype=np.float32).reshape(4, 12)
    later_actions = raw_actions[:2] + 100
    with h5py.File(dataset_path, "w") as file:
        demo = file.create_group("data/demo_0")
        demo.create_dataset("actions", data=raw_actions)
        obs = demo.create_group("obs")
        obs.create_dataset("ee_pos", data=np.zeros((4, 3), dtype=np.float32))
        # RoboCasa success filtering can leave gaps in the demo numbering.
        demo = file.create_group("data/demo_2")
        demo.create_dataset("actions", data=later_actions)
        obs = demo.create_group("obs")
        obs.create_dataset("ee_pos", data=np.ones((2, 3), dtype=np.float32))

    replay = _convert_robomimic_to_replay(
        store=zarr.MemoryStore(),
        shape_meta={
            "obs": {"ee_pos": {"shape": [3], "type": "low_dim"}},
            "action": {"shape": [7]},
        },
        dataset_path=str(dataset_path),
        abs_action=False,
        rotation_transformer=RotationTransformer(
            from_rep="axis_angle",
            to_rep="rotation_6d",
        ),
        action_indices=list(range(7)),
    )

    expected_actions = np.concatenate(
        [raw_actions[:, :7], later_actions[:, :7]],
        axis=0,
    )
    np.testing.assert_array_equal(replay["action"], expected_actions)
    np.testing.assert_array_equal(replay.episode_ends[:], [4, 6])
