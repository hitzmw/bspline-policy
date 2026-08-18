"""Unit tests for the RoboCasa online-runner boundary."""

from collections import deque

import numpy as np
import pytest
import torch

from bspline_policy.env_runner.robocasa_image_runner import (
    MOBILE_BASE_ACTION,
    RoboCasaImageRunner,
    adapt_policy_action,
    stack_observation_history,
)


def test_adapt_policy_action_appends_stationary_mobile_base():
    manipulation = np.arange(7, dtype=np.float32)
    actual = adapt_policy_action(manipulation, env_action_dim=12)
    np.testing.assert_array_equal(actual[:7], manipulation)
    np.testing.assert_array_equal(actual[7:], MOBILE_BASE_ACTION)


def test_adapt_policy_action_rejects_unknown_dimensions():
    with pytest.raises(ValueError, match="Cannot adapt"):
        adapt_policy_action(np.zeros(6), env_action_dim=12)


def test_runner_embeds_reference_delta_pose_controller(tmp_path):
    runner = RoboCasaImageRunner(output_dir=str(tmp_path))
    controller = runner._controller_config()
    assert controller["type"] == "OSC_POSE"
    assert controller["control_delta"] is True
    assert controller["output_max"] == [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]


def test_stack_observation_history_pads_and_converts_images():
    image = np.full((4, 5, 3), 255, dtype=np.uint8)
    frame = {
        "robot0_agentview_left_rgb": image,
        "ee_pos": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    }
    result = stack_observation_history(
        deque([frame]),
        n_obs_steps=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert result["robot0_agentview_left_rgb"].shape == (1, 2, 3, 4, 5)
    assert result["ee_pos"].shape == (1, 2, 3)
    torch.testing.assert_close(
        result["robot0_agentview_left_rgb"],
        torch.ones((1, 2, 3, 4, 5)),
    )
    torch.testing.assert_close(result["ee_pos"][0, 0], result["ee_pos"][0, 1])
