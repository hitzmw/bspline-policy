import numpy as np
import torch
from scipy.interpolate import BSpline

from bspline_policy.common.bspline_action import BSplineChunkSampler
from bspline_policy.common.torch_bspline import (
    RAW_TIME_CLAMP,
    SUPPORT_LINSPACE,
    decode_bspline_parameters,
    project_monotonic_knots,
)


def _parameters(knots, controls):
    result = np.zeros((len(knots), controls.shape[-1] + 1), dtype=np.float64)
    result[:, 0] = knots
    result[: len(controls), 1:] = controls
    return result


def test_torch_decoder_matches_scipy_with_endpoint_hold():
    degree = 3
    knots = np.asarray([0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4], dtype=np.float64)
    controls = np.arange(14, dtype=np.float64).reshape(7, 2)
    parameters = _parameters(knots, controls)
    raw_times = np.asarray([-2.0, 0.0, 0.5, 2.0, 4.0, 8.0])

    decoded, info = decode_bspline_parameters(
        torch.from_numpy(parameters).unsqueeze(0),
        torch.from_numpy(raw_times).unsqueeze(0),
        degree=degree,
        decode_mode=RAW_TIME_CLAMP,
    )
    scipy_times = np.clip(raw_times, knots[degree], knots[-degree - 1])
    reference = BSpline(knots, controls, degree, extrapolate=False)(scipy_times)

    np.testing.assert_allclose(decoded[0].numpy(), reference, atol=1e-10)
    np.testing.assert_allclose(info["evaluation_times"][0].numpy(), scipy_times)
    assert torch.isfinite(decoded).all()


def test_support_linspace_matches_scipy_and_is_distinct_ablation():
    degree = 3
    knots = np.asarray([-2, -1, 0, 1, 2, 4, 6, 8, 9, 10, 11], dtype=np.float64)
    controls = np.arange(14, dtype=np.float64).reshape(7, 2)
    parameters = torch.from_numpy(_parameters(knots, controls)).unsqueeze(0)
    requested = torch.arange(5, dtype=torch.float64).unsqueeze(0)
    decoded, info = decode_bspline_parameters(
        parameters,
        requested,
        degree=degree,
        decode_mode=SUPPORT_LINSPACE,
    )
    reference = BSpline(knots, controls, degree)(
        np.linspace(knots[degree], knots[-degree - 1], 5)
    )
    np.testing.assert_allclose(decoded[0].numpy(), reference, atol=1e-10)
    assert not torch.equal(info["evaluation_times"], requested)


def test_projection_is_strict_and_decoder_backpropagates_to_knots_and_controls():
    parameters = torch.randn(2, 11, 3, dtype=torch.float64, requires_grad=True)
    projected = project_monotonic_knots(
        parameters[..., 0], min_delta=1e-3, straight_through=True
    )
    assert torch.all(projected[:, 1:] - projected[:, :-1] >= 1e-3 - 1e-12)

    decoded, _ = decode_bspline_parameters(
        parameters,
        torch.arange(4, dtype=torch.float64).expand(2, -1),
        degree=3,
        project_knots=True,
        min_knot_delta=1e-3,
        straight_through=True,
    )
    decoded.square().mean().backward()
    assert torch.isfinite(parameters.grad).all()
    assert parameters.grad[..., 0].abs().sum() > 0
    assert parameters.grad[..., 1:].abs().sum() > 0


def test_raw_action_tail_is_zero_padded_and_masked_without_nan():
    sampler = BSplineChunkSampler.__new__(BSplineChunkSampler)
    sampler.valid_timesteps = np.asarray([2], dtype=np.int64)
    sampler.episode_ends = np.asarray([4], dtype=np.int64)
    sampler.replay_buffer = {
        "action": np.asarray(
            [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.float32
        )
    }
    sampler.action_key = "action"
    sampler.degree = 3
    sampler.relative_knots = False

    knots = np.asarray([0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3], dtype=np.float32)
    controls = np.zeros((7, 2), dtype=np.float32)
    reconstruction = sampler.sample_reconstruction_sequence(
        0, num_actions=4, action_params=_parameters(knots, controls)
    )

    np.testing.assert_array_equal(
        reconstruction["raw_action"],
        np.asarray([[4, 5], [6, 7], [0, 0], [0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        reconstruction["raw_action_episode_mask"], [True, True, False, False]
    )
    np.testing.assert_array_equal(
        reconstruction["raw_action_mask"], [True, True, False, False]
    )
    assert not np.isnan(reconstruction["raw_action"]).any()
