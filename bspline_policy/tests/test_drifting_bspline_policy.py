"""End-to-end tensor smoke test for Drifting-BSpline Push-T."""

from __future__ import annotations

import numpy as np
import torch
import unittest

from bspline_policy.policy.drifting_unet_pusht_bspline_image_policy import (
    DriftingUnetPushTBSplineImagePolicy,
)
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer


def _normalizer() -> LinearNormalizer:
    normalizer = LinearNormalizer()
    minimum = np.tile(
        np.asarray([-4.0, 0.0, 0.0], dtype=np.float32),
        16,
    )
    maximum = np.tile(
        np.asarray([20.0, 512.0, 512.0], dtype=np.float32),
        16,
    )
    action_stats = {
        "min": minimum,
        "max": maximum,
        "mean": (minimum + maximum) / 2,
        "std": (maximum - minimum) / np.sqrt(12),
    }
    normalizer["action"] = get_range_normalizer_from_stat(action_stats)
    normalizer["agent_pos"] = get_range_normalizer_from_stat(
        array_to_stats(
            np.asarray(
                [[0.0, 0.0], [512.0, 512.0]],
                dtype=np.float32,
            )
        )
    )
    normalizer["image"] = get_image_range_normalizer()
    return normalizer


def _make_policy(device: torch.device):
    policy = DriftingUnetPushTBSplineImagePolicy(
        shape_meta={
            "obs": {
                "image": {
                    "shape": [3, 96, 96],
                    "type": "rgb",
                },
                "agent_pos": {
                    "shape": [2],
                    "type": "low_dim",
                },
            },
            "action": {"shape": [2]},
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
        gen_per_label=8,
        bspline_degree=3,
    )
    policy.set_normalizer(_normalizer())
    return policy.to(device)


def _make_batch(device: torch.device):
    generator = torch.Generator().manual_seed(5)
    knots = torch.linspace(-3.0, 19.0, 16).reshape(1, 16, 1)
    controls = torch.rand(1, 16, 2, generator=generator) * 512
    return {
        "obs": {
            "image": torch.rand(
                1,
                2,
                3,
                96,
                96,
                generator=generator,
            ).to(device),
            "agent_pos": (
                torch.rand(1, 2, 2, generator=generator) * 512
            ).to(device),
        },
        "action": torch.cat([knots, controls], dim=-1).to(device),
    }


def _run_forward_backward_and_decoded_prediction(device_name):
    device = torch.device(device_name)
    policy = _make_policy(device)
    batch = _make_batch(device)

    policy.train()
    loss, diagnostics = policy.compute_loss(batch, return_info=True)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert diagnostics.keys() == {
        "scale",
        "loss_0.02",
        "loss_0.05",
        "loss_0.2",
    }
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in policy.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    policy.eval()
    if device.type == "cuda":
        prediction_generator = torch.Generator(device=device).manual_seed(11)
    else:
        prediction_generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        result = policy.predict_action(
            batch["obs"],
            generator=prediction_generator,
        )

    assert result["action_pred"].shape == (1, 16, 3)
    assert result["bspline_action"].shape == (1, 16, 3)
    assert result["projected_bspline_action"].shape == (1, 16, 3)
    assert result["action"].shape == (1, 8, 2)
    projected_knots = result["projected_bspline_action"][0, :, 0]
    # Projection is strict in float64 for SciPy decoding. Casting the
    # diagnostic tensor back to float32 can round a 1e-6 increment to equality.
    assert torch.all(projected_knots[1:] >= projected_knots[:-1])
    assert torch.isfinite(result["action"]).all()


def test_cpu_forward_backward_and_decoded_prediction():
    _run_forward_backward_and_decoded_prediction("cpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
def test_cuda_forward_backward_and_decoded_prediction():
    _run_forward_backward_and_decoded_prediction("cuda")
