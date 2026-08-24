"""Tensor smoke tests for direct-action Drifting."""

import numpy as np
import torch

from bspline_policy.policy.drifting_unet_hybrid_image_policy import (
    DriftingUnetHybridImagePolicy,
)
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer


def _make_normalizer() -> LinearNormalizer:
    normalizer = LinearNormalizer()
    normalizer["action"] = get_identity_normalizer_from_stat(
        array_to_stats(np.asarray([[-1.0] * 7, [1.0] * 7], dtype=np.float32))
    )
    normalizer["agent_pos"] = get_range_normalizer_from_stat(
        array_to_stats(
            np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        )
    )
    normalizer["image"] = get_image_range_normalizer()
    return normalizer


def test_direct_drifting_forward_backward_and_execution_slice():
    policy = DriftingUnetHybridImagePolicy(
        shape_meta={
            "obs": {
                "image": {"shape": [3, 96, 96], "type": "rgb"},
                "agent_pos": {"shape": [2], "type": "low_dim"},
            },
            "action": {"shape": [7]},
        },
        horizon=16,
        n_action_steps=8,
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
    )
    policy.set_normalizer(_make_normalizer())

    generator = torch.Generator().manual_seed(7)
    batch = {
        "obs": {
            "image": torch.rand(1, 2, 3, 96, 96, generator=generator),
            "agent_pos": torch.rand(1, 2, 2, generator=generator),
        },
        "action": torch.rand(1, 16, 7, generator=generator) * 2 - 1,
    }

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
    with torch.no_grad():
        result = policy.predict_action(
            batch["obs"],
            generator=torch.Generator().manual_seed(11),
        )

    assert result.keys() == {"action", "action_pred"}
    assert result["action_pred"].shape == (1, 16, 7)
    assert result["action"].shape == (1, 8, 7)
    torch.testing.assert_close(
        result["action"],
        result["action_pred"][:, 1:9],
    )
