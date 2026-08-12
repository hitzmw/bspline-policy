import torch

from bspline_policy.common.torch_bspline import decode_bspline_parameters
from bspline_policy.policy.imeanflow_unet_pusht_bspline_image_policy import (
    IMeanFlowUnetPushTBSplineImagePolicy,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer


def _make_policy_and_batch():
    torch.manual_seed(7)
    shape_meta = {
        "obs": {"agent_pos": {"shape": [2], "type": "low_dim"}},
        "action": {"shape": [2]},
    }
    policy = IMeanFlowUnetPushTBSplineImagePolicy(
        shape_meta=shape_meta,
        horizon=8,
        n_action_steps=8,
        execution_action_steps=4,
        n_obs_steps=2,
        obs_as_global_cond=True,
        crop_shape=None,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        kernel_size=3,
        n_groups=4,
        cond_predict_scale=True,
        obs_encoder_group_norm=True,
        eval_fixed_crop=True,
        use_auxiliary_head=True,
        num_inference_steps=1,
        clip_sample=True,
        bspline_degree=3,
        relative_knots=False,
        reconstruction_loss_weight=0.05,
        reconstruction_warmup_ratio=0.1,
        min_knot_delta=1e-3,
    )

    knots = torch.tensor([0, 0, 0, 0, 3, 3, 3, 3], dtype=torch.float32)
    controls = torch.tensor(
        [[0.0, 0.0], [0.3, -0.2], [0.6, 0.4], [1.0, 0.8]],
        dtype=torch.float32,
    )
    parameters = torch.zeros(2, 8, 3)
    parameters[:, :, 0] = knots
    parameters[:, :4, 1:] = controls
    raw_times = torch.arange(4, dtype=torch.float32).expand(2, -1)
    raw_action, _ = decode_bspline_parameters(parameters, raw_times, degree=3)

    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": torch.cat([parameters - 1.0, parameters + 1.0], dim=0),
            "raw_action": torch.cat([raw_action - 1.0, raw_action + 1.0], dim=0),
            "agent_pos": torch.tensor(
                [[[-1.0, -1.0], [1.0, 1.0]], [[-0.5, 0.5], [0.5, -0.5]]]
            ),
        },
        last_n_dims=1,
        mode="limits",
    )
    policy.set_normalizer(normalizer)
    batch = {
        "obs": {"agent_pos": torch.randn(2, 2, 2)},
        "action": parameters,
        "raw_action": raw_action,
        "raw_action_time": raw_times,
        "raw_action_mask": torch.ones(2, 4, dtype=torch.bool),
        "raw_action_episode_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    return policy, batch


def test_joint_loss_backward_and_required_diagnostics_are_finite():
    policy, batch = _make_policy_and_batch()
    policy.set_training_step(5, 100)
    loss, metrics = policy.compute_loss(batch, return_info=True)
    loss.backward()
    with torch.no_grad():
        validation_loss, _ = policy.compute_loss(batch, return_info=True)

    assert torch.isfinite(loss)
    assert torch.isfinite(validation_loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.model.parameters()
    )
    expected = {
        "loss_total",
        "loss_imeanflow",
        "loss_imeanflow_u",
        "loss_imeanflow_v",
        "loss_reconstruction",
        "reconstruction_weight",
        "error_fit",
        "error_parameter_to_curve",
        "error_final",
        "clip_fraction",
        "generated_support_coverage",
    }
    assert expected.issubset(metrics)
    assert all(torch.isfinite(metrics[key]) for key in expected)
    assert torch.allclose(
        metrics["reconstruction_weight"], torch.tensor(0.025)
    )


def test_rollout_uses_one_step_clip_projection_and_raw_integer_times():
    policy, batch = _make_policy_and_batch()
    prediction = policy.predict_action(batch["obs"])

    assert prediction["action"].shape == (2, 4, 2)
    assert prediction["bspline_action"].shape == (2, 8, 3)
    assert torch.isfinite(prediction["action"]).all()
    normalized = prediction["normalized_bspline_action"]
    assert normalized.min() >= -1.0
    assert normalized.max() <= 1.0
    projected_knots = prediction["projected_bspline_action"][..., 0]
    assert torch.all(projected_knots[:, 1:] > projected_knots[:, :-1])
