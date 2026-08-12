import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.diffusion.conditional_imeanflow_unet1d import (
    ConditionalIMeanFlowUnet1D,
)
from diffusion_policy.model.diffusion.improved_mean_flow import (
    improved_mean_flow_loss,
    improved_mean_flow_sample,
    sample_time_pairs,
)
from diffusion_policy.policy.imeanflow_unet_lowdim_policy import (
    IMeanFlowUnetLowdimPolicy,
)


class TinyDualVelocityModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.u = nn.Linear(feature_dim, feature_dim)
        self.v = nn.Linear(feature_dim, feature_dim)
        self.calls = 0

    def forward(self, sample, timestep, end_timestep, **kwargs):
        self.calls += 1
        interval = (timestep - end_timestep).reshape(-1, 1, 1)
        return self.u(sample) + interval, self.v(sample)


def test_logit_normal_time_pairs_are_ordered_and_include_boundary_samples():
    generator = torch.Generator().manual_seed(7)
    t, r, flow_mask = sample_time_pairs(
        batch_size=10,
        device=torch.device("cpu"),
        dtype=torch.float32,
        data_proportion=0.4,
        generator=generator,
    )
    assert torch.all((0.0 <= r) & (r <= t) & (t <= 1.0))
    assert flow_mask.sum().item() == 4
    assert torch.equal(t[flow_mask], r[flow_mask])


def test_improved_mean_flow_loss_backpropagates_through_both_heads():
    torch.manual_seed(3)
    model = TinyDualVelocityModel(feature_dim=2)
    clean = torch.randn(4, 5, 2)
    condition_mask = torch.zeros_like(clean, dtype=torch.bool)
    condition_mask[:, :1] = True

    loss, metrics = improved_mean_flow_loss(
        model=model,
        clean_trajectory=clean,
        condition_mask=condition_mask,
        adaptive_loss=False,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.u.weight.grad is not None
    assert model.v.weight.grad is not None
    assert torch.isfinite(model.u.weight.grad).all()
    assert torch.isfinite(model.v.weight.grad).all()
    assert set(metrics) == {
        "loss", "loss_u", "loss_v", "mean_t", "mean_interval", "flow_ratio"
    }


def test_single_step_sampler_calls_network_once_and_reapplies_conditioning():
    model = TinyDualVelocityModel(feature_dim=3)
    condition_data = torch.randn(2, 4, 3)
    condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
    condition_mask[:, 0] = True

    output = improved_mean_flow_sample(
        model=model,
        condition_data=condition_data,
        condition_mask=condition_mask,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(11),
    )

    assert model.calls == 1
    assert output.shape == condition_data.shape
    assert torch.equal(output[condition_mask], condition_data[condition_mask])
    assert torch.isfinite(output).all()


def test_imeanflow_unet_returns_trajectory_sized_dual_heads():
    model = ConditionalIMeanFlowUnet1D(
        input_dim=3,
        global_cond_dim=4,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        kernel_size=3,
        n_groups=4,
        use_auxiliary_head=True,
    )
    sample = torch.randn(2, 8, 3)
    t = torch.tensor([1.0, 0.7])
    r = torch.tensor([0.0, 0.2])
    u, v = model(sample, t, r, global_cond=torch.randn(2, 4))

    assert u.shape == sample.shape
    assert v.shape == sample.shape


def test_lowdim_policy_preserves_diffusion_policy_action_interface():
    model = ConditionalIMeanFlowUnet1D(
        input_dim=2,
        global_cond_dim=6,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        n_groups=4,
        use_auxiliary_head=True,
    )
    policy = IMeanFlowUnetLowdimPolicy(
        model=model,
        horizon=8,
        obs_dim=3,
        action_dim=2,
        n_action_steps=4,
        n_obs_steps=2,
        obs_as_global_cond=True,
        oa_step_convention=True,
        adaptive_loss=False,
    )
    normalizer = LinearNormalizer()
    normalizer["obs"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)

    batch = {
        "obs": torch.randn(2, 8, 3),
        "action": torch.randn(2, 8, 2).clamp(-1.0, 1.0),
    }
    loss = policy.compute_loss(batch)
    loss.backward()
    result = policy.predict_action({"obs": batch["obs"][:, :2]})
    with torch.no_grad():
        validation_loss = policy.compute_loss(batch)

    assert torch.isfinite(loss)
    assert torch.isfinite(validation_loss)
    assert result["action"].shape == (2, 4, 2)
    assert result["action_pred"].shape == (2, 8, 2)
