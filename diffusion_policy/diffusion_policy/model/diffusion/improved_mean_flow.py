"""PyTorch improved MeanFlow objective and solver for trajectory policies."""

from typing import Callable, Optional, Tuple

import torch


def sample_time_pairs(
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        data_proportion: float = 0.5,
        p_mean: float = -0.4,
        p_std: float = 1.0,
        generator: Optional[torch.Generator] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample logit-normal ``(t, r)`` pairs with ``0 <= r <= t <= 1``.

    A fixed proportion of the batch is changed to flow-matching boundary
    samples by setting ``r=t``.  This is the time sampler used by iMeanFlow.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= data_proportion <= 1.0:
        raise ValueError("data_proportion must be in [0, 1]")

    normal = torch.randn(
        (batch_size, 2), device=device, dtype=dtype, generator=generator)
    samples = torch.sigmoid(normal * p_std + p_mean)
    t = torch.maximum(samples[:, 0], samples[:, 1])
    r = torch.minimum(samples[:, 0], samples[:, 1])

    num_flow_matching = int(batch_size * data_proportion)
    flow_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    if num_flow_matching > 0:
        indices = torch.randperm(
            batch_size, device=device, generator=generator)[:num_flow_matching]
        flow_mask[indices] = True
        r = torch.where(flow_mask, t, r)
    return t, r, flow_mask


def _split_model_output(output):
    if isinstance(output, (tuple, list)):
        return output[0], output[1]
    return output, None


def _masked_objective(
        error: torch.Tensor,
        loss_mask: torch.Tensor,
        adaptive: bool,
        norm_p: float,
        norm_eps: float) -> torch.Tensor:
    squared_error = error.square() * loss_mask.to(error.dtype)
    if adaptive:
        per_sample = squared_error.flatten(start_dim=1).sum(dim=1)
        weight = (per_sample + norm_eps).pow(-norm_p).detach()
        return (weight * per_sample).mean()

    denominator = loss_mask.to(error.dtype).sum().clamp_min(1.0)
    return squared_error.sum() / denominator


def improved_mean_flow_loss(
        model: Callable,
        clean_trajectory: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        data_proportion: float = 0.5,
        p_mean: float = -0.4,
        p_std: float = 1.0,
        adaptive_loss: bool = True,
        norm_p: float = 1.0,
        norm_eps: float = 0.01,
        auxiliary_loss_weight: float = 1.0,
        generator: Optional[torch.Generator] = None,
        noise: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, dict]:
    """Compute the improved MeanFlow v-loss.

    The JVP tangent uses the model's boundary velocity prediction rather than
    the ground-truth velocity.  Its derivative is stopped in the composite
    prediction ``V = u + (t-r) du/dt``, matching the iMeanFlow objective.
    """
    if clean_trajectory.shape != condition_mask.shape:
        raise ValueError("condition_mask must have the trajectory shape")

    batch_size = clean_trajectory.shape[0]
    t, r, flow_mask = sample_time_pairs(
        batch_size=batch_size,
        device=clean_trajectory.device,
        dtype=clean_trajectory.dtype,
        data_proportion=data_proportion,
        p_mean=p_mean,
        p_std=p_std,
        generator=generator,
    )
    time_shape = (batch_size,) + (1,) * (clean_trajectory.ndim - 1)
    t_broadcast = t.reshape(time_shape)
    interval = (t - r).reshape(time_shape)

    if noise is None:
        noise = torch.randn(
            clean_trajectory.shape,
            device=clean_trajectory.device,
            dtype=clean_trajectory.dtype,
            generator=generator,
        )
    elif noise.shape != clean_trajectory.shape:
        raise ValueError("noise must have the clean trajectory shape")
    instantaneous_velocity = noise - clean_trajectory
    noisy_trajectory = (
        (1.0 - t_broadcast) * clean_trajectory + t_broadcast * noise)

    # Inpainted observations are fixed conditions, not part of the flow.
    noisy_trajectory = torch.where(
        condition_mask, clean_trajectory, noisy_trajectory)
    loss_mask = ~condition_mask
    instantaneous_velocity = torch.where(
        loss_mask, instantaneous_velocity, torch.zeros_like(instantaneous_velocity))

    current_output = model(
        noisy_trajectory,
        t,
        r,
        local_cond=local_cond,
        global_cond=global_cond,
    )
    u, auxiliary_v = _split_model_output(current_output)

    # iMeanFlow scheme B uses the auxiliary head at h=0.  Models without an
    # auxiliary head fall back to scheme A's boundary identity v(z_t,t)=u(z_t,t,t).
    with torch.no_grad():
        boundary_output = model(
            noisy_trajectory,
            t,
            t,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        boundary_u, boundary_v = _split_model_output(boundary_output)
        predicted_velocity = boundary_v if boundary_v is not None else boundary_u
        predicted_velocity = torch.where(
            loss_mask, predicted_velocity, torch.zeros_like(predicted_velocity))

    def u_function(z, t_value, r_value):
        output = model(
            z,
            t_value,
            r_value,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        return _split_model_output(output)[0]

    # create_graph=False is intentional: iMeanFlow applies stop-gradient to
    # du/dt.  The separately evaluated primal u retains the parameter gradient.
    _, du_dt = torch.autograd.functional.jvp(
        u_function,
        (noisy_trajectory, t, r),
        (
            predicted_velocity,
            torch.ones_like(t),
            torch.zeros_like(r),
        ),
        create_graph=False,
        strict=False,
    )
    composite_velocity = u + interval * du_dt.detach()
    target_velocity = instantaneous_velocity.detach()

    loss_u = _masked_objective(
        composite_velocity - target_velocity,
        loss_mask,
        adaptive=adaptive_loss,
        norm_p=norm_p,
        norm_eps=norm_eps,
    )
    loss = loss_u
    loss_v = clean_trajectory.new_zeros(())
    if auxiliary_v is not None:
        loss_v = _masked_objective(
            auxiliary_v - target_velocity,
            loss_mask,
            adaptive=adaptive_loss,
            norm_p=norm_p,
            norm_eps=norm_eps,
        )
        loss = loss + auxiliary_loss_weight * loss_v

    metrics = {
        "loss": loss.detach(),
        "loss_u": loss_u.detach(),
        "loss_v": loss_v.detach(),
        "mean_t": t.mean().detach(),
        "mean_interval": (t - r).mean().detach(),
        "flow_ratio": flow_mask.to(clean_trajectory.dtype).mean().detach(),
    }
    return loss, metrics


@torch.no_grad()
def improved_mean_flow_sample(
        model: Callable,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        num_inference_steps: int = 1,
        generator: Optional[torch.Generator] = None,
        clip_sample: bool = True,
        ) -> torch.Tensor:
    """Generate a trajectory with iMeanFlow interval updates.

    With the default single step this is exactly
    ``z_0 = z_1 - u(z_1, 1, 0)``.  More intervals can be requested without
    changing the trained model.
    """
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if condition_data.shape != condition_mask.shape:
        raise ValueError("condition_mask must have the trajectory shape")

    trajectory = torch.randn(
        condition_data.shape,
        device=condition_data.device,
        dtype=condition_data.dtype,
        generator=generator,
    )
    batch_size = trajectory.shape[0]
    time_steps = torch.linspace(
        1.0,
        0.0,
        num_inference_steps + 1,
        device=trajectory.device,
        dtype=trajectory.dtype,
    )

    for index in range(num_inference_steps):
        trajectory = torch.where(condition_mask, condition_data, trajectory)
        t = time_steps[index].expand(batch_size)
        r = time_steps[index + 1].expand(batch_size)
        output = model(
            trajectory,
            t,
            r,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        u, _ = _split_model_output(output)
        interval_shape = (batch_size,) + (1,) * (trajectory.ndim - 1)
        trajectory = trajectory - (t - r).reshape(interval_shape) * u
        if clip_sample:
            trajectory.clamp_(-1.0, 1.0)

    return torch.where(condition_mask, condition_data, trajectory)
