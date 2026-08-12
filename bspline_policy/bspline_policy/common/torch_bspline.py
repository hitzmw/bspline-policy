"""Differentiable PyTorch utilities for fixed-shape B-spline actions.

The policy representation is ``[knot, control...]`` with shape
``[B, knot_count, 1 + action_dim]``.  This module is the single decoding path
used by both reconstruction training and rollout inference.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


RAW_TIME_CLAMP = "raw_time_clamp"
SUPPORT_LINSPACE = "support_linspace"


def project_monotonic_knots(
    knots: torch.Tensor,
    min_delta: float = 1e-4,
    straight_through: bool = True,
) -> torch.Tensor:
    """Project knots to a strictly increasing sequence.

    ``cummax(k_i - i*delta) + i*delta`` is the Euclidean-order projection used
    by the previous rollout adapter, expressed in PyTorch.  With
    ``straight_through=True`` the forward pass uses the valid projection while
    the backward pass treats it as the identity.
    """
    if knots.ndim < 2:
        raise ValueError("knots must have shape [..., knot_count]")
    if min_delta <= 0:
        raise ValueError("min_delta must be positive")

    offsets = torch.arange(
        knots.shape[-1], device=knots.device, dtype=knots.dtype
    ) * float(min_delta)
    projected = torch.cummax(knots - offsets, dim=-1).values + offsets
    if straight_through:
        return knots + (projected - knots).detach()
    return projected


def _batched_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``values[B,N,...]`` at ``indices[B,T]`` along dimension one."""
    if values.shape[0] != indices.shape[0]:
        raise ValueError("values and indices must have the same batch size")
    expanded_indices = indices
    for _ in range(values.ndim - 2):
        expanded_indices = expanded_indices.unsqueeze(-1)
    expanded_indices = expanded_indices.expand(
        indices.shape + values.shape[2:]
    )
    return torch.gather(values, 1, expanded_indices)


def _prepare_times(
    times: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    decode_mode: str,
) -> torch.Tensor:
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(left.shape[0], -1)
    if times.ndim != 2 or times.shape[0] != left.shape[0]:
        raise ValueError("times must have shape [T] or [B,T]")
    times = times.to(device=left.device, dtype=left.dtype)

    if decode_mode == RAW_TIME_CLAMP:
        # Endpoint-hold rule: every requested physical timestamp is clamped to
        # the effective spline domain. This is deliberately shared by training
        # and inference, so extrapolation can never produce NaN.
        return torch.maximum(
            torch.minimum(times, right.unsqueeze(-1)), left.unsqueeze(-1)
        )
    if decode_mode == SUPPORT_LINSPACE:
        if times.shape[1] <= 1:
            fraction = torch.zeros(
                (1,), device=times.device, dtype=times.dtype
            )
        else:
            fraction = torch.linspace(
                0.0,
                1.0,
                times.shape[1],
                device=times.device,
                dtype=times.dtype,
            )
        return left.unsqueeze(-1) + fraction * (right - left).unsqueeze(-1)
    raise ValueError(
        f"Unknown decode_mode {decode_mode!r}; expected "
        f"{RAW_TIME_CLAMP!r} or {SUPPORT_LINSPACE!r}"
    )


def decode_bspline_parameters(
    parameters: torch.Tensor,
    times: torch.Tensor,
    degree: int = 3,
    decode_mode: str = RAW_TIME_CLAMP,
    project_knots: bool = False,
    min_knot_delta: float = 1e-4,
    straight_through: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Decode batched B-spline parameters with the de Boor algorithm.

    The effective domain is ``[knots[degree], knots[-degree-1]]``.  Requested
    timestamps outside that domain use endpoint-hold in ``raw_time_clamp``
    mode. ``support_linspace`` is retained only as an evaluation ablation.
    """
    if parameters.ndim == 2:
        parameters = parameters.unsqueeze(0)
    if parameters.ndim != 3:
        raise ValueError("parameters must have shape [B,K,1+D] or [K,1+D]")
    if degree < 0:
        raise ValueError("degree must be non-negative")

    knot_count = parameters.shape[1]
    n_control = knot_count - degree - 1
    if n_control <= degree:
        raise ValueError(
            f"Need at least degree+1 control points, got {n_control}"
        )

    raw_knots = parameters[..., 0]
    if project_knots:
        knots = project_monotonic_knots(
            raw_knots,
            min_delta=min_knot_delta,
            straight_through=straight_through,
        )
    else:
        knots = raw_knots
    controls = parameters[:, :n_control, 1:]

    left = knots[:, degree]
    right = knots[:, -(degree + 1)]
    if torch.any(right <= left):
        raise ValueError("B-spline effective support must have positive width")
    evaluation_times = _prepare_times(times, left, right, decode_mode)

    # searchsorted selects the polynomial span. Clamping to n_control-1 gives
    # SciPy-compatible left-limit behavior at the right support endpoint.
    spans = torch.searchsorted(
        knots.contiguous(), evaluation_times.contiguous(), right=True
    ) - 1
    spans = spans.clamp(min=degree, max=n_control - 1)

    work = []
    for offset in range(degree + 1):
        control_index = spans - degree + offset
        work.append(_batched_gather(controls, control_index))

    tiny = torch.finfo(parameters.dtype).eps
    x = evaluation_times.unsqueeze(-1)
    for recursion in range(1, degree + 1):
        for offset in range(degree, recursion - 1, -1):
            knot_index = spans - degree + offset
            lower = _batched_gather(knots, knot_index)
            upper = _batched_gather(
                knots, knot_index + degree - recursion + 1
            )
            denominator = upper - lower
            safe_denominator = torch.where(
                denominator.abs() > tiny,
                denominator,
                torch.ones_like(denominator),
            )
            alpha = (x - lower.unsqueeze(-1)) / safe_denominator.unsqueeze(-1)
            alpha = torch.where(
                (denominator.abs() > tiny).unsqueeze(-1),
                alpha,
                torch.zeros_like(alpha),
            )
            work[offset] = (
                (1.0 - alpha) * work[offset - 1] + alpha * work[offset]
            )

    projected_parameters = torch.cat(
        [knots.unsqueeze(-1), parameters[:, :, 1:]], dim=-1
    )
    info = {
        "projected_parameters": projected_parameters,
        "evaluation_times": evaluation_times,
        "support_left": left,
        "support_right": right,
        "raw_knots": raw_knots,
        "knots": knots,
    }
    return work[degree], info


def support_mask(
    parameters: torch.Tensor,
    times: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    """Return whether requested raw timestamps lie in the effective domain."""
    if parameters.ndim == 2:
        parameters = parameters.unsqueeze(0)
    knots = parameters[..., 0]
    left = knots[:, degree]
    right = knots[:, -(degree + 1)]
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(parameters.shape[0], -1)
    times = times.to(device=parameters.device, dtype=parameters.dtype)
    return (times >= left.unsqueeze(-1)) & (times <= right.unsqueeze(-1))
