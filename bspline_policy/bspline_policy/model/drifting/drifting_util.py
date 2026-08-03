"""PyTorch implementation of the Drifting distribution-matching loss.

This is the numerically verified implementation carried by the repository's
``drifting_policy`` reference.  It lives in the ``bspline_policy`` namespace
to avoid the two sibling projects' conflicting ``diffusion_policy`` package
names.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _cdist(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise L2 distance for ``[B,N,D]`` and ``[B,M,D]`` inputs."""
    xy_dot = torch.einsum("bnd,bmd->bnm", x, y)
    x_norms = torch.einsum("bnd,bnd->bn", x, x)
    y_norms = torch.einsum("bmd,bmd->bm", y, y)
    squared_distance = (
        x_norms[:, :, None] + y_norms[:, None, :] - 2 * xy_dot
    )
    return torch.sqrt(torch.clamp(squared_distance, min=eps))


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    R_list: Sequence[float] = (0.02, 0.05, 0.2),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the normalized Drifting loss.

    Args:
        gen: Generated samples with shape ``[B, G, D]``.
        fixed_pos: Positive data samples with shape ``[B, P, D]``.
        fixed_neg: Optional negative samples with shape ``[B, N, D]``.
        weight_gen: Optional weights with shape ``[B, G]``.
        weight_pos: Optional weights with shape ``[B, P]``.
        weight_neg: Optional weights with shape ``[B, N]``.
        R_list: Kernel temperatures used to construct the drift field.

    Returns:
        A per-batch-element loss of shape ``[B]`` and detached diagnostic
        scalars.  Gradients flow only through ``generated``.
    """
    generated = gen
    fixed_positive = fixed_pos
    fixed_negative = fixed_neg
    generated_weight = weight_gen
    positive_weight = weight_pos
    negative_weight = weight_neg
    temperatures = R_list

    if generated.ndim != 3 or fixed_positive.ndim != 3:
        raise ValueError("generated and fixed_positive must have shape [B,C,D]")
    if generated.shape[0] != fixed_positive.shape[0]:
        raise ValueError("generated and fixed_positive batch sizes must match")
    if generated.shape[2] != fixed_positive.shape[2]:
        raise ValueError("generated and fixed_positive feature sizes must match")
    if not temperatures:
        raise ValueError("temperatures must contain at least one value")
    if any(float(value) <= 0 for value in temperatures):
        raise ValueError("temperatures must be positive")

    batch_size, generated_count, feature_size = generated.shape
    positive_count = fixed_positive.shape[1]

    if fixed_negative is None:
        fixed_negative = generated.new_zeros(batch_size, 0, feature_size)
    if fixed_negative.shape[0] != batch_size:
        raise ValueError("fixed_negative batch size must match generated")
    if fixed_negative.shape[2] != feature_size:
        raise ValueError("fixed_negative feature size must match generated")
    negative_count = fixed_negative.shape[1]

    if generated_weight is None:
        generated_weight = generated.new_ones(batch_size, generated_count)
    if positive_weight is None:
        positive_weight = generated.new_ones(batch_size, positive_count)
    if negative_weight is None:
        negative_weight = generated.new_ones(batch_size, negative_count)

    # The reference computes the field in float32 even under mixed precision.
    generated = generated.float()
    fixed_positive = fixed_positive.float()
    fixed_negative = fixed_negative.float()
    generated_weight = generated_weight.float()
    positive_weight = positive_weight.float()
    negative_weight = negative_weight.float()

    old_generated = generated.detach()
    targets = torch.cat(
        [old_generated, fixed_negative, fixed_positive],
        dim=1,
    )
    target_weights = torch.cat(
        [generated_weight, negative_weight, positive_weight],
        dim=1,
    )

    with torch.no_grad():
        diagnostics: dict[str, torch.Tensor] = {}
        distances = _cdist(old_generated, targets)
        weighted_distances = distances * target_weights[:, None, :]
        scale = weighted_distances.mean() / target_weights.mean()
        diagnostics["scale"] = scale

        input_scale = torch.clamp(
            scale / (feature_size**0.5),
            min=1e-3,
        )
        old_generated_scaled = old_generated / input_scale
        targets_scaled = targets / input_scale
        normalized_distances = distances / torch.clamp(scale, min=1e-3)

        diagonal_mask = torch.eye(
            generated_count,
            device=generated.device,
            dtype=generated.dtype,
        )
        connection_mask = F.pad(
            diagonal_mask,
            (0, negative_count + positive_count),
        ).unsqueeze(0)
        normalized_distances = normalized_distances + connection_mask * 100.0

        force_across_temperatures = torch.zeros_like(old_generated_scaled)
        for temperature in temperatures:
            logits = -normalized_distances / float(temperature)

            affinity = torch.softmax(logits, dim=-1)
            transposed_affinity = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt(
                torch.clamp(affinity * transposed_affinity, min=1e-6)
            )
            affinity = affinity * target_weights[:, None, :]

            split_index = generated_count + negative_count
            negative_affinity = affinity[:, :, :split_index]
            positive_affinity = affinity[:, :, split_index:]

            positive_sum = positive_affinity.sum(dim=-1, keepdim=True)
            negative_coefficients = -negative_affinity * positive_sum
            negative_sum = negative_affinity.sum(dim=-1, keepdim=True)
            positive_coefficients = positive_affinity * negative_sum
            coefficients = torch.cat(
                [negative_coefficients, positive_coefficients],
                dim=2,
            )

            force = torch.einsum(
                "biy,byx->bix",
                coefficients,
                targets_scaled,
            )
            coefficient_sum = coefficients.sum(dim=-1)
            force = (
                force
                - coefficient_sum.unsqueeze(-1) * old_generated_scaled
            )

            force_norm = force.square().mean()
            diagnostics[f"loss_{temperature}"] = force_norm
            force_scale = torch.sqrt(torch.clamp(force_norm, min=1e-8))
            force_across_temperatures = (
                force_across_temperatures + force / force_scale
            )

        goal_scaled = old_generated_scaled + force_across_temperatures

    generated_scaled = generated / input_scale.detach()
    difference = generated_scaled - goal_scaled.detach()
    loss = difference.square().mean(dim=(-1, -2))
    diagnostics = {
        key: value.mean().detach() for key, value in diagnostics.items()
    }
    return loss, diagnostics
