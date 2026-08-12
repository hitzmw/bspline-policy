"""Improved MeanFlow policy over B-spline parameters with curve loss."""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import torch

from bspline_policy.common.torch_bspline import (
    RAW_TIME_CLAMP,
    SUPPORT_LINSPACE,
    decode_bspline_parameters,
    support_mask,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.improved_mean_flow import (
    improved_mean_flow_loss,
)
from diffusion_policy.policy.imeanflow_unet_hybrid_image_policy import (
    IMeanFlowUnetHybridImagePolicy,
)


def _masked_curve_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target curve shapes must match")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("curve mask must have shape [B,T]")
    weighted_error = (prediction - target).square() * mask.unsqueeze(-1)
    denominator = (
        mask.to(prediction.dtype).sum() * prediction.shape[-1]
    ).clamp_min(1.0)
    return weighted_error.sum() / denominator


def _masked_fraction(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dtype = torch.float32 if not values.is_floating_point() else values.dtype
    denominator = mask.to(dtype).sum().clamp_min(1.0)
    return (values.to(dtype) * mask.to(dtype)).sum() / denominator


class IMeanFlowUnetBSplineImagePolicy(IMeanFlowUnetHybridImagePolicy):
    """Replace DDPM with iMeanFlow and train in parameter and curve spaces.

    ``shape_meta`` describes physical actions.  One knot channel is added only
    for the model and B-spline parameter normalizer.
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        execution_action_steps: int = 8,
        bspline_degree: int = 3,
        relative_knots: bool = False,
        reconstruction_loss_weight: float = 0.05,
        reconstruction_warmup_ratio: float = 0.1,
        min_knot_delta: float = 1e-4,
        rollout_decode_mode: str = RAW_TIME_CLAMP,
        **kwargs,
    ):
        if relative_knots:
            raise ValueError(
                "The first iMeanFlow-BSpline implementation requires "
                "relative_knots=false"
            )
        if rollout_decode_mode not in {RAW_TIME_CLAMP, SUPPORT_LINSPACE}:
            raise ValueError(f"Unsupported rollout_decode_mode: {rollout_decode_mode}")
        if int(n_action_steps) != int(horizon):
            raise ValueError(
                "A B-spline policy must generate the complete parameter chunk: "
                "n_action_steps must equal horizon"
            )
        if reconstruction_loss_weight < 0:
            raise ValueError("reconstruction_loss_weight must be non-negative")
        if not 0.0 <= reconstruction_warmup_ratio <= 1.0:
            raise ValueError("reconstruction_warmup_ratio must be in [0,1]")

        bspline_shape_meta = copy.deepcopy(shape_meta)
        physical_action_shape = bspline_shape_meta["action"]["shape"]
        if len(physical_action_shape) != 1:
            raise ValueError("shape_meta.action.shape must be one-dimensional")
        self.regular_action_dim = int(physical_action_shape[0])
        bspline_shape_meta["action"]["shape"] = [self.regular_action_dim + 1]

        super().__init__(
            shape_meta=bspline_shape_meta,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            **kwargs,
        )
        if not self.obs_as_global_cond:
            raise ValueError(
                "IMeanFlowUnetBSplineImagePolicy requires obs_as_global_cond=true"
            )
        if self.num_inference_steps != 1:
            raise ValueError(
                "IMeanFlowUnetBSplineImagePolicy requires one inference step"
            )
        if not self.clip_sample:
            raise ValueError(
                "clip_sample must be true so reconstruction and rollout agree"
            )

        self.execution_action_steps = int(execution_action_steps)
        self.bspline_degree = int(bspline_degree)
        self.relative_knots = False
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.reconstruction_warmup_ratio = float(reconstruction_warmup_ratio)
        self.min_knot_delta = float(min_knot_delta)
        self.rollout_decode_mode = rollout_decode_mode
        self._current_reconstruction_weight = self.reconstruction_loss_weight

    def set_training_step(self, optimizer_step: int, total_optimizer_steps: int):
        """Update the reconstruction weight using optimizer-step progress."""
        warmup_steps = int(
            math.ceil(total_optimizer_steps * self.reconstruction_warmup_ratio)
        )
        if warmup_steps <= 0:
            fraction = 1.0
        else:
            fraction = min(1.0, max(0.0, optimizer_step / warmup_steps))
        self._current_reconstruction_weight = (
            self.reconstruction_loss_weight * fraction
        )

    def _encode_observation(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if "past_action" in obs_dict:
            raise ValueError("past_action conditioning is not implemented")
        normalized_observation = self.normalizer.normalize(obs_dict)
        value = next(iter(normalized_observation.values()))
        batch_size = value.shape[0]
        sliced_observation = dict_apply(
            normalized_observation,
            lambda tensor: tensor[:, : self.n_obs_steps].reshape(
                -1, *tensor.shape[2:]
            ),
        )
        features = self.obs_encoder(sliced_observation)
        return features.reshape(batch_size, -1)

    def _one_step_normalized_parameters(
        self,
        global_condition: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Shared differentiable ``1 -> 0`` map for training and rollout."""
        batch_size = global_condition.shape[0]
        expected_shape = (batch_size, self.horizon, self.action_dim)
        if noise is None:
            noise = torch.randn(
                expected_shape,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
        elif tuple(noise.shape) != expected_shape:
            raise ValueError(
                f"Expected noise shape {expected_shape}, got {tuple(noise.shape)}"
            )
        t = torch.ones(batch_size, device=noise.device, dtype=noise.dtype)
        r = torch.zeros_like(t)
        output = self.model(
            noise,
            t,
            r,
            global_cond=global_condition,
        )
        average_velocity = output[0] if isinstance(output, (tuple, list)) else output
        unclipped = noise - average_velocity
        clipped = unclipped.clamp(-1.0, 1.0)
        return clipped, {
            "noise": noise,
            "unclipped": unclipped,
            "clip_fraction": ((unclipped < -1.0) | (unclipped > 1.0))
            .to(unclipped.dtype)
            .mean(),
        }

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        decode_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        global_condition = self._encode_observation(obs_dict)
        normalized_parameters, generation_info = (
            self._one_step_normalized_parameters(
                global_condition,
                generator=generator,
            )
        )
        parameters = self.normalizer["action"].unnormalize(
            normalized_parameters
        )
        raw_times = torch.arange(
            self.execution_action_steps,
            device=parameters.device,
            dtype=parameters.dtype,
        ).expand(parameters.shape[0], -1)
        decoded, decode_info = decode_bspline_parameters(
            parameters,
            raw_times,
            degree=self.bspline_degree,
            decode_mode=decode_mode or self.rollout_decode_mode,
            project_knots=True,
            min_knot_delta=self.min_knot_delta,
            straight_through=True,
        )
        return {
            "action": decoded,
            "action_pred": parameters,
            "bspline_action": parameters,
            "projected_bspline_action": decode_info["projected_parameters"],
            "normalized_bspline_action": normalized_parameters,
            "clip_fraction": generation_info["clip_fraction"],
        }

    def compute_loss(
        self,
        batch: dict,
        reconstruction_weight: Optional[float] = None,
        return_info: bool = False,
    ):
        required = {
            "obs",
            "action",
            "raw_action",
            "raw_action_time",
            "raw_action_mask",
        }
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"Missing reconstruction batch fields: {sorted(missing)}")

        normalized_parameters = self.normalizer["action"].normalize(
            batch["action"]
        )
        batch_size, horizon, action_dim = normalized_parameters.shape
        if horizon != self.horizon or action_dim != self.action_dim:
            raise ValueError(
                "Expected normalized B-spline parameters with shape "
                f"[B,{self.horizon},{self.action_dim}], got "
                f"{tuple(normalized_parameters.shape)}"
            )
        global_condition = self._encode_observation(batch["obs"])
        condition_mask = torch.zeros_like(normalized_parameters, dtype=torch.bool)
        noise = torch.randn_like(normalized_parameters)

        imeanflow_loss, imeanflow_metrics = improved_mean_flow_loss(
            model=self.model,
            clean_trajectory=normalized_parameters,
            condition_mask=condition_mask,
            global_cond=global_condition,
            data_proportion=self.data_proportion,
            p_mean=self.p_mean,
            p_std=self.p_std,
            adaptive_loss=self.adaptive_loss,
            norm_p=self.norm_p,
            norm_eps=self.norm_eps,
            auxiliary_loss_weight=self.auxiliary_loss_weight,
            noise=noise,
        )

        generated_normalized, generation_info = (
            self._one_step_normalized_parameters(global_condition, noise=noise)
        )
        generated_parameters = self.normalizer["action"].unnormalize(
            generated_normalized
        )
        raw_times = batch["raw_action_time"].to(
            device=generated_parameters.device,
            dtype=generated_parameters.dtype,
        )
        generated_curve, generated_decode_info = decode_bspline_parameters(
            generated_parameters,
            raw_times,
            degree=self.bspline_degree,
            decode_mode=RAW_TIME_CLAMP,
            project_knots=True,
            min_knot_delta=self.min_knot_delta,
            straight_through=True,
        )
        target_curve, _ = decode_bspline_parameters(
            batch["action"],
            raw_times,
            degree=self.bspline_degree,
            decode_mode=RAW_TIME_CLAMP,
            project_knots=False,
        )

        normalized_generated_curve = self.normalizer["raw_action"].normalize(
            generated_curve
        )
        normalized_target_curve = self.normalizer["raw_action"].normalize(
            target_curve
        )
        normalized_raw_action = self.normalizer["raw_action"].normalize(
            batch["raw_action"]
        )
        reconstruction_mask = batch["raw_action_mask"].to(torch.bool)
        episode_mask = batch.get("raw_action_episode_mask", reconstruction_mask)
        episode_mask = episode_mask.to(torch.bool)

        fit_error = _masked_curve_mse(
            normalized_target_curve,
            normalized_raw_action,
            reconstruction_mask,
        )
        parameter_to_curve_error = _masked_curve_mse(
            normalized_generated_curve,
            normalized_target_curve,
            reconstruction_mask,
        )
        final_error = _masked_curve_mse(
            normalized_generated_curve,
            normalized_raw_action,
            reconstruction_mask,
        )
        weight = (
            self._current_reconstruction_weight
            if reconstruction_weight is None
            else float(reconstruction_weight)
        )
        total_loss = imeanflow_loss + weight * final_error

        projected_parameters = generated_decode_info["projected_parameters"]
        generated_coverage = support_mask(
            projected_parameters,
            raw_times,
            degree=self.bspline_degree,
        )
        target_coverage = support_mask(
            batch["action"], raw_times, degree=self.bspline_degree
        )
        raw_knots = generated_parameters[..., 0]
        knot_violation_fraction = (
            raw_knots[:, 1:]
            < raw_knots[:, :-1] + self.min_knot_delta
        ).to(raw_knots.dtype).mean()

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_imeanflow": imeanflow_loss.detach(),
            "loss_imeanflow_u": imeanflow_metrics["loss_u"],
            "loss_imeanflow_v": imeanflow_metrics["loss_v"],
            "loss_reconstruction": final_error.detach(),
            "reconstruction_weight": normalized_parameters.new_tensor(weight),
            "error_fit": fit_error.detach(),
            "error_parameter_to_curve": parameter_to_curve_error.detach(),
            "error_final": final_error.detach(),
            "clip_fraction": generation_info["clip_fraction"].detach(),
            "knot_violation_fraction": knot_violation_fraction.detach(),
            "generated_support_coverage": _masked_fraction(
                generated_coverage, episode_mask
            ).detach(),
            "target_support_coverage": _masked_fraction(
                target_coverage, episode_mask
            ).detach(),
            "valid_reconstruction_fraction": (
                reconstruction_mask.to(torch.float32).sum()
                / episode_mask.to(torch.float32).sum().clamp_min(1.0)
            ).detach(),
            "mean_t": imeanflow_metrics["mean_t"],
            "mean_interval": imeanflow_metrics["mean_interval"],
            "flow_ratio": imeanflow_metrics["flow_ratio"],
        }
        self.loss_metrics = metrics
        if return_info:
            return total_loss, metrics
        return total_loss
