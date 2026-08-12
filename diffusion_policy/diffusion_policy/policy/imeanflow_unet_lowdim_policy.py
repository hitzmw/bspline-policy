from typing import Dict, Optional

import torch

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_imeanflow_unet1d import (
    ConditionalIMeanFlowUnet1D,
)
from diffusion_policy.model.diffusion.improved_mean_flow import (
    improved_mean_flow_loss,
    improved_mean_flow_sample,
)
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class IMeanFlowUnetLowdimPolicy(BaseLowdimPolicy):
    """Improved MeanFlow replacement for the low-dimensional DDPM policy."""

    def __init__(
            self,
            model: ConditionalIMeanFlowUnet1D,
            horizon: int,
            obs_dim: int,
            action_dim: int,
            n_action_steps: int,
            n_obs_steps: int,
            num_inference_steps: int = 1,
            obs_as_local_cond: bool = False,
            obs_as_global_cond: bool = False,
            pred_action_steps_only: bool = False,
            oa_step_convention: bool = False,
            data_proportion: float = 0.5,
            p_mean: float = -0.4,
            p_std: float = 1.0,
            adaptive_loss: bool = True,
            norm_p: float = 1.0,
            norm_eps: float = 0.01,
            auxiliary_loss_weight: float = 1.0,
            clip_sample: bool = True,
            noise_scheduler=None,
            **kwargs):
        super().__init__()
        if obs_as_local_cond and obs_as_global_cond:
            raise ValueError("local and global observation conditioning are exclusive")
        if pred_action_steps_only and not obs_as_global_cond:
            raise ValueError("pred_action_steps_only requires global conditioning")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.data_proportion = data_proportion
        self.p_mean = p_mean
        self.p_std = p_std
        self.adaptive_loss = adaptive_loss
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.clip_sample = clip_sample
        self.loss_metrics = {}

    def conditional_sample(
            self,
            condition_data: torch.Tensor,
            condition_mask: torch.Tensor,
            local_cond: Optional[torch.Tensor] = None,
            global_cond: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            **kwargs) -> torch.Tensor:
        return improved_mean_flow_sample(
            model=self.model,
            condition_data=condition_data,
            condition_mask=condition_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            clip_sample=self.clip_sample,
        )

    def predict_action(
            self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "obs" not in obs_dict:
            raise KeyError("obs_dict must contain 'obs'")
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action conditioning is not implemented")

        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        batch_size, _, obs_dim = nobs.shape
        if obs_dim != self.obs_dim:
            raise ValueError(f"expected obs_dim={self.obs_dim}, got {obs_dim}")
        horizon = self.horizon
        obs_steps = self.n_obs_steps
        action_dim = self.action_dim

        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            local_cond = torch.zeros(
                (batch_size, horizon, obs_dim), device=self.device, dtype=self.dtype)
            local_cond[:, :obs_steps] = nobs[:, :obs_steps]
            shape = (batch_size, horizon, action_dim)
        elif self.obs_as_global_cond:
            global_cond = nobs[:, :obs_steps].reshape(batch_size, -1)
            shape = (batch_size, horizon, action_dim)
            if self.pred_action_steps_only:
                shape = (batch_size, self.n_action_steps, action_dim)
        else:
            shape = (batch_size, horizon, action_dim + obs_dim)

        condition_data = torch.zeros(shape, device=self.device, dtype=self.dtype)
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            condition_data[:, :obs_steps, action_dim:] = nobs[:, :obs_steps]
            condition_mask[:, :obs_steps, action_dim:] = True

        nsample = self.conditional_sample(
            condition_data,
            condition_mask,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        normalized_action = nsample[..., :action_dim]
        action_pred = self.normalizer["action"].unnormalize(normalized_action)

        if self.pred_action_steps_only:
            action = action_pred
            start = 0
            end = self.n_action_steps
        else:
            start = obs_steps - 1 if self.oa_step_convention else obs_steps
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            normalized_obs = nsample[..., action_dim:]
            obs_pred = self.normalizer["obs"].unnormalize(normalized_obs)
            result["action_obs_pred"] = obs_pred[:, start:end]
            result["obs_pred"] = obs_pred
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: dict) -> torch.Tensor:
        if "valid_mask" in batch:
            raise NotImplementedError("valid_mask is not supported")
        normalized_batch = self.normalizer.normalize(batch)
        obs = normalized_batch["obs"]
        action = normalized_batch["action"]

        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            local_cond = obs.clone()
            local_cond[:, self.n_obs_steps:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, :self.n_obs_steps].reshape(obs.shape[0], -1)
            if self.pred_action_steps_only:
                start = (
                    self.n_obs_steps - 1
                    if self.oa_step_convention else self.n_obs_steps)
                trajectory = action[:, start:start + self.n_action_steps]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        loss, metrics = improved_mean_flow_loss(
            model=self.model,
            clean_trajectory=trajectory,
            condition_mask=condition_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            data_proportion=self.data_proportion,
            p_mean=self.p_mean,
            p_std=self.p_std,
            adaptive_loss=self.adaptive_loss,
            norm_p=self.norm_p,
            norm_eps=self.norm_eps,
            auxiliary_loss_weight=self.auxiliary_loss_weight,
        )
        self.loss_metrics = metrics
        return loss
