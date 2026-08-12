from typing import Dict, Optional

import torch
import torch.nn as nn

from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_imeanflow_unet1d import (
    ConditionalIMeanFlowUnet1D,
)
from diffusion_policy.model.diffusion.improved_mean_flow import (
    improved_mean_flow_loss,
    improved_mean_flow_sample,
)
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.models.base_nets as rmbn
import robomimic.utils.obs_utils as ObsUtils


class IMeanFlowUnetHybridImagePolicy(BaseImagePolicy):
    """Image-conditioned Diffusion Policy with DDPM replaced by iMeanFlow."""

    def __init__(
            self,
            shape_meta: dict,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            num_inference_steps: int = 1,
            obs_as_global_cond: bool = True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim: int = 256,
            down_dims=(256, 512, 1024),
            kernel_size: int = 5,
            n_groups: int = 8,
            cond_predict_scale: bool = True,
            obs_encoder_group_norm: bool = False,
            eval_fixed_crop: bool = False,
            use_auxiliary_head: bool = True,
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
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) != 1:
            raise ValueError("only vector actions are supported")
        action_dim = action_shape[0]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = {}
        for key, attr in shape_meta["obs"].items():
            obs_key_shapes[key] = list(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                obs_config["rgb"].append(key)
            elif obs_type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"unsupported observation type: {obs_type}")

        config = get_robomimic_config(
            algo_name="bc_rnn",
            hdf5_type="image",
            task_name="square",
            dataset_type="ph",
        )
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            if crop_shape is None:
                for modality in config.observation.encoder.values():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                crop_height, crop_width = crop_shape
                for modality in config.observation.encoder.values():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = crop_height
                        modality.obs_randomizer_kwargs.crop_width = crop_width

        ObsUtils.initialize_obs_utils_with_config(config)
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )
        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    num_groups=module.num_features // 16,
                    num_channels=module.num_features,
                ),
            )
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda module: isinstance(module, rmbn.CropRandomizer),
                func=lambda module: dmvc.CropRandomizer(
                    input_shape=module.input_shape,
                    crop_height=module.crop_height,
                    crop_width=module.crop_width,
                    num_crops=module.num_crops,
                    pos_enc=module.pos_enc,
                ),
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalIMeanFlowUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            use_auxiliary_head=use_auxiliary_head,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.data_proportion = data_proportion
        self.p_mean = p_mean
        self.p_std = p_std
        self.adaptive_loss = adaptive_loss
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.clip_sample = clip_sample
        self.loss_metrics = {}

        print("iMeanFlow params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))

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
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action conditioning is not implemented")
        normalized_obs = self.normalizer.normalize(obs_dict)
        value = next(iter(normalized_obs.values()))
        batch_size = value.shape[0]
        obs_steps = self.n_obs_steps

        if self.obs_as_global_cond:
            this_obs = dict_apply(
                normalized_obs,
                lambda value: value[:, :obs_steps].reshape(-1, *value.shape[2:]),
            )
            obs_features = self.obs_encoder(this_obs)
            global_cond = obs_features.reshape(batch_size, -1)
            condition_data = torch.zeros(
                (batch_size, self.horizon, self.action_dim),
                device=self.device,
                dtype=self.dtype,
            )
            condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        else:
            this_obs = dict_apply(
                normalized_obs,
                lambda value: value[:, :obs_steps].reshape(-1, *value.shape[2:]),
            )
            obs_features = self.obs_encoder(this_obs).reshape(batch_size, obs_steps, -1)
            condition_data = torch.zeros(
                (batch_size, self.horizon, self.action_dim + self.obs_feature_dim),
                device=self.device,
                dtype=self.dtype,
            )
            condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
            condition_data[:, :obs_steps, self.action_dim:] = obs_features
            condition_mask[:, :obs_steps, self.action_dim:] = True
            global_cond = None

        normalized_sample = self.conditional_sample(
            condition_data,
            condition_mask,
            global_cond=global_cond,
        )
        normalized_action = normalized_sample[..., :self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(normalized_action)
        start = obs_steps - 1
        action = action_pred[:, start:start + self.n_action_steps]
        return {"action": action, "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: dict) -> torch.Tensor:
        if "valid_mask" in batch:
            raise NotImplementedError("valid_mask is not supported")
        normalized_obs = self.normalizer.normalize(batch["obs"])
        normalized_action = self.normalizer["action"].normalize(batch["action"])
        batch_size, horizon = normalized_action.shape[:2]

        trajectory = normalized_action
        global_cond = None
        if self.obs_as_global_cond:
            this_obs = dict_apply(
                normalized_obs,
                lambda value: value[:, :self.n_obs_steps].reshape(
                    -1, *value.shape[2:]),
            )
            obs_features = self.obs_encoder(this_obs)
            global_cond = obs_features.reshape(batch_size, -1)
        else:
            this_obs = dict_apply(
                normalized_obs,
                lambda value: value.reshape(-1, *value.shape[2:]),
            )
            obs_features = self.obs_encoder(this_obs).reshape(batch_size, horizon, -1)
            trajectory = torch.cat([normalized_action, obs_features], dim=-1)

        condition_mask = self.mask_generator(trajectory.shape)
        loss, metrics = improved_mean_flow_loss(
            model=self.model,
            clean_trajectory=trajectory,
            condition_mask=condition_mask,
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
