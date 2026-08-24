"""One-step Drifting policy over dense image-conditioned action chunks."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from bspline_policy.model.drifting.drifting_util import drift_loss
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.models.base_nets as rmbn
import robomimic.utils.obs_utils as ObsUtils


class DriftingUnetHybridImagePolicy(BaseImagePolicy):
    """Predict a dense action trajectory with one conditional UNet call.

    This is the repository's native Drifting image policy ported into the
    package used by the RoboCasa experiments.  In particular, ``action`` is a
    physical action sequence with shape ``[B, horizon, action_dim]``: no knot,
    control-point, projection, fitting, or B-spline decoding is involved.
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        obs_as_global_cond: bool = True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        obs_encoder_group_norm: bool = False,
        eval_fixed_crop: bool = False,
        temperatures=(0.02, 0.05, 0.2),
        per_timestep_loss: bool = True,
        gen_per_label: int = 8,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError(
                "DriftingUnetHybridImagePolicy requires "
                "obs_as_global_cond=True"
            )
        if int(gen_per_label) < 2:
            raise ValueError(
                "gen_per_label must be at least 2 for the Drifting self-mask"
            )

        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) != 1:
            raise ValueError("shape_meta.action.shape must be one-dimensional")
        self.action_dim = int(action_shape[0])

        observation_config = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": [],
        }
        observation_key_shapes = {}
        for key, attributes in shape_meta["obs"].items():
            observation_key_shapes[key] = list(attributes["shape"])
            observation_type = attributes.get("type", "low_dim")
            if observation_type == "rgb":
                observation_config["rgb"].append(key)
            elif observation_type == "low_dim":
                observation_config["low_dim"].append(key)
            else:
                raise RuntimeError(
                    f"Unsupported observation type: {observation_type}"
                )

        robomimic_config = get_robomimic_config(
            algo_name="bc_rnn",
            hdf5_type="image",
            task_name="square",
            dataset_type="ph",
        )
        with robomimic_config.unlocked():
            robomimic_config.observation.modalities.obs = observation_config
            if crop_shape is None:
                for modality in robomimic_config.observation.encoder.values():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                crop_height, crop_width = crop_shape
                for modality in robomimic_config.observation.encoder.values():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = crop_height
                        modality.obs_randomizer_kwargs.crop_width = crop_width

        ObsUtils.initialize_obs_utils_with_config(robomimic_config)
        encoder_policy: PolicyAlgo = algo_factory(
            algo_name=robomimic_config.algo_name,
            config=robomimic_config,
            obs_key_shapes=observation_key_shapes,
            ac_dim=self.action_dim,
            device="cpu",
        )
        self.obs_encoder = (
            encoder_policy.nets["policy"].nets["encoder"].nets["obs"]
        )
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=self.obs_encoder,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    num_groups=module.num_features // 16,
                    num_channels=module.num_features,
                ),
            )
        if eval_fixed_crop:
            replace_submodules(
                root_module=self.obs_encoder,
                predicate=lambda module: isinstance(
                    module, rmbn.CropRandomizer
                ),
                func=lambda module: CropRandomizer(
                    input_shape=module.input_shape,
                    crop_height=module.crop_height,
                    crop_width=module.crop_width,
                    num_crops=module.num_crops,
                    pos_enc=module.pos_enc,
                ),
            )

        self.obs_feature_dim = int(self.obs_encoder.output_shape()[0])
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=self.obs_feature_dim * int(n_obs_steps),
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_global_cond = True
        self.temperatures = tuple(float(value) for value in temperatures)
        self.per_timestep_loss = bool(per_timestep_loss)
        self.gen_per_label = int(gen_per_label)

        action_start = self.n_obs_steps - 1
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        if self.n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        if action_start + self.n_action_steps > self.horizon:
            raise ValueError(
                "The requested execution chunk does not fit in the horizon: "
                f"start={action_start}, n_action_steps={self.n_action_steps}, "
                f"horizon={self.horizon}"
            )

        print(
            "Drifting params: %e"
            % sum(parameter.numel() for parameter in self.model.parameters())
        )
        print(
            "Vision params: %e"
            % sum(
                parameter.numel()
                for parameter in self.obs_encoder.parameters()
            )
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _encode_observation(
        self,
        obs_dict: Dict[str, torch.Tensor],
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

    def _generate_normalized_action(
        self,
        global_condition: torch.Tensor,
        samples_per_observation: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size = global_condition.shape[0]
        repeated_condition = global_condition.repeat_interleave(
            samples_per_observation,
            dim=0,
        )
        noise = torch.randn(
            (
                batch_size * samples_per_observation,
                self.horizon,
                self.action_dim,
            ),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        timesteps = torch.zeros(
            batch_size * samples_per_observation,
            device=self.device,
            dtype=torch.long,
        )
        generated = self.model(
            noise,
            timesteps,
            global_cond=repeated_condition,
        )
        return generated.reshape(
            batch_size,
            samples_per_observation,
            self.horizon,
            self.action_dim,
        )

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        global_condition = self._encode_observation(obs_dict)
        normalized_action = self._generate_normalized_action(
            global_condition,
            samples_per_observation=1,
            generator=generator,
        )[:, 0]
        action_prediction = self.normalizer["action"].unnormalize(
            normalized_action
        )
        action_start = self.n_obs_steps - 1
        action_end = action_start + self.n_action_steps
        return {
            "action": action_prediction[:, action_start:action_end],
            "action_pred": action_prediction,
        }

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        return_info: bool = False,
    ):
        normalized_observation = self.normalizer.normalize(batch["obs"])
        normalized_action = self.normalizer["action"].normalize(
            batch["action"]
        )
        batch_size, horizon, action_dim = normalized_action.shape
        if horizon != self.horizon or action_dim != self.action_dim:
            raise ValueError(
                "Expected dense actions with shape "
                f"[B,{self.horizon},{self.action_dim}], got "
                f"{tuple(normalized_action.shape)}"
            )

        sliced_observation = dict_apply(
            normalized_observation,
            lambda tensor: tensor[:, : self.n_obs_steps].reshape(
                -1, *tensor.shape[2:]
            ),
        )
        observation_features = self.obs_encoder(sliced_observation)
        global_condition = observation_features.reshape(batch_size, -1)
        generated = self._generate_normalized_action(
            global_condition,
            samples_per_observation=self.gen_per_label,
        )

        if self.per_timestep_loss:
            loss, diagnostics = self._per_timestep_drift_loss(
                generated,
                normalized_action,
            )
        else:
            generated_flat = generated.reshape(
                batch_size,
                self.gen_per_label,
                -1,
            )
            target_flat = normalized_action.reshape(batch_size, 1, -1)
            per_item_loss, diagnostics = drift_loss(
                generated_flat,
                target_flat,
                R_list=self.temperatures,
            )
            loss = per_item_loss.mean()

        if return_info:
            return loss, diagnostics
        return loss

    def _per_timestep_drift_loss(
        self,
        generated: torch.Tensor,
        normalized_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total_loss = normalized_action.new_zeros(())
        diagnostics: dict[str, list[torch.Tensor]] = {}
        for timestep in range(self.horizon):
            timestep_loss, timestep_diagnostics = drift_loss(
                generated[:, :, timestep, :],
                normalized_action[:, timestep, :].unsqueeze(1),
                R_list=self.temperatures,
            )
            total_loss = total_loss + timestep_loss.mean()
            for key, value in timestep_diagnostics.items():
                diagnostics.setdefault(key, []).append(value)
        return total_loss / self.horizon, {
            key: torch.stack(values).mean().detach()
            for key, values in diagnostics.items()
        }
