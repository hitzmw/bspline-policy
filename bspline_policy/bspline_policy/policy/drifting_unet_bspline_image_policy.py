"""One-step Drifting policy over complete B-spline parameter chunks."""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn

from bspline_policy.common.bspline_action import (
    decode_bspline_action_torch,
    project_monotonic_knots_torch,
)
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


class DriftingUnetBSplineImagePolicy(BaseImagePolicy):
    """Predict a full B-spline parameter matrix with one UNet evaluation.

    ``shape_meta`` continues to describe the physical action dimensions.  The
    policy adds one model channel for knots, matching the B-spline datasets'
    ``[knot, control_point...]`` action representation.  The opt-in dense-action
    target supports three training modes:

    - ``joint`` preserves the original 15D joint-Drifting implementation for
      checkpoint compatibility.
    - ``auxiliary`` keeps Drifting strictly in the 8D B-spline space and uses a
      separate observation-conditioned head for normalized dense-action MSE.
    - ``decode_consistency`` keeps the 8D Drifting model and directly compares
      differentiably decoded B-spline samples with time-aligned dense actions.

    Rollout always decodes only the B-spline prediction.
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
        bspline_degree: int = 3,
        raw_action_concat: bool = False,
        raw_action_training_mode: str = "joint",
        raw_action_loss_weight: float = 0.1,
        raw_action_hidden_dim: int = 256,
        raw_action_consistency_detach_knots: bool = True,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError(
                "DriftingUnetBSplineImagePolicy requires "
                "obs_as_global_cond=True"
            )
        if int(gen_per_label) < 2:
            raise ValueError(
                "gen_per_label must be at least 2 for the Drifting self-mask"
            )
        if not per_timestep_loss:
            raise ValueError(
                "The canonical Drifting-BSpline configuration requires "
                "per_timestep_loss=True"
            )

        bspline_shape_meta = copy.deepcopy(shape_meta)
        physical_action_shape = bspline_shape_meta["action"]["shape"]
        if len(physical_action_shape) != 1:
            raise ValueError("shape_meta.action.shape must be one-dimensional")
        self.regular_action_dim = int(physical_action_shape[0])
        self.bspline_action_dim = self.regular_action_dim + 1
        self.raw_action_concat = bool(raw_action_concat)
        self.raw_action_training_mode = str(raw_action_training_mode)
        if self.raw_action_training_mode not in {
            "joint",
            "auxiliary",
            "decode_consistency",
        }:
            raise ValueError(
                "raw_action_training_mode must be 'joint', 'auxiliary', or "
                "'decode_consistency'"
            )
        if (
            not self.raw_action_concat
            and self.raw_action_training_mode != "joint"
        ):
            raise ValueError(
                "A non-joint raw_action_training_mode requires "
                "raw_action_concat=True"
            )
        self.raw_action_loss_weight = float(raw_action_loss_weight)
        if self.raw_action_loss_weight < 0:
            raise ValueError("raw_action_loss_weight must be non-negative")
        if int(raw_action_hidden_dim) < 1:
            raise ValueError("raw_action_hidden_dim must be positive")
        self.raw_action_consistency_detach_knots = bool(
            raw_action_consistency_detach_knots
        )

        self.action_dim = self.bspline_action_dim + (
            self.regular_action_dim if self.raw_action_concat else 0
        )
        self.model_action_dim = (
            self.bspline_action_dim
            if self.raw_action_training_mode in {
                "auxiliary",
                "decode_consistency",
            }
            else self.action_dim
        )
        bspline_shape_meta["action"]["shape"] = [self.model_action_dim]

        observation_shape_meta = bspline_shape_meta["obs"]
        observation_config = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": [],
        }
        observation_key_shapes = {}
        for key, attributes in observation_shape_meta.items():
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
            ac_dim=self.model_action_dim,
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
        global_condition_dim = self.obs_feature_dim * int(n_obs_steps)
        self.model = ConditionalUnet1D(
            input_dim=self.model_action_dim,
            local_cond_dim=None,
            global_cond_dim=global_condition_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.raw_action_head: nn.Module | None = None
        if self.raw_action_training_mode == "auxiliary":
            self.raw_action_head = nn.Sequential(
                nn.Linear(global_condition_dim, int(raw_action_hidden_dim)),
                nn.Mish(),
                nn.Linear(
                    int(raw_action_hidden_dim),
                    int(horizon) * self.regular_action_dim,
                ),
            )
        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_global_cond = True
        self.temperatures = tuple(float(value) for value in temperatures)
        self.per_timestep_loss = True
        self.gen_per_label = int(gen_per_label)
        self.bspline_degree = int(bspline_degree)

        if self.n_action_steps != self.horizon:
            raise ValueError(
                "A B-spline policy must return its complete parameter chunk: "
                "n_action_steps must equal horizon"
            )

        print(
            "Drifting params: %e"
            % sum(parameter.numel() for parameter in self.model.parameters())
        )
        if self.raw_action_head is not None:
            print(
                "Raw-action auxiliary params: %e"
                % sum(
                    parameter.numel()
                    for parameter in self.raw_action_head.parameters()
                )
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

    def _action_normalizer_slice(
        self,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-row scale and offset for one joint-action channel slice."""
        action_normalizer = self.normalizer["action"]
        scale = action_normalizer.params_dict["scale"]
        offset = action_normalizer.params_dict["offset"]
        expected_size = self.horizon * self.action_dim
        if scale.numel() != expected_size or offset.numel() != expected_size:
            raise RuntimeError(
                "Expected flattened action normalizer parameters with "
                f"{expected_size} values, got scale={scale.numel()} and "
                f"offset={offset.numel()}"
            )
        scale = scale.reshape(self.horizon, self.action_dim)[..., start:end]
        offset = offset.reshape(self.horizon, self.action_dim)[..., start:end]
        return scale, offset

    def _unnormalize_action_slice(
        self,
        normalized_action: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        scale, offset = self._action_normalizer_slice(start, end)
        return (normalized_action - offset) / scale

    def _normalize_action_slice(
        self,
        action: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        scale, offset = self._action_normalizer_slice(start, end)
        return action * scale + offset

    def _decode_bspline_prediction(
        self,
        bspline_action: torch.Tensor,
        num_actions: int,
        detach_knots: bool = False,
    ) -> torch.Tensor:
        projected = project_monotonic_knots_torch(bspline_action)
        if detach_knots:
            projected = torch.cat(
                [projected[..., :1].detach(), projected[..., 1:]],
                dim=-1,
            )
        return decode_bspline_action_torch(
            projected,
            degree=self.bspline_degree,
            num_actions=num_actions,
        )

    def _predict_normalized_raw_action(
        self,
        global_condition: torch.Tensor,
    ) -> torch.Tensor:
        if self.raw_action_head is None:
            raise RuntimeError("Raw-action auxiliary head is not configured")
        return self.raw_action_head(global_condition).reshape(
            global_condition.shape[0],
            self.horizon,
            self.regular_action_dim,
        )

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
                self.model_action_dim,
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
            self.model_action_dim,
        )

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        global_condition = self._encode_observation(obs_dict)
        normalized_model_action = self._generate_normalized_action(
            global_condition,
            samples_per_observation=1,
            generator=generator,
        )[:, 0]
        if self.raw_action_training_mode in {
            "auxiliary",
            "decode_consistency",
        }:
            bspline_action_prediction = self._unnormalize_action_slice(
                normalized_model_action,
                0,
                self.bspline_action_dim,
            )
            if self.raw_action_training_mode == "auxiliary":
                normalized_raw_action = self._predict_normalized_raw_action(
                    global_condition
                )
                raw_action_prediction = self._unnormalize_action_slice(
                    normalized_raw_action,
                    self.bspline_action_dim,
                    self.action_dim,
                )
            else:
                raw_action_prediction = self._decode_bspline_prediction(
                    bspline_action_prediction,
                    num_actions=self.horizon,
                )
            joint_action_prediction = torch.cat(
                [bspline_action_prediction, raw_action_prediction],
                dim=-1,
            )
        else:
            joint_action_prediction = self.normalizer["action"].unnormalize(
                normalized_model_action
            )
            bspline_action_prediction = joint_action_prediction[
                ..., : self.bspline_action_dim
            ]
        result = {
            "action": bspline_action_prediction,
            "action_pred": bspline_action_prediction,
            "bspline_action": bspline_action_prediction,
        }
        if self.raw_action_concat:
            result["joint_action_pred"] = joint_action_prediction
            result["raw_action_pred"] = joint_action_prediction[
                ..., self.bspline_action_dim:
            ]
        return result

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
                "Expected normalized B-spline training targets with shape "
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

        drifting_target = normalized_action
        if self.raw_action_training_mode in {
            "auxiliary",
            "decode_consistency",
        }:
            drifting_target = normalized_action[..., : self.bspline_action_dim]

        total_loss = normalized_action.new_zeros(())
        diagnostics: dict[str, list[torch.Tensor]] = {}
        for timestep in range(horizon):
            timestep_loss, timestep_diagnostics = drift_loss(
                generated[:, :, timestep, :],
                drifting_target[:, timestep, :].unsqueeze(1),
                R_list=self.temperatures,
            )
            total_loss = total_loss + timestep_loss.mean()
            for key, value in timestep_diagnostics.items():
                diagnostics.setdefault(key, []).append(value)
        drifting_loss = total_loss / horizon
        mean_diagnostics = {
            key: torch.stack(values).mean().detach()
            for key, values in diagnostics.items()
        }
        loss = drifting_loss
        if self.raw_action_training_mode == "auxiliary":
            normalized_raw_prediction = self._predict_normalized_raw_action(
                global_condition
            )
            normalized_raw_target = normalized_action[
                ..., self.bspline_action_dim:
            ]
            raw_action_aux_loss = nn.functional.mse_loss(
                normalized_raw_prediction,
                normalized_raw_target,
            )
            weighted_raw_action_loss = (
                self.raw_action_loss_weight * raw_action_aux_loss
            )
            loss = drifting_loss + weighted_raw_action_loss
            mean_diagnostics.update(
                {
                    "drift_loss": drifting_loss.detach(),
                    "raw_action_aux_loss": raw_action_aux_loss.detach(),
                    "raw_action_weighted_loss": (
                        weighted_raw_action_loss.detach()
                    ),
                }
            )
        elif self.raw_action_training_mode == "decode_consistency":
            generated_bspline_action = self._unnormalize_action_slice(
                generated,
                0,
                self.bspline_action_dim,
            )
            decoded_raw_action = self._decode_bspline_prediction(
                generated_bspline_action,
                num_actions=self.horizon,
                detach_knots=self.raw_action_consistency_detach_knots,
            )
            normalized_decoded_raw_action = self._normalize_action_slice(
                decoded_raw_action,
                self.bspline_action_dim,
                self.action_dim,
            )
            normalized_raw_target = normalized_action[
                ..., self.bspline_action_dim:
            ].unsqueeze(1)
            per_sample_consistency_loss = nn.functional.mse_loss(
                normalized_decoded_raw_action,
                normalized_raw_target.expand_as(normalized_decoded_raw_action),
                reduction="none",
            ).mean(dim=(-1, -2))
            # Drifting deliberately produces multiple samples. Penalizing all
            # of them with MSE would collapse that diversity, so supervise only
            # the sample closest to the demonstrated dense trajectory.
            raw_action_consistency_loss = (
                per_sample_consistency_loss.min(dim=1).values.mean()
            )
            weighted_raw_action_loss = (
                self.raw_action_loss_weight * raw_action_consistency_loss
            )
            loss = drifting_loss + weighted_raw_action_loss
            mean_diagnostics.update(
                {
                    "drift_loss": drifting_loss.detach(),
                    "raw_action_consistency_loss": (
                        raw_action_consistency_loss.detach()
                    ),
                    "raw_action_weighted_loss": (
                        weighted_raw_action_loss.detach()
                    ),
                }
            )
        if return_info:
            return loss, mean_diagnostics
        return loss
