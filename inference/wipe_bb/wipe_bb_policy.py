"""Inference-only wipe_bb adapter; reuses an installed diffusion_policy.

No bspline_policy, training workspace, dataset, Hydra instantiation or robot
driver is imported. The model and normalization weights remain float32.
"""

from collections import deque
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from scipy.interpolate import BSpline

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from robomimic.algo import algo_factory
import robomimic.models.base_nets as rmbn
import robomimic.utils.obs_utils as ObsUtils


class _InferenceNetwork(BaseImagePolicy):
    """Same module names and architecture as the exported Franka EMA model."""

    def __init__(self, cfg):
        super().__init__()
        obs_meta = cfg["shape_meta"]["obs"]
        modalities = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        shapes = {}
        for key, meta in obs_meta.items():
            modalities[meta.get("type", "low_dim")].append(key)
            shapes[key] = list(meta["shape"])
        config = get_robomimic_config(
            algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph"
        )
        with config.unlocked():
            config.observation.modalities.obs = modalities
            for modality in config.observation.encoder.values():
                if modality.obs_randomizer_class == "CropRandomizer":
                    modality.obs_randomizer_kwargs.crop_height = cfg["crop_shape"][0]
                    modality.obs_randomizer_kwargs.crop_width = cfg["crop_shape"][1]
        ObsUtils.initialize_obs_utils_with_config(config)
        encoder_policy = algo_factory(
            algo_name=config.algo_name, config=config, obs_key_shapes=shapes,
            ac_dim=8, device="cpu",
        )
        self.obs_encoder = encoder_policy.nets["policy"].nets["encoder"].nets["obs"]
        replace_submodules(
            self.obs_encoder,
            predicate=lambda module: isinstance(module, nn.BatchNorm2d),
            func=lambda module: nn.GroupNorm(module.num_features // 16, module.num_features),
        )
        replace_submodules(
            self.obs_encoder,
            predicate=lambda module: isinstance(module, rmbn.CropRandomizer),
            func=lambda module: CropRandomizer(
                input_shape=module.input_shape, crop_height=module.crop_height,
                crop_width=module.crop_width, num_crops=module.num_crops,
                pos_enc=module.pos_enc,
            ),
        )
        self.horizon = int(cfg["horizon"])
        self.n_obs_steps = int(cfg["n_obs_steps"])
        self.model = ConditionalUnet1D(
            input_dim=8, local_cond_dim=None,
            global_cond_dim=int(self.obs_encoder.output_shape()[0]) * self.n_obs_steps,
            diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
            down_dims=cfg["down_dims"], kernel_size=cfg["kernel_size"],
            n_groups=cfg["n_groups"], cond_predict_scale=cfg["cond_predict_scale"],
        )
        self.normalizer = LinearNormalizer()

    def predict_parameters(self, obs, generator=None, noise=None):
        batch_size = next(iter(obs.values())).shape[0]
        normalized = self.normalizer.normalize(obs)
        encoder_input = {
            key: value.reshape(-1, *value.shape[2:])
            for key, value in normalized.items()
        }
        condition = self.obs_encoder(encoder_input).reshape(batch_size, -1)
        if noise is None:
            noise = torch.randn(
                batch_size, self.horizon, 8, device=self.device,
                dtype=self.dtype, generator=generator,
            )
        prediction = self.model(
            noise,
            torch.zeros(batch_size, dtype=torch.long, device=self.device),
            global_cond=condition,
        )
        # The checkpoint normalizes 15 target channels, although the UNet
        # predicts only knot + 7 controls. Select the matching first 8 channels.
        params = self.normalizer["action"].params_dict
        scale = params["scale"].reshape(self.horizon, 15)[:, :8]
        offset = params["offset"].reshape(self.horizon, 15)[:, :8]
        return (prediction - offset) / scale


def _decode(parameters, degree=3, num_actions=8):
    """Exact float64 knot projection + integer-tick Franka decoder."""
    projected = np.asarray(parameters, dtype=np.float64).copy()
    for index in range(1, len(projected)):
        projected[index, 0] = max(projected[index, 0], projected[index - 1, 0] + 1e-6)
    knots = projected[:, 0]
    controls = projected[: -(degree + 1), 1:]
    t_min, t_max = float(knots[degree]), float(knots[-(degree + 1)])
    if t_max <= t_min:
        raise ValueError("Invalid B-spline interval")
    if num_actions > 1 and t_max - t_min >= num_actions - 1:
        start = min(max(0.0, t_min), t_max - (num_actions - 1))
        times = start + np.arange(num_actions, dtype=np.float64)
    else:
        times = np.linspace(t_min, t_max, num_actions)
    actions = BSpline(knots, controls, degree, extrapolate=False)(times)
    if np.isnan(actions).any():
        actions = BSpline(knots, controls, degree, extrapolate=True)(times)
    return projected.astype(np.float32), actions.astype(np.float32)


class WipeBBPolicy:
    """Local Python entry point. Does not connect to or command a robot.

    predict_action(obs): DP-style batched tensors, raw low_dim and RGB [0,1].
    predict(frames): two chronological raw frames -> numpy action (8,7).
    step(frame, infer=True): maintain two-frame history on every control tick.
    """

    ALIASES = {
        "sideview_image": ("sideview_image", "D435_color", "D455_color"),
        "wrist_image": ("wrist_image", "D405_color"),
        "ee_pose": ("ee_pose",),
        "joint_positions": ("joint_positions",),
    }

    def __init__(self, bundle_dir=None, device="cuda:0", image_color_order="rgb"):
        folder = Path(bundle_dir) if bundle_dir else Path(__file__).resolve().parent
        self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        cfg = self.metadata["policy_config"]
        if self.metadata["format"] != "wipe_bb_ema_fp32_v1":
            raise ValueError("Unsupported deployment bundle format")
        if (
            cfg["shape_meta"]["action"]["shape"] != [7]
            or cfg["raw_action_training_mode"] != "decode_consistency"
            or not cfg["raw_action_concat"]
            or not cfg["obs_encoder_group_norm"]
            or not cfg["eval_fixed_crop"]
            or cfg["horizon"] != 16
            or cfg["n_obs_steps"] != 2
        ):
            raise ValueError("This adapter only supports the exported wipe_bb architecture")
        if image_color_order.lower() not in ("rgb", "bgr"):
            raise ValueError("image_color_order must be rgb or bgr")
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use device='cpu' for an offline check")
        self.image_color_order = image_color_order.lower()
        self.obs_shape_meta = cfg["shape_meta"]["obs"]
        self.n_obs_steps = int(cfg["n_obs_steps"])
        self.n_action_steps = int(cfg["execution_action_steps"])
        self.degree = int(cfg["bspline_degree"])
        self.network = _InferenceNetwork(cfg)
        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = True
        state = torch.load(str(folder / "ema_weights.pt"), **load_kwargs)
        self.network.load_state_dict(state, strict=True)
        del state
        self.network.eval().requires_grad_(False).to(requested_device)
        self.history = deque(maxlen=self.n_obs_steps)

    @property
    def device(self):
        return self.network.device

    def eval(self):
        self.network.eval()
        return self

    def to(self, device):
        self.network.to(device=device)
        return self

    def reset(self):
        self.history.clear()

    def _validate_tensor_observations(self, observations):
        result = {}
        batch_size = None
        for key, meta in self.obs_shape_meta.items():
            value = torch.as_tensor(observations[key], device=self.device, dtype=torch.float32)
            expected = (self.n_obs_steps,) + tuple(meta["shape"])
            if value.ndim != len(expected) + 1 or tuple(value.shape[1:]) != expected:
                raise ValueError("{} must have shape (B, {}), got {}".format(key, expected, tuple(value.shape)))
            if not torch.isfinite(value).all():
                raise ValueError("{} contains NaN/Inf".format(key))
            if meta.get("type") == "rgb" and (value.min() < 0 or value.max() > 1):
                raise ValueError("{} must be RGB float32 in [0,1]".format(key))
            if batch_size is not None and value.shape[0] != batch_size:
                raise ValueError("Observation batch sizes differ")
            batch_size = value.shape[0]
            result[key] = value
        return result

    @torch.no_grad()
    def predict_action(self, obs_dict, generator=None, noise=None):
        obs = self._validate_tensor_observations(obs_dict)
        if noise is not None:
            noise = torch.as_tensor(noise, dtype=torch.float32, device=self.device)
            expected = (next(iter(obs.values())).shape[0], 16, 8)
            if tuple(noise.shape) != expected or not torch.isfinite(noise).all():
                raise ValueError("noise must be finite with shape {}".format(expected))
        parameters = self.network.predict_parameters(obs, generator=generator, noise=noise)
        if not torch.isfinite(parameters).all():
            raise FloatingPointError("Predicted B-spline parameters contain NaN/Inf")
        decoded = [_decode(p, self.degree, self.n_action_steps) for p in parameters.cpu().numpy()]
        projected, actions = (np.stack(values) for values in zip(*decoded))
        if not np.isfinite(actions).all():
            raise FloatingPointError("Decoded joint targets contain NaN/Inf")
        # All seven values are joint targets. There is no gripper channel to clip.
        return {
            "action": torch.from_numpy(actions).to(self.device),
            "action_pred": parameters,
            "bspline_action": parameters,
            "projected_bspline_action": torch.from_numpy(projected).to(self.device),
        }

    def prepare_frame(self, frame):
        converted = {}
        for key, meta in self.obs_shape_meta.items():
            aliases = self.ALIASES[key]
            value = next((frame[name] for name in aliases if name in frame), None)
            if value is None:
                raise KeyError("Missing {}; accepted keys: {}".format(key, aliases))
            value = np.asarray(value)
            if meta.get("type") == "rgb":
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                    raise ValueError("{} raw image must be uint8 HWC, 3 channels".format(key))
                if self.image_color_order == "bgr":
                    value = value[..., ::-1]
                _, height, width = meta["shape"]
                if value.shape[:2] != (height, width):
                    resampling = getattr(Image, "Resampling", Image)
                    value = np.asarray(Image.fromarray(value).resize((width, height), resampling.LANCZOS))
                value = np.moveaxis(value, -1, 0).astype(np.float32) / 255.0
            else:
                if value.size != int(np.prod(meta["shape"])):
                    raise ValueError("{} must contain {} elements".format(key, meta["shape"]))
                value = value.astype(np.float32).reshape(meta["shape"])
            if not np.isfinite(value).all():
                raise ValueError("{} contains NaN/Inf".format(key))
            converted[key] = np.ascontiguousarray(value).copy()
        return converted

    def _predict_prepared(self, frames, generator=None):
        obs = {key: np.stack([frame[key] for frame in frames])[None] for key in self.obs_shape_meta}
        return self.predict_action(obs, generator=generator)["action"][0].cpu().numpy()

    def predict(self, frames, generator=None):
        if len(frames) != self.n_obs_steps:
            raise ValueError("Expected exactly two chronological observations")
        return self._predict_prepared([self.prepare_frame(frame) for frame in frames], generator)

    def step(self, frame, infer=True, generator=None):
        # Copy and preprocess immediately: camera buffers may be reused by the caller.
        self.history.append(self.prepare_frame(frame))
        if not infer or len(self.history) < self.n_obs_steps:
            return None
        return self._predict_prepared(list(self.history), generator)
