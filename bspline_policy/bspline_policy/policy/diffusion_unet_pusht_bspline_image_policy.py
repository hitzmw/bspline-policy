from __future__ import annotations

import numpy as np
import torch

from bspline_policy.common.bspline_action import decode_bspline_action
from bspline_policy.policy.diffusion_unet_bspline_image_policy import (
    DiffusionUnetBSplineImagePolicy,
)


def _project_monotonic_knots(action_params: np.ndarray, delta: float = 1e-6):
    projected = np.asarray(action_params, dtype=np.float64).copy()
    for idx in range(1, projected.shape[0]):
        projected[idx, 0] = max(
            projected[idx, 0],
            projected[idx - 1, 0] + delta,
        )
    return projected


class DiffusionUnetPushTBSplineImagePolicy(DiffusionUnetBSplineImagePolicy):
    """B-spline policy that decodes parameter segments for Push-T rollouts."""

    def __init__(self, execution_action_steps: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.execution_action_steps = int(execution_action_steps)

    def predict_action(self, obs_dict):
        result = super().predict_action(obs_dict)
        bspline_action = result["action"]
        decoded = []
        for params in bspline_action.detach().cpu().numpy():
            params = _project_monotonic_knots(params)
            decoded.append(
                decode_bspline_action(
                    params,
                    degree=self.bspline_degree,
                    num_actions=self.execution_action_steps,
                    relative_knots=False,
                )
            )
        result["bspline_action"] = bspline_action
        result["action"] = torch.as_tensor(
            np.stack(decoded),
            device=bspline_action.device,
            dtype=bspline_action.dtype,
        )
        return result
