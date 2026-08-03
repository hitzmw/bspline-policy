"""Push-T decoding adapter for the Drifting B-spline image policy."""

from __future__ import annotations

import numpy as np
import torch

from bspline_policy.common.bspline_action import decode_bspline_action
from bspline_policy.policy.diffusion_unet_pusht_bspline_image_policy import (
    _project_monotonic_knots,
)
from bspline_policy.policy.drifting_unet_bspline_image_policy import (
    DriftingUnetBSplineImagePolicy,
)


class DriftingUnetPushTBSplineImagePolicy(
    DriftingUnetBSplineImagePolicy
):
    """Generate B-spline parameters once, then decode environment actions."""

    def __init__(self, execution_action_steps: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.execution_action_steps = int(execution_action_steps)

    def predict_action(self, obs_dict, generator=None):
        result = super().predict_action(obs_dict, generator=generator)
        bspline_action = result["action"]
        projected_parameters = []
        decoded_actions = []
        for parameters in bspline_action.detach().cpu().numpy():
            projected = _project_monotonic_knots(parameters)
            projected_parameters.append(projected)
            decoded_actions.append(
                decode_bspline_action(
                    projected,
                    degree=self.bspline_degree,
                    num_actions=self.execution_action_steps,
                    relative_knots=False,
                )
            )

        result["bspline_action"] = bspline_action
        result["projected_bspline_action"] = torch.as_tensor(
            np.stack(projected_parameters),
            device=bspline_action.device,
            dtype=bspline_action.dtype,
        )
        result["action"] = torch.as_tensor(
            np.stack(decoded_actions),
            device=bspline_action.device,
            dtype=bspline_action.dtype,
        )
        return result
