"""Franka decoding adapter for the Drifting B-spline image policy.

Identical decode contract as the PushT adapter: ``predict_action`` keeps the
raw B-spline parameters under ``action_pred``/``bspline_action`` and exposes
the decoded physical action chunk (8 steps of the 8-dim ``control`` vector:
7 absolute joint targets + gripper command) under ``action`` for real-robot
rollout scripts.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline

from bspline_policy.policy.diffusion_unet_pusht_bspline_image_policy import (
    _project_monotonic_knots,
)
from bspline_policy.policy.drifting_unet_bspline_image_policy import (
    DriftingUnetBSplineImagePolicy,
)


def _decode_integer_times(parameters: np.ndarray, degree: int, num_actions: int):
    """Scheme-C decode: evaluate at integer control ticks, not linspace.

    Oracle audit on franka_collect: integer-time decode reaches the 0.01 rad
    fit budget (mean 0.0097) while linspace-over-domain is 11x worse
    (mean 0.109), because adaptive knot placement rarely starts the decode
    domain at t=0 on this data (t_min p50 = 2).
    """
    knots = parameters[:, 0]
    control_points = parameters[: -(degree + 1), 1:]
    t_min = float(knots[degree])
    t_max = float(knots[-(degree + 1)])
    n = int(num_actions)
    if t_max <= t_min:
        raise ValueError(f"Invalid B-spline range: [{t_min}, {t_max}]")
    if n > 1 and t_max - t_min >= n - 1:
        start = min(max(0.0, t_min), t_max - (n - 1))
        t_eval = start + np.arange(n, dtype=np.float64)
    else:
        t_eval = np.linspace(t_min, t_max, n)
    decoded = BSpline(knots, control_points, degree, extrapolate=False)(t_eval)
    if np.isnan(decoded).any():
        decoded = BSpline(knots, control_points, degree, extrapolate=True)(t_eval)
    return decoded.astype(np.float32)


class DriftingUnetFrankaBSplineImagePolicy(
    DriftingUnetBSplineImagePolicy
):
    """Generate B-spline parameters once, then decode control actions."""

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
                _decode_integer_times(
                    projected,
                    degree=self.bspline_degree,
                    num_actions=self.execution_action_steps,
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
