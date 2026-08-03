"""Robomimic rollout adapter for the Drifting B-spline image policy."""

from bspline_policy.policy.drifting_unet_pusht_bspline_image_policy import (
    DriftingUnetPushTBSplineImagePolicy,
)


class DriftingUnetRobomimicBSplineImagePolicy(
    DriftingUnetPushTBSplineImagePolicy
):
    """Decode one predicted B-spline into dense Robomimic actions.

    The decoder used by Push-T is independent of the physical action
    dimension. This explicit task adapter keeps Robomimic configurations from
    depending on a class whose public name says Push-T while preserving the
    already-tested projection and SciPy decoding path.
    """
