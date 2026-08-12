"""Workspace entry point for iMeanFlow B-spline image policies."""

from bspline_policy.workspace.train_drifting_bspline_image_workspace import (
    TrainDriftingBSplineImageWorkspace,
)


class TrainIMeanFlowBSplineImageWorkspace(TrainDriftingBSplineImageWorkspace):
    """Use the verified image-policy loop with optimizer-step loss warmup."""

    pass
