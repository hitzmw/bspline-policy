"""Training workspace for direct-action Drifting image policies."""

from bspline_policy.workspace.train_drifting_bspline_image_workspace import (
    _TrainDriftingImageWorkspaceBase,
)


class TrainDriftingUnetHybridWorkspace(
    _TrainDriftingImageWorkspaceBase
):
    """Train Drifting directly on dense action trajectories.

    The inherited implementation contains only the representation-agnostic
    optimizer, validation, EMA, and checkpoint loop.  All B-spline behavior
    lives in separate dataset and policy classes, neither of which is used
    here.
    """

    training_label = "Drifting"
