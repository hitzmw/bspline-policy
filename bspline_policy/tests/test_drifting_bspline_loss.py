"""Regression tests for the Drifting loss used by B-spline policies."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from bspline_policy.model.drifting.drifting_util import drift_loss


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = (
    REPOSITORY_ROOT
    / "drifting_policy"
    / "diffusion_policy"
    / "model"
    / "drifting"
    / "drifting_util.py"
)


def _load_reference_module():
    specification = importlib.util.spec_from_file_location(
        "reference_drifting_util",
        REFERENCE_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load Drifting reference: {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_drift_loss_exactly_matches_repository_reference():
    reference = _load_reference_module()
    generator = torch.Generator().manual_seed(91)
    generated = torch.randn(3, 8, 3, generator=generator)
    positive = torch.randn(3, 1, 3, generator=generator)
    negative = torch.randn(3, 2, 3, generator=generator)
    temperatures = (0.02, 0.05, 0.2)

    actual_loss, actual_info = drift_loss(
        generated,
        positive,
        fixed_neg=negative,
        R_list=temperatures,
    )
    expected_loss, expected_info = reference.drift_loss(
        generated,
        positive,
        fixed_neg=negative,
        R_list=temperatures,
    )

    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert actual_info.keys() == expected_info.keys()
    for key in actual_info:
        torch.testing.assert_close(
            actual_info[key],
            expected_info[key],
            rtol=0,
            atol=0,
        )


def test_drift_loss_backpropagates_to_all_generated_candidates():
    generated = torch.randn(2, 8, 3, requires_grad=True)
    positive = torch.randn(2, 1, 3)
    loss, diagnostics = drift_loss(generated, positive)
    loss.mean().backward()

    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert torch.count_nonzero(generated.grad).item() > 0
    assert diagnostics.keys() == {
        "scale",
        "loss_0.02",
        "loss_0.05",
        "loss_0.2",
    }


def test_drift_loss_rejects_single_candidate_only_at_policy_boundary():
    generated = torch.randn(2, 1, 3, requires_grad=True)
    positive = torch.randn(2, 1, 3)
    loss, _ = drift_loss(generated, positive)
    assert torch.isfinite(loss).all()
    # The mathematical primitive permits G=1; the policy constructor rejects
    # it because the generated self-connection is masked and provides no useful
    # candidate-to-candidate distribution estimate.
