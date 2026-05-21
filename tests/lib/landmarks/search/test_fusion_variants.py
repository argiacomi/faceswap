#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.core.fusion_variants` (Ticket 3)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.core.fusion_variants import fuse_candidate, fuse_variant
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.search.candidate_search import Candidate


def _truth_face() -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")
    points[0:17, 0] = np.linspace(40, 160, 17)
    points[0:17, 1] = 120 + 30 * np.sin(np.linspace(0, np.pi, 17))
    points[17:22, 0] = np.linspace(50, 90, 5)
    points[17:22, 1] = 70
    points[22:27, 0] = np.linspace(110, 150, 5)
    points[22:27, 1] = 70
    points[27:36, 0] = 100
    points[27:36, 1] = np.linspace(75, 110, 9)
    points[36:42, 0] = np.linspace(60, 80, 6)
    points[36:42, 1] = 85
    points[42:48, 0] = np.linspace(120, 140, 6)
    points[42:48, 1] = 85
    points[48:60, 0] = np.linspace(70, 130, 12)
    points[48:60, 1] = 130
    points[60:68, 0] = np.linspace(80, 120, 8)
    points[60:68, 1] = 130
    return points


def _predictions() -> list[LandmarkPrediction]:
    truth = _truth_face()
    return [
        LandmarkPrediction(truth, source="hrnet"),
        LandmarkPrediction(truth + 0.5, source="spiga"),
    ]


def test_fuse_variant_plain_average_no_weights_needed() -> None:
    """``plain_average`` is the canonical weight-free variant."""
    fused = fuse_variant("plain_average", _predictions(), models=("hrnet", "spiga"))
    assert fused.shape == (68, 2)


def test_fuse_variant_legacy_alias_resolves_via_registry() -> None:
    """Legacy alias names (e.g. ``static_weighted_outliers``) keep working."""
    weights = {"hrnet": [0.5] * 68, "spiga": [0.5] * 68}
    fused = fuse_variant(
        "static_weighted_outliers",
        _predictions(),
        models=("hrnet", "spiga"),
        weights=weights,
        outlier_threshold=3.5,
    )
    assert fused.shape == (68, 2)


def test_fuse_variant_weighted_median_requires_weights() -> None:
    """Weighted variants raise when no weights are supplied."""
    with pytest.raises(ValueError, match="weighted_median"):
        fuse_variant(
            "weighted_median",
            _predictions(),
            models=("hrnet", "spiga"),
            weights=None,
        )


def test_fuse_candidate_uses_candidate_strategy_and_threshold() -> None:
    """``fuse_candidate`` reads strategy + threshold from the Candidate dataclass."""
    candidate = Candidate(
        models=("hrnet", "spiga"),
        weight_generator="equal",
        weight_generator_params=(),
        strategy="static_weighted_downweight",
        outlier_threshold=2.5,
        bbox_source="manifest",
        crop_scale=1.6,
    )
    truth = _truth_face()
    weights = {model: [0.5] * 68 for model in candidate.models}
    fused = fuse_candidate(candidate, [truth, truth + 0.5], weights=weights)
    assert fused.shape == (68, 2)


def test_fuse_candidate_plain_average_skips_weights_requirement() -> None:
    """``plain_average`` candidates don't require a weights mapping."""
    candidate = Candidate(
        models=("hrnet",),
        weight_generator="equal",
        weight_generator_params=(),
        strategy="plain_average",
        outlier_threshold=None,
        bbox_source="manifest",
        crop_scale=1.6,
    )
    truth = _truth_face()
    fused = fuse_candidate(candidate, [truth], weights={})
    assert fused.shape == (68, 2)
