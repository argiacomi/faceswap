#!/usr/bin/env python3
"""Smoke + error contract for :mod:`lib.landmarks.core.fusion_variants`.

Exhaustive registry coverage (canonical names, alias resolution, threshold
requirements) lives in ``ensemble_strategies_test.py``.  Underlying fusion
math is covered by the ``plain_average``/``static_weighted``/``weighted_median``
unit tests.  This file only verifies that ``fuse_variant`` and
``fuse_candidate`` shape-route through the right strategy and raise on
ill-formed input.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.core.fusion_variants import fuse_candidate, fuse_variant
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.search.candidate_search import Candidate


def _predictions() -> list[LandmarkPrediction]:
    base = np.full((68, 2), 0.5, dtype="float32")
    return [
        LandmarkPrediction(base, source="hrnet"),
        LandmarkPrediction(base + 0.5, source="spiga"),
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
    base = np.full((68, 2), 0.5, dtype="float32")
    weights = {model: [0.5] * 68 for model in candidate.models}
    fused = fuse_candidate(candidate, [base, base + 0.5], weights=weights)
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
    fused = fuse_candidate(candidate, [np.full((68, 2), 0.5, dtype="float32")], weights={})
    assert fused.shape == (68, 2)
