#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.evaluation.effective_ensemble` (issue #79)."""

from __future__ import annotations

import pytest

from lib.landmarks.ensemble.weights import LANDMARK_COUNT
from lib.landmarks.evaluation.effective_ensemble import (
    DEFAULT_EFFECTIVE_MODELS_FLOOR,
    diagnose,
)


def _columns(distribution: dict[str, float]) -> dict[str, list[float]]:
    """Return a (model -> 68-vec) weight matrix where every column matches ``distribution``."""
    return {model: [weight] * LANDMARK_COUNT for model, weight in distribution.items()}


def test_equal_weights_yield_max_effective_models_and_no_strict_winner() -> None:
    """Three-way equal weights: effective_models == 3 and no strict majority share."""
    weights = _columns({"hrnet": 1.0, "spiga": 1.0, "orformer": 1.0})

    diag = diagnose(weights, strategy="static_weighted")

    assert diag.mean_effective_models == pytest.approx(3.0, rel=1e-6)
    # Strict-winner share: ties don't count toward any model.
    for share in diag.landmark_share_by_model.values():
        assert share == pytest.approx(0.0)
    assert diag.collapsed is False
    assert diag.collapsed_dominant_model == ""


def test_dominant_model_collapse_detected_for_static_weighted() -> None:
    """Static-weighted ensemble dominated by one model is flagged as collapsed."""
    weights = _columns({"hrnet": 0.9, "spiga": 0.05, "orformer": 0.05})

    diag = diagnose(weights, strategy="static_weighted")

    assert diag.collapsed is True
    assert diag.collapsed_dominant_model == "hrnet"
    assert diag.landmark_share_by_model["hrnet"] == pytest.approx(1.0)
    assert diag.mean_effective_models < 1.25


def test_weighted_median_collapse_detection_uses_half_threshold() -> None:
    """For ``weighted_median``, any model > 0.5 in every column wins every median."""
    weights = _columns({"spiga": 0.6, "orformer": 0.4})

    diag = diagnose(weights, strategy="weighted_median")

    assert diag.weighted_median_collapsed is True
    assert diag.weighted_median_dominant_model == "spiga"
    assert diag.collapsed is True
    assert diag.collapsed_dominant_model == "spiga"


def test_weighted_median_balanced_columns_are_not_collapsed() -> None:
    """A balanced weighted median is not collapsed even if one model edges others."""
    weights = _columns({"spiga": 0.45, "orformer": 0.4, "hrnet": 0.15})

    diag = diagnose(weights, strategy="weighted_median")

    assert diag.weighted_median_collapsed is False
    assert diag.collapsed is False


def test_diagnose_requires_non_negative_weights() -> None:
    """Negative weights raise before any column normalization."""
    bad = {"hrnet": [-0.1] * LANDMARK_COUNT, "spiga": [0.5] * LANDMARK_COUNT}
    with pytest.raises(ValueError, match="non-negative"):
        diagnose(bad, strategy="static_weighted")


def test_diagnose_requires_positive_column_sum() -> None:
    """A landmark column with all-zero weights is invalid."""
    column = [0.0] * LANDMARK_COUNT
    bad = {"hrnet": column, "spiga": column}
    with pytest.raises(ValueError, match="non-zero"):
        diagnose(bad, strategy="static_weighted")


def test_diagnose_rejects_invalid_effective_models_floor() -> None:
    """Effective-models floors must exceed 1.0 (single-model degenerate case)."""
    weights = _columns({"hrnet": 1.0, "spiga": 1.0})
    with pytest.raises(ValueError, match="effective_models_floor"):
        diagnose(weights, strategy="static_weighted", effective_models_floor=1.0)


def test_diagnose_uses_explicit_model_order() -> None:
    """Passing ``models`` reorders columns for the diagnostics output."""
    weights = _columns({"hrnet": 0.6, "spiga": 0.4})

    diag = diagnose(weights, strategy="static_weighted", models=("spiga", "hrnet"))

    assert diag.models == ("spiga", "hrnet")
    assert diag.majority_model_by_landmark[0] == "hrnet"


def test_effective_models_floor_is_tunable() -> None:
    """A stricter floor flags marginal ensembles as collapsed."""
    # (0.7, 0.3) → effective_models = 1 / (0.49 + 0.09) ≈ 1.72
    weights = _columns({"hrnet": 0.7, "spiga": 0.3})

    permissive = diagnose(weights, strategy="static_weighted", effective_models_floor=1.5)
    strict = diagnose(weights, strategy="static_weighted", effective_models_floor=2.0)

    assert permissive.collapsed is False  # 1.72 >= 1.5 → effective ensemble
    assert strict.collapsed is True  # 1.72 < 2.0  → too narrow for "real" ensemble
    assert permissive.effective_models_floor == pytest.approx(1.5)
    assert strict.effective_models_floor == pytest.approx(2.0)


def test_default_effective_models_floor_is_constant() -> None:
    """Pinned for the gate config that consumes it in #77."""
    assert DEFAULT_EFFECTIVE_MODELS_FLOOR == 1.5


def test_diagnose_reports_per_landmark_majority() -> None:
    """The per-landmark majority list has length 68 and references real model names."""
    weights = _columns({"hrnet": 0.7, "spiga": 0.3})

    diag = diagnose(weights, strategy="static_weighted")

    assert len(diag.majority_model_by_landmark) == LANDMARK_COUNT
    assert set(diag.majority_model_by_landmark).issubset({"hrnet", "spiga"})
