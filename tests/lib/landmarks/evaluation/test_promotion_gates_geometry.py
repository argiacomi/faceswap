#!/usr/bin/env python3
"""Unit tests for the geometry-first gate branches in :mod:`promotion_gates` (#77)."""

from __future__ import annotations

import pytest

from lib.landmarks.ensemble.weights import LANDMARK_COUNT
from lib.landmarks.evaluation.effective_ensemble import diagnose
from lib.landmarks.search.candidate_search import (
    Candidate,
    CandidateMetrics,
    CandidateResult,
)
from lib.landmarks.search.promotion_gates import (
    GateConfig,
    GeometryScore,
    apply_gates,
)


def _make_result(
    *,
    candidate_id: str,
    score: float,
    is_baseline: bool,
    nme: float = 0.05,
    bucket_regression: float = 0.0,
) -> CandidateResult:
    models = ("hrnet",) if is_baseline else ("hrnet", "spiga", "orformer")
    candidate = Candidate(
        models=models,
        weight_generator="equal",
        strategy="plain_average" if is_baseline else "static_weighted",
        outlier_threshold=None,
    )
    weights = {model: [1.0 / len(models)] * LANDMARK_COUNT for model in models}
    diagnostics = diagnose(weights, strategy=candidate.strategy)
    return CandidateResult(
        candidate=candidate,
        candidate_id=candidate_id,
        weights=weights,
        weights_hash=f"sha256:{candidate_id}-w",
        score=score,
        objective="alignment_geometry_v1",
        regression_epsilon_nme=0.001,
        metrics=CandidateMetrics(
            sample_count=10,
            overall_nme=nme,
            failure_rate=0.0,
            auc=0.5,
            regression_rate_vs_best_single=0.0,
            bucket_regression_rate_vs_best_single=bucket_regression,
            per_bucket={},
            best_single_model="hrnet",
        ),
        fit_diagnostics={},
        effective_ensemble=diagnostics,
        is_single_model_baseline=is_baseline,
    )


def _geometry(
    *,
    overall: float,
    catastrophic: float = 0.0,
    p95_transform: float = 0.01,
    p95_crop: float = 0.01,
    p95_roll: float = 1.0,
    mean_hull_iou: float = 0.9,
    max_bucket_regression: float = 0.0,
) -> GeometryScore:
    return GeometryScore(
        overall_score=overall,
        catastrophic_failure_rate=catastrophic,
        p95_translation_normalized=p95_transform,
        p95_roi_center_normalized=p95_crop,
        p95_roll_degrees=p95_roll,
        mean_hull_iou=mean_hull_iou,
        p05_hull_iou=mean_hull_iou - 0.05,
        max_bucket_regression_score=max_bucket_regression,
    )


def test_geometry_improvement_gate_selects_lower_score_candidate() -> None:
    """The first candidate beating the baseline's geometry score is promoted."""
    baseline = _make_result(candidate_id="hrnet", score=0.10, is_baseline=True)
    better = _make_result(candidate_id="ensemble-a", score=0.05, is_baseline=False)
    worse = _make_result(candidate_id="ensemble-b", score=0.20, is_baseline=False)
    results = [better, worse, baseline]  # sorted by ensemble score

    geometry_scores = {
        "hrnet": _geometry(overall=0.20),
        "ensemble-a": _geometry(overall=0.10),
        "ensemble-b": _geometry(overall=0.30),
    }
    config = GateConfig(require_geometry_improvement=True)
    application = apply_gates(results, config, geometry_scores=geometry_scores)

    assert application.promoted is better
    assert application.promoted_outcome.geometry_score == pytest.approx(0.10)


def test_geometry_improvement_gate_rejects_worse_candidate() -> None:
    """No candidate is promoted when none beats the baseline geometry score."""
    baseline = _make_result(candidate_id="hrnet", score=0.10, is_baseline=True)
    worse = _make_result(candidate_id="ensemble", score=0.05, is_baseline=False)
    results = [worse, baseline]

    geometry_scores = {
        "hrnet": _geometry(overall=0.10),
        "ensemble": _geometry(overall=0.20),
    }
    config = GateConfig(require_geometry_improvement=True)
    application = apply_gates(results, config, geometry_scores=geometry_scores)

    assert application.promoted is None
    failed = application.outcomes[0]
    assert "require_geometry_improvement" in failed.failed_gates


def test_max_catastrophic_geometry_failure_rate_gate() -> None:
    """Configured catastrophic-rate ceiling rejects high-catastrophe ensembles."""
    baseline = _make_result(candidate_id="hrnet", score=0.10, is_baseline=True)
    risky = _make_result(candidate_id="ensemble", score=0.01, is_baseline=False)
    results = [risky, baseline]
    geometry_scores = {
        "hrnet": _geometry(overall=0.10),
        "ensemble": _geometry(overall=0.05, catastrophic=0.10),
    }
    config = GateConfig(max_catastrophic_geometry_failure_rate=0.05)
    application = apply_gates(results, config, geometry_scores=geometry_scores)

    assert application.promoted is None
    assert "max_catastrophic_geometry_failure_rate" in application.outcomes[0].failed_gates


def test_max_p95_transform_error_gate() -> None:
    """P95 transform-error ceiling drops candidates with large alignment drift."""
    baseline = _make_result(candidate_id="hrnet", score=0.10, is_baseline=True)
    drifty = _make_result(candidate_id="ensemble", score=0.01, is_baseline=False)
    results = [drifty, baseline]
    geometry_scores = {
        "hrnet": _geometry(overall=0.10, p95_transform=0.02),
        "ensemble": _geometry(overall=0.05, p95_transform=0.10),
    }
    config = GateConfig(max_p95_transform_error=0.05)
    application = apply_gates(results, config, geometry_scores=geometry_scores)
    assert application.promoted is None
    assert "max_p95_transform_error" in application.outcomes[0].failed_gates


def test_min_hull_iou_gate() -> None:
    """A hull-IoU floor blocks candidates whose hull alignment regressed."""
    baseline = _make_result(candidate_id="hrnet", score=0.10, is_baseline=True)
    bad_hull = _make_result(candidate_id="ensemble", score=0.01, is_baseline=False)
    results = [bad_hull, baseline]
    geometry_scores = {
        "hrnet": _geometry(overall=0.10, mean_hull_iou=0.9),
        "ensemble": _geometry(overall=0.05, mean_hull_iou=0.4),
    }
    config = GateConfig(min_hull_iou=0.6)
    application = apply_gates(results, config, geometry_scores=geometry_scores)
    assert application.promoted is None
    assert "min_hull_iou" in application.outcomes[0].failed_gates


def test_geometry_gates_fail_when_scores_missing() -> None:
    """Configured geometry gates with no scores produce a clear failure reason."""
    baseline = _make_result(candidate_id="hrnet", score=0.10, is_baseline=True)
    candidate = _make_result(candidate_id="ensemble", score=0.01, is_baseline=False)
    results = [candidate, baseline]
    config = GateConfig(require_geometry_improvement=True)
    application = apply_gates(results, config, geometry_scores=None)
    assert application.promoted is None
    assert any("geometry_metrics_unavailable" in o.failed_gates for o in application.outcomes)


def test_max_hard_slice_regression_rate_gate() -> None:
    """Worst-bucket regression above the configured ceiling blocks promotion."""
    baseline = _make_result(candidate_id="hrnet", score=0.10, is_baseline=True)
    candidate = _make_result(candidate_id="ensemble", score=0.01, is_baseline=False)
    results = [candidate, baseline]
    geometry_scores = {
        "hrnet": _geometry(overall=0.10),
        "ensemble": _geometry(overall=0.08, max_bucket_regression=0.05),
    }
    config = GateConfig(max_hard_slice_regression_rate=0.02)
    application = apply_gates(results, config, geometry_scores=geometry_scores)
    assert application.promoted is None
    assert "max_hard_slice_regression_rate" in application.outcomes[0].failed_gates


def test_geometry_gates_pass_when_candidate_meets_all_thresholds() -> None:
    """A candidate that beats baseline on every gate is promoted."""
    baseline = _make_result(candidate_id="hrnet", score=0.10, is_baseline=True)
    candidate = _make_result(candidate_id="ensemble", score=0.01, is_baseline=False)
    results = [candidate, baseline]
    geometry_scores = {
        "hrnet": _geometry(
            overall=0.30, catastrophic=0.10, p95_transform=0.05, mean_hull_iou=0.70
        ),
        "ensemble": _geometry(
            overall=0.10, catastrophic=0.0, p95_transform=0.02, mean_hull_iou=0.95
        ),
    }
    config = GateConfig(
        require_geometry_improvement=True,
        max_catastrophic_geometry_failure_rate=0.05,
        max_p95_transform_error=0.05,
        min_hull_iou=0.80,
    )
    application = apply_gates(results, config, geometry_scores=geometry_scores)
    assert application.promoted is candidate
