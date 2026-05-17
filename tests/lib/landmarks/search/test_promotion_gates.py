#!/usr/bin/env python3
"""Tests for search promotion gates."""

from __future__ import annotations

from lib.landmarks.search.candidate_search import CandidateMetrics, CandidateResult
from lib.landmarks.search.candidates import Candidate
from lib.landmarks.search.promotion_gates import (
    GateConfig,
    GeometryScore,
    apply_gates,
    no_promotion_payload,
)


def _metrics(nme: float = 0.1) -> CandidateMetrics:
    return CandidateMetrics(
        sample_count=10,
        overall_nme=nme,
        failure_rate=0.0,
        auc=0.0,
        regression_rate_vs_best_single=0.0,
        bucket_regression_rate_vs_best_single=0.0,
        best_single_model="hrnet",
    )


def _result(
    candidate_id: str,
    *,
    models: tuple[str, ...] = ("hrnet",),
    nme: float = 0.1,
    strategy: str = "plain_average",
    single: bool = False,
) -> CandidateResult:
    candidate = Candidate(models=models, weight_generator="equal", strategy=strategy)
    return CandidateResult(
        candidate=candidate,
        candidate_id=candidate_id,
        weights={model: [1.0 / len(models)] * 68 for model in models},
        weights_hash=f"sha256:{candidate_id}",
        score=nme,
        objective="alignment_geometry_v1",
        regression_epsilon_nme=0.001,
        metrics=_metrics(nme),
        is_single_model_baseline=single,
    )


def _geometry(
    score: float = 0.1,
    *,
    per_bucket: dict[str, dict[str, float]] | None = None,
) -> GeometryScore:
    return GeometryScore(
        overall_score=score,
        catastrophic_failure_rate=0.0,
        p95_translation_normalized=0.0,
        p95_roi_center_normalized=0.0,
        p95_roll_degrees=0.0,
        mean_hull_iou=0.95,
        p05_hull_iou=0.90,
        per_bucket=per_bucket or {
            "fixture:clean": {
                "catastrophic_failure_rate": 0.0,
                "p05_hull_iou": 0.90,
            }
        },
    )


def test_per_bucket_geometry_gates_fail_on_hidden_slice_regressions() -> None:
    """Per-bucket catastrophic and hull floors catch bad slices hidden by averages."""
    baseline = _result("baseline", single=True)
    candidate = _result("candidate", models=("hrnet", "orformer"), strategy="static_weighted")
    scores = {
        "baseline": _geometry(0.2),
        "candidate": _geometry(
            0.1,
            per_bucket={
                "fixture:clean": {
                    "catastrophic_failure_rate": 0.0,
                    "p05_hull_iou": 0.95,
                },
                "fixture:occluded": {
                    "catastrophic_failure_rate": 0.5,
                    "p05_hull_iou": 0.2,
                },
            },
        ),
    }

    application = apply_gates(
        [candidate, baseline],
        GateConfig(
            max_per_bucket_catastrophic_geometry_failure_rate=0.2,
            min_per_bucket_p05_hull_iou=0.5,
        ),
        geometry_scores=scores,
    )

    outcome = application.outcomes[0]
    assert outcome.candidate_id == "candidate"
    assert outcome.passed is False
    assert "max_per_bucket_catastrophic_geometry_failure_rate" in outcome.failed_gates
    assert "min_per_bucket_p05_hull_iou" in outcome.failed_gates
    assert outcome.worst_catastrophic_geometry_bucket == "fixture:occluded"
    assert outcome.worst_hull_iou_bucket == "fixture:occluded"


def test_single_model_baselines_are_comparison_only_by_default() -> None:
    """A passing single-model baseline should produce no promotion unless explicit."""
    baseline = _result("baseline", single=True, nme=0.1)
    application = apply_gates(
        [baseline],
        GateConfig(max_catastrophic_geometry_failure_rate=1.0),
        geometry_scores={"baseline": _geometry(0.1)},
    )

    assert application.promoted is None
    outcome = application.outcomes[0]
    assert outcome.passed is False
    assert "single_model_promotion_not_allowed" in outcome.failed_gates
    payload = no_promotion_payload(application)
    assert payload["status"] == "no_promotion"
    assert "keep the current best single-model baseline" in payload["reason"]


def test_single_model_promotion_must_beat_best_single_baseline() -> None:
    """Even when explicit, a single model cannot promote by tying itself."""
    baseline = _result("baseline", single=True, nme=0.1)
    application = apply_gates(
        [baseline],
        GateConfig(
            allow_single_model_promotion=True,
            max_catastrophic_geometry_failure_rate=1.0,
        ),
        geometry_scores={"baseline": _geometry(0.1)},
    )

    assert application.promoted is None
    outcome = application.outcomes[0]
    assert outcome.passed is False
    assert "single_model_does_not_beat_best_baseline_report" in outcome.failed_gates
    assert "single_model_does_not_beat_best_baseline_geometry" in outcome.failed_gates
