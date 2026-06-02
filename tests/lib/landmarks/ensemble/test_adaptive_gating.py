#!/usr/bin/env python3
"""Tests for Phase 5 #10 mixture-of-experts gating over the enlarged candidate set.

The learned scorer is the gate: per-bucket (#8) and region-level (#9) fusion
candidates are appended to the runtime candidate pool, and the scorer ranks them
alongside the single-model and standard-fusion candidates using its existing
runtime features. These tests confirm the enlarged set is scored and selectable,
and that adding the new candidate one-hots does not reorder the stable runtime
feature contract.
"""

from __future__ import annotations

import numpy as np

from lib.landmarks.ensemble import runtime_resolver
from lib.landmarks.ensemble.runtime_features import (
    RUNTIME_PREFERRED_FEATURE_ORDER,
    candidate_feature_map,
    runtime_feature_order,
)
from lib.landmarks.ensemble.runtime_resolver import (
    ModelPrediction,
    RuntimeBucketResult,
    RuntimeResolverConfig,
    bucket_candidate_name,
    resolve_runtime,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import (
    RuntimeResolverScorer,
    write_runtime_resolver_scorer,
)

MODELS = ("hrnet", "spiga", "orformer")


def _face() -> np.ndarray:
    points: np.ndarray = np.zeros((68, 2), dtype="float32")
    points[0:17, 0] = np.linspace(40, 160, 17)
    points[0:17, 1] = 120 + 30 * np.sin(np.linspace(0, np.pi, 17))
    points[17:27, 0] = np.linspace(50, 150, 10)
    points[17:27, 1] = 70
    points[27:36, 0] = 100
    points[27:36, 1] = np.linspace(75, 110, 9)
    points[36:48, 0] = np.linspace(60, 140, 12)
    points[36:48, 1] = 85
    points[48:68, 0] = np.linspace(70, 130, 20)
    points[48:68, 1] = 130
    return points


def _predictions() -> list[ModelPrediction]:
    base = _face()
    return [
        ModelPrediction("hrnet", base + 0.1),
        ModelPrediction("spiga", base),
        ModelPrediction("orformer", base + 0.2),
    ]


def test_scorer_can_select_bucket_weighted_candidate(monkeypatch, tmp_path) -> None:
    """A learned scorer that favors the per-bucket candidate selects it (#8 + #10)."""
    monkeypatch.setattr(
        runtime_resolver,
        "infer_runtime_bucket",
        lambda **kwargs: RuntimeBucketResult(bucket="profile_left", features={}),
    )
    name = bucket_candidate_name("static_weighted", "profile")
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=(f"candidate_name={name}",),
            coefficients=(-5.0,),
            intercept=0.0,
        ),
        tmp_path / "scorer.json",
    )
    result = resolve_runtime(
        _predictions(),
        RuntimeResolverConfig(
            policy="learned_quality_v2",
            scorer_path=str(scorer_path),
            weights={model: [1.0 / 3.0] * 68 for model in MODELS},
            bucket_weights={
                "profile": {"hrnet": [3.0] * 68, "spiga": [1.0] * 68, "orformer": [1.0] * 68}
            },
        ),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
    )
    assert name in result.metadata["candidate_scores"]
    assert result.selected_candidate == name


def test_scorer_can_select_region_weighted_candidate(monkeypatch, tmp_path) -> None:
    """A learned scorer that favors the region-weighted candidate selects it (#9 + #10)."""
    monkeypatch.setattr(
        runtime_resolver,
        "infer_runtime_bucket",
        lambda **kwargs: RuntimeBucketResult(bucket="frontal", features={}),
    )
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=region_weighted",),
            coefficients=(-5.0,),
            intercept=0.0,
        ),
        tmp_path / "scorer.json",
    )
    result = resolve_runtime(
        _predictions(),
        RuntimeResolverConfig(
            policy="learned_quality_v2",
            scorer_path=str(scorer_path),
            weights={model: [1.0 / 3.0] * 68 for model in MODELS},
            region_weights={
                region: {"hrnet": 1.0, "spiga": 1.0, "orformer": 1.0}
                for region in (
                    "visible_jaw",
                    "brows",
                    "nose",
                    "visible_eye",
                    "mouth_outer",
                    "mouth_inner",
                )
            },
        ),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
    )
    assert "region_weighted" in result.metadata["candidate_scores"]
    assert result.selected_candidate == "region_weighted"


def test_adaptive_candidates_absent_without_weights(monkeypatch, tmp_path) -> None:
    """Without bucket/region weights the candidate pool is unchanged (v1 parity)."""
    monkeypatch.setattr(
        runtime_resolver,
        "infer_runtime_bucket",
        lambda **kwargs: RuntimeBucketResult(bucket="profile_left", features={}),
    )
    scorer_path = write_runtime_resolver_scorer(
        RuntimeResolverScorer(
            features=("candidate_name=hrnet",), coefficients=(-1.0,), intercept=0.0
        ),
        tmp_path / "scorer.json",
    )
    result = resolve_runtime(
        _predictions(),
        RuntimeResolverConfig(
            policy="learned_quality_v2",
            scorer_path=str(scorer_path),
            weights={model: [1.0 / 3.0] * 68 for model in MODELS},
        ),
        detector_bbox=(35.0, 65.0, 165.0, 155.0),
    )
    scored = set(result.metadata["candidate_scores"])
    assert not any("@" in name for name in scored)
    assert "region_weighted" not in scored


def test_runtime_feature_order_is_stable_under_new_candidate_one_hots() -> None:
    """New candidate one-hots are additive and never reorder the preferred prefix."""

    class _Metric:
        geometry_veto_reasons = ()
        shape_veto_reasons = ()
        roll_degrees = 1.0
        yaw_degrees = 2.0

    class _Candidate:
        def __init__(self, name: str, is_fusion: bool) -> None:
            self.name = name
            self.is_fusion = is_fusion

    metric = _Metric()
    legacy = [
        candidate_feature_map(_Candidate("hrnet", False), metric),
        candidate_feature_map(_Candidate("static_weighted", True), metric),
    ]
    enlarged = legacy + [
        candidate_feature_map(_Candidate("static_weighted@profile", True), metric),
        candidate_feature_map(_Candidate("region_weighted", True), metric),
    ]

    legacy_order = runtime_feature_order(legacy)
    enlarged_order = runtime_feature_order(enlarged)

    # The preferred (non-one-hot) feature prefix is identical in both orderings.
    preferred_legacy = [name for name in legacy_order if name in RUNTIME_PREFERRED_FEATURE_ORDER]
    preferred_enlarged = [
        name for name in enlarged_order if name in RUNTIME_PREFERRED_FEATURE_ORDER
    ]
    assert preferred_legacy == preferred_enlarged
    # The new candidate one-hots are present, appended after the preferred block.
    assert "candidate_name=region_weighted" in enlarged_order
    assert "candidate_name=static_weighted@profile" in enlarged_order
