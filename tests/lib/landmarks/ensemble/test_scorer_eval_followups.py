#!/usr/bin/env python3
"""Regression tests for scorer-eval follow-up fixes."""

from __future__ import annotations

import types
from typing import cast

from lib.landmarks.ensemble.runtime_resolver_scorer_data import SampleCandidateContext
from lib.landmarks.ensemble.scorer_eval import choose_scorer


def _metric(veto: tuple[str, ...] = ()) -> types.SimpleNamespace:
    return types.SimpleNamespace(geometry_veto_reasons=veto)


def _candidate(name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(name=name, is_fusion=False)


def test_profile_context_without_v3_rows_is_not_marked_degraded_fallback() -> None:
    context = types.SimpleNamespace(
        sample_id="profile-production",
        nme_by_candidate={"hrnet": 0.1, "spiga": 0.2},
        metrics={"hrnet": _metric(), "spiga": _metric()},
        scorer_rows=[],  # production/no-GT path: no v3 validity evidence
        candidates=[_candidate("hrnet"), _candidate("spiga")],
        candidate_extra_features={},
        condition="profile_left",
        runtime_bucket="profile_left",
        runtime_bucket_source="stored_manifest_landmark_ensemble",
    )

    chosen, fallback_used, fallback_reason, _rejected, _replacement = choose_scorer(
        cast(SampleCandidateContext, context),
        {"hrnet": 0.2, "spiga": 0.1},
        risk_floor_for_safe_fallback=999.0,
        safe_fallback_min_delta=999.0,
    )

    assert chosen == "spiga"
    assert fallback_used is False
    assert fallback_reason != "profile_all_invalid_degraded_fallback"
