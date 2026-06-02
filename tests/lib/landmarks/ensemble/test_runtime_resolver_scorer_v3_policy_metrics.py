#!/usr/bin/env python3
"""Focused v3 policy-metric tests for scorer and baseline regret accounting."""

from __future__ import annotations

import typing as T
from types import SimpleNamespace

import pytest

from lib.landmarks.ensemble import scorer_eval as scorer_eval_impl
from lib.landmarks.ensemble.runtime_resolver_scorer_data import SampleCandidateContext


def _v3_row(
    candidate_name: str,
    *,
    transform_cost_v3: float,
    transform_oracle_cost_v3: float = 0.10,
    transform_oracle_candidate_v3: str = "candidate_a",
    transform_oracle_gap_v3: float = 0.25,
    rankable_v3: bool = True,
    hard_invalid_v3: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        candidate_name=candidate_name,
        feature_values={f"candidate_name={candidate_name}": 1.0},
        transform_cost_v3=transform_cost_v3,
        transform_oracle_cost_v3=transform_oracle_cost_v3,
        transform_regret_v3=max(transform_cost_v3 - transform_oracle_cost_v3, 0.0),
        transform_oracle_candidate_v3=transform_oracle_candidate_v3,
        transform_oracle_gap_v3=transform_oracle_gap_v3,
        rankable_v3=rankable_v3,
        hard_invalid_v3=hard_invalid_v3,
        hard_invalid_reasons_v3=("hard_invalid",) if hard_invalid_v3 else (),
        soft_structural_penalty_v3=0.0,
    )


def _v3_context() -> SimpleNamespace:
    rows = (
        _v3_row("candidate_a", transform_cost_v3=0.10),
        _v3_row("candidate_b", transform_cost_v3=0.40),
        _v3_row("static_weighted_downweight", transform_cost_v3=0.25),
    )
    return SimpleNamespace(
        sample_id="sample_a",
        source="gt_hard",
        dataset="gt_hard",
        condition="profile",
        runtime_bucket="profile",
        runtime_bucket_source="stored_manifest_landmark_ensemble",
        hard_case_tags=("profile",),
        nme_by_candidate={
            "candidate_a": 0.01,
            "candidate_b": 0.04,
            "static_weighted_downweight": 0.02,
        },
        failure_by_candidate={
            "candidate_a": False,
            "candidate_b": False,
            "static_weighted_downweight": False,
        },
        oracle="candidate_a",
        scorer_rows=rows,
    )


def test_baseline_transform_regret_uses_same_v3_row_cost_as_scorer_policy() -> None:
    context = _v3_context()
    contexts = T.cast(list[SampleCandidateContext], [context])
    source_by_sample_id: dict[str, str] = {}

    scorer_summary = scorer_eval_impl.policy_summary(
        contexts,
        {context.sample_id: "candidate_b"},
        source_by_sample_id=source_by_sample_id,
    )
    baseline_summary = scorer_eval_impl.policy_summary(
        contexts,
        {context.sample_id: "static_weighted_downweight"},
        source_by_sample_id=source_by_sample_id,
    )
    oracle_summary = scorer_eval_impl.policy_summary(
        contexts,
        {context.sample_id: "candidate_a"},
        source_by_sample_id=source_by_sample_id,
    )

    by_candidate = {row.candidate_name: row for row in context.scorer_rows}

    assert scorer_summary["mean_transform_regret_v3"] == pytest.approx(
        by_candidate["candidate_b"].transform_regret_v3
    )
    assert baseline_summary["mean_transform_regret_v3"] == pytest.approx(
        by_candidate["static_weighted_downweight"].transform_regret_v3
    )
    assert oracle_summary["mean_transform_regret_v3"] == pytest.approx(0.0)
    assert scorer_summary["transform_eval_count_v3"] == 1
    assert baseline_summary["transform_eval_count_v3"] == 1
    assert oracle_summary["transform_eval_count_v3"] == 1


def test_policy_metric_bundle_baselines_share_v3_transform_regret_accounting() -> None:
    context = _v3_context()
    contexts = T.cast(list[SampleCandidateContext], [context])
    bundle = scorer_eval_impl.policy_metric_bundle(
        contexts,
        candidates=("candidate_a", "candidate_b", "static_weighted_downweight"),
        scorer_policy_name="learned_quality_v3",
        scorer_choices={context.sample_id: "candidate_b"},
        current_choices={context.sample_id: "candidate_a"},
        oracle_choices={context.sample_id: "candidate_a"},
        source_by_sample_id={},
    )

    by_candidate = {row.candidate_name: row for row in context.scorer_rows}

    assert bundle["learned_quality_v3"]["mean_transform_regret_v3"] == pytest.approx(
        by_candidate["candidate_b"].transform_regret_v3
    )
    assert bundle["static_weighted_downweight"]["mean_transform_regret_v3"] == pytest.approx(
        by_candidate["static_weighted_downweight"].transform_regret_v3
    )
    assert bundle["best_single"]["mean_transform_regret_v3"] == pytest.approx(0.0)
    assert bundle["oracle"]["mean_transform_regret_v3"] == pytest.approx(0.0)
