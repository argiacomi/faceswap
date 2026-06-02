#!/usr/bin/env python3
"""v3 scorer-row tests for transform labels, diagnostics, and CSV schema."""

from __future__ import annotations

import csv
import typing as T

import numpy as np
import pytest

from lib.landmarks.ensemble.runtime_resolver import CandidateMetrics, CandidateRecord
from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_V3_MAX_SHAPE_PLAUSIBILITY_SCORE,
    CandidateQualityRow,
    SampleCandidateContext,
    _v3_hard_invalid_reasons,
    _v3_soft_suspect_reasons,
    rows_for_context,
    write_rows_csv,
)
from lib.landmarks.ensemble.scorer_dataset import _fieldnames
from lib.landmarks.ensemble.scorer_target_config import (
    REGRESSION_TARGETS,
    TARGET_TRANSFORM_COST_V3,
    TARGET_TRANSFORM_REGRET_V3,
)
from lib.landmarks.ensemble.scorer_training import (
    MIN_V3_ORACLE_GAP,
    _lambdarank_label_v3,
    _v3_lambdarank_item_weights,
    grouped_rankable_rows_v3,
    scorer_target_value,
    v3_lambdarank_query_weight,
    write_tagged_rows_csv,
)

V3_FIELDS = (
    "transform_cost_v3",
    "center_delta_v3",
    "scale_delta_v3",
    "roll_delta_degrees_v3",
    "fit_delta_v3",
    "transform_oracle_cost_v3",
    "transform_regret_v3",
    "transform_oracle_candidate_v3",
    "transform_oracle_gap_v3",
    "rankable_v3",
    "hard_invalid_v3",
    "hard_invalid_reasons_v3",
    "soft_structural_penalty_v3",
)


def _truth_face() -> np.ndarray:
    points: np.ndarray = np.zeros((68, 2), dtype="float64")
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


def _metric(**overrides: object) -> CandidateMetrics:
    values: dict[str, T.Any] = {
        "roll_degrees": None,
        "yaw_degrees": None,
        "pitch_degrees": None,
    }
    values.update(overrides)
    return CandidateMetrics(**values)


def _row(**overrides: object) -> CandidateQualityRow:
    values: dict[str, T.Any] = {
        "sample_id": "sample",
        "face_index": 0,
        "dataset": "dataset",
        "condition": "frontal",
        "candidate_name": "candidate_a",
        "candidate_nme": 0.10,
        "oracle_nme": 0.05,
        "regret_vs_oracle": 0.05,
        "normalized_regret": 1.0,
        "failure_label": False,
        "large_regret_label": False,
        "candidate_failure_or_high_gap": False,
        "selection_cost": 0.25,
        "transform_cost_v3": 0.40,
        "center_delta_v3": 0.05,
        "scale_delta_v3": 0.05,
        "roll_delta_degrees_v3": 1.0,
        "fit_delta_v3": 0.05,
        "transform_oracle_cost_v3": 0.10,
        "transform_regret_v3": 0.30,
        "transform_oracle_candidate_v3": "candidate_b",
        "transform_oracle_gap_v3": 0.20,
        "rankable_v3": True,
        "hard_invalid_v3": False,
        "hard_invalid_reasons_v3": (),
        "soft_structural_penalty_v3": 0.0,
        "is_oracle": False,
        "was_selected_by_current_policy": True,
        "gap_vs_oracle": 0.05,
        "runtime_bucket": "frontal",
        "hard_case_tags": (),
        "risk_route": "direct",
        "feature_values": {"feature_a": 1.0},
        "selected_by_current_policy": "candidate_a",
        "selected_candidate_missing_from_eval": False,
        "oracle": "candidate_b",
        "runtime_bucket_source": "test",
        "geometry_veto_reasons": (),
    }
    values.update(overrides)
    return CandidateQualityRow(**values)


def _csv_header(path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return set(next(csv.reader(handle)))


def test_soft_suspect_uses_threshold_not_any_positive_plausibility_score() -> None:
    below = _metric(
        shape_plausibility_score=DEFAULT_V3_MAX_SHAPE_PLAUSIBILITY_SCORE - 1e-6,
    )
    above = _metric(
        shape_plausibility_score=DEFAULT_V3_MAX_SHAPE_PLAUSIBILITY_SCORE + 1e-6,
    )

    assert "low_plausibility" not in _v3_soft_suspect_reasons(below)
    assert "low_plausibility" in _v3_soft_suspect_reasons(above)


def test_soft_suspect_reasons_do_not_repeat_hard_invalid_reasons() -> None:
    metric = _metric(
        geometry_veto_reasons=(
            "hull_area_too_small",
            "borderline_roi_warning",
        )
    )

    hard = _v3_hard_invalid_reasons(metric)
    soft = _v3_soft_suspect_reasons(metric, hard_invalid_reasons=hard)

    assert "hull_area_too_small" in hard
    assert "hull_area_too_small" not in soft
    assert "borderline_roi_warning" in soft


def test_rows_for_context_all_invalid_group_has_no_v3_oracle() -> None:
    truth = _truth_face()
    candidates = (
        CandidateRecord(
            name="candidate_a",
            landmarks=truth.copy(),
            is_fusion=False,
            contributing_models=("candidate_a",),
        ),
        CandidateRecord(
            name="candidate_b",
            landmarks=truth + np.asarray([3.0, 0.0], dtype="float64"),
            is_fusion=False,
            contributing_models=("candidate_b",),
        ),
    )
    metrics = {
        candidate.name: _metric(geometry_veto_reasons=("cloud_collapse",))
        for candidate in candidates
    }
    context = SampleCandidateContext(
        sample_id="sample",
        face_index=0,
        dataset="dataset",
        source="gt",
        condition="frontal",
        candidates=candidates,
        metrics=metrics,
        truth_landmarks=truth,
        normalizer=100.0,
        visibility=None,
        nme_by_candidate={"candidate_a": 0.10, "candidate_b": 0.12},
        failure_by_candidate={"candidate_a": False, "candidate_b": False},
        runtime_bucket="frontal",
        hard_case_tags=(),
        risk_route="direct",
        current_policy_choice="candidate_a",
        oracle="candidate_a",
        model_predictions_available={},
        roll_estimate=None,
        yaw_estimate=None,
        candidate_yaw_disagreement=None,
        max_disagreement_px=0.0,
        runtime_bucket_source="test",
        selected_candidate_missing_from_eval=False,
        candidate_extra_features={},
    )

    rows = rows_for_context(context)

    assert rows
    assert all(not row.rankable_v3 for row in rows)
    assert all(row.hard_invalid_v3 for row in rows)
    assert all(row.transform_oracle_candidate_v3 == "" for row in rows)
    assert all(row.transform_regret_v3 == pytest.approx(0.0) for row in rows)


def test_transform_regret_v3_is_registered_training_target() -> None:
    row = _row(transform_cost_v3=0.625, transform_regret_v3=0.375)

    # v3 regret is the only registered learned-quality training target (#214); the
    # raw transform cost is retained as an offline diagnostic component only.
    assert REGRESSION_TARGETS == (TARGET_TRANSFORM_REGRET_V3,)
    assert TARGET_TRANSFORM_COST_V3 not in REGRESSION_TARGETS
    assert TARGET_TRANSFORM_COST_V3 == "transform_alignment_cost_v3"
    assert TARGET_TRANSFORM_REGRET_V3 == "transform_alignment_regret_v3"
    assert scorer_target_value(row, TARGET_TRANSFORM_REGRET_V3) == pytest.approx(0.375)
    with pytest.raises(ValueError, match="unsupported scorer target"):
        scorer_target_value(row, TARGET_TRANSFORM_COST_V3)


def test_v3_columns_are_carried_through_csv_schemas(tmp_path) -> None:
    row = _row()

    runtime_rows = write_rows_csv([row], tmp_path / "runtime_rows.csv")
    tagged_rows = write_tagged_rows_csv([(row, "gt")], tmp_path / "tagged_rows.csv")

    assert set(V3_FIELDS).issubset(_fieldnames(("feature_a",)))
    assert set(V3_FIELDS).issubset(_csv_header(runtime_rows))
    assert set(V3_FIELDS).issubset(_csv_header(tagged_rows))


def test_v3_rankable_filter_drops_invalid_rows_and_counts_abstain_groups() -> None:
    rankable = _row(
        sample_id="sample_a",
        candidate_name="candidate_a",
        rankable_v3=True,
        hard_invalid_v3=False,
        transform_regret_v3=0.0,
        transform_oracle_gap_v3=0.20,
    )
    rankable_same_group = _row(
        sample_id="sample_a",
        candidate_name="candidate_b",
        rankable_v3=True,
        hard_invalid_v3=False,
        transform_regret_v3=0.2,
        transform_oracle_gap_v3=0.20,
    )
    invalid_same_group = _row(
        sample_id="sample_a",
        candidate_name="candidate_c",
        rankable_v3=False,
        hard_invalid_v3=True,
        hard_invalid_reasons_v3=("cloud_collapse",),
        transform_regret_v3=0.0,
    )
    invalid_all_group = _row(
        sample_id="sample_b",
        candidate_name="candidate_c",
        rankable_v3=False,
        hard_invalid_v3=True,
        hard_invalid_reasons_v3=("eye_mouth_flip",),
        transform_regret_v3=0.0,
    )

    groups, _query_weights, stats = grouped_rankable_rows_v3(
        [
            (rankable, "gt"),
            (rankable_same_group, "gt"),
            (invalid_same_group, "gt"),
            (invalid_all_group, "gt"),
        ]
    )

    assert groups == [[(rankable, "gt"), (rankable_same_group, "gt")]]
    assert stats["total_group_count"] == 2
    assert stats["rankable_pair_group_count"] == 1
    assert stats["fallback_abstain_group_count"] == 1
    assert stats["fallback_abstain_row_count"] == 1
    assert stats["hard_invalid_row_count"] == 2


def test_grouped_rankable_rows_v3_excludes_invalid_candidates() -> None:
    rankable = _row(
        sample_id="sample_a",
        candidate_name="candidate_a",
        rankable_v3=True,
        hard_invalid_v3=False,
        transform_oracle_gap_v3=0.20,
    )
    rankable_same_group = _row(
        sample_id="sample_a",
        candidate_name="candidate_b",
        rankable_v3=True,
        hard_invalid_v3=False,
        transform_regret_v3=0.2,
        transform_oracle_gap_v3=0.20,
    )
    invalid_same_group = _row(
        sample_id="sample_a",
        candidate_name="candidate_c",
        rankable_v3=False,
        hard_invalid_v3=True,
        hard_invalid_reasons_v3=("cloud_collapse",),
    )

    groups, query_weights, stats = grouped_rankable_rows_v3(
        [
            (rankable, "gt"),
            (rankable_same_group, "gt"),
            (invalid_same_group, "gt"),
        ]
    )

    assert groups == [[(rankable, "gt"), (rankable_same_group, "gt")]]
    assert query_weights == pytest.approx([v3_lambdarank_query_weight(rankable, "gt")])
    assert stats["rankable_pair_group_count"] == 1
    assert stats["rankable_row_count"] == 2
    assert stats["hard_invalid_row_count"] == 1


def test_v3_rankable_filter_skips_near_tie_groups() -> None:
    first = _row(
        sample_id="near_tie",
        candidate_name="candidate_a",
        rankable_v3=True,
        hard_invalid_v3=False,
        transform_regret_v3=0.0,
        transform_oracle_gap_v3=MIN_V3_ORACLE_GAP / 2.0,
    )
    second = _row(
        sample_id="near_tie",
        candidate_name="candidate_b",
        rankable_v3=True,
        hard_invalid_v3=False,
        transform_regret_v3=MIN_V3_ORACLE_GAP / 2.0,
        transform_oracle_gap_v3=MIN_V3_ORACLE_GAP / 2.0,
    )

    groups, _query_weights, stats = grouped_rankable_rows_v3([(first, "gt"), (second, "gt")])

    assert groups == []
    assert stats["rankable_pair_group_count"] == 0
    assert stats["near_tie_group_count"] == 1
    assert stats["near_tie_row_count"] == 2
    assert stats["min_v3_oracle_gap"] == pytest.approx(MIN_V3_ORACLE_GAP)


def test_v3_lambdarank_label_uses_transform_regret_not_nme_or_selection_cost() -> None:
    transform_oracle = _row(
        candidate_name="transform_oracle",
        candidate_nme=0.90,
        oracle_nme=0.01,
        selection_cost=3.0,
        transform_regret_v3=0.0,
    )
    transform_bad = _row(
        candidate_name="transform_bad",
        candidate_nme=0.01,
        oracle_nme=0.01,
        selection_cost=0.0,
        transform_regret_v3=1.0,
    )

    assert _lambdarank_label_v3(transform_oracle) > _lambdarank_label_v3(transform_bad)


def test_v3_query_weight_ignores_candidate_specific_signals() -> None:
    clean = _row(
        condition="profile",
        runtime_bucket="profile",
        geometry_veto_reasons=(),
        soft_structural_penalty_v3=0.0,
        hard_invalid_v3=False,
    )
    candidate_penalized = _row(
        condition="profile",
        runtime_bucket="profile",
        geometry_veto_reasons=("jaw_crop_warning",),
        soft_structural_penalty_v3=0.5,
        hard_invalid_v3=True,
        hard_invalid_reasons_v3=("cloud_collapse",),
    )

    assert v3_lambdarank_query_weight(candidate_penalized, "gt") == pytest.approx(
        v3_lambdarank_query_weight(clean, "gt")
    )


def test_v3_item_weights_repeat_query_weight_for_whole_group() -> None:
    first = _row(
        sample_id="profile_group",
        candidate_name="candidate_a",
        condition="profile",
        runtime_bucket="profile",
        geometry_veto_reasons=(),
        soft_structural_penalty_v3=0.0,
    )
    second = _row(
        sample_id="profile_group",
        candidate_name="candidate_b",
        condition="profile",
        runtime_bucket="profile",
        geometry_veto_reasons=("eye_warning",),
        soft_structural_penalty_v3=0.75,
    )
    groups, query_weights, _stats = grouped_rankable_rows_v3([(first, "gt"), (second, "gt")])

    weights = _v3_lambdarank_item_weights(query_weights, [len(groups[0])])

    assert weights.tolist() == pytest.approx([query_weights[0]] * 2)
