#!/usr/bin/env python3
"""Tests for profile scorer routing, reporting, and specialist training filter (#218)."""

from __future__ import annotations

import json
import types
from pathlib import Path

from lib.landmarks.ensemble import scorer_eval
from lib.landmarks.ensemble.scorer_reports import write_profile_scorer_report
from lib.landmarks.ensemble.scorer_training import (
    SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE,
    filter_profile_specialist_rows,
    is_profile_specialist_row,
)


def _ctx(sample_id: str, condition: str = "", runtime_bucket: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        sample_id=sample_id,
        condition=condition,
        runtime_bucket=runtime_bucket,
        hard_case_tags=(),
        source="",
        dataset="",
        runtime_bucket_source="",
        candidates=(),
        metrics={},
        nme_by_candidate={},
        failure_by_candidate={},
    )


def test_profile_report_bucket_classification() -> None:
    assert (
        scorer_eval._profile_report_bucket(_ctx("a", runtime_bucket="profile_left")) == "profile"
    )
    assert (
        scorer_eval._profile_report_bucket(_ctx("b", condition="profile_occlusion"))
        == "profile_occlusion"
    )
    assert scorer_eval._profile_report_bucket(_ctx("c", condition="occlusion")) == "occlusion"
    assert scorer_eval._profile_report_bucket(_ctx("d", runtime_bucket="frontal")) == "anchor"


def test_profile_scorer_report_groups_all_buckets(monkeypatch) -> None:
    # Isolate bucket grouping / route counting from the v3 row builder.
    monkeypatch.setattr(scorer_eval, "_context_rows_for_eval", lambda context: [])
    contexts = [
        _ctx("po", condition="profile_occlusion"),
        _ctx("p", runtime_bucket="profile_right"),
        _ctx("o", condition="occlusion"),
        _ctx("n", runtime_bucket="frontal"),
    ]
    report = scorer_eval.profile_scorer_report(contexts, {}, source_by_sample_id={})
    assert set(report["buckets"]) == {"profile_occlusion", "profile", "occlusion", "anchor"}
    # profile + occlusion + profile_occlusion contexts are on the profile route.
    assert report["profile_route_context_count"] == 3


def test_write_profile_scorer_report(tmp_path: Path) -> None:
    report = {
        "buckets": {
            "profile_occlusion": {
                "transform_group_count_v3": 4,
                "mean_transform_regret_v3": 0.05,
                "p95_transform_regret_v3": 0.12,
                "avoidable_invalid_selection_rate_v3": 0.0,
                "all_invalid_selected_rate_v3": 0.25,
                "validity_stage_valid_rankable_count_v3": 3,
                "validity_stage_profile_soft_valid_count_v3": 0,
                "validity_stage_all_invalid_count_v3": 1,
                "profile_all_invalid_degraded_fallback_count": 1,
            },
            "anchor": {"transform_group_count_v3": 2},
        },
        "profile_route": {"mean_transform_regret_v3": 0.05},
        "profile_route_context_count": 4,
    }
    out = write_profile_scorer_report(report=report, output_dir=tmp_path)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["profile_route_context_count"] == 4
    csv_text = (tmp_path / "profile_region_report.csv").read_text(encoding="utf-8")
    assert "profile_occlusion" in csv_text
    assert "anchor" in csv_text


def test_filter_profile_specialist_rows() -> None:
    profile_row = types.SimpleNamespace(
        condition="profile_left", runtime_bucket="profile_left", hard_case_tags=()
    )
    occ_row = types.SimpleNamespace(condition="occlusion", runtime_bucket="", hard_case_tags=())
    normal_row = types.SimpleNamespace(
        condition="normal", runtime_bucket="frontal", hard_case_tags=()
    )
    rows = [(profile_row, "gt"), (occ_row, "gt"), (normal_row, "gt")]
    assert is_profile_specialist_row(profile_row) is True  # type: ignore[arg-type]
    assert is_profile_specialist_row(normal_row) is False  # type: ignore[arg-type]
    filtered = filter_profile_specialist_rows(rows)  # type: ignore[arg-type]
    assert [row.condition for row, _ in filtered] == ["profile_left", "occlusion"]


def test_profile_policy_name_constant() -> None:
    assert SCORER_VERSION_LEARNED_QUALITY_V3_PROFILE == "learned_quality_v3_profile"
