#!/usr/bin/env python3
"""Unit tests for the runtime resolver evaluation harness."""

from __future__ import annotations

import numpy as np
import pytest

from tools.landmarks.evaluate_runtime_resolver import (
    CandidateMetrics,
    CandidateRecord,
    aggregate_reports,
    apply_policy,
    render_overlays,
    select_worst_samples,
    write_csv_report,
    write_failures_csv,
    write_worst_samples_json,
)


def _candidate(name: str) -> CandidateRecord:
    return CandidateRecord(
        name=name,
        landmarks=np.zeros((68, 2), dtype="float32"),
        is_fusion=False,
        contributing_models=(name,),
    )


def _metric(nme: float, *, roll: float | None) -> CandidateMetrics:
    return CandidateMetrics(
        nme=nme,
        failure=nme > 0.08,
        roll_degrees=roll,
        yaw_degrees=None,
        pitch_degrees=None,
    )


def test_roll_aware_veto_vetoes_roll_outlier_and_picks_survivor() -> None:
    candidates = [_candidate("static_weighted"), _candidate("orformer"), _candidate("spiga")]
    metrics = {
        "static_weighted": _metric(0.02, roll=75.0),
        "orformer": _metric(0.04, roll=2.0),
        "spiga": _metric(0.05, roll=3.0),
    }

    decision = apply_policy("roll_aware_veto", candidates, metrics)

    assert decision.chosen == "orformer"
    assert decision.vetoed == ("static_weighted",)
    assert decision.consensus_roll_deg == pytest.approx(3.0)


def test_roll_aware_veto_falls_back_when_every_candidate_is_vetoed() -> None:
    candidates = [_candidate("a"), _candidate("b")]
    metrics = {
        "a": _metric(0.02, roll=None),
        "b": _metric(0.04, roll=None),
    }

    decision = apply_policy("roll_aware_veto", candidates, metrics)

    assert decision.chosen == "a"
    assert decision.vetoed == ("b",)
    assert decision.diagnostics["fallback_reason"] == "no_candidate_pose_estimate"


def test_aggregate_reports_summarizes_empty_input() -> None:
    payload = aggregate_reports([], policy="roll_aware_veto", failure_threshold=0.08)

    assert payload == {"policy": "roll_aware_veto", "sample_count": 0}


def test_output_helpers_handle_empty_reports(tmp_path) -> None:
    csv_path = tmp_path / "report.csv"
    failures_path = tmp_path / "failures.csv"
    worst_path = tmp_path / "worst.json"
    overlay_dir = tmp_path / "overlays"

    write_csv_report([], csv_path)
    write_failures_csv([], failures_path, failure_threshold=0.08)
    write_worst_samples_json(select_worst_samples([], count=10), worst_path)
    overlay_counts = render_overlays([], overlay_dir, failure_threshold=0.08, worst_count=10)

    assert csv_path.read_text(encoding="utf-8") == ""
    assert failures_path.read_text(encoding="utf-8").startswith("sample_id,dataset")
    assert worst_path.read_text(encoding="utf-8") == '{\n  "samples": []\n}\n'
    assert overlay_counts == {"requested": 0, "written": 0}
