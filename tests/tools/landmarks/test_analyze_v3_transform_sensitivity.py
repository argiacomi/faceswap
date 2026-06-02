#!/usr/bin/env python3
"""Tests for the v3 oracle/cost sensitivity analyzer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.landmarks import analyze_v3_transform_sensitivity as analyzer

_COMPONENT_HEADER = (
    "sample_id,face_index,candidate_name,runtime_bucket,rankable_v3,hard_invalid_v3,"
    "corner_delta_v3,fit_delta_v3,soft_structural_penalty_v3,transform_cost_v3,"
    "was_selected_by_current_policy"
)


def _write_component_csv(path: Path, *, soft: bool = False) -> None:
    """Write a small component-column scorer-row CSV."""
    soft_value = 0.05 if soft else 0.0
    lines = [
        _COMPONENT_HEADER,
        f"f1,0,fan,frontal,1,0,0.020,0.002,{soft_value},0.0205,1",
        "f1,0,hrnet,frontal,1,0,0.018,0.010,0.0,0.0205,0",
        "f1,0,spiga,frontal,1,0,0.050,0.001,0.0,0.0502,0",
        "f2,0,fan,profile,1,0,0.040,0.001,0.0,0.0402,1",
        "f2,0,hrnet,profile,1,0,0.041,0.001,0.0,0.0412,0",
        "f3,0,fan,intermediate,1,0,0.030,0.020,0.0,0.0350,0",
        "f3,0,hrnet,intermediate,1,0,0.031,0.001,0.0,0.0312,1",
        "f3,0,spiga,intermediate,0,1,0.0,0.0,0.0,0.0,0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_analyzer_weighted_components_envelope_and_flip_cost(tmp_path: Path) -> None:
    """Component columns drive a real corner/fit/soft sweep with envelope + flip cost."""
    rows_csv = tmp_path / "rows.csv"
    _write_component_csv(rows_csv)
    out = tmp_path / "out"

    assert analyzer.main(["--rows", str(rows_csv), "--output-dir", str(out), "--no-progress"]) == 0

    summary = json.loads((out / "v3_cost_sensitivity_summary.json").read_text())
    # Item 6: labelled as oracle/cost sensitivity, not promotion evidence.
    assert summary["analysis_type"] == "oracle_cost_sensitivity"
    assert summary["promotion_evidence"] is False
    assert summary["components_available"] is True
    assert summary["component_mode"] == "weighted_components"
    # Item 4: soft tier inert is surfaced explicitly.
    assert summary["soft_structural_penalty_active"] is False
    # Item 1: a reasonable envelope is reported separately from the stress grid.
    assert summary["envelope_config_counts"]["baseline"] == 1
    assert summary["envelope_config_counts"]["single_axis"] > 0
    assert summary["reasonable_envelope"]["config_count"] >= 1
    assert "stress_grid" in summary
    assert 0.0 <= summary["reasonable_envelope"]["mean_oracle_winner_stability"] <= 1.0
    # Item 2: flip-cost metrics exist on every config record.
    record = summary["configs"][0]
    for key in (
        "oracle_flip_count",
        "oracle_flip_rate",
        "oracle_flip_mean_delta_regret",
        "oracle_flip_p95_delta_regret",
        "flip_count_by_baseline_gap",
    ):
        assert key in record

    # Item 5: per-bucket gate pass/fail is emitted.
    bucket_rows = list(csv.DictReader((out / "v3_cost_sensitivity_by_bucket.csv").open()))
    assert bucket_rows
    assert "cost_sensitivity_gate_pass" in bucket_rows[0]
    assert "oracle_flip_p95_delta_regret" in bucket_rows[0]

    gates = json.loads((out / "v3_cost_sensitivity_gates.json").read_text())
    assert gates and "cost_sensitivity_gate_pass" in gates[0]


def test_analyzer_reports_active_soft_tier(tmp_path: Path) -> None:
    """A non-zero soft penalty flips the soft-active metadata to True."""
    rows_csv = tmp_path / "rows.csv"
    _write_component_csv(rows_csv, soft=True)
    out = tmp_path / "out"

    analyzer.main(["--rows", str(rows_csv), "--output-dir", str(out), "--no-progress"])

    summary = json.loads((out / "v3_cost_sensitivity_summary.json").read_text())
    assert summary["soft_structural_penalty_active"] is True


def test_analyzer_degrades_without_component_columns(tmp_path: Path) -> None:
    """Without corner/fit columns the sweep collapses to transform_cost_v3 only."""
    rows_csv = tmp_path / "rows.csv"
    rows_csv.write_text(
        "sample_id,face_index,candidate_name,runtime_bucket,rankable_v3,hard_invalid_v3,"
        "transform_cost_v3\n"
        "f1,0,fan,frontal,1,0,0.05\n"
        "f1,0,hrnet,frontal,1,0,0.04\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"

    analyzer.main(["--rows", str(rows_csv), "--output-dir", str(out), "--no-progress"])

    summary = json.loads((out / "v3_cost_sensitivity_summary.json").read_text())
    assert summary["components_available"] is False
    assert summary["component_mode"] == "existing_transform_cost_v3_only"
    # Only min_gap varies when components are absent.
    assert {config["corner"] for config in summary["configs"]} == {
        analyzer.BASELINE_WEIGHTS["corner"]
    }
