#!/usr/bin/env python3
"""Tests for the runtime resolver evaluation harness."""

from __future__ import annotations

import json
import numpy as np
import pytest

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from tools.landmarks.evaluate_runtime_resolver import (
    CandidateMetrics,
    CandidateRecord,
    aggregate_reports,
    apply_policy,
    main,
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


def _truth_points() -> np.ndarray:
    points = np.zeros((68, 2), dtype="float32")
    points[:, 0] = np.linspace(10.0, 90.0, 68)
    points[:, 1] = 50.0 + np.sin(np.linspace(0.0, np.pi, 68)) * 30.0
    points[36] = [35.0, 45.0]
    points[45] = [65.0, 45.0]
    return points


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


def test_main_writes_all_resolver_outputs_for_minimal_cache(tmp_path) -> None:
    truth = _truth_points()
    landmarks_path = tmp_path / "landmarks.npy"
    np.save(str(landmarks_path), truth)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample-1",
                        "dataset": "aflw2000-3d",
                        "condition": "large_roll",
                        "image": str(tmp_path / "missing.jpg"),
                        "landmarks": str(landmarks_path),
                        "normalizer": 100.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cache = DiskPredictionCache(tmp_path / "cache")
    cache.write("sample-1", LandmarkPrediction(truth, model_name="spiga"), refresh=True)
    cache.write("sample-1", LandmarkPrediction(truth + 1.0, model_name="orformer"), refresh=True)

    weights_path = tmp_path / "weights.json"
    weights_path.write_text(
        json.dumps({"spiga": [0.5] * 68, "orformer": [0.5] * 68}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "resolver"

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--weights",
            str(weights_path),
            "--candidates",
            "spiga,orformer",
            "--output-dir",
            str(output_dir),
            "--worst-count",
            "0",
        ]
    ) == 0

    expected = {
        "resolver_policy_report.json",
        "resolver_policy_report.csv",
        "resolver_failures.csv",
        "resolver_worst_samples.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    payload = json.loads((output_dir / "resolver_policy_report.json").read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["chosen"]["pick_counts"]["spiga"] == 1
