#!/usr/bin/env python3
"""CLI integration tests for ``validate_geometry_signals`` (#80)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.weights import LANDMARK_COUNT, save_weights
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction
from tools.landmarks.validate_geometry_signals import main as validate_main


def _truth_face() -> np.ndarray:
    points = np.zeros((LANDMARK_COUNT, 2), dtype="float32")
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


def _build_fixture(tmp_path: Path, models: tuple[str, ...]) -> dict[str, Path]:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    truth = _truth_face()
    np.save(str(dataset_dir / "truth.npy"), truth)
    cache_dir = tmp_path / "cache"
    cache = DiskPredictionCache(cache_dir)
    samples: list[dict[str, object]] = []
    for idx in range(8):
        sid = f"face-{idx:03d}"
        samples.append(
            {
                "sample_id": sid,
                "dataset": "fixture",
                "condition": "profile_left" if idx < 4 else "profile_right",
                "image": "image.png",
                "landmarks": "truth.npy",
                "metadata": {"face_bbox": [40.0, 60.0, 160.0, 150.0]},
            }
        )
        for model_idx, model in enumerate(models):
            noise = float(model_idx) * 0.5
            cache.write(sid, LandmarkPrediction(truth + noise, model_name=model))
    manifest = dataset_dir / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return {"manifest": manifest, "cache": cache_dir}


def test_validate_signals_writes_all_outputs(tmp_path: Path) -> None:
    """CLI emits candidate index + signal + selector reports."""
    fixture = _build_fixture(tmp_path, ("hrnet", "spiga", "orformer"))
    output = tmp_path / "out"

    rc = validate_main(
        [
            "--manifest",
            str(fixture["manifest"]),
            "--cache-dir",
            str(fixture["cache"]),
            "--models",
            "hrnet,spiga,orformer",
            "--output-dir",
            str(output),
        ]
    )
    assert rc == 0
    for name in (
        "candidate_index.csv",
        "signal_validation_report.json",
        "signal_validation_report.csv",
        "selector_ablations.json",
        "selector_ablations.csv",
    ):
        assert (output / name).is_file(), f"missing required output {name}"


def test_candidate_index_has_one_row_per_sample_and_candidate(tmp_path: Path) -> None:
    """8 samples × 3 candidates → 24 rows in the candidate index."""
    fixture = _build_fixture(tmp_path, ("hrnet", "spiga", "orformer"))
    output = tmp_path / "out"
    assert (
        validate_main(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                "hrnet,spiga,orformer",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader((output / "candidate_index.csv").open(encoding="utf-8")))
    assert len(rows) == 24
    sample_ids = {row["sample_id"] for row in rows}
    assert len(sample_ids) == 8


def test_signal_validation_report_lists_every_named_signal(tmp_path: Path) -> None:
    """The JSON report contains a row for every default signal."""
    fixture = _build_fixture(tmp_path, ("hrnet", "spiga"))
    output = tmp_path / "out"
    assert (
        validate_main(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                "hrnet,spiga",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads((output / "signal_validation_report.json").read_text(encoding="utf-8"))
    names = {entry["name"] for entry in payload["signals"]}
    assert {"nme", "transform_normalized", "hull_iou", "geometry_score"}.issubset(names)


def test_validate_signals_includes_variants_when_weights_given(tmp_path: Path) -> None:
    """``--variants`` adds fused-ensemble candidate rows alongside single models."""
    fixture = _build_fixture(tmp_path, ("hrnet", "spiga", "orformer"))
    weights_path = tmp_path / "weights.json"
    save_weights(
        weights_path,
        {m: [1.0 / 3] * LANDMARK_COUNT for m in ("hrnet", "spiga", "orformer")},
    )
    output = tmp_path / "out"

    assert (
        validate_main(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                "hrnet,spiga,orformer",
                "--variants",
                "plain_average,static_weighted",
                "--weights",
                str(weights_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    rows = list(csv.DictReader((output / "candidate_index.csv").open(encoding="utf-8")))
    labels = {row["candidate_label"] for row in rows}
    assert {"hrnet", "spiga", "orformer", "plain_average", "static_weighted"}.issubset(labels)


def test_selector_ablations_csv_has_one_row_per_named_selector(tmp_path: Path) -> None:
    """Every named selector contributes one ablation row."""
    fixture = _build_fixture(tmp_path, ("hrnet", "spiga"))
    output = tmp_path / "out"
    assert (
        validate_main(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                "hrnet,spiga",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    rows = list(csv.DictReader((output / "selector_ablations.csv").open(encoding="utf-8")))
    names = {row["name"] for row in rows}
    assert {"lowest_nme", "composite_geometry", "highest_hull_iou"}.issubset(names)
