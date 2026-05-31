#!/usr/bin/env python3
"""CLI integration tests for ``evaluate_alignment_geometry`` (#76)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.weights import LANDMARK_COUNT, save_weights
from tools.landmarks.evaluate_alignment_geometry import main as evaluate_main


def _truth_face() -> np.ndarray:
    points = np.zeros((LANDMARK_COUNT, 2), dtype="float32")  # type: ignore[var-annotated]
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


def _bbox() -> list[float]:
    return [40.0, 60.0, 160.0, 150.0]


def _build_fixture(tmp_path: Path, models: tuple[str, ...]) -> dict[str, Path]:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    truth_path = dataset_dir / "truth.npy"
    truth = _truth_face()
    np.save(str(truth_path), truth)
    cache_dir = tmp_path / "cache"
    cache = DiskPredictionCache(cache_dir)

    samples: list[dict[str, object]] = []
    for idx in range(6):
        sid = f"face-{idx:03d}"
        samples.append(
            {
                "sample_id": sid,
                "dataset": "fixture",
                "condition": "clean" if idx < 3 else "profile",
                "image": "image.png",
                "landmarks": "truth.npy",
                "metadata": {"face_bbox": _bbox()},
            }
        )
        for model_idx, model in enumerate(models):
            noise = float(model_idx) * 0.5
            cache.write(sid, LandmarkPrediction(truth + noise, model_name=model))
    manifest = dataset_dir / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return {"manifest": manifest, "cache": cache_dir}


def _run(args: list[str]) -> int:
    return evaluate_main(args)


def test_cli_writes_all_required_outputs(tmp_path: Path) -> None:
    """CLI emits the five output files per the roadmap."""
    models = ("hrnet", "spiga", "orformer")
    fixture = _build_fixture(tmp_path, models)
    output = tmp_path / "out"

    assert (
        _run(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                ",".join(models),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    for name in (
        "geometry_metrics.json",
        "geometry_metrics.csv",
        "per_region_geometry.csv",
        "catastrophic_geometry_failures.csv",
    ):
        assert (output / name).is_file(), f"missing required output {name}"
    assert (output / "worst_geometry_failures" / "worst_samples.json").is_file()


def test_cli_payload_includes_geometry_aggregates(tmp_path: Path) -> None:
    """JSON output exposes per-label geometry aggregates."""
    models = ("hrnet", "spiga", "orformer")
    fixture = _build_fixture(tmp_path, models)
    output = tmp_path / "out"

    assert (
        _run(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                ",".join(models),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads((output / "geometry_metrics.json").read_text(encoding="utf-8"))
    assert payload["objective"] == "alignment_geometry_v1"
    assert payload["best_single_label"] == "hrnet"
    for model in models:
        aggregate = payload["aggregates"][model]
        assert aggregate["sample_count"] == 6
        assert "p95_translation_normalized" in aggregate
        assert "catastrophic_failure_rate" in aggregate
        assert "per_bucket" in aggregate
        assert set(aggregate["per_bucket"]) >= {"fixture:clean", "fixture:profile"}


def test_cli_per_region_csv_has_one_row_per_region_per_sample(tmp_path: Path) -> None:
    """The per-region CSV contains one row per (sample, label, region)."""
    models = ("hrnet",)
    fixture = _build_fixture(tmp_path, models)
    output = tmp_path / "out"

    assert (
        _run(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                ",".join(models),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    with (output / "per_region_geometry.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    region_names = {row["region"] for row in rows}
    assert region_names == {"eyes", "nose", "mouth", "jaw", "brows"}
    # 6 samples × 1 label × 5 regions = 30 rows.
    assert len(rows) == 30


def test_cli_catastrophic_csv_is_empty_for_clean_predictions(tmp_path: Path) -> None:
    """No catastrophic flags should fire on the canonical happy-path fixture."""
    models = ("hrnet",)
    fixture = _build_fixture(tmp_path, models)
    output = tmp_path / "out"

    assert (
        _run(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                ",".join(models),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    with (output / "catastrophic_geometry_failures.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []


def test_cli_supports_ensemble_variants_with_weights(tmp_path: Path) -> None:
    """``--variants`` adds fused-ensemble rows alongside single-model rows."""
    models = ("hrnet", "spiga", "orformer")
    fixture = _build_fixture(tmp_path, models)
    weights_path = tmp_path / "weights.json"
    save_weights(weights_path, {m: [1.0 / len(models)] * LANDMARK_COUNT for m in models})
    output = tmp_path / "out"

    assert (
        _run(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                ",".join(models),
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

    payload = json.loads((output / "geometry_metrics.json").read_text(encoding="utf-8"))
    assert "plain_average" in payload["aggregates"]
    assert "static_weighted" in payload["aggregates"]


def test_cli_rejects_missing_cached_predictions(tmp_path: Path) -> None:
    """Manifest sample missing a model's cached prediction fails clearly."""
    fixture = _build_fixture(tmp_path, ("hrnet", "spiga"))
    output = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="cached predictions"):
        _run(
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


def test_cli_worst_samples_index_ranks_by_overall_score(tmp_path: Path) -> None:
    """``worst_samples.json`` contains the top-N samples per label by descending score."""
    models = ("hrnet",)
    fixture = _build_fixture(tmp_path, models)
    output = tmp_path / "out"

    assert (
        _run(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                ",".join(models),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(
        (output / "worst_geometry_failures" / "worst_samples.json").read_text(encoding="utf-8")
    )
    assert payload["objective"] == "alignment_geometry_v1"
    assert "hrnet" in payload["by_label"]
    scores = [row["overall_score"] for row in payload["by_label"]["hrnet"]]
    assert scores == sorted(scores, reverse=True)
