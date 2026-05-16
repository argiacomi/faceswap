#!/usr/bin/env python3
"""CLI integration tests for ``evaluate_aflw_profile`` (issue #76)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.ensemble.weights import LANDMARK_COUNT, save_weights
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction
from tools.landmarks.evaluate_aflw_profile import main as evaluate_main


def _truth_points() -> np.ndarray:
    return np.stack(
        (
            np.linspace(0.0, 100.0, LANDMARK_COUNT, dtype="float32"),
            np.linspace(0.0, 200.0, LANDMARK_COUNT, dtype="float32"),
        ),
        axis=1,
    )


def _bbox() -> list[float]:
    truth = _truth_points()
    return [
        float(truth[:, 0].min()),
        float(truth[:, 1].min()),
        float(truth[:, 0].max()),
        float(truth[:, 1].max()),
    ]


def _build_fixture(tmp_path: Path, models: tuple[str, ...]) -> dict[str, Path]:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    truth = dataset_dir / "truth.npy"
    np.save(str(truth), _truth_points())
    cache_dir = tmp_path / "cache"
    cache = DiskPredictionCache(cache_dir)

    samples: list[dict[str, object]] = []
    for idx in range(6):
        sid = f"profile-{idx:03d}"
        samples.append(
            {
                "sample_id": sid,
                "dataset": "aflw2000-3d",
                "condition": "profile",
                "image": "image.png",
                "landmarks": "truth.npy",
                "metadata": {"face_bbox": _bbox()},
            }
        )
        for model_idx, model in enumerate(models):
            # Vary noise by model so they receive different scores.
            noise = float(model_idx) * 0.5
            points = _truth_points() + noise
            cache.write(sid, LandmarkPrediction(points, model_name=model))

    manifest = dataset_dir / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return {"manifest": manifest, "cache": cache_dir, "dataset_dir": dataset_dir}


def _run(args: list[str]) -> int:
    return evaluate_main(args)


def test_evaluate_aflw_profile_writes_expected_outputs(tmp_path: Path) -> None:
    """The CLI emits the three required output files for a small fixture."""
    models = ("hrnet", "spiga", "orformer")
    fixture = _build_fixture(tmp_path, models)
    output_dir = tmp_path / "out"

    exit_code = _run(
        [
            "--manifest",
            str(fixture["manifest"]),
            "--cache-dir",
            str(fixture["cache"]),
            "--models",
            ",".join(models),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    for filename in (
        "aflw_profile_metrics.json",
        "aflw_profile_metrics.csv",
        "aflw_region_failures.csv",
    ):
        assert (output_dir / filename).is_file()


def test_evaluate_aflw_profile_payload_contains_required_blocks(tmp_path: Path) -> None:
    """JSON payload exposes per-label aggregates, regression rates, and PCK."""
    models = ("hrnet", "spiga", "orformer")
    fixture = _build_fixture(tmp_path, models)
    output_dir = tmp_path / "out"

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
                str(output_dir),
            ]
        )
        == 0
    )

    payload = json.loads((output_dir / "aflw_profile_metrics.json").read_text(encoding="utf-8"))
    assert payload["objective"] == "profile_alignment_v1"
    assert set(payload["aggregates"]).issuperset(set(models))
    assert payload["best_single_label"] == "hrnet"  # lowest noise wins
    for model in models:
        aggregate = payload["aggregates"][model]
        assert aggregate["sample_count"] == 6
        assert "overall_score" in aggregate
        assert "pck_at" in aggregate
        assert "per_region_error" in aggregate
        assert "per_region_failure_rate" in aggregate


def test_evaluate_aflw_profile_includes_ensemble_variants(tmp_path: Path) -> None:
    """Passing ``--variants`` adds fused ensemble rows alongside single-model rows."""
    models = ("hrnet", "spiga", "orformer")
    fixture = _build_fixture(tmp_path, models)
    weights_path = tmp_path / "weights.json"
    save_weights(
        weights_path,
        {model: [1.0 / len(models)] * LANDMARK_COUNT for model in models},
    )
    output_dir = tmp_path / "out"

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
                str(output_dir),
            ]
        )
        == 0
    )

    payload = json.loads((output_dir / "aflw_profile_metrics.json").read_text(encoding="utf-8"))
    assert "plain_average" in payload["aggregates"]
    assert "static_weighted" in payload["aggregates"]


def test_evaluate_aflw_profile_rows_csv_has_region_and_pck_columns(tmp_path: Path) -> None:
    """Per-sample CSV exposes region error/failure flags + per-threshold PCK columns."""
    models = ("hrnet",)
    fixture = _build_fixture(tmp_path, models)
    output_dir = tmp_path / "out"

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
                str(output_dir),
            ]
        )
        == 0
    )

    csv_text = (output_dir / "aflw_profile_metrics.csv").read_text(encoding="utf-8")
    header = csv_text.splitlines()[0].split(",")
    assert "region_error_visible_jaw" in header
    assert "region_failure_nose" in header
    assert "pck@0.03" in header


def test_evaluate_aflw_profile_skips_samples_without_bbox(tmp_path: Path) -> None:
    """Samples lacking a face_bbox fall back to landmark extrema and still score."""
    models = ("hrnet",)
    fixture = _build_fixture(tmp_path, models)
    # Strip face_bbox from the manifest so the CLI must use landmark fallback.
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        sample["metadata"] = {}
    fixture["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "out"

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
                str(output_dir),
            ]
        )
        == 0
    )

    payload = json.loads((output_dir / "aflw_profile_metrics.json").read_text(encoding="utf-8"))
    assert payload["aggregates"]["hrnet"]["sample_count"] == 6
    assert payload["skipped_sample_ids"] == []


def test_evaluate_aflw_profile_normalizer_choice(tmp_path: Path) -> None:
    """The ``--normalizer`` flag changes the recorded normalizer method."""
    models = ("hrnet",)
    fixture = _build_fixture(tmp_path, models)
    output_dir = tmp_path / "out"

    assert (
        _run(
            [
                "--manifest",
                str(fixture["manifest"]),
                "--cache-dir",
                str(fixture["cache"]),
                "--models",
                ",".join(models),
                "--normalizer",
                "bbox_sqrt_area",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    payload = json.loads((output_dir / "aflw_profile_metrics.json").read_text(encoding="utf-8"))
    assert payload["normalizer_method"] == "bbox_sqrt_area"


def test_evaluate_aflw_profile_rejects_missing_cached_prediction(tmp_path: Path) -> None:
    """A manifest sample with no cached prediction for a model fails clearly."""
    fixture = _build_fixture(tmp_path, ("hrnet", "spiga"))
    output_dir = tmp_path / "out"

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
                str(output_dir),
            ]
        )
