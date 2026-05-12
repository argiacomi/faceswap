#!/usr/bin/env python3
"""Tests for disk cache and cache-driven landmark harness."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.landmarks.datasets import build_cofw_manifest, build_manifest, build_wflw_manifest
from lib.landmarks.ensemble.weights import weights_from_errors
from lib.landmarks.eval.harness import run_quality_harness
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction


def _points(offset: float = 0.0) -> np.ndarray:
    points = np.stack(
        (
            np.linspace(0, 67, 68, dtype="float32"),
            np.linspace(10, 77, 68, dtype="float32"),
        ),
        axis=1,
    )
    return points + offset


def test_disk_prediction_cache_round_trip(tmp_path: Path) -> None:
    """Cache stores prediction arrays and metadata."""
    cache = DiskPredictionCache(tmp_path)
    prediction = LandmarkPrediction(
        landmarks=_points(),
        model_name="hrnet",
        coordinate_space="frame",
    )

    cache.write("sample", prediction, checkpoint="ckpt", config={"version": 1})
    loaded = cache.read("sample", "hrnet")

    assert cache.available_models("sample") == ("hrnet",)
    assert loaded.model_name == "hrnet"
    np.testing.assert_array_equal(loaded.landmarks, prediction.landmarks)


def test_dataset_builder_and_harness(tmp_path: Path) -> None:
    """Prepared WFLW/COFW-style folders can be evaluated from cached predictions."""
    source = tmp_path / "source"
    source.mkdir()
    cv2.imwrite(str(source / "sample.png"), np.zeros((8, 8, 3), dtype="uint8"))
    np.save(str(source / "sample.npy"), _points())
    manifest = build_manifest(source, tmp_path / "dataset", dataset="wflw")
    cache = DiskPredictionCache(tmp_path / "cache")
    cache.write(
        "sample",
        LandmarkPrediction(_points(0.0), model_name="hrnet"),
        config="hrnet",
    )
    cache.write(
        "sample",
        LandmarkPrediction(_points(1.0), model_name="spiga"),
        config="spiga",
    )

    result = run_quality_harness(
        manifest,
        tmp_path / "cache",
        variants=("plain_average", "static_weighted_outliers", "weighted_median"),
        output_dir=tmp_path / "metrics",
        failure_threshold=1.0,
    )

    assert (tmp_path / "metrics" / "metrics.json").is_file()
    assert (tmp_path / "metrics" / "metrics.csv").is_file()
    assert result["overall"]["hrnet"]["nme"] == 0.0
    assert "plain_average" in result["overall"]
    assert "static_weighted_outliers" in result["overall"]
    assert "weighted_median" in result["overall"]
    assert "wflw" in result["datasets"]
    assert result["regions"]


def test_weights_from_errors_prefers_lower_error() -> None:
    """Static weight computation favors lower validation error per landmark."""
    weights = weights_from_errors(
        {"hrnet": [1.0] * 68, "spiga": [2.0] * 68, "orformer": [4.0] * 68}
    )

    assert weights["hrnet"][0] > weights["spiga"][0] > weights["orformer"][0]
    assert sum(weights[model][0] for model in weights) == pytest.approx(1.0)


def test_manifest_audit_written(tmp_path: Path) -> None:
    """Dataset builder writes a dataset audit."""
    source = tmp_path / "source"
    source.mkdir()
    cv2.imwrite(str(source / "a.jpg"), np.zeros((4, 4, 3), dtype="uint8"))
    np.save(str(source / "a.npy"), _points())

    build_manifest(source, tmp_path / "dataset", dataset="cofw", scenario="occluded")
    audit = json.loads((tmp_path / "dataset" / "dataset_audit.json").read_text())

    assert audit["total_entries"] == 1
    assert audit["condition_counts"] == {"occluded": 1}


def test_wflw_annotation_builder_maps_98_to_68(tmp_path: Path) -> None:
    """WFLW text annotations are written as canonical 68-point landmark files."""
    points = np.stack(
        (
            np.linspace(0, 97, 98, dtype="float32"),
            np.linspace(100, 197, 98, dtype="float32"),
        ),
        axis=1,
    )
    annotation = tmp_path / "wflw.txt"
    annotation.write_text(
        " ".join(str(value) for value in points.reshape(-1)) + " images/sample.jpg\n",
        encoding="utf-8",
    )

    manifest_path = build_wflw_manifest(annotation, tmp_path / "wflw")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    landmarks = np.load(tmp_path / "wflw" / payload["samples"][0]["landmarks"])

    assert payload["dataset"] == "wflw"
    assert landmarks.shape == (68, 2)


def test_cofw_json_builder_writes_manifest(tmp_path: Path) -> None:
    """COFW JSON exports are converted into harness-compatible manifests."""
    source = tmp_path / "cofw.json"
    source.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "face",
                        "image": "face.png",
                        "landmarks": _points().tolist(),
                        "conditions": {"scenario": "occluded"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest_path = build_cofw_manifest(source, tmp_path / "cofw")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads((tmp_path / "cofw" / "dataset_audit.json").read_text())

    assert payload["samples"][0]["condition"] == "occluded"
    assert np.load(tmp_path / "cofw" / payload["samples"][0]["landmarks"]).shape == (68, 2)
    assert audit["condition_counts"] == {"occluded": 1}
