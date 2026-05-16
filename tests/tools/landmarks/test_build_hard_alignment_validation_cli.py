#!/usr/bin/env python3
"""CLI integration tests for ``build_hard_alignment_validation`` (#82)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.weights import LANDMARK_COUNT
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction
from tools.landmarks.build_hard_alignment_validation import main as build_main


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


def _yaw_sample(sid: str, yaw_radians: float | None) -> dict:
    payload = {
        "sample_id": sid,
        "dataset": "aflw2000-3d",
        "image": "image.png",
        "landmarks": "truth.npy",
        "metadata": {"face_bbox": [40.0, 60.0, 160.0, 150.0]},
    }
    if yaw_radians is not None:
        payload["metadata"]["Pose_Para"] = [
            0.0,
            yaw_radians,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    return payload


def _build_fixture(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    np.save(str(dataset_dir / "truth.npy"), _truth_face())
    manifest_path = dataset_dir / "manifest.json"
    samples = [
        _yaw_sample("frontal", 0.0),
        _yaw_sample("profile_l", math.radians(-35.0)),
        _yaw_sample("profile_r", math.radians(35.0)),
        _yaw_sample("extreme", math.radians(70.0)),
        _yaw_sample("unposed", None),
    ]
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return manifest_path


def _populate_cache(tmp_path: Path, manifest_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    cache = DiskPredictionCache(cache_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        for noise, model in ((0.0, "hrnet"), (0.5, "spiga")):
            cache.write(
                sample["sample_id"], LandmarkPrediction(_truth_face() + noise, model_name=model)
            )
    return cache_dir


def test_build_hard_manifest_keeps_only_hard_samples(tmp_path: Path) -> None:
    """Default invocation drops frontal / intermediate / no-pose samples."""
    manifest = _build_fixture(tmp_path)
    out_dir = tmp_path / "out"

    rc = build_main(["--manifest", str(manifest), "--output-dir", str(out_dir)])
    assert rc == 0
    payload = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = sorted(item["sample_id"] for item in payload["samples"])
    assert ids == ["extreme", "profile_l", "profile_r"]
    summary = json.loads((out_dir / "hard_slice_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_sample_count"] == 3
    assert summary["source_sample_count"] == 5
    assert summary["bucket_counts"]["frontal"] == 1
    assert summary["bucket_counts"]["no_pose"] == 1


def test_build_hard_manifest_tags_each_sample_with_bucket(tmp_path: Path) -> None:
    """The filtered manifest carries ``hard_slice`` and ``condition`` fields."""
    manifest = _build_fixture(tmp_path)
    out_dir = tmp_path / "out"
    assert build_main(["--manifest", str(manifest), "--output-dir", str(out_dir)]) == 0
    payload = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    by_id = {item["sample_id"]: item for item in payload["samples"]}
    assert by_id["profile_r"]["hard_slice"] == "profile_right"
    assert by_id["profile_r"]["condition"] == "profile_right"
    assert "yaw_degrees" in by_id["profile_r"]


def test_build_hard_manifest_includes_unposed_when_requested(tmp_path: Path) -> None:
    """``--include-unposed`` + ``--no-hard-only`` keeps every sample tagged."""
    manifest = _build_fixture(tmp_path)
    out_dir = tmp_path / "out"
    assert (
        build_main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(out_dir),
                "--include-unposed",
                "--no-hard-only",
            ]
        )
        == 0
    )
    payload = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = sorted(item["sample_id"] for item in payload["samples"])
    assert ids == ["extreme", "frontal", "profile_l", "profile_r", "unposed"]


def test_build_hard_manifest_runs_geometry_eval_when_cache_supplied(tmp_path: Path) -> None:
    """Passing ``--cache-dir`` triggers the geometry-metrics CLI on the hard set."""
    manifest = _build_fixture(tmp_path)
    cache = _populate_cache(tmp_path, manifest)
    out_dir = tmp_path / "out"
    assert (
        build_main(
            [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(out_dir),
                "--cache-dir",
                str(cache),
                "--models",
                "hrnet,spiga",
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
        assert (out_dir / name).is_file(), f"missing {name}"
    payload = json.loads((out_dir / "geometry_metrics.json").read_text(encoding="utf-8"))
    # Geometry rows only cover the three hard-case samples kept in the manifest.
    assert payload["aggregates"]["hrnet"]["sample_count"] == 3
