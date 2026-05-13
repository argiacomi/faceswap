#!/usr/bin/env python3
"""End-to-end landmark ensemble smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from lib.landmarks.datasets import build_manifest
from lib.landmarks.eval.harness import run_quality_harness
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction


def _points(offset: float = 0.0) -> np.ndarray:
    points = np.stack(
        (
            np.linspace(0, 67, 68, dtype="float32"),
            np.linspace(20, 87, 68, dtype="float32"),
        ),
        axis=1,
    )
    return points + offset


def test_cached_three_model_pipeline_runs_all_ensemble_variants(tmp_path: Path) -> None:
    """A tiny manifest and cached HRNet/SPIGA/ORFormer predictions run end-to-end."""
    source = tmp_path / "source"
    source.mkdir()
    cv2.imwrite(str(source / "sample.png"), np.zeros((16, 16, 3), dtype="uint8"))
    np.save(str(source / "sample.npy"), _points())
    manifest = build_manifest(source, tmp_path / "dataset", dataset="wflw")
    cache = DiskPredictionCache(tmp_path / "cache")
    for model_name, offset in (("hrnet", 0.0), ("spiga", 0.5), ("orformer", 50.0)):
        cache.write(
            "sample",
            LandmarkPrediction(
                _points(offset),
                model_name=model_name,
                coordinate_space="frame",
            ),
            checkpoint=f"{model_name}-fixture",
            config={"model": model_name, "fixture": True},
        )

    result = run_quality_harness(
        manifest,
        tmp_path / "cache",
        models=("hrnet", "spiga", "orformer"),
        variants=(
            "plain_average",
            "static_weighted",
            "static_weighted_outliers",
            "static_weighted_downweight",
            "weighted_median",
        ),
        output_dir=tmp_path / "metrics",
        failure_threshold=1.0,
        outlier_threshold=2.0,
    )

    metrics_json = tmp_path / "metrics" / "metrics.json"
    metrics_csv = tmp_path / "metrics" / "metrics.csv"
    assert metrics_json.is_file()
    assert metrics_csv.is_file()
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert payload == result
    for label in (
        "hrnet",
        "spiga",
        "orformer",
        "plain_average",
        "static_weighted",
        "static_weighted_outliers",
        "static_weighted_downweight",
        "weighted_median",
    ):
        assert label in result["overall"]
    assert "sample" in metrics_csv.read_text(encoding="utf-8")
