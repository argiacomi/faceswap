#!/usr/bin/env python3
"""Tests for the parallel 39-point profile row loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.profile39_dataset import (
    is_profile39_sample,
    load_profile39_rows,
    profile39_mix_report,
    profile39_samples,
)
from lib.landmarks.ensemble.weights import save_weights


def _face68(offset: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    face: np.ndarray = (rng.uniform(0.0, 100.0, size=(68, 2)) + offset).astype("float32")
    return face


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    cache_dir = tmp_path / "cache"
    weights_path = tmp_path / "weights.json"
    save_weights(
        weights_path,
        {"hrnet": [1.0] * 68, "spiga": [0.0] * 68, "orformer": [0.0] * 68},
    )
    cache = DiskPredictionCache(cache_dir)
    samples = []
    # One 39-point profile sample (left) + one canonical-68 sample (must be skipped).
    truth39: np.ndarray = _face68(seed=5)[:39].astype("float32")  # (39, 2)
    np.save(str(tmp_path / "p39.npy"), truth39)
    np.save(str(tmp_path / "c68.npy"), _face68(seed=9))
    for model, offset in {"hrnet": 1.0, "spiga": 16.0, "orformer": 6.0}.items():
        cache.write(
            "p39", LandmarkPrediction(_face68(offset, seed=1), model_name=model), checkpoint="t"
        )
        cache.write(
            "c68", LandmarkPrediction(_face68(offset, seed=2), model_name=model), checkpoint="t"
        )
    samples.append(
        {
            "sample_id": "p39",
            "image": "p39.jpg",
            "landmarks": "p39.npy",
            "dataset": "multipie",
            "condition": "profile_left",
            "conditions": ["profile_left"],
            "source_schema": "2d_39",
            "normalizer": 100.0,
            "face_bbox": [0.0, 20.0, 100.0, 110.0],
        }
    )
    samples.append(
        {
            "sample_id": "c68",
            "image": "c68.jpg",
            "landmarks": "c68.npy",
            "dataset": "wflw",
            "condition": "frontal",
            "normalizer": 100.0,
        }
    )
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return manifest_path, cache_dir, weights_path


def test_only_39pt_samples_are_selected(tmp_path: Path) -> None:
    manifest_path, _cache, _weights = _write_fixture(tmp_path)
    selected = profile39_samples(manifest_path)
    assert [s.sample_id for s in selected] == ["p39"]
    assert all(is_profile39_sample(s) for s in selected)


def test_load_profile39_rows_produces_scored_rows(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    rows = load_profile39_rows(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "orformer"),
    )
    assert rows, "expected at least one profile39 row"
    assert all(row.sample_id == "p39" for row in rows)
    assert all(row.side == "left" for row in rows)
    # exactly one oracle, all regrets non-negative, oracle regret == 0
    oracle_rows = [row for row in rows if row.profile39_is_oracle]
    assert len(oracle_rows) == 1
    assert oracle_rows[0].profile39_transform_regret == 0.0
    assert all(row.profile39_transform_regret >= 0.0 for row in rows)
    # rows carry runtime + profile features (profile route -> profile features present)
    assert any("profile_is_left" in row.feature_values for row in rows)
    assert all(row.profile39_visible_side_error >= 0.0 for row in rows)


def test_profile39_mix_report(tmp_path: Path) -> None:
    manifest_path, cache_dir, weights_path = _write_fixture(tmp_path)
    rows = load_profile39_rows(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet", "spiga", "orformer"),
    )
    report = profile39_mix_report(rows)
    assert report["sample_count"] == 1
    assert report["side_counts"]["left"] >= 1
    assert report["oracle_row_count"] == 1
    assert report["mean_transform_regret"] >= 0.0


def test_unresolved_side_is_skipped(tmp_path: Path) -> None:
    # A 39-point sample with no side-resolving labels must be skipped.
    manifest_path = tmp_path / "m.json"
    cache_dir = tmp_path / "c"
    weights_path = tmp_path / "w.json"
    save_weights(weights_path, {"hrnet": [1.0] * 68})
    cache = DiskPredictionCache(cache_dir)
    cache.write("x", LandmarkPrediction(_face68(seed=3), model_name="hrnet"), checkpoint="t")
    np.save(str(tmp_path / "x.npy"), _face68(seed=4)[:39].astype("float32"))
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "x",
                        "image": "x.jpg",
                        "landmarks": "x.npy",
                        "dataset": "menpo2d",
                        "condition": "occlusion",
                        "source_schema": "2d_39",
                        "normalizer": 100.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = load_profile39_rows(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=("hrnet",),
    )
    assert rows == []
