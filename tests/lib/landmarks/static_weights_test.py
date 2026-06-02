#!/usr/bin/env python3
"""Tests for static landmark ensemble reliability weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.core.metrics import per_landmark_error
from lib.landmarks.core.schema import LandmarkPrediction
from lib.landmarks.ensemble.weights import (
    LANDMARK_COUNT,
    MODEL_NAMES,
    load_weights,
    normalize_static_weights,
    save_weights,
    weights_from_errors,
)
from lib.landmarks.evaluation.harness import load_manifest


def _points(offset: float = 0.0) -> np.ndarray:
    points = np.stack(
        (
            np.linspace(0, 67, LANDMARK_COUNT, dtype="float32"),
            np.linspace(10, 77, LANDMARK_COUNT, dtype="float32"),
        ),
        axis=1,
    )
    return points + offset  # type: ignore[no-any-return]


def _write_manifest(tmp_path: Path, sample_ids: tuple[str, ...]) -> Path:
    landmark_dir = tmp_path / "truth"
    landmark_dir.mkdir()
    samples = []
    for sample_id in sample_ids:
        path = landmark_dir / f"{sample_id}.npy"
        np.save(str(path), _points())
        samples.append(
            {
                "sample_id": sample_id,
                "image": f"{sample_id}.png",
                "landmarks": str(path.relative_to(tmp_path)),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}) + "\n", encoding="utf-8")
    return manifest


def _compute_weights_from_cache(
    manifest_path: str | Path,
    cache_dir: str | Path,
    models: tuple[str, ...] = MODEL_NAMES,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    cache = DiskPredictionCache(cache_dir)
    samples = load_manifest(manifest_path)
    errors: dict[str, list[np.ndarray]] = {model: [] for model in models}
    for sample in samples:
        truth = np.load(sample.landmarks).astype("float32")
        for model in models:
            prediction = cache.read(sample.sample_id, model)
            errors[model].append(per_landmark_error(prediction.landmarks, truth))
    mean_errors = {
        model: np.stack(model_errors, axis=0).mean(axis=0).astype("float32").tolist()
        for model, model_errors in errors.items()
    }
    return weights_from_errors(mean_errors), mean_errors


def test_weights_from_errors_normalizes_each_landmark() -> None:
    """Lower per-landmark validation error receives higher normalized weight."""
    errors = {
        "hrnet": [1.0] * LANDMARK_COUNT,
        "spiga": [2.0] * LANDMARK_COUNT,
        "orformer": [4.0] * LANDMARK_COUNT,
    }

    weights = weights_from_errors(errors)

    assert tuple(weights) == MODEL_NAMES
    assert all(len(values) == LANDMARK_COUNT for values in weights.values())
    assert weights["hrnet"][0] > weights["spiga"][0] > weights["orformer"][0]
    for landmark_index in range(LANDMARK_COUNT):
        assert sum(values[landmark_index] for values in weights.values()) == pytest.approx(1.0)


def test_normalize_static_weights_rejects_non_68_landmarks() -> None:
    """Static weights must cover exactly the canonical 68 landmarks."""
    with pytest.raises(ValueError, match="shape"):
        normalize_static_weights({"hrnet": [1.0] * (LANDMARK_COUNT - 1)})


def test_save_and_load_static_weights_round_trip(tmp_path: Path) -> None:
    """Serialization preserves schema, required model keys, and normalized sums."""
    path = tmp_path / "static_landmark_weights.json"
    weights = weights_from_errors(
        {
            "hrnet": np.linspace(1.0, 2.0, LANDMARK_COUNT).tolist(),
            "spiga": np.linspace(2.0, 1.0, LANDMARK_COUNT).tolist(),
            "orformer": [3.0] * LANDMARK_COUNT,
        }
    )

    save_weights(path, weights)
    payload = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_weights(path)

    assert payload["schema"] == "2d_68"
    assert tuple(payload["weights"]) == MODEL_NAMES
    assert tuple(loaded) == MODEL_NAMES
    for landmark_index in range(LANDMARK_COUNT):
        assert sum(loaded[model][landmark_index] for model in MODEL_NAMES) == pytest.approx(1.0)


def test_static_weights_from_cache_and_ground_truth(tmp_path: Path) -> None:
    """Validation errors from cached predictions drive per-model landmark weights."""
    manifest = _write_manifest(tmp_path, ("a", "b"))
    cache = DiskPredictionCache(tmp_path / "cache")
    for sample_id in ("a", "b"):
        cache.write(sample_id, LandmarkPrediction(_points(1.0), model_name="hrnet"))
        cache.write(sample_id, LandmarkPrediction(_points(2.0), model_name="spiga"))
        cache.write(sample_id, LandmarkPrediction(_points(4.0), model_name="orformer"))

    weights, mean_errors = _compute_weights_from_cache(manifest, tmp_path / "cache")

    assert tuple(weights) == MODEL_NAMES
    assert all(len(mean_errors[model]) == LANDMARK_COUNT for model in MODEL_NAMES)
    assert mean_errors["hrnet"][0] < mean_errors["spiga"][0] < mean_errors["orformer"][0]
    assert weights["hrnet"][0] > weights["spiga"][0] > weights["orformer"][0]
    for landmark_index in range(LANDMARK_COUNT):
        assert sum(weights[model][landmark_index] for model in MODEL_NAMES) == pytest.approx(1.0)


def test_compute_static_weights_skips_non_canonical_68_gt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A mixed manifest with 39-point profile GT must not crash static-weight fitting."""
    import logging

    from lib.landmarks.ensemble.static_weight_fit import compute_static_weights

    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    np.save(str(truth_dir / "good.npy"), _points())
    np.save(str(truth_dir / "profile.npy"), np.zeros((39, 2), dtype="float32"))
    samples = [
        {
            "sample_id": "good",
            "image": "good.png",
            "landmarks": "truth/good.npy",
            "source_schema": "2d_68",
        },
        {
            "sample_id": "profile",
            "image": "profile.png",
            "landmarks": "truth/profile.npy",
            "source_schema": "2d_39",
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}) + "\n", encoding="utf-8")

    cache = DiskPredictionCache(tmp_path / "cache")
    for sample_id in ("good", "profile"):
        for model in MODEL_NAMES:
            cache.write(sample_id, LandmarkPrediction(_points(1.0), model_name=model))

    with caplog.at_level(logging.WARNING, logger="lib.landmarks.datasets.manifest_io"):
        weights, mean_errors = compute_static_weights(manifest, tmp_path / "cache")

    # Only the canonical-68 sample contributed; the 39-point profile sample was skipped.
    assert tuple(weights) == MODEL_NAMES
    assert all(len(mean_errors[model]) == LANDMARK_COUNT for model in MODEL_NAMES)
    assert "non-canonical-68" in caplog.text


def test_compute_bucket_weights_falls_back_when_below_min_samples(tmp_path: Path) -> None:
    """Buckets below ``min_samples`` are omitted so the runtime uses global weights."""
    from lib.landmarks.ensemble.static_weight_fit import compute_bucket_weights

    manifest = _write_manifest(tmp_path, ("a", "b"))
    cache = DiskPredictionCache(tmp_path / "cache")
    for sample_id in ("a", "b"):
        cache.write(sample_id, LandmarkPrediction(_points(1.0), model_name="hrnet"))
        cache.write(sample_id, LandmarkPrediction(_points(2.0), model_name="spiga"))
        cache.write(sample_id, LandmarkPrediction(_points(4.0), model_name="orformer"))

    global_weights, mean_errors, bucket_weights = compute_bucket_weights(
        manifest, tmp_path / "cache", min_samples=1000
    )

    assert tuple(global_weights) == MODEL_NAMES
    assert all(len(mean_errors[model]) == LANDMARK_COUNT for model in MODEL_NAMES)
    assert bucket_weights == {}


def test_compute_bucket_weights_emits_normalized_bucket_sets(tmp_path: Path) -> None:
    """With a low threshold every assigned bucket gets normalized per-model weights."""
    from lib.landmarks.ensemble.hard_condition_taxonomy import WEIGHT_BUCKETS
    from lib.landmarks.ensemble.static_weight_fit import compute_bucket_weights

    manifest = _write_manifest(tmp_path, ("a", "b"))
    cache = DiskPredictionCache(tmp_path / "cache")
    for sample_id in ("a", "b"):
        cache.write(sample_id, LandmarkPrediction(_points(1.0), model_name="hrnet"))
        cache.write(sample_id, LandmarkPrediction(_points(2.0), model_name="spiga"))
        cache.write(sample_id, LandmarkPrediction(_points(4.0), model_name="orformer"))

    _, _, bucket_weights = compute_bucket_weights(manifest, tmp_path / "cache", min_samples=1)

    assert bucket_weights, "expected at least one fitted bucket"
    for bucket, columns in bucket_weights.items():
        assert bucket in WEIGHT_BUCKETS
        assert tuple(columns) == MODEL_NAMES
        for landmark_index in range(LANDMARK_COUNT):
            total = sum(columns[model][landmark_index] for model in MODEL_NAMES)
            assert total == pytest.approx(1.0)
