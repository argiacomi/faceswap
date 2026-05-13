#!/usr/bin/env python3
"""Tests for static landmark ensemble reliability weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.ensemble.weights import (
    LANDMARK_COUNT,
    MODEL_NAMES,
    load_weights,
    normalize_static_weights,
    save_weights,
    weights_from_errors,
)
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction
from tools.landmarks.compute_static_weights import DEFAULT_OUTPUT, compute_static_weights, main


def _points(offset: float = 0.0) -> np.ndarray:
    points = np.stack(
        (
            np.linspace(0, 67, LANDMARK_COUNT, dtype="float32"),
            np.linspace(10, 77, LANDMARK_COUNT, dtype="float32"),
        ),
        axis=1,
    )
    return points + offset


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


def test_compute_static_weights_reads_cache_and_ground_truth(tmp_path: Path) -> None:
    """Validation errors from cached predictions drive per-model landmark weights."""
    manifest = _write_manifest(tmp_path, ("a", "b"))
    cache = DiskPredictionCache(tmp_path / "cache")
    for sample_id in ("a", "b"):
        cache.write(sample_id, LandmarkPrediction(_points(1.0), model_name="hrnet"))
        cache.write(sample_id, LandmarkPrediction(_points(2.0), model_name="spiga"))
        cache.write(sample_id, LandmarkPrediction(_points(4.0), model_name="orformer"))

    weights, mean_errors = compute_static_weights(manifest, tmp_path / "cache")

    assert tuple(weights) == MODEL_NAMES
    assert all(len(mean_errors[model]) == LANDMARK_COUNT for model in MODEL_NAMES)
    assert mean_errors["hrnet"][0] < mean_errors["spiga"][0] < mean_errors["orformer"][0]
    assert weights["hrnet"][0] > weights["spiga"][0] > weights["orformer"][0]
    for landmark_index in range(LANDMARK_COUNT):
        assert sum(weights[model][landmark_index] for model in MODEL_NAMES) == pytest.approx(1.0)


def test_compute_static_weights_cli_default_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI defaults to the ensemble static-weight config path."""
    manifest = _write_manifest(tmp_path, ("sample",))
    cache = DiskPredictionCache(tmp_path / "cache")
    for model, offset in zip(MODEL_NAMES, (1.0, 2.0, 4.0), strict=True):
        cache.write("sample", LandmarkPrediction(_points(offset), model_name=model))
    monkeypatch.chdir(tmp_path)

    result = main(["--manifest", str(manifest), "--cache-dir", str(tmp_path / "cache")])

    assert result == 0
    output = tmp_path / DEFAULT_OUTPUT
    assert output.is_file()
    weights = load_weights(output)
    assert tuple(weights) == MODEL_NAMES
