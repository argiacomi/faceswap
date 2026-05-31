#!/usr/bin/env python3
"""Tests for cache_predictions model-running mode."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lib.landmarks.adapters import LandmarkAdapterConfig, StaticLandmarkAdapter
from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from tools.landmarks import cache_predictions


def _truth_points(offset: float = 0.0) -> np.ndarray:
    """Return deterministic frame-space canonical landmarks."""
    return (  # type: ignore[no-any-return]
        np.stack(
            (
                np.linspace(20, 80, 68, dtype="float32"),
                np.linspace(25, 85, 68, dtype="float32"),
            ),
            axis=1,
        )
        + offset
    )


def _points_98_normalized() -> np.ndarray:
    """Return a deterministic normalized 98-point prediction."""
    return np.stack(  # type: ignore[no-any-return]
        (
            np.linspace(0.1, 0.9, 98, dtype="float32"),
            np.linspace(0.2, 0.8, 98, dtype="float32"),
        ),
        axis=1,
    )


def _write_manifest(tmp_path: Path, *, include_bbox: bool = True) -> Path:
    """Write a minimal image+landmark manifest."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image = np.zeros((128, 128, 3), dtype="uint8")  # type: ignore[var-annotated]
    assert cv2.imwrite(str(dataset / "sample.png"), image)
    np.save(str(dataset / "truth.npy"), _truth_points())
    sample = {
        "sample_id": "sample",
        "image": "sample.png",
        "landmarks": "truth.npy",
        "dataset": "fixture",
        "condition": "clean",
    }
    if include_bbox:
        sample["face_bbox"] = [20, 25, 80, 85]  # type: ignore[assignment]
    manifest = dataset / "manifest.json"
    manifest.write_text(json.dumps({"samples": [sample]}), encoding="utf-8")
    return manifest


class _CountingStaticAdapter(StaticLandmarkAdapter):
    """Static adapter that records batch invocations."""

    def __init__(self, config: LandmarkAdapterConfig, points: np.ndarray) -> None:
        super().__init__(config, points)
        self.calls = 0

    def predict_batch(self, images, *, matrices=None, faces=None):
        self.calls += 1
        return super().predict_batch(images, matrices=matrices, faces=faces)


class _RecordingPlugin:
    """Minimal plugin double that records the ROI passed to pre_process."""

    input_size = 256

    def __init__(self) -> None:
        self.seen: np.ndarray | None = None

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Record raw ROI input and return the plugin-owned crop ROI."""
        self.seen = batch.copy()
        return batch.astype("float32", copy=False)


def test_run_model_predictions_writes_cache_with_complete_metadata(tmp_path: Path) -> None:
    """A fake model adapter can generate frame-space cached predictions from a manifest."""
    manifest = _write_manifest(tmp_path)
    adapter = _CountingStaticAdapter(
        LandmarkAdapterConfig("hrnet", coordinate_space="normalized_crop"),
        np.full((68, 2), 0.5, dtype="float32"),
    )

    written, reused = cache_predictions._run_model_predictions(
        manifest_path=manifest,
        models=("hrnet",),
        cache_dir=tmp_path / "cache",
        checkpoint="validation-v1",
        batch_size=8,
        device="cpu",
        refresh=False,
        allow_gt_roi=True,
        adapters={"hrnet": adapter},
    )

    cache = DiskPredictionCache(tmp_path / "cache")
    metadata = cache.load_metadata("sample")["hrnet"]
    cached = cache.read("sample", "hrnet")
    assert written == 1
    assert reused == 0
    assert adapter.calls == 1
    assert cached.landmarks.shape == (68, 2)
    assert metadata["model_name"] == "hrnet"
    assert metadata["checkpoint"] == "validation-v1"
    assert metadata["schema"] == "2d_68"
    assert metadata["coordinate_space"] == "frame"
    assert metadata["source_landmark_count"] == 68
    assert len(metadata["config_hash"]) == 64
    assert len(metadata["prediction_hash"]) == 64


def test_plugin_preprocess_receives_raw_manifest_bbox(tmp_path: Path) -> None:
    """Faceswap plugins own crop expansion; the runner must not pre-expand ROI boxes."""
    manifest = _write_manifest(tmp_path)
    adapter = _CountingStaticAdapter(
        LandmarkAdapterConfig("hrnet", coordinate_space="normalized_crop"),
        np.full((68, 2), 0.5, dtype="float32"),
    )
    plugin = _RecordingPlugin()
    adapter.plugin = plugin  # type:ignore[attr-defined]

    cache_predictions._run_model_predictions(
        manifest_path=manifest,
        models=("hrnet",),
        cache_dir=tmp_path / "cache",
        checkpoint="validation-v1",
        batch_size=1,
        device="cpu",
        refresh=False,
        allow_gt_roi=True,
        adapters={"hrnet": adapter},
    )

    assert plugin.seen is not None
    np.testing.assert_allclose(plugin.seen[0], np.array([20, 25, 80, 85], dtype="float32"))


def test_run_model_predictions_reuses_fresh_cache_and_refresh_forces_regeneration(
    tmp_path: Path,
) -> None:
    """Fresh metadata skips model execution unless refresh is requested."""
    manifest = _write_manifest(tmp_path)
    adapter = _CountingStaticAdapter(
        LandmarkAdapterConfig("hrnet", coordinate_space="normalized_crop"),
        np.full((68, 2), 0.25, dtype="float32"),
    )
    kwargs = dict(
        manifest_path=manifest,
        models=("hrnet",),
        cache_dir=tmp_path / "cache",
        checkpoint="validation-v1",
        device="cpu",
        allow_gt_roi=True,
        adapters={"hrnet": adapter},
    )

    assert cache_predictions._run_model_predictions(batch_size=1, refresh=False, **kwargs) == (  # type: ignore[arg-type]
        1,
        0,
    )
    assert cache_predictions._run_model_predictions(batch_size=8, refresh=False, **kwargs) == (  # type: ignore[arg-type]
        0,
        1,
    )
    assert adapter.calls == 1
    assert cache_predictions._run_model_predictions(batch_size=1, refresh=True, **kwargs) == (1, 0)  # type: ignore[arg-type]
    assert adapter.calls == 2


def test_run_model_predictions_uses_gt_roi_when_bbox_missing(tmp_path: Path) -> None:
    """GT-derived ROIs are accepted for validation-only prediction generation."""
    manifest = _write_manifest(tmp_path, include_bbox=False)
    adapter = _CountingStaticAdapter(
        LandmarkAdapterConfig("hrnet", coordinate_space="normalized_crop"),
        np.full((68, 2), 0.5, dtype="float32"),
    )

    written, reused = cache_predictions._run_model_predictions(
        manifest_path=manifest,
        models=("hrnet",),
        cache_dir=tmp_path / "cache",
        checkpoint="validation-v1",
        batch_size=1,
        device="cpu",
        refresh=False,
        allow_gt_roi=True,
        adapters={"hrnet": adapter},
    )

    assert (written, reused) == (1, 0)
    assert DiskPredictionCache(tmp_path / "cache").prediction_path("sample", "hrnet").is_file()


def test_run_model_predictions_fails_when_bbox_missing_and_gt_roi_disabled(tmp_path: Path) -> None:
    """The runner fails clearly when neither explicit nor GT-derived ROI is available."""
    manifest = _write_manifest(tmp_path, include_bbox=False)
    adapter = _CountingStaticAdapter(
        LandmarkAdapterConfig("hrnet", coordinate_space="normalized_crop"),
        np.full((68, 2), 0.5, dtype="float32"),
    )

    with pytest.raises(ValueError, match="missing face_bbox"):
        cache_predictions._run_model_predictions(
            manifest_path=manifest,
            models=("hrnet",),
            cache_dir=tmp_path / "cache",
            checkpoint="validation-v1",
            batch_size=1,
            device="cpu",
            refresh=False,
            allow_gt_roi=False,
            adapters={"hrnet": adapter},
        )


def test_run_model_predictions_normalizes_98_point_outputs(tmp_path: Path) -> None:
    """98-point model outputs are stored as canonical frame-space 68-point predictions."""
    manifest = _write_manifest(tmp_path)
    adapter = _CountingStaticAdapter(
        LandmarkAdapterConfig("spiga", schema="2d_98", coordinate_space="normalized_crop"),
        _points_98_normalized(),
    )

    written, reused = cache_predictions._run_model_predictions(
        manifest_path=manifest,
        models=("spiga",),
        cache_dir=tmp_path / "cache",
        checkpoint="validation-v1",
        batch_size=1,
        device="cpu",
        refresh=False,
        allow_gt_roi=True,
        adapters={"spiga": adapter},
    )

    cached = DiskPredictionCache(tmp_path / "cache").read("sample", "spiga")
    assert (written, reused) == (1, 0)
    assert cached.landmarks.shape == (68, 2)
    assert cached.source_landmark_count == 98


def test_model_runner_dispatch_builds_selected_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model names are dispatched through the shared adapter factory."""
    seen = []

    def fake_build_landmark_adapter(model: str, **kwargs):
        seen.append((model, kwargs))
        return _CountingStaticAdapter(
            LandmarkAdapterConfig(model, coordinate_space="normalized_crop"),
            np.full((68, 2), 0.5, dtype="float32"),
        )

    monkeypatch.setattr(cache_predictions, "build_landmark_adapter", fake_build_landmark_adapter)

    adapters = cache_predictions._build_model_adapters(
        ("hrnet", "spiga", "orformer"), device="cpu"
    )

    assert tuple(adapters) == ("hrnet", "spiga", "orformer")
    assert [name for name, _kwargs in seen] == ["hrnet", "spiga", "orformer"]
    assert all(kwargs["device"] == "cpu" for _name, kwargs in seen)
