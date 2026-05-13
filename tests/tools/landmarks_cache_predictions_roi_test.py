#!/usr/bin/env python3
"""Tests for landmark prediction ROI selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.eval.harness import LandmarkSample
from lib.landmarks.schema import normalize_landmarks
from tools.landmarks import cache_predictions


def _points_98() -> np.ndarray:
    """Return deterministic WFLW-style 98-point landmarks."""
    return np.stack(
        (
            np.linspace(10, 206, 98, dtype="float32"),
            np.linspace(30, 226, 98, dtype="float32"),
        ),
        axis=1,
    )


def _sample(tmp_path: Path, *, dataset: str = "wflw") -> LandmarkSample:
    points = _points_98()
    path = tmp_path / "truth.npy"
    np.save(str(path), points)
    return LandmarkSample(
        sample_id="sample",
        image=str(tmp_path / "image.jpg"),
        landmarks=str(path),
        dataset=dataset,
        condition="default",
    )


def test_wflw_98_ignores_annotation_bbox_for_validation_roi(tmp_path: Path) -> None:
    """WFLW annotation bbox should not be treated as the model crop ROI."""
    sample = _sample(tmp_path)
    entry = {
        "dataset": "wflw",
        "source_schema": "2d_98",
        "metadata": {"bbox": [1000, 1000, 2000, 2000]},
    }

    roi, source = cache_predictions._base_roi_for_sample(  # pylint:disable=protected-access
        sample,
        entry,
        allow_gt_roi=True,
        gt_roi_scale=1.0,
    )

    canonical = normalize_landmarks(_points_98(), source_schema="2d_98")
    expected = np.asarray(
        [
            np.min(canonical[:, 0]),
            np.min(canonical[:, 1]),
            np.max(canonical[:, 0]),
            np.max(canonical[:, 1]),
        ],
        dtype="float32",
    )
    assert source == "gt_landmarks_wflw_98"
    np.testing.assert_allclose(roi, expected)


def test_wflw_98_explicit_face_bbox_wins(tmp_path: Path) -> None:
    """An explicit face_bbox remains authoritative for WFLW validation."""
    sample = _sample(tmp_path)
    entry = {
        "dataset": "wflw",
        "source_schema": "2d_98",
        "metadata": {
            "bbox": [1000, 1000, 2000, 2000],
            "face_bbox": [1, 2, 3, 4],
        },
    }

    roi, source = cache_predictions._base_roi_for_sample(  # pylint:disable=protected-access
        sample,
        entry,
        allow_gt_roi=True,
        gt_roi_scale=1.0,
    )

    assert source == "manifest_face_bbox"
    np.testing.assert_allclose(roi, np.asarray([1, 2, 3, 4], dtype="float32"))


def test_non_wflw_keeps_manifest_bbox(tmp_path: Path) -> None:
    """Generic manifest bbox handling is unchanged for non-WFLW datasets."""
    sample = _sample(tmp_path, dataset="cofw")
    entry = {"dataset": "cofw", "source_schema": "2d_68", "metadata": {"bbox": [5, 6, 7, 8]}}

    roi, source = cache_predictions._base_roi_for_sample(  # pylint:disable=protected-access
        sample,
        entry,
        allow_gt_roi=True,
        gt_roi_scale=1.0,
    )

    assert source == "manifest_bbox"
    np.testing.assert_allclose(roi, np.asarray([5, 6, 7, 8], dtype="float32"))


def test_wflw_98_without_gt_roi_errors_on_annotation_bbox(tmp_path: Path) -> None:
    """--no-gt-roi should reject WFLW annotation bbox as a crop ROI."""
    sample = _sample(tmp_path)
    entry = {
        "dataset": "wflw",
        "source_schema": "2d_98",
        "metadata": {"bbox": [1000, 1000, 2000, 2000]},
    }

    with pytest.raises(ValueError, match="WFLW 98-point"):
        cache_predictions._base_roi_for_sample(  # pylint:disable=protected-access
            sample,
            entry,
            allow_gt_roi=False,
            gt_roi_scale=1.0,
        )
