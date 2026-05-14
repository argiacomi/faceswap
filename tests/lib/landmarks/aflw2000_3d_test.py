#!/usr/bin/env python3
"""Tests for native AFLW2000-3D manifest building."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.aflw2000_3d import build_aflw2000_3d_manifest


def _points_68_2d() -> np.ndarray:
    """Return deterministic 68-point landmarks as 2x68."""
    return np.stack(
        (
            np.linspace(10, 77, 68, dtype="float32"),
            np.linspace(20, 87, 68, dtype="float32"),
        ),
        axis=0,
    )


def _points_68_3d() -> np.ndarray:
    """Return deterministic 68-point landmarks as 3x68."""
    xy = _points_68_2d()
    z = np.linspace(1, 68, 68, dtype="float32")[None, :]
    return np.concatenate((xy, z), axis=0)


def _write_png(path: Path) -> None:
    """Write a tiny valid PNG image."""
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype="uint8")
    image[..., 0] = 255
    assert cv2.imwrite(str(path), image)


def test_build_aflw2000_3d_manifest_prefers_pt3d_68_over_sparse_pt2d(tmp_path: Path) -> None:
    """AFLW2000-3D builder uses pt3d_68 XY, not sparse 21-point pt2d."""
    scipy_io = pytest.importorskip("scipy.io")
    root = tmp_path / "aflw2000"
    root.mkdir()
    _write_png(root / "face.jpg")
    sparse_pt2d = np.stack(
        (
            np.linspace(1, 21, 21, dtype="float32"),
            np.linspace(2, 22, 21, dtype="float32"),
        ),
        axis=0,
    )
    scipy_io.savemat(
        str(root / "face.mat"),
        {
            "pt2d": sparse_pt2d,
            "pt3d_68": _points_68_3d(),
            "Pose_Para": np.asarray([[1, 2, 3]], dtype="float32"),
        },
    )

    manifest = build_aflw2000_3d_manifest(root / "out", source_dir=root)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    audit = json.loads((root / "out" / "dataset_audit.json").read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    landmarks = np.load(root / "out" / sample["landmarks"])

    assert payload["dataset"] == "aflw2000-3d"
    assert sample["source_schema"] == "2d_68"
    assert sample["metadata"]["landmark_source_key"] == "pt3d_68"
    assert sample["metadata"]["pt2d_shape"] == [2, 21]
    assert sample["metadata"]["pt3d_68_shape"] == [3, 68]
    assert sample["metadata"]["face_bbox_source"] == "aflw2000_3d_pt3d_68_xy_extrema"
    assert sample["metadata"]["Pose_Para"] == [1.0, 2.0, 3.0]
    assert landmarks.shape == (68, 2)
    np.testing.assert_allclose(landmarks, _points_68_2d().T)
    assert audit["count_per_source_schema"] == {"2d_68": 1}
    assert audit["landmark_shape_counts"] == {"68x2": 1}


def test_build_aflw2000_3d_manifest_falls_back_to_68_point_pt2d(tmp_path: Path) -> None:
    """68-point pt2d remains supported when pt3d_68 is absent."""
    scipy_io = pytest.importorskip("scipy.io")
    root = tmp_path / "aflw2000"
    root.mkdir()
    _write_png(root / "face.jpg")
    scipy_io.savemat(str(root / "face.mat"), {"pt2d": _points_68_2d()})

    manifest = build_aflw2000_3d_manifest(root / "out", source_dir=root)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    landmarks = np.load(root / "out" / sample["landmarks"])

    assert sample["metadata"]["landmark_source_key"] == "pt2d"
    assert sample["metadata"]["face_bbox_source"] == "aflw2000_3d_pt2d_xy_extrema"
    assert landmarks.shape == (68, 2)
