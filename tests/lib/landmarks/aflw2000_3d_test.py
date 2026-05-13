#!/usr/bin/env python3
"""Tests for native AFLW2000-3D manifest building."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.aflw2000_3d import build_aflw2000_3d_manifest


def _points_68() -> np.ndarray:
    """Return deterministic 68-point landmarks as 2x68 pt2d."""
    return np.stack(
        (
            np.linspace(10, 77, 68, dtype="float32"),
            np.linspace(20, 87, 68, dtype="float32"),
        ),
        axis=0,
    )


def _write_png(path: Path) -> None:
    """Write a tiny valid PNG image."""
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype="uint8")
    image[..., 0] = 255
    assert cv2.imwrite(str(path), image)


def test_build_aflw2000_3d_manifest_from_native_mat_and_image(tmp_path: Path) -> None:
    """AFLW2000-3D builder consumes paired image/.mat files with pt2d."""
    scipy_io = pytest.importorskip("scipy.io")
    root = tmp_path / "aflw2000"
    root.mkdir()
    _write_png(root / "face.jpg")
    scipy_io.savemat(
        str(root / "face.mat"),
        {
            "pt2d": _points_68(),
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
    assert sample["metadata"]["face_bbox_source"] == "aflw2000_3d_pt2d_extrema"
    assert sample["metadata"]["Pose_Para"] == [1.0, 2.0, 3.0]
    assert landmarks.shape == (68, 2)
    assert audit["count_per_source_schema"] == {"2d_68": 1}
    assert audit["landmark_shape_counts"] == {"68x2": 1}
