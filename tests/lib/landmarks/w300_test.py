#!/usr/bin/env python3
"""Tests for 300W dataset manifest building."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.w300 import build_300w_manifest


def _points_68() -> np.ndarray:
    """Return deterministic 68-point landmarks."""
    return np.stack(
        (
            np.linspace(10, 77, 68, dtype="float32"),
            np.linspace(20, 87, 68, dtype="float32"),
        ),
        axis=1,
    )


def _write_png(path: Path) -> None:
    """Write a tiny valid PNG image."""
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype="uint8")
    image[..., 2] = 255
    assert cv2.imwrite(str(path), image)


def _write_pts(path: Path, points: np.ndarray) -> None:
    """Write a standard Menpo/300W .pts annotation file."""
    rows = ["version: 1", "n_points: 68", "{"]
    rows.extend(f"{float(x):.4f} {float(y):.4f}" for x, y in points)
    rows.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_300w_manifest_from_pts_and_images(tmp_path: Path) -> None:
    """300W builder emits canonical 68-point samples with split labels."""
    root = tmp_path / "300w"
    points = _points_68()
    _write_png(root / "ibug" / "face.jpg")
    _write_pts(root / "ibug" / "face.pts", points)
    _write_png(root / "helen" / "common.jpg")
    _write_pts(root / "helen" / "common.pts", points + 1)

    manifest = build_300w_manifest(root / "out", source_dir=root)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    audit = json.loads((root / "out" / "dataset_audit.json").read_text(encoding="utf-8"))
    samples = payload["samples"]

    assert payload["dataset"] == "300w"
    assert {sample["condition"] for sample in samples} == {"challenging", "common"}
    assert {sample["source_schema"] for sample in samples} == {"2d_68"}
    assert audit["count_per_source_schema"] == {"2d_68": 2}
    assert audit["landmark_shape_counts"] == {"68x2": 2}
    for sample in samples:
        landmarks = np.load(root / "out" / sample["landmarks"])
        assert landmarks.shape == (68, 2)
        assert sample["metadata"]["face_bbox_source"] == "300w_68_landmark_extrema"
