#!/usr/bin/env python3
"""Tests for native MERL-RAV organized manifest building."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.merl_rav import build_merl_rav_manifest


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
    image[..., 1] = 255
    assert cv2.imwrite(str(path), image)


def _write_pts(path: Path, points: np.ndarray) -> None:
    """Write a MERL-RAV/300W-style .pts file."""
    rows = ["version: 1", "n_points: 68", "{"]
    rows.extend(f"{float(x):.4f} {float(y):.4f}" for x, y in points)
    rows.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_build_merl_rav_manifest_from_organized_pts_and_images(tmp_path: Path) -> None:
    """MERL-RAV builder consumes organized AFLW image plus .pts pairs."""
    root = tmp_path / "merl_rav_organized"
    points = _points_68()
    points[5] *= -1  # external occlusion, estimated location should become abs(x/y)
    sample_dir = root / "left" / "testset"
    _write_png(sample_dir / "face.jpg")
    _write_pts(sample_dir / "face.pts", points)

    manifest = build_merl_rav_manifest(root / "out", source_dir=root)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    landmarks = np.load(root / "out" / sample["landmarks"])

    assert payload["dataset"] == "merl-rav"
    assert sample["source_schema"] == "2d_68"
    assert sample["conditions"] == ["left", "testset", "occlusion"]
    assert sample["metadata"]["externally_occluded_count"] == 1
    assert sample["metadata"]["self_occluded_count"] == 0
    assert sample["metadata"]["face_bbox_source"] == "merl_rav_68_landmark_extrema"
    assert sample["metadata"]["aflw_images_required"] is True
    assert landmarks.shape == (68, 2)
    assert landmarks[5, 0] > 0
    assert landmarks[5, 1] > 0


def test_merl_rav_skips_self_occluded_unestimated_landmarks(tmp_path: Path) -> None:
    """Self-occluded -1/-1 samples are skipped until harness supports GT masks."""
    root = tmp_path / "merl_rav_organized"
    points = _points_68()
    points[0] = [-1, -1]
    sample_dir = root / "frontal" / "testset"
    _write_png(sample_dir / "face.jpg")
    _write_pts(sample_dir / "face.pts", points)

    with pytest.raises(FileNotFoundError, match="No usable MERL-RAV"):
        build_merl_rav_manifest(root / "out", source_dir=root)
