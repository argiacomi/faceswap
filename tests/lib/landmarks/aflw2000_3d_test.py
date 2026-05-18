#!/usr/bin/env python3
"""Tests for native AFLW2000-3D manifest building."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.aflw2000_3d import (
    _pt3d_z_coordinates,
    _visibility_from_depth,
    build_aflw2000_3d_manifest,
)


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


def _frontal_pt3d_68() -> np.ndarray:
    """Return a 3x68 array with near-uniform depth (frontal face)."""
    xy = _points_68_2d()
    z = np.full((1, 68), 50.0, dtype="float32")
    return np.concatenate((xy, z), axis=0)


def _profile_pt3d_68(yawed_right: bool = True) -> np.ndarray:
    """Return a 3x68 array where one side's jawline sits well behind the
    central face plane (simulates a profile pose).

    Central landmarks (eye corners, nose, mouth corners) stay near z=0.
    Half the jawline gets pushed back along z by a large delta so the
    depth-based occlusion heuristic flags it.
    """
    xy = _points_68_2d()
    z = np.zeros((68,), dtype="float32")
    # XY face extent ≈ sqrt(67^2 + 67^2) ≈ 94.75 in this fixture; 60 is
    # comfortably above the 0.3 * extent = 28.4 threshold.
    far_side = slice(0, 8) if yawed_right else slice(9, 17)
    z[far_side] = 60.0
    return np.concatenate((xy, z[None, :]), axis=0)


def test_pt3d_z_coordinates_extracts_depth_column_for_3x68() -> None:
    """``_pt3d_z_coordinates`` returns the Z row for canonical (3, 68) input."""
    pt3d = _frontal_pt3d_68()
    z = _pt3d_z_coordinates({"pt3d_68": pt3d})
    assert z is not None
    assert z.shape == (68,)
    np.testing.assert_allclose(z, 50.0)


def test_pt3d_z_coordinates_returns_none_when_pt3d_missing() -> None:
    """Missing ``pt3d_68`` yields ``None`` so callers skip visibility derivation."""
    assert _pt3d_z_coordinates({}) is None
    assert _pt3d_z_coordinates({"pt2d": _points_68_2d()}) is None


def test_visibility_from_depth_returns_none_for_frontal_uniform_depth() -> None:
    """A face with near-uniform depth has nothing to mask out."""
    xy = _points_68_2d().T  # (68, 2)
    z = np.full((68,), 50.0, dtype="float32")
    assert _visibility_from_depth(xy, z) is None


def test_visibility_from_depth_flags_far_side_jawline_on_profile() -> None:
    """Profile poses produce a 68-bool mask with the far jawline marked invisible."""
    pt3d = _profile_pt3d_68(yawed_right=True)
    xy = pt3d[:2].T
    z = pt3d[2]
    visibility = _visibility_from_depth(xy, z)
    assert visibility is not None
    assert len(visibility) == 68
    # The right-jawline (indices 0..7) was pushed back → flagged invisible.
    assert all(not value for value in visibility[0:8])
    # Everything else stays visible.
    assert all(value for value in visibility[8:])


def test_visibility_from_depth_returns_none_when_xy_extent_collapses() -> None:
    """Degenerate input (no XY extent) returns ``None`` rather than dividing by zero."""
    xy = np.zeros((68, 2), dtype="float32")
    z = np.linspace(0, 60, 68, dtype="float32")
    assert _visibility_from_depth(xy, z) is None


def test_build_aflw2000_3d_manifest_records_visibility_for_profile_pose(tmp_path: Path) -> None:
    """A pt3d_68 with one-sided depth produces a visibility entry on the sample."""
    scipy_io = pytest.importorskip("scipy.io")
    root = tmp_path / "aflw2000"
    root.mkdir()
    _write_png(root / "face.jpg")
    scipy_io.savemat(
        str(root / "face.mat"),
        {
            "pt3d_68": _profile_pt3d_68(yawed_right=True),
            "Pose_Para": np.asarray([[0, 1.0, 0]], dtype="float32"),
        },
    )

    manifest = build_aflw2000_3d_manifest(root / "out", source_dir=root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]

    visibility = sample.get("visibility") or sample["metadata"].get("visibility")
    assert visibility is not None
    assert len(visibility) == 68
    assert sum(1 for v in visibility if not v) == 8
    assert sample["metadata"]["visibility_source"] == "aflw2000_3d_pt3d_68_depth"


def test_build_aflw2000_3d_manifest_omits_visibility_for_frontal_pose(tmp_path: Path) -> None:
    """Frontal faces (uniform depth) do not emit a visibility entry."""
    scipy_io = pytest.importorskip("scipy.io")
    root = tmp_path / "aflw2000"
    root.mkdir()
    _write_png(root / "face.jpg")
    scipy_io.savemat(
        str(root / "face.mat"),
        {"pt3d_68": _frontal_pt3d_68()},
    )

    manifest = build_aflw2000_3d_manifest(root / "out", source_dir=root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]

    assert "visibility" not in sample
    assert "visibility" not in sample["metadata"]
    assert "visibility_source" not in sample["metadata"]
