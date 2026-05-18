#!/usr/bin/env python3
"""Tests for native AFLW2000-3D manifest building."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.aflw2000_3d import (
    _profile_safe_normalizer,
    _pt3d_z_coordinates,
    _visibility_from_zbuffer,
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


def _profile_pt3d_68() -> np.ndarray:
    """Return a 3x68 array where back-of-face landmarks share XY positions
    with front-of-face landmarks but sit farther from the camera in Z.

    Setup: two clusters far enough apart that 0.08 * face_extent never
    bridges them. Cluster A at (10, 10) contains a front layer (indices
    0..7 at z=60, "closer to camera") and a back layer (indices 8..15
    at z=0). Cluster B at (90, 90) holds the remaining 52 landmarks at
    z=0; their large XY separation from cluster A keeps them out of
    each other's neighborhood. The z-buffer test should hide exactly
    indices 8..15.
    """
    xy = np.zeros((68, 2), dtype="float32")
    xy[0:8] = [10.0, 10.0]  # front layer of cluster A
    xy[8:16] = [10.0, 10.0]  # back layer of cluster A (XY-coincident with front)
    xy[16:] = [90.0, 90.0]  # cluster B (isolated from cluster A)
    z = np.zeros((68,), dtype="float32")
    z[0:8] = 60.0  # extent ≈ 113 → 60 > depth_margin (0.10 * 113 ≈ 11)
    return np.concatenate((xy.T, z[None, :]), axis=0)


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


def test_visibility_from_zbuffer_returns_none_for_uniform_depth() -> None:
    """A face with uniform depth has no front-of-back relationships to flag."""
    xy = _points_68_2d().T  # (68, 2)
    z = np.full((68,), 50.0, dtype="float32")
    assert _visibility_from_zbuffer(xy, z) is None


def test_visibility_from_zbuffer_hides_landmarks_behind_xy_neighbors() -> None:
    """Landmarks whose XY-neighbors sit in front in Z get flagged invisible."""
    pt3d = _profile_pt3d_68()
    xy = pt3d[:2].T
    z = pt3d[2]
    visibility = _visibility_from_zbuffer(xy, z)
    assert visibility is not None
    assert len(visibility) == 68
    # Indices 8..15 (back cluster) are stacked on top of 0..7 in XY but
    # sit further in Z → they should be the occluded set.
    assert visibility[0:8] == [True] * 8
    assert visibility[8:16] == [False] * 8
    assert all(visibility[16:])


def test_visibility_from_zbuffer_returns_none_when_xy_extent_collapses() -> None:
    """Degenerate input (no XY extent) returns ``None`` rather than dividing by zero."""
    xy = np.zeros((68, 2), dtype="float32")
    z = np.linspace(0, 60, 68, dtype="float32")
    assert _visibility_from_zbuffer(xy, z) is None


def test_profile_safe_normalizer_returns_none_for_frontal_faces() -> None:
    """When the eye corners are well separated, defer to the harness default."""
    xy = _points_68_2d().T.copy()
    # Inter-ocular ≈ 0.62 * sqrt(bbox_w * bbox_h) for a real frontal face;
    # set eye corners 36/45 explicitly so the ratio is comfortably above 0.30.
    xy[36] = [20.0, 50.0]
    xy[45] = [80.0, 50.0]
    assert _profile_safe_normalizer(xy) is None


def test_profile_safe_normalizer_returns_bbox_sqrt_when_eyes_collapse() -> None:
    """Extreme yaw — eye corners almost on top of each other — switches to sqrt(wh)."""
    xy = _points_68_2d().T.copy()
    # Inter-ocular ≈ 1 pixel on a ~100x100 face → ratio ≪ 0.30.
    xy[36] = [49.0, 50.0]
    xy[45] = [50.0, 50.0]
    bbox_w = float(xy[:, 0].max() - xy[:, 0].min())
    bbox_h = float(xy[:, 1].max() - xy[:, 1].min())
    expected = (bbox_w * bbox_h) ** 0.5
    result = _profile_safe_normalizer(xy)
    assert result is not None
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_profile_safe_normalizer_handles_degenerate_bbox() -> None:
    """A collapsed bbox (zero area) returns None instead of crashing."""
    xy = np.zeros((68, 2), dtype="float32")
    assert _profile_safe_normalizer(xy) is None


def test_build_aflw2000_3d_manifest_records_visibility_for_profile_pose(tmp_path: Path) -> None:
    """A pt3d_68 with self-occlusion produces visibility on the sample."""
    scipy_io = pytest.importorskip("scipy.io")
    root = tmp_path / "aflw2000"
    root.mkdir()
    _write_png(root / "face.jpg")
    scipy_io.savemat(
        str(root / "face.mat"),
        {
            "pt3d_68": _profile_pt3d_68(),
            "Pose_Para": np.asarray([[0, 1.0, 0]], dtype="float32"),
        },
    )

    manifest = build_aflw2000_3d_manifest(root / "out", source_dir=root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]

    visibility = sample.get("visibility") or sample["metadata"].get("visibility")
    assert visibility is not None
    assert len(visibility) == 68
    # Eight back-of-face landmarks were stacked on top of the front
    # cluster and pushed deeper in Z; the z-buffer test should hide them.
    occluded = sum(1 for v in visibility if not v)
    assert occluded == 8
    assert sample["metadata"]["visibility_source"] == "aflw2000_3d_pt3d_68_zbuffer"


def test_build_aflw2000_3d_manifest_records_profile_safe_normalizer(tmp_path: Path) -> None:
    """When eye corners collapse, the entry carries an explicit bbox-sqrt normalizer."""
    scipy_io = pytest.importorskip("scipy.io")
    root = tmp_path / "aflw2000"
    root.mkdir()
    _write_png(root / "face.jpg")
    # Force inter-ocular ≪ 0.30 * sqrt(w*h) by colocating eye corners.
    xy = _points_68_2d().T.copy()
    xy[36] = [49.0, 50.0]
    xy[45] = [50.0, 50.0]
    pt3d = np.concatenate((xy.T, np.full((1, 68), 25.0, dtype="float32")), axis=0)
    scipy_io.savemat(str(root / "face.mat"), {"pt3d_68": pt3d})

    manifest = build_aflw2000_3d_manifest(root / "out", source_dir=root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sample = payload["samples"][0]

    assert "normalizer" in sample
    assert sample["normalizer"] > 0
    assert sample["metadata"]["normalizer_source"] == "aflw2000_3d_bbox_sqrt_wh"


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
