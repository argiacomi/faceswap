#!/usr/bin/env python3
"""AFLW2000-3D dataset manifest builder.

AFLW2000-3D is distributed as image files paired with MATLAB ``.mat`` files.
The native 3DDFA annotation file contains 68-point 3D landmarks under the
``pt3d_68`` key. For the 68-point ensemble pipeline we consume the XY projection
of ``pt3d_68`` as canonical 68-point ground truth and preserve additional
3D/Pose metadata when present.

Some AFLW2000-3D ``.mat`` files also contain a ``pt2d`` key, but that key may be
a 21-point sparse annotation and is not the 68-point target used by the standard
AFLW2000-3D dataset loaders.
"""

from __future__ import annotations

import contextlib
import logging
import math
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.core.schema import normalize_landmarks
from lib.landmarks.datasets import (
    IMAGE_EXTS,
    _condition_labels_from_metadata,
    _explicit_scenario_groups,
    _filter_samples,
    _source_root,
    _write_manifest_and_audit,
)
from lib.landmarks.datasets.sources import (
    DEFAULT_CACHE_DIR,
    DatasetSourceSpec,
    is_archive,
    resolve_dataset_source,
)

logger = logging.getLogger(__name__)

AFLW2000_3D_URL = (
    "http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip"
)
AFLW2000_3D_SHA256 = "252bc35274d65ff27b6e573aa96c2f4c116ad88452cc984fb882258c0ed6e2d8"
AFLW2000_3D_SOURCE = DatasetSourceSpec(
    dataset="AFLW2000-3D",
    cache_subdir="aflw2000-3d",
    canonical_archive="AFLW2000-3D.zip",
    cache_aliases=("AFLW2000_3D.zip", "aflw2000-3d.zip", "aflw2000_3d.zip"),
    extracted_aliases=("AFLW2000-3D", "AFLW2000_3D", "aflw2000-3d", "aflw2000_3d"),
    url=AFLW2000_3D_URL,
    sha256=AFLW2000_3D_SHA256,
    manual_hint="Provide an extracted AFLW2000-3D directory/archive with image+.mat pairs.",
)


def _load_mat(path: Path) -> dict[str, T.Any]:
    """Load a MATLAB annotation file."""
    try:
        from scipy.io import loadmat
    except ImportError as err:  # pragma: no cover - depends on local environment
        raise ImportError("AFLW2000-3D parsing requires scipy") from err
    return dict(loadmat(str(path)))


def _matching_image(annotation: Path) -> Path | None:
    """Return the image with the same stem as a ``.mat`` annotation."""
    for ext in IMAGE_EXTS:
        candidate = annotation.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _as_68x2(points: np.ndarray, *, source: Path, key: str) -> np.ndarray:
    """Normalize a MATLAB landmark array to 68x2 XY coordinates."""
    points = np.asarray(points, dtype="float32")
    points = np.squeeze(points)
    if points.shape == (3, 68):
        points = points[:2].T
    elif points.shape == (2, 68):
        points = points.T
    elif points.shape == (68, 3) or points.shape == (68, 2):
        points = points[:, :2]
    if points.shape != (68, 2):
        raise ValueError(
            f"AFLW2000-3D {key} must resolve to 68x2 XY landmarks, got {points.shape}: {source}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"AFLW2000-3D {key} contains NaN/Inf values: {source}")
    return np.ascontiguousarray(points, dtype="float32")


def _pt3d_z_coordinates(payload: dict[str, T.Any]) -> np.ndarray | None:
    """Return the per-landmark Z coordinate of ``pt3d_68`` when available.

    AFLW2000-3D ships ``pt3d_68`` as a (3, 68) array of 3D landmarks in the
    face-aligned camera frame. Only XY is used for the canonical 2D ground
    truth; this helper exposes the Z column so visibility can be derived
    from actual 3D depth instead of a yaw-bin heuristic.
    """
    if "pt3d_68" not in payload:
        return None
    raw = np.asarray(payload["pt3d_68"], dtype="float32")
    raw = np.squeeze(raw)
    if raw.shape == (3, 68):
        z = raw[2]
    elif raw.shape == (68, 3):
        z = raw[:, 2]
    else:
        return None
    if not np.all(np.isfinite(z)):
        return None
    return z.astype("float32")


# Z-buffer self-occlusion parameters. A landmark is flagged as occluded
# when another landmark projects within ``neighborhood_ratio * extent``
# of its XY position AND sits at least ``depth_margin_ratio * extent``
# closer to the camera in AFLW2000-3D's pt3d_68 frame (smaller z).
# Calibrated against the AFLW2000-3D 60-90° yaw bin so that the full
# far-side jaw/cheek/ear arc gets hidden instead of leaving 1-8 stray
# back-side landmarks visible (the failure mode of the previous depth-
# vs-central-plane heuristic).
DEFAULT_ZBUFFER_NEIGHBORHOOD_RATIO: float = 0.08
DEFAULT_ZBUFFER_DEPTH_MARGIN_RATIO: float = 0.10


# Below this inter-ocular / sqrt(bbox_w * bbox_h) ratio the eye corners
# are too collapsed to use as an NME denominator. On the AFLW2000-3D
# test set, frontal-to-45° samples sit at IO/sqrt(wh) >= 0.37, while
# 60°+ profiles drop to <0.30 (>=85% of samples). 0.30 cleanly separates
# the two regimes.
DEFAULT_INTEROCULAR_COLLAPSE_RATIO: float = 0.30


def _visibility_from_zbuffer(
    points_xy: np.ndarray,
    z: np.ndarray,
    *,
    neighborhood_ratio: float = DEFAULT_ZBUFFER_NEIGHBORHOOD_RATIO,
    depth_margin_ratio: float = DEFAULT_ZBUFFER_DEPTH_MARGIN_RATIO,
) -> list[bool] | None:
    """Return a 68-bool visibility list using a z-buffer self-occlusion
    test against AFLW2000-3D's pt3d_68 depths.

    AFLW2000-3D's pt3d_68 stores camera-space coordinates where larger
    ``z`` means closer to the camera (the 3DMM face normal points out
    of the face toward the viewer). For each landmark ``i``, check
    every other landmark ``j``: when ``|xy_j - xy_i|`` is within
    ``neighborhood_ratio * face_extent`` AND ``z_j`` is at least
    ``depth_margin_ratio * face_extent`` greater than ``z_i`` (i.e. j
    sits in front of i toward the camera), ``i`` is flagged invisible.
    Returns ``None`` when no landmark is occluded so the manifest stays
    compact and the harness's ``visibility=None`` path applies as
    before.
    """
    if points_xy.shape != (68, 2) or z.shape != (68,):
        return None
    face_extent = float(np.linalg.norm(points_xy.max(axis=0) - points_xy.min(axis=0)))
    if face_extent <= 0:
        return None
    neighborhood = neighborhood_ratio * face_extent
    depth_margin = depth_margin_ratio * face_extent
    xy = points_xy.astype(np.float64)
    pairwise_xy_dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(pairwise_xy_dist, np.inf)
    within = pairwise_xy_dist < neighborhood
    # For each i, largest z among XY-neighbors (or -inf if no neighbor).
    candidate_z = np.where(within, z[None, :], -np.inf)
    nearest_in_front_z = candidate_z.max(axis=1)
    occluded = (nearest_in_front_z - z) > depth_margin
    visibility = (~occluded).tolist()
    if all(visibility):
        return None
    return [bool(value) for value in visibility]


def _profile_safe_normalizer(points_xy: np.ndarray) -> float | None:
    """Return ``sqrt(bbox_w * bbox_h)`` when the inter-ocular distance
    collapses for profile views, else ``None`` to keep the harness's
    default inter-ocular normalizer.

    The default NME denominator is the distance between landmarks 36 and
    45 (outer eye corners). Under extreme yaw the two outer corners
    project onto nearly the same pixel and that distance approaches
    zero, inflating NME by an order of magnitude or more. When the
    inter-ocular distance drops below ``DEFAULT_INTEROCULAR_COLLAPSE_RATIO``
    of the GT bounding-box diagonal we fall back to the published
    AFLW2000-3D normalizer ``sqrt(bbox_w * bbox_h)`` (Zhu et al., 3DDFA),
    which stays bounded across the full yaw range.
    """
    if points_xy.shape != (68, 2):
        return None
    inter_ocular = float(np.linalg.norm(points_xy[36] - points_xy[45]))
    bbox_w = float(points_xy[:, 0].max() - points_xy[:, 0].min())
    bbox_h = float(points_xy[:, 1].max() - points_xy[:, 1].min())
    bbox_sqrt = math.sqrt(max(bbox_w * bbox_h, 0.0))
    if bbox_sqrt <= 0:
        return None
    if inter_ocular < DEFAULT_INTEROCULAR_COLLAPSE_RATIO * bbox_sqrt:
        return bbox_sqrt
    return None


def _points_68(payload: dict[str, T.Any], *, source: Path) -> tuple[np.ndarray, str]:
    """Return canonical 68-point XY landmarks and the source key used."""
    if "pt3d_68" in payload:
        return _as_68x2(np.asarray(payload["pt3d_68"]), source=source, key="pt3d_68"), "pt3d_68"
    if "pt2d_68" in payload:
        return _as_68x2(np.asarray(payload["pt2d_68"]), source=source, key="pt2d_68"), "pt2d_68"
    if "pt2d" in payload:
        return _as_68x2(np.asarray(payload["pt2d"]), source=source, key="pt2d"), "pt2d"
    raise ValueError(f"AFLW2000-3D annotation missing pt3d_68/pt2d_68/pt2d landmarks: {source}")


def _landmark_bbox(points: np.ndarray) -> list[float]:
    """Return left/top/right/bottom from landmark extrema."""
    left, top = np.min(points, axis=0)
    right, bottom = np.max(points, axis=0)
    return [float(left), float(top), float(right), float(bottom)]


def _serializable_vector(payload: dict[str, T.Any], key: str) -> list[float] | None:
    """Return a flattened MATLAB vector as JSON-serializable floats."""
    if key not in payload:
        return None
    values = np.asarray(payload[key]).reshape(-1)
    if values.size == 0:
        return None
    return [float(value) for value in values.astype("float32").tolist()]


def _metadata(
    payload: dict[str, T.Any],
    annotation: Path,
    image: Path,
    root: Path,
    *,
    landmark_source_key: str,
) -> dict[str, T.Any]:
    """Return metadata preserved from an AFLW2000-3D annotation."""
    metadata: dict[str, T.Any] = {
        "image_id": image.relative_to(root).as_posix(),
        "annotation_file": annotation.relative_to(root).as_posix(),
        "landmark_source_key": landmark_source_key,
    }
    if "pt2d" in payload:
        metadata["pt2d_shape"] = list(np.squeeze(np.asarray(payload["pt2d"])).shape)
    if "pt3d_68" in payload:
        metadata["pt3d_68_shape"] = list(np.squeeze(np.asarray(payload["pt3d_68"])).shape)
    for key in ("Pose_Para", "Shape_Para", "Exp_Para"):
        values = _serializable_vector(payload, key)
        if values is not None:
            metadata[key] = values
    return metadata


def _build_from_root(
    root: Path,
    output_dir: str | Path,
    *,
    scenario: str,
    scenarios: T.Sequence[str] | None,
    samples_per_scenario: int | None,
    manifest_mode: str,
    allow_overlap: bool,
    write_overlays: bool,
) -> Path:
    """Build a manifest from an extracted AFLW2000-3D root."""
    scenario_groups = _explicit_scenario_groups(scenarios)
    samples: list[dict[str, T.Any]] = []
    for annotation in sorted(root.rglob("*.mat")):
        image = _matching_image(annotation)
        if image is None:
            logger.debug("Skipping AFLW2000-3D annotation without matching image: %s", annotation)
            continue
        payload = _load_mat(annotation)
        points, landmark_source_key = _points_68(payload, source=annotation)
        metadata = _metadata(
            payload,
            annotation,
            image,
            root,
            landmark_source_key=landmark_source_key,
        )
        metadata["face_bbox"] = _landmark_bbox(points)
        metadata["face_bbox_source"] = f"aflw2000_3d_{landmark_source_key}_xy_extrema"
        visibility: list[bool] | None = None
        if landmark_source_key == "pt3d_68":
            z_values = _pt3d_z_coordinates(payload)
            if z_values is not None:
                visibility = _visibility_from_zbuffer(points, z_values)
        if visibility is not None:
            metadata["visibility"] = visibility
            metadata["visibility_source"] = "aflw2000_3d_pt3d_68_zbuffer"
        normalizer = _profile_safe_normalizer(points)
        if normalizer is not None:
            metadata["normalizer"] = normalizer
            metadata["normalizer_source"] = "aflw2000_3d_bbox_sqrt_wh"
        condition_labels = _condition_labels_from_metadata({}, metadata, default=scenario)
        sample_id = annotation.relative_to(root).with_suffix("").as_posix()
        entry: dict[str, T.Any] = {
            "sample_id": sample_id,
            "dataset": "aflw2000-3d",
            "condition": condition_labels[0],
            "conditions": condition_labels,
            "image": str(image.resolve()),
            "source_schema": "2d_68",
            "source": {"dataset": "aflw2000-3d", "source_id": sample_id},
            "metadata": metadata,
            "points": normalize_landmarks(points, source_schema="2d_68"),
        }
        if visibility is not None:
            entry["visibility"] = visibility
        if normalizer is not None:
            entry["normalizer"] = normalizer
        samples.append(entry)
    if not samples:
        raise FileNotFoundError(f"No AFLW2000-3D .mat/image pairs found under {root}")
    return _write_manifest_and_audit(
        _filter_samples(samples, scenario_groups, samples_per_scenario),
        Path(output_dir),
        "aflw2000-3d",
        scenario,
        manifest_mode=manifest_mode,
        allow_overlap=allow_overlap,
        write_overlays=write_overlays,
        scenario_groups=scenario_groups,
    )


def build_aflw2000_3d_manifest(
    output_dir: str | Path,
    *,
    source_dir: str | Path | None = None,
    source_zip: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
    scenario: str = "default",
    scenarios: T.Sequence[str] | None = None,
    samples_per_scenario: int | None = None,
    manifest_mode: str = "replace",
    allow_overlap: bool = False,
    write_overlays: bool = False,
) -> Path:
    """Build an AFLW2000-3D manifest from native 3DDFA ``.mat`` annotations."""
    resolved = resolve_dataset_source(
        AFLW2000_3D_SOURCE,
        cache_dir=cache_dir,
        source_dir=source_dir,
        source_zip=source_zip,
        download_url=download_url,
        force_download=force_download,
        no_download=no_download,
    )
    cleanup: contextlib.AbstractContextManager[Path] | None = None
    try:
        if resolved.is_file() and not is_archive(resolved):
            raise ValueError("AFLW2000-3D source must be an extracted directory or archive")
        cleanup = _source_root(resolved)
        root = cleanup.__enter__()
        return _build_from_root(
            root,
            output_dir,
            scenario=scenario,
            scenarios=scenarios,
            samples_per_scenario=samples_per_scenario,
            manifest_mode=manifest_mode,
            allow_overlap=allow_overlap,
            write_overlays=write_overlays,
        )
    finally:
        if cleanup is not None:
            cleanup.__exit__(None, None, None)
