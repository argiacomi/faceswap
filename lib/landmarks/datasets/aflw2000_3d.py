#!/usr/bin/env python3
"""AFLW2000-3D dataset manifest builder.

AFLW2000-3D is distributed as image files paired with MATLAB ``.mat`` files.
The native 3DDFA annotation file contains 2D landmark projections under the
``pt2d`` key. For the 68-point ensemble pipeline we consume that 2D projection
as canonical 68-point ground truth and preserve additional 3D/Pose metadata when
present.
"""

from __future__ import annotations

import contextlib
import logging
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.datasets import (
    AFLW2000_3D_SOURCE,
    IMAGE_EXTS,
    _condition_labels_from_metadata,
    _explicit_scenario_groups,
    _filter_samples,
    _source_root,
    _write_manifest_and_audit,
)
from lib.landmarks.datasets.sources import DEFAULT_CACHE_DIR, is_archive, resolve_dataset_source
from lib.landmarks.schema import normalize_landmarks

logger = logging.getLogger(__name__)


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


def _pt2d(payload: dict[str, T.Any], *, source: Path) -> np.ndarray:
    """Return 68x2 2D landmarks from a 3DDFA AFLW2000 annotation payload."""
    raw = payload.get("pt2d")
    if raw is None:
        raise ValueError(f"AFLW2000-3D annotation missing 'pt2d': {source}")
    points = np.asarray(raw, dtype="float32")
    points = np.squeeze(points)
    if points.shape == (2, 68):
        points = points.T
    elif points.shape == (3, 68):
        points = points[:2].T
    elif points.shape == (68, 3):
        points = points[:, :2]
    if points.shape != (68, 2):
        raise ValueError(f"AFLW2000-3D pt2d must resolve to 68x2, got {points.shape}: {source}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"AFLW2000-3D pt2d contains NaN/Inf values: {source}")
    return np.ascontiguousarray(points, dtype="float32")


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


def _metadata(payload: dict[str, T.Any], annotation: Path, image: Path, root: Path) -> dict[str, T.Any]:
    """Return metadata preserved from an AFLW2000-3D annotation."""
    metadata: dict[str, T.Any] = {
        "image_id": image.relative_to(root).as_posix(),
        "annotation_file": annotation.relative_to(root).as_posix(),
    }
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
        points = _pt2d(payload, source=annotation)
        metadata = _metadata(payload, annotation, image, root)
        metadata["face_bbox"] = _landmark_bbox(points)
        metadata["face_bbox_source"] = "aflw2000_3d_pt2d_extrema"
        condition_labels = _condition_labels_from_metadata({}, metadata, default=scenario)
        sample_id = annotation.relative_to(root).with_suffix("").as_posix()
        samples.append(
            {
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
        )
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
