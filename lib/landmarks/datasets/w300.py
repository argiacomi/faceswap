#!/usr/bin/env python3
"""300W dataset manifest builder.

300W uses native 68-point landmarks, which makes it a direct fit for the
canonical 68-point ensemble pipeline. This builder consumes extracted 300W-style
folders containing ``.pts`` annotation files and matching image files.
"""

from __future__ import annotations

import contextlib
import logging
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.datasets import (
    IMAGE_EXTS,
    _explicit_scenario_groups,
    _fallback_condition_label,
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
from lib.landmarks.schema import normalize_landmarks

logger = logging.getLogger(__name__)

W300_SOURCE = DatasetSourceSpec(
    dataset="300W",
    cache_subdir="300w",
    canonical_archive="300w.zip",
    cache_aliases=("300W.zip", "300W.tar.gz", "300W.tgz", "300w_68.json"),
    extracted_aliases=("300W", "300w", "300-W", "300_W"),
    manual_hint=(
        "Provide --source-dir/--source-zip containing extracted 300W .pts annotations "
        "and matching images, or place an archive/extracted dataset under "
        ".fs_cache/landmark_quality/300w."
    ),
)


def _parse_pts(path: Path) -> np.ndarray:
    """Parse a standard 300W/Menpo-style ``.pts`` file into a 68x2 array."""
    rows: list[list[float]] = []
    inside = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            inside = True
            continue
        if stripped.startswith("}"):
            break
        if not inside and any(stripped.lower().startswith(prefix) for prefix in ("version", "n_points")):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except ValueError as err:
            raise ValueError(f"invalid 300W .pts row {line_number} in {path}: {stripped}") from err
    points = np.asarray(rows, dtype="float32")
    if points.shape != (68, 2):
        raise ValueError(f"300W .pts file must contain 68 x/y points, got {points.shape}: {path}")
    return points


def _matching_image(annotation: Path) -> Path | None:
    """Return the image matching a 300W ``.pts`` annotation."""
    for ext in IMAGE_EXTS:
        candidate = annotation.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _condition_for_path(path: Path, *, default: str) -> tuple[str, ...]:
    """Return split-style condition labels from a 300W annotation path."""
    parts = {part.lower().replace("-", "_") for part in path.parts}
    joined = "/".join(parts)
    if "ibug" in parts or "/ibug/" in joined:
        return ("challenging",)
    if "helen" in parts or "lfpw" in parts or "/helen/" in joined or "/lfpw/" in joined:
        return ("common",)
    if "afw" in parts or "/afw/" in joined:
        return ("train",)
    return (_fallback_condition_label(default),)


def _landmark_bbox(points: np.ndarray) -> list[float]:
    """Return left/top/right/bottom landmark extrema."""
    left, top = np.min(points, axis=0)
    right, bottom = np.max(points, axis=0)
    return [float(left), float(top), float(right), float(bottom)]


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
    scenario_groups = _explicit_scenario_groups(scenarios)
    samples: list[dict[str, T.Any]] = []
    for annotation in sorted(root.rglob("*.pts")):
        image = _matching_image(annotation)
        if image is None:
            logger.debug("Skipping 300W annotation without matching image: %s", annotation)
            continue
        points = _parse_pts(annotation)
        condition_labels = _condition_for_path(annotation.relative_to(root), default=scenario)
        sample_id = annotation.relative_to(root).with_suffix("").as_posix()
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": "300w",
                "condition": condition_labels[0],
                "conditions": condition_labels,
                "image": str(image.resolve()),
                "source_schema": "2d_68",
                "source": {"dataset": "300w", "source_id": sample_id},
                "metadata": {
                    "image_id": image.relative_to(root).as_posix(),
                    "annotation_file": annotation.relative_to(root).as_posix(),
                    "face_bbox": _landmark_bbox(points),
                    "face_bbox_source": "300w_68_landmark_extrema",
                },
                "points": normalize_landmarks(points, source_schema="2d_68"),
            }
        )
    if not samples:
        raise FileNotFoundError(f"No 300W .pts/image pairs found under {root}")
    return _write_manifest_and_audit(
        _filter_samples(samples, scenario_groups, samples_per_scenario),
        Path(output_dir),
        "300w",
        scenario,
        manifest_mode=manifest_mode,
        allow_overlap=allow_overlap,
        write_overlays=write_overlays,
        scenario_groups=scenario_groups,
    )


def build_300w_manifest(
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
    """Build a 300W manifest from extracted ``.pts`` annotations and images."""
    resolved = resolve_dataset_source(
        W300_SOURCE,
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
            raise ValueError("300W source must be an extracted directory or archive, not a JSON file")
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
