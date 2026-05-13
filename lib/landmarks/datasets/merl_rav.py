#!/usr/bin/env python3
"""MERL-RAV dataset manifest builder.

MERL-RAV provides 68-point reannotations over AFLW images. The labels are
public, but AFLW images require approval and must be supplied by the user. This
builder consumes an already organized MERL-RAV directory containing matching
``.pts`` and image files, such as the output of the upstream
``organize_merl_rav_using_aflw_and_our_labels.py`` script.

MERL-RAV label semantics:

* positive ``x y``: visible landmark
* negative ``-x -y``: externally occluded, estimated at ``abs(x), abs(y)``
* ``-1 -1``: self-occluded, location not estimated

The current 68-point ensemble harness has no per-landmark GT mask, so samples
with self-occluded ``-1 -1`` landmarks are skipped rather than evaluated against
invalid coordinates.
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

MERL_RAV_LABELS_URL = "https://github.com/abhi1kumar/MERL-RAV_dataset/archive/refs/heads/master.zip"
MERL_RAV_AFLW_URL = "https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw"
MERL_RAV_AFLW_REQUEST_FORM = "https://cloud.tugraz.at/index.php/apps/forms/s/R7nejN42iY58E754eqHMfDKS"
MERL_RAV_SOURCE = DatasetSourceSpec(
    dataset="MERL-RAV",
    cache_subdir="merl-rav",
    canonical_archive="merl-rav.zip",
    cache_aliases=("MERL-RAV.zip", "MERL_RAV.zip", "MERL-RAV_dataset-master.zip"),
    extracted_aliases=("merl_rav_organized", "MERL-RAV_dataset-master", "MERL-RAV", "MERL_RAV"),
    manual_hint=(
        "MERL-RAV labels are public, but AFLW images require separate approval. "
        f"Labels: {MERL_RAV_LABELS_URL}. AFLW request: {MERL_RAV_AFLW_REQUEST_FORM}. "
        "Provide an organized image+pts directory generated after obtaining AFLW images."
    ),
)


def _matching_image(annotation: Path) -> Path | None:
    """Return the image matching a MERL-RAV ``.pts`` annotation."""
    for ext in IMAGE_EXTS:
        candidate = annotation.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _parse_pts(path: Path) -> tuple[np.ndarray | None, dict[str, T.Any]]:
    """Parse MERL-RAV ``.pts`` labels and return points plus visibility metadata."""
    rows: list[tuple[float, float]] = []
    inside = False
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
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
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError as err:
            raise ValueError(f"invalid MERL-RAV .pts row {line_number} in {path}: {stripped}") from err
    if len(rows) != 68:
        raise ValueError(f"MERL-RAV .pts file must contain 68 points, got {len(rows)}: {path}")
    visibility: list[str] = []
    points: list[list[float]] = []
    self_occluded = 0
    externally_occluded = 0
    for x_value, y_value in rows:
        if x_value == -1 and y_value == -1:
            visibility.append("self_occluded")
            self_occluded += 1
            points.append([np.nan, np.nan])
        elif x_value < 0 and y_value < 0:
            visibility.append("externally_occluded")
            externally_occluded += 1
            points.append([abs(x_value), abs(y_value)])
        else:
            visibility.append("visible")
            points.append([x_value, y_value])
    metadata = {
        "visibility": visibility,
        "self_occluded_count": self_occluded,
        "externally_occluded_count": externally_occluded,
    }
    if self_occluded:
        return None, metadata
    return np.asarray(points, dtype="float32"), metadata


def _landmark_bbox(points: np.ndarray) -> list[float]:
    """Return left/top/right/bottom from landmark extrema."""
    left, top = np.min(points, axis=0)
    right, bottom = np.max(points, axis=0)
    return [float(left), float(top), float(right), float(bottom)]


def _labels_from_path(path: Path) -> tuple[str, ...]:
    """Return MERL-RAV pose/split labels from the organized path."""
    parts = tuple(part.lower().replace("-", "_") for part in path.parts)
    labels: list[str] = []
    for pose in ("frontal", "left", "lefthalf", "right", "righthalf"):
        if pose in parts:
            labels.append(pose)
            break
    for split in ("testset", "trainset"):
        if split in parts:
            labels.append(split)
            break
    return tuple(labels) or ("default",)


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
    """Build a MERL-RAV manifest from organized image/pts pairs."""
    scenario_groups = _explicit_scenario_groups(scenarios)
    samples: list[dict[str, T.Any]] = []
    skipped_self_occluded = 0
    skipped_missing_image = 0
    for annotation in sorted(root.rglob("*.pts")):
        image = _matching_image(annotation)
        if image is None:
            skipped_missing_image += 1
            continue
        points, visibility_metadata = _parse_pts(annotation)
        if points is None:
            skipped_self_occluded += 1
            continue
        condition_labels = _labels_from_path(annotation.relative_to(root))
        if visibility_metadata.get("externally_occluded_count", 0):
            condition_labels = tuple(dict.fromkeys((*condition_labels, "occlusion")))
        sample_id = annotation.relative_to(root).with_suffix("").as_posix()
        metadata: dict[str, T.Any] = {
            "image_id": image.relative_to(root).as_posix(),
            "annotation_file": annotation.relative_to(root).as_posix(),
            **visibility_metadata,
            "face_bbox": _landmark_bbox(points),
            "face_bbox_source": "merl_rav_68_landmark_extrema",
            "aflw_images_required": True,
            "aflw_source_url": MERL_RAV_AFLW_URL,
            "aflw_request_form": MERL_RAV_AFLW_REQUEST_FORM,
        }
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": "merl-rav",
                "condition": condition_labels[0],
                "conditions": condition_labels,
                "image": str(image.resolve()),
                "source_schema": "2d_68",
                "source": {"dataset": "merl-rav", "source_id": sample_id},
                "metadata": metadata,
                "points": normalize_landmarks(points, source_schema="2d_68"),
            }
        )
    if skipped_missing_image:
        logger.warning("Skipped MERL-RAV labels without matching AFLW image: %d", skipped_missing_image)
    if skipped_self_occluded:
        logger.warning(
            "Skipped MERL-RAV samples with self-occluded unestimated landmarks: %d",
            skipped_self_occluded,
        )
    if not samples:
        raise FileNotFoundError(
            f"No usable MERL-RAV .pts/image pairs found under {root}. "
            "AFLW images must be supplied after approval and organized beside labels."
        )
    return _write_manifest_and_audit(
        _filter_samples(samples, scenario_groups, samples_per_scenario),
        Path(output_dir),
        "merl-rav",
        scenario,
        manifest_mode=manifest_mode,
        allow_overlap=allow_overlap,
        write_overlays=write_overlays,
        scenario_groups=scenario_groups,
    )


def build_merl_rav_manifest(
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
    """Build a MERL-RAV manifest from organized native ``.pts`` labels and images."""
    resolved = resolve_dataset_source(
        MERL_RAV_SOURCE,
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
            raise ValueError("MERL-RAV source must be an organized directory or archive")
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
