#!/usr/bin/env python3
"""MultiPIE dataset integration from MenpoBenchmark.

The MenpoBenchmark MultiPIE package does not ship one annotation file per image.
Instead it provides flat list files (``MultiPIE_semifrontal_train.txt`` and
``MultiPIE_profile_train.txt``) where every line describes a single face::

    image/<pose>/<name>.jpg  x1 y1 x2 y2  <5 detector points>  <68|39 dense points>

i.e. the image path, a 4-value detection bounding box, 5 reference points (10
values) and then the dense ground-truth landmarks (68 for ``semifrontal``, 39
for ``profile``). This module parses those list files directly and reuses the
shared MenpoBenchmark sample helpers to build a canonical manifest.
"""

from __future__ import annotations

import logging
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.datasets import (
    _explicit_scenario_groups,
    _filter_samples,
    _write_manifest_and_audit,
)
from lib.landmarks.datasets.manifest_io import coerce_bbox
from lib.landmarks.datasets.menpo_benchmark import (
    IMAGE_EXTS,
    _conditions,
    _normalizer,
    _points_for_manifest,
    _pose_group,
    _source_root,
    _yaw_side,
)
from lib.landmarks.datasets.sources import (
    DEFAULT_CACHE_DIR,
    DatasetSourceSpec,
    resolve_dataset_source,
)

logger = logging.getLogger(__name__)

# MenpoBenchmark README:
# https://github.com/jiankangdeng/MenpoBenchmark
# MultiPIE Google Drive file id from:
# https://drive.google.com/file/d/18JFjBTAZqthpORmEf2LuT14IuMYNyD_h/view
MULTIPIE_GOOGLE_DRIVE_FILE_ID = "18JFjBTAZqthpORmEf2LuT14IuMYNyD_h"

# Each list-file line has 4 bounding-box values followed by 5 detector reference
# points (10 values) before the dense ground-truth landmarks begin.
_HEADER_VALUES = 14
_DENSE_VALUES = {136: 68, 78: 39}

MULTIPIE_SOURCE = DatasetSourceSpec(
    dataset="MultiPIE",
    cache_subdir="multipie",
    canonical_archive="MultiPIE.zip",
    cache_aliases=("multipie.zip", "MultiPIE.tar.gz", "multipie.tgz"),
    extracted_aliases=("MultiPIE", "multipie"),
    google_drive_file_id=MULTIPIE_GOOGLE_DRIVE_FILE_ID,
    manual_hint=(
        "MultiPIE annotations/package are distributed by MenpoBenchmark via Google Drive. "
        "Install the optional Google Drive downloader dependency if needed, "
        "or place MultiPIE.zip/extracted MultiPIE under .fs_cache/landmark_quality/multipie."
    ),
)


def _is_list_file(path: Path) -> bool:
    """Return ``True`` if ``path`` looks like a MenpoBenchmark list file.

    Parameters
    ----------
    path
        The candidate ``.txt`` file

    Returns
    -------
    ``True`` if the first non-empty line starts with an image path followed by numeric values
    """
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                tokens = raw.split()
                if not tokens:
                    continue
                if Path(tokens[0]).suffix.lower() not in IMAGE_EXTS or len(tokens) < 2:
                    return False
                try:
                    float(tokens[1])
                except ValueError:
                    return False
                return True
    except OSError:
        return False
    return False


def _find_list_files(root: Path) -> list[Path]:
    """Locate the MenpoBenchmark MultiPIE list files under ``root``.

    Parameters
    ----------
    root
        The extracted dataset root

    Returns
    -------
    Sorted list of list-file paths
    """
    return sorted(p for p in root.rglob("*.txt") if p.is_file() and _is_list_file(p))


def _parse_line(line: str) -> tuple[str, tuple[float, float, float, float], np.ndarray] | None:
    """Parse one list-file line into an image path, detector bbox, and landmarks.

    Parameters
    ----------
    line
        A single line from a MenpoBenchmark list file

    Returns
    -------
    A ``(image relative path, ltrb bbox, (N, 2) landmark array)`` tuple, or ``None`` if the
    line is blank or does not contain a supported number of dense landmark values
    """
    tokens = line.split()
    if not tokens:
        return None
    image_rel = tokens[0]
    try:
        values = [float(token) for token in tokens[1:]]
    except ValueError:
        return None
    dense_count = len(values) - _HEADER_VALUES
    if dense_count not in _DENSE_VALUES:
        return None
    bbox = coerce_bbox(values[:4])
    if bbox is None:
        return None
    points = np.asarray(values[_HEADER_VALUES:], dtype="float32").reshape(-1, 2)
    return image_rel, bbox, points


def _samples_from_list_file(
    list_file: Path,
    root: Path,
    dataset_name: str,
    scenario: str,
    *,
    include_39pt_profile: bool,
) -> list[dict[str, T.Any]]:
    """Build manifest samples from a single MenpoBenchmark list file.

    Parameters
    ----------
    list_file
        The list file to parse
    root
        The extracted dataset root (used for sample ids)
    dataset_name
        The dataset name to record on each sample
    scenario
        The fallback scenario condition
    include_39pt_profile
        ``False`` to skip 39-point profile annotations

    Returns
    -------
    The parsed samples for this list file
    """
    samples: list[dict[str, T.Any]] = []
    for raw in list_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = _parse_line(raw)
        if parsed is None:
            continue
        image_rel, face_bbox, raw_points = parsed
        try:
            points, source_schema = _points_for_manifest(raw_points)
        except ValueError:
            continue
        if source_schema == "2d_39" and not include_39pt_profile:
            continue

        # Image paths are stored relative to the list file's directory.
        image = (list_file.parent / image_rel).resolve()
        if not image.is_file():
            logger.warning("Skipping missing MultiPIE image: %s", image)
            continue

        rel_path = image.relative_to(root.resolve()).with_suffix("")
        rel = rel_path.as_posix()
        condition_path = Path(image_rel)
        normalizer, normalizer_source = _normalizer(points, rel)
        labels = _conditions(dataset_name, condition_path, points, scenario)
        metadata: dict[str, T.Any] = {
            "face_bbox": list(face_bbox),
            "face_bbox_source": "multipie_menpobenchmark_detection",
            "source_landmark_count": int(points.shape[0]),
            "source_annotation": str(list_file.resolve()),
            "normalizer_source": normalizer_source,
            "menpo_benchmark_pose_group": _pose_group(condition_path, points),
        }
        side = _yaw_side(condition_path)
        if side:
            metadata["yaw_side"] = side

        samples.append(
            {
                "sample_id": rel,
                "dataset": dataset_name,
                "condition": labels[0],
                "conditions": labels,
                "image": str(image),
                "source_schema": source_schema,
                "face_bbox": list(face_bbox),
                "source": {"dataset": dataset_name, "source_id": rel},
                "metadata": metadata,
                "normalizer": normalizer,
                "points": points,
            }
        )
    return samples


def build_multipie_manifest(
    output_dir: str | Path,
    *,
    source_dir: str | Path | None = None,
    source_zip: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
    scenario: str = "multipie_profile",
    scenarios: T.Sequence[str] | None = None,
    samples_per_scenario: int | None = None,
    manifest_mode: str = "replace",
    allow_overlap: bool = False,
    write_overlays: bool = False,
    include_39pt_profile: bool = False,
) -> Path:
    """Build a canonical manifest from the MenpoBenchmark MultiPIE package."""
    resolved = resolve_dataset_source(
        MULTIPIE_SOURCE,
        cache_dir=cache_dir,
        source_dir=source_dir,
        source_zip=source_zip,
        download_url=download_url,
        force_download=force_download,
        no_download=no_download,
    )
    dataset_name = "multipie"
    scenario_groups = _explicit_scenario_groups(scenarios)
    with _source_root(resolved) as root:
        list_files = _find_list_files(root)
        if not list_files:
            raise FileNotFoundError(
                f"No MultiPIE MenpoBenchmark list files found under {root}. Expected "
                "'MultiPIE_semifrontal_train.txt' / 'MultiPIE_profile_train.txt' style files "
                "with 'image/<pose>/<name>.jpg x1 y1 x2 y2 <5 points> <68|39 points>' lines."
            )

        samples: list[dict[str, T.Any]] = []
        for list_file in list_files:
            samples.extend(
                _samples_from_list_file(
                    list_file,
                    root,
                    dataset_name,
                    scenario,
                    include_39pt_profile=include_39pt_profile,
                )
            )

        if not samples:
            raise FileNotFoundError(
                f"No usable MultiPIE image/landmark pairs found under {root}. Parsed "
                f"{len(list_files)} list file(s) but produced no 68/39-point samples with "
                "matching images."
            )

        logger.info(
            "Parsed %s MultiPIE samples from %s list file(s)", len(samples), len(list_files)
        )
        return _write_manifest_and_audit(
            _filter_samples(samples, scenario_groups, samples_per_scenario),
            Path(output_dir),
            dataset_name,
            scenario,
            manifest_mode=manifest_mode,
            allow_overlap=allow_overlap,
            write_overlays=write_overlays,
            scenario_groups=scenario_groups,
        )
