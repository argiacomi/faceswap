#!/usr/bin/env python3
"""300W dataset manifest builder.

300W uses native 68-point landmarks, which makes it a direct fit for the
canonical 68-point ensemble pipeline. This builder consumes extracted 300W-style
folders containing ``.pts`` annotation files and matching image files.

The official iBUG 300-W package is distributed as four split zip parts behind an
iBUG download form. We therefore do not pretend this is a normal silent network
fallback. Instead, the builder records the official URLs, detects the four parts
when the user places them in the standard cache, concatenates them into a local
``300w.zip``, and extracts that archive.
"""

from __future__ import annotations

import contextlib
import logging
import os
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
    extract_archive_to_cache,
    is_archive,
    resolve_dataset_source,
)
from lib.landmarks.schema import normalize_landmarks

logger = logging.getLogger(__name__)

W300_OFFICIAL_PAGE = "https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/"
W300_OFFICIAL_PART_URLS = (
    "https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.001",
    "https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.002",
    "https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.003",
    "https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.004",
)
W300_OFFICIAL_PART_NAMES = tuple(Path(url).name for url in W300_OFFICIAL_PART_URLS)
W300_COMBINED_ARCHIVE = "300w.zip"

W300_SOURCE = DatasetSourceSpec(
    dataset="300W",
    cache_subdir="300w",
    canonical_archive=W300_COMBINED_ARCHIVE,
    cache_aliases=("300W.zip", "300W.tar.gz", "300W.tgz"),
    extracted_aliases=("300W", "300w", "300-W", "300_W"),
    manual_hint=(
        "Official 300-W is distributed by iBUG as four split zip parts behind "
        "their download form. Download part1-part4 from "
        f"{W300_OFFICIAL_PAGE} and place them in .fs_cache/landmark_quality/300w "
        "as 300w.zip.001, 300w.zip.002, 300w.zip.003, 300w.zip.004; or pass "
        "--source-dir/--source-zip for an already extracted/combined source. "
        "iBUG states the annotations are for research purposes only."
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


def _cache_root(cache_dir: str | Path) -> Path:
    """Return the 300W cache root."""
    return Path(cache_dir) / W300_SOURCE.cache_root_name


def _official_part_paths(cache_dir: str | Path) -> list[Path]:
    """Return official split part paths under the standard 300W cache."""
    root = _cache_root(cache_dir)
    return [root / name for name in W300_OFFICIAL_PART_NAMES]


def _official_source_help(cache_dir: str | Path, *, missing: T.Sequence[Path] | None = None) -> str:
    """Return a setup message for official 300-W sources."""
    root = _cache_root(cache_dir)
    lines = [
        f"300W official source not ready in {root}.",
        f"Download the four official iBUG parts from {W300_OFFICIAL_PAGE}",
        "Expected cache files:",
        *(f"  {root / name}" for name in W300_OFFICIAL_PART_NAMES),
        "Official part URLs:",
        *(f"  {url}" for url in W300_OFFICIAL_PART_URLS),
        "The iBUG page requires a download form and states the annotations are for research purposes only, so the pipeline will not silently fetch these files without user setup.",
    ]
    if missing:
        lines.insert(1, "Missing parts: " + ", ".join(path.name for path in missing))
    return "\n".join(lines)


def _looks_like_html(path: Path) -> bool:
    """Return whether a downloaded official part looks like the iBUG form HTML."""
    try:
        prefix = path.read_bytes()[:128].lstrip().lower()
    except OSError:
        return False
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _combine_official_parts(cache_dir: str | Path, *, force: bool = False) -> Path | None:
    """Combine cached official 300-W split zip parts into one local zip archive."""
    root = _cache_root(cache_dir)
    parts = _official_part_paths(cache_dir)
    existing = [path for path in parts if path.is_file()]
    if not existing:
        return None
    missing = [path for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(_official_source_help(cache_dir, missing=missing))
    html_parts = [path for path in parts if _looks_like_html(path)]
    if html_parts:
        raise ValueError(
            "300W cached part appears to be the iBUG download form HTML, not the archive payload: "
            + ", ".join(str(path) for path in html_parts)
            + "\n"
            + _official_source_help(cache_dir)
        )
    combined = root / W300_COMBINED_ARCHIVE
    if combined.is_file() and not force:
        logger.info("Using cached combined 300W archive: %s", combined)
        return combined
    root.mkdir(parents=True, exist_ok=True)
    tmp = combined.with_suffix(combined.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    logger.info("Combining official 300W split archives into: %s", combined)
    with tmp.open("wb") as outfile:
        for part in parts:
            with part.open("rb") as infile:
                while True:
                    chunk = infile.read(1024 * 1024)
                    if not chunk:
                        break
                    outfile.write(chunk)
    os.replace(tmp, combined)
    return combined


def _resolve_300w_source(
    *,
    cache_dir: str | Path,
    source_dir: str | Path | None,
    source_zip: str | Path | None,
    download_url: str | None,
    force_download: bool,
    no_download: bool,
) -> Path:
    """Resolve a 300W source, including the official multipart cache layout."""
    try:
        return resolve_dataset_source(
            W300_SOURCE,
            cache_dir=cache_dir,
            source_dir=source_dir,
            source_zip=source_zip,
            download_url=download_url,
            force_download=force_download,
            no_download=no_download,
        )
    except FileNotFoundError as err:
        if source_dir is not None or source_zip is not None or download_url:
            raise
        combined = _combine_official_parts(cache_dir, force=force_download)
        if combined is None:
            raise FileNotFoundError(_official_source_help(cache_dir)) from err
        return extract_archive_to_cache(
            combined,
            _cache_root(cache_dir) / "extracted",
            force=force_download,
            label="300W official multipart archive",
        )


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
                    "official_source_page": W300_OFFICIAL_PAGE,
                    "official_part_urls": list(W300_OFFICIAL_PART_URLS),
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
    resolved = _resolve_300w_source(
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
