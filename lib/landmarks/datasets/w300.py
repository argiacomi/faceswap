#!/usr/bin/env python3
"""300W dataset manifest builder.

300W uses native 68-point landmarks, which makes it a direct fit for the
canonical 68-point ensemble pipeline. This builder consumes extracted 300W-style
folders containing ``.pts`` annotation files and matching image files.

The default automated source is the Oxford VGG DVE paper reproduction copy of
300W, distributed as a single tarball with a published SHA1. The original iBUG
300-W split archive layout is still supported as a manual/cache fallback.
"""

from __future__ import annotations

import contextlib
import hashlib
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
    download,
    extract_archive_to_cache,
    is_archive,
    resolve_dataset_source,
)
from lib.landmarks.schema import normalize_landmarks

logger = logging.getLogger(__name__)

W300_DVE_URL = "http://www.robots.ox.ac.uk/~vgg/research/DVE/data/datasets/300w.tar.gz"
W300_DVE_SHA1 = "885b09159c61fa29998437747d589c65cfc4ccd3"
W300_DVE_ARCHIVE = "300w.tar.gz"
W300_DVE_PAPER = "Unsupervised learning of landmarks via vector exchange, ICCV 2019"

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
    canonical_archive=W300_DVE_ARCHIVE,
    cache_aliases=(W300_COMBINED_ARCHIVE, "300W.zip", "300W.tar.gz", "300W.tgz"),
    extracted_aliases=("300W", "300w", "300-W", "300_W"),
    manual_hint=(
        "Default source is the Oxford VGG DVE 300W tarball at "
        f"{W300_DVE_URL} with SHA1 {W300_DVE_SHA1}. The original iBUG 300-W "
        f"split archive can also be obtained from {W300_OFFICIAL_PAGE}; place "
        "300w.zip.001, 300w.zip.002, 300w.zip.003, and 300w.zip.004 under "
        ".fs_cache/landmark_quality/300w, or pass --source-dir/--source-zip."
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


def _sha1_file(path: Path) -> str:
    """Return the SHA1 hex digest for ``path``."""
    sha = hashlib.sha1()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _verify_dve_sha1(path: Path) -> None:
    """Raise when a cached/downloaded DVE tarball has the wrong SHA1."""
    actual = _sha1_file(path)
    if actual.lower() != W300_DVE_SHA1.lower():
        raise ValueError(
            f"300W DVE archive checksum mismatch for {path.name}: "
            f"expected SHA1 {W300_DVE_SHA1}, got {actual}. "
            "Remove the cached file and retry with --force-download."
        )


def _has_300w_content(path: Path) -> bool:
    """Return whether an extracted source appears to contain 300W samples."""
    return path.is_dir() and any(path.rglob("*.pts"))


def _dve_archive_path(cache_dir: str | Path) -> Path:
    """Return the default cached DVE tarball path."""
    return _cache_root(cache_dir) / W300_DVE_ARCHIVE


def _download_or_use_dve_archive(
    cache_dir: str | Path,
    *,
    force_download: bool,
    no_download: bool,
) -> Path | None:
    """Return a verified Oxford DVE 300W archive, downloading when allowed."""
    archive = _dve_archive_path(cache_dir)
    if archive.is_file() and not force_download:
        _verify_dve_sha1(archive)
        logger.info("Using cached 300W DVE archive: %s", archive)
        return archive
    if no_download:
        return None
    downloaded = download(
        W300_DVE_URL,
        archive,
        force=force_download,
        label="300W DVE archive",
    )
    _verify_dve_sha1(downloaded)
    return downloaded


def _extract_dve_archive(cache_dir: str | Path, archive: Path, *, force: bool) -> Path:
    """Extract a verified DVE 300W archive into the managed cache."""
    _verify_dve_sha1(archive)
    return extract_archive_to_cache(
        archive,
        _cache_root(cache_dir) / "extracted",
        force=force,
        label="300W DVE archive",
    )


def _official_part_paths(cache_dir: str | Path) -> list[Path]:
    """Return official split part paths under the standard 300W cache."""
    root = _cache_root(cache_dir)
    return [root / name for name in W300_OFFICIAL_PART_NAMES]


def _source_help(cache_dir: str | Path, *, missing: T.Sequence[Path] | None = None) -> str:
    """Return a setup message for 300W sources."""
    root = _cache_root(cache_dir)
    lines = [
        f"300W source not ready in {root}.",
        "Default automated source:",
        f"  {W300_DVE_URL}",
        f"  SHA1: {W300_DVE_SHA1}",
        f"Expected downloaded cache file: {root / W300_DVE_ARCHIVE}",
        "Manual iBUG fallback cache files:",
        *(f"  {root / name}" for name in W300_OFFICIAL_PART_NAMES),
        f"iBUG source page: {W300_OFFICIAL_PAGE}",
        "You can also pass --source-dir/--source-zip for an already extracted or archived 300W source.",
    ]
    if missing:
        lines.insert(1, "Missing iBUG split parts: " + ", ".join(path.name for path in missing))
    return "\n".join(lines)


def _looks_like_html(path: Path) -> bool:
    """Return whether a downloaded official part looks like an HTML form page."""
    try:
        prefix = path.read_bytes()[:128].lstrip().lower()
    except OSError:
        return False
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")


def _combine_official_parts(cache_dir: str | Path, *, force: bool = False) -> Path | None:
    """Combine cached iBUG 300-W split zip parts into one local zip archive."""
    root = _cache_root(cache_dir)
    parts = _official_part_paths(cache_dir)
    existing = [path for path in parts if path.is_file()]
    if not existing:
        return None
    missing = [path for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(_source_help(cache_dir, missing=missing))
    html_parts = [path for path in parts if _looks_like_html(path)]
    if html_parts:
        raise ValueError(
            "300W cached part appears to be HTML, not archive payload: "
            + ", ".join(str(path) for path in html_parts)
            + "\n"
            + _source_help(cache_dir)
        )
    combined = root / W300_COMBINED_ARCHIVE
    if combined.is_file() and not force:
        logger.info("Using cached combined 300W archive: %s", combined)
        return combined
    root.mkdir(parents=True, exist_ok=True)
    tmp = combined.with_suffix(combined.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    logger.info("Combining iBUG 300W split archives into: %s", combined)
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
    """Resolve a 300W source, including DVE tarball and iBUG split fallback."""
    if source_dir is not None or source_zip is not None or download_url:
        return resolve_dataset_source(
            W300_SOURCE,
            cache_dir=cache_dir,
            source_dir=source_dir,
            source_zip=source_zip,
            download_url=download_url,
            force_download=force_download,
            no_download=no_download,
        )

    extracted = _cache_root(cache_dir) / "extracted"
    if not force_download and _has_300w_content(extracted):
        logger.info("Using cached extracted 300W source: %s", extracted)
        return extracted

    dve_archive = _download_or_use_dve_archive(
        cache_dir,
        force_download=force_download,
        no_download=no_download,
    )
    if dve_archive is not None:
        return _extract_dve_archive(cache_dir, dve_archive, force=force_download)

    combined = _combine_official_parts(cache_dir, force=force_download)
    if combined is not None:
        return extract_archive_to_cache(
            combined,
            extracted,
            force=force_download,
            label="300W iBUG multipart archive",
        )

    raise FileNotFoundError(_source_help(cache_dir))


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
                    "dve_source_url": W300_DVE_URL,
                    "dve_sha1": W300_DVE_SHA1,
                    "dve_paper": W300_DVE_PAPER,
                    "ibug_source_page": W300_OFFICIAL_PAGE,
                    "ibug_part_urls": list(W300_OFFICIAL_PART_URLS),
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
