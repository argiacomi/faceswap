#!/usr/bin/env python3
"""MERL-RAV dataset manifest builder.

MERL-RAV provides 68-point reannotations over AFLW images. The labels are
public and can be downloaded from the MERL-RAV repository. AFLW source images can
be supplied from an already organized MERL-RAV directory, the Oxford VGG DVE
recropped AFLW tarball, or a user-provided AFLW image directory/archive.

The LMDIS ``aflw_data.tar.gz`` package contains split text files and 5-point
keypoint ``.mat`` files, not AFLW source images, so it is intentionally not used
as an automatic image source.

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
import hashlib
import logging
import re
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
    download,
    extract_archive_to_cache,
    is_archive,
    resolve_dataset_source,
)
from lib.landmarks.schema import normalize_landmarks

logger = logging.getLogger(__name__)

MERL_RAV_LABELS_URL = (
    "https://github.com/abhi1kumar/MERL-RAV_dataset/archive/refs/heads/master.zip"
)
MERL_RAV_AFLW_URL = "https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw"
MERL_RAV_AFLW_REQUEST_FORM = (
    "https://cloud.tugraz.at/index.php/apps/forms/s/R7nejN42iY58E754eqHMfDKS"
)

AFLW_DVE_URL = "http://www.robots.ox.ac.uk/~vgg/research/DVE/data/datasets/aflw-recrop.tar.gz"
AFLW_DVE_SHA1 = "939fdce0e6262a14159832c71d4f84a9d516de5e"
AFLW_DVE_ARCHIVE = "aflw-recrop.tar.gz"
AFLW_DVE_PAPER = "Unsupervised learning of landmarks via vector exchange, ICCV 2019"

MERL_RAV_SOURCE = DatasetSourceSpec(
    dataset="MERL-RAV",
    cache_subdir="merl-rav",
    canonical_archive="MERL-RAV_dataset-master.zip",
    cache_aliases=("merl-rav.zip", "MERL-RAV.zip", "MERL_RAV.zip"),
    extracted_aliases=("merl_rav_organized", "MERL-RAV_dataset-master", "MERL-RAV", "MERL_RAV"),
    url=MERL_RAV_LABELS_URL,
    manual_hint=(
        "MERL-RAV labels are public and default to the MERL-RAV GitHub archive. "
        "AFLW images default to the Oxford DVE recropped AFLW tarball. "
        "Original AFLW remains approval-gated."
    ),
)

_IMAGE_TOKEN_RE = re.compile(r"image\d+|\d+")


def _sha1_file(path: Path) -> str:
    """Return the SHA1 hex digest for ``path``."""
    sha = hashlib.sha1()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _verify_sha1(path: Path, expected_sha1: str, *, label: str) -> None:
    """Raise when ``path`` does not match ``expected_sha1``."""
    actual = _sha1_file(path)
    if actual.lower() != expected_sha1.lower():
        raise ValueError(
            f"{label} checksum mismatch for {path.name}: expected SHA1 {expected_sha1}, got {actual}. "
            "Remove the cached file and retry with --force-download."
        )


def _aflw_cache_root(cache_dir: str | Path) -> Path:
    """Return the shared AFLW source-image cache root."""
    return Path(cache_dir) / "aflw"


def _aflw_source_help(cache_dir: str | Path) -> str:
    """Return a setup message for AFLW image sources."""
    root = _aflw_cache_root(cache_dir)
    return "\n".join(
        [
            f"AFLW image source not ready in {root}.",
            "Default source candidate:",
            f"  Oxford DVE AFLW recrop: {AFLW_DVE_URL}",
            f"  SHA1: {AFLW_DVE_SHA1}",
            f"Expected cache file: {root / AFLW_DVE_ARCHIVE}",
            "You can also pass --merl-rav-aflw-source-dir or --merl-rav-aflw-source-zip.",
            "Do not use LMDIS aflw_data.tar.gz as an image source; it contains split text files and 5-point .mat metadata, not source images.",
            f"Original AFLW approval/request page: {MERL_RAV_AFLW_REQUEST_FORM}",
        ]
    )


def _has_images(path: Path) -> bool:
    """Return whether ``path`` contains any supported image files."""
    if not path.is_dir():
        return False
    return any(child.is_file() and child.suffix.lower() in IMAGE_EXTS for child in path.rglob("*"))


def _require_images(path: Path, *, label: str) -> Path:
    """Return ``path`` when it contains images, otherwise raise a clear error."""
    if _has_images(path):
        return path
    mat_files = sum(1 for child in path.rglob("*.mat") if child.is_file()) if path.is_dir() else 0
    txt_files = sum(1 for child in path.rglob("*.txt") if child.is_file()) if path.is_dir() else 0
    raise FileNotFoundError(
        f"{label} does not contain AFLW source images: {path}. "
        f"Found txt_files={txt_files}, mat_files={mat_files}. "
        "For MERL-RAV, provide actual AFLW image files. LMDIS aflw_data.tar.gz is metadata/keypoints only and is not a usable image source."
    )


def _download_or_use_archive(
    url: str,
    destination: Path,
    *,
    force_download: bool,
    no_download: bool,
    sha1: str | None = None,
    label: str,
) -> Path | None:
    """Return a cached/downloaded archive, verifying SHA1 when provided."""
    if destination.is_file() and not force_download:
        if sha1 is not None:
            _verify_sha1(destination, sha1, label=label)
        logger.info("Using cached %s: %s", label, destination)
        return destination
    if no_download:
        return None
    archive = download(url, destination, force=force_download, label=label)
    if sha1 is not None:
        _verify_sha1(archive, sha1, label=label)
    return archive


def _resolve_aflw_source(
    *,
    cache_dir: str | Path,
    source_dir: str | Path | None = None,
    source_zip: str | Path | None = None,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
) -> Path:
    """Resolve an AFLW image source for matching MERL-RAV labels."""
    if source_dir is not None:
        directory = Path(source_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"AFLW source image directory not found: {directory}")
        return _require_images(directory, label="AFLW source image directory")
    if source_zip is not None:
        archive = Path(source_zip)
        if not archive.is_file():
            raise FileNotFoundError(f"AFLW source image archive not found: {archive}")
        extracted = extract_archive_to_cache(
            archive,
            _aflw_cache_root(cache_dir) / "extracted",
            force=force_download,
            label="AFLW source image archive",
        )
        return _require_images(extracted, label="AFLW source image archive")
    if download_url:
        archive = download(
            download_url,
            _aflw_cache_root(cache_dir) / Path(download_url).name,
            force=force_download,
            label="AFLW source image archive",
        )
        extracted = extract_archive_to_cache(
            archive,
            _aflw_cache_root(cache_dir) / "extracted",
            force=force_download,
            label="AFLW source image archive",
        )
        return _require_images(extracted, label="AFLW source image archive")

    extracted = _aflw_cache_root(cache_dir) / "extracted"
    if not force_download and _has_images(extracted):
        logger.info("Using cached extracted AFLW source images: %s", extracted)
        return extracted

    dve_archive = _download_or_use_archive(
        AFLW_DVE_URL,
        _aflw_cache_root(cache_dir) / AFLW_DVE_ARCHIVE,
        force_download=force_download,
        no_download=no_download,
        sha1=AFLW_DVE_SHA1,
        label="AFLW DVE recrop archive",
    )
    if dve_archive is not None:
        extracted = extract_archive_to_cache(
            dve_archive,
            extracted,
            force=force_download,
            expected_sha256=None,
            label="AFLW DVE recrop archive",
        )
        return _require_images(extracted, label="AFLW DVE recrop archive")

    raise FileNotFoundError(_aflw_source_help(cache_dir))


def _matching_image(annotation: Path) -> Path | None:
    """Return the same-stem image beside a MERL-RAV ``.pts`` annotation."""
    for ext in IMAGE_EXTS:
        candidate = annotation.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _image_lookup_keys(path: Path) -> set[str]:
    """Return AFLW image lookup keys for a path."""
    keys = {path.name.lower(), path.stem.lower(), path.as_posix().lower()}
    keys.update(part.lower() for part in path.stem.split("_") if part)
    keys.update(match.group(0).lower() for match in _IMAGE_TOKEN_RE.finditer(path.stem.lower()))
    return keys


def _build_aflw_image_index(root: Path) -> dict[str, Path]:
    """Build a deterministic AFLW image lookup from an extracted image source."""
    index: dict[str, Path] = {}
    image_paths = sorted(
        child
        for child in root.rglob("*")
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS
    )
    for image in image_paths:
        relative = image.relative_to(root)
        for key in _image_lookup_keys(relative):
            index.setdefault(key, image)
        for key in _image_lookup_keys(image):
            index.setdefault(key, image)
    return index


def _annotation_lookup_keys(annotation: Path) -> tuple[str, ...]:
    """Return possible AFLW image keys for a MERL-RAV annotation filename."""
    stem = annotation.stem.lower()
    first = stem.split("_")[0]
    keys = [stem, first]
    keys.extend(part for part in stem.split("_") if part)
    keys.extend(match.group(0).lower() for match in _IMAGE_TOKEN_RE.finditer(stem))
    return tuple(dict.fromkeys(keys))


def _find_aflw_image(annotation: Path, image_index: dict[str, Path]) -> Path | None:
    """Return the AFLW image that corresponds to a MERL-RAV label file."""
    for key in _annotation_lookup_keys(annotation):
        if key in image_index:
            return image_index[key]
    first = annotation.stem.lower().split("_")[0]
    if not first:
        return None
    for key, image in image_index.items():
        if first in key:
            return image
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
        if not inside and any(
            stripped.lower().startswith(prefix) for prefix in ("version", "n_points")
        ):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError as err:
            raise ValueError(
                f"invalid MERL-RAV .pts row {line_number} in {path}: {stripped}"
            ) from err
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


def _find_label_roots(root: Path) -> list[Path]:
    """Return roots that contain native MERL-RAV label folders."""
    candidates = [path for path in root.rglob("merl_rav_labels") if path.is_dir()]
    if candidates:
        return candidates
    if any(path.name == "labels" and path.is_dir() for path in root.rglob("*")):
        return [root]
    return [root]


def _label_files(root: Path) -> list[Path]:
    """Return MERL-RAV label files below one root."""
    label_files = [path for path in root.rglob("*.pts") if path.is_file()]
    return sorted(label_files)


def _sample_from_label(
    *,
    annotation: Path,
    label_root: Path,
    image: Path,
    image_root: Path | None,
    points: np.ndarray,
    visibility_metadata: dict[str, T.Any],
) -> dict[str, T.Any]:
    """Return one manifest sample from a MERL-RAV label and matched AFLW image."""
    relative_annotation = annotation.relative_to(label_root)
    condition_labels = _labels_from_path(relative_annotation)
    if visibility_metadata.get("externally_occluded_count", 0):
        condition_labels = tuple(dict.fromkeys((*condition_labels, "occlusion")))
    sample_id = relative_annotation.with_suffix("").as_posix()
    image_id = image.name if image_root is None else image.relative_to(image_root).as_posix()
    metadata: dict[str, T.Any] = {
        "image_id": image_id,
        "annotation_file": relative_annotation.as_posix(),
        **visibility_metadata,
        "face_bbox": _landmark_bbox(points),
        "face_bbox_source": "merl_rav_68_landmark_extrema",
        "aflw_image_source": "resolved_aflw_image_cache" if image_root else "organized_merl_rav",
        "aflw_dve_url": AFLW_DVE_URL,
        "aflw_dve_sha1": AFLW_DVE_SHA1,
        "aflw_dve_paper": AFLW_DVE_PAPER,
        "aflw_original_source_url": MERL_RAV_AFLW_URL,
        "aflw_request_form": MERL_RAV_AFLW_REQUEST_FORM,
    }
    return {
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


def _build_samples_from_labels(
    label_root: Path,
    *,
    image_root: Path | None = None,
) -> tuple[list[dict[str, T.Any]], dict[str, int]]:
    """Build manifest sample dictionaries from labels and optional AFLW image root."""
    image_index = _build_aflw_image_index(image_root) if image_root is not None else {}
    samples: list[dict[str, T.Any]] = []
    stats = {
        "labels": 0,
        "matched_images": 0,
        "skipped_missing_image": 0,
        "skipped_self_occluded": 0,
    }
    for annotation in _label_files(label_root):
        stats["labels"] += 1
        image = _matching_image(annotation)
        if image is None and image_root is not None:
            image = _find_aflw_image(annotation, image_index)
        if image is None:
            stats["skipped_missing_image"] += 1
            continue
        points, visibility_metadata = _parse_pts(annotation)
        if points is None:
            stats["skipped_self_occluded"] += 1
            continue
        stats["matched_images"] += 1
        samples.append(
            _sample_from_label(
                annotation=annotation,
                label_root=label_root,
                image=image,
                image_root=image_root,
                points=points,
                visibility_metadata=visibility_metadata,
            )
        )
    return samples, stats


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
    aflw_image_root: Path | None = None,
) -> Path:
    """Build a MERL-RAV manifest from organized pairs or separate labels/images."""
    scenario_groups = _explicit_scenario_groups(scenarios)
    all_samples: list[dict[str, T.Any]] = []
    aggregate = {
        "labels": 0,
        "matched_images": 0,
        "skipped_missing_image": 0,
        "skipped_self_occluded": 0,
    }
    for label_root in _find_label_roots(root):
        samples, stats = _build_samples_from_labels(label_root, image_root=aflw_image_root)
        all_samples.extend(samples)
        for key, value in stats.items():
            aggregate[key] += value
    if aggregate["skipped_missing_image"]:
        logger.warning(
            "Skipped MERL-RAV labels without matching AFLW image: %d",
            aggregate["skipped_missing_image"],
        )
    if aggregate["skipped_self_occluded"]:
        logger.warning(
            "Skipped MERL-RAV samples with self-occluded unestimated landmarks: %d",
            aggregate["skipped_self_occluded"],
        )
    if not all_samples:
        detail = (
            f"labels={aggregate['labels']} matched_images={aggregate['matched_images']} "
            f"missing_images={aggregate['skipped_missing_image']} "
            f"self_occluded={aggregate['skipped_self_occluded']}"
        )
        raise FileNotFoundError(
            f"No usable MERL-RAV label/AFLW image pairs found under {root}. {detail}. "
            "Provide a compatible AFLW image source or an already organized image+pts directory."
        )
    return _write_manifest_and_audit(
        _filter_samples(all_samples, scenario_groups, samples_per_scenario),
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
    aflw_source_dir: str | Path | None = None,
    aflw_source_zip: str | Path | None = None,
    aflw_download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
    scenario: str = "default",
    scenarios: T.Sequence[str] | None = None,
    samples_per_scenario: int | None = None,
    manifest_mode: str = "replace",
    allow_overlap: bool = False,
    write_overlays: bool = False,
) -> Path:
    """Build a MERL-RAV manifest from native labels and AFLW source images."""
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
            raise ValueError("MERL-RAV source must be a directory or archive")
        cleanup = _source_root(resolved)
        root = cleanup.__enter__()
        organized_samples, _organized_stats = _build_samples_from_labels(root, image_root=None)
        if organized_samples:
            return _build_from_root(
                root,
                output_dir,
                scenario=scenario,
                scenarios=scenarios,
                samples_per_scenario=samples_per_scenario,
                manifest_mode=manifest_mode,
                allow_overlap=allow_overlap,
                write_overlays=write_overlays,
                aflw_image_root=None,
            )
        aflw_root = _resolve_aflw_source(
            cache_dir=cache_dir,
            source_dir=aflw_source_dir,
            source_zip=aflw_source_zip,
            download_url=aflw_download_url,
            force_download=force_download,
            no_download=no_download,
        )
        return _build_from_root(
            root,
            output_dir,
            scenario=scenario,
            scenarios=scenarios,
            samples_per_scenario=samples_per_scenario,
            manifest_mode=manifest_mode,
            allow_overlap=allow_overlap,
            write_overlays=write_overlays,
            aflw_image_root=aflw_root,
        )
    finally:
        if cleanup is not None:
            cleanup.__exit__(None, None, None)
