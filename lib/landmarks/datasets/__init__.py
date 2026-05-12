#!/usr/bin/env python3
"""Landmark quality dataset manifest helpers."""

from __future__ import annotations

import contextlib
import json
import logging
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.schema import normalize_landmarks
from lib.landmarks.datasets.sources import (
    DEFAULT_CACHE_DIR,
    DatasetSourceSpec,
    extract_archive_to_temp,
    is_archive,
    resolve_dataset_source,
)

logger = logging.getLogger(__name__)
SUPPORTED_DATASETS = ("wflw", "cofw", "directory")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
WFLW_SOURCE = DatasetSourceSpec(
    dataset="WFLW",
    cache_subdir="wflw",
    canonical_archive="wflw.zip",
    cache_aliases=("WFLW.zip", "WFLW.tar.gz", "WFLW.tgz"),
    extracted_aliases=("WFLW", "WFLW_images"),
    manual_hint=(
        "Provide --wflw-annotations and --image-root, or place a WFLW archive/extracted "
        "dataset under .fs_cache/landmark_quality/wflw."
    ),
)
COFW_SOURCE = DatasetSourceSpec(
    dataset="COFW",
    cache_subdir="cofw",
    canonical_archive="cofw.zip",
    cache_aliases=("COFW.zip", "COFW.tar.gz", "COFW.tgz", "cofw_68.json"),
    extracted_aliases=("COFW", "cofw"),
    manual_hint=(
        "Provide --cofw-json, or place cofw_68.json/an archive/extracted dataset under "
        ".fs_cache/landmark_quality/cofw."
    ),
)


def build_manifest(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    scenario: str = "default",
) -> Path:
    """Build a simple manifest from ``*.npy`` landmarks and matching images.

    This MVP supports local WFLW/COFW-style prepared folders where each sample
    has ``<id>.npy`` landmarks and an image with the same stem.
    """
    dataset_name = dataset.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset '{dataset}'")
    src = Path(source_dir)
    samples: list[dict[str, T.Any]] = []
    for landmarks in sorted(src.glob("*.npy")):
        image = _matching_image(landmarks)
        if image is None:
            continue
        samples.append(
            {
                "sample_id": landmarks.stem,
                "dataset": dataset_name,
                "condition": scenario,
                "image": str(image.resolve()),
                "landmarks": str(landmarks.resolve()),
            }
        )
    return _write_manifest_and_audit(samples, Path(output_dir), dataset_name, scenario)


def build_directory_manifest(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str = "directory",
    scenario: str = "default",
) -> Path:
    """Build a manifest from a directory tree of ``*.npy`` landmarks.

    Each landmark file is paired with an image of the same stem. Unlike
    :func:`build_manifest`, this helper scans recursively and is useful for
    prepared benchmark exports.
    """
    dataset_name = dataset.lower()
    src = Path(source_dir)
    samples = []
    for landmarks in sorted(src.rglob("*.npy")):
        image = _matching_image(landmarks)
        if image is None:
            continue
        sample_id = landmarks.relative_to(src).with_suffix("").as_posix()
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "condition": scenario,
                "image": str(image.resolve()),
                "landmarks": str(landmarks.resolve()),
            }
        )
    return _write_manifest_and_audit(samples, Path(output_dir), dataset_name, scenario)


def build_wflw_manifest(
    annotation_file: str | Path | None,
    output_dir: str | Path,
    *,
    image_root: str | Path | None = None,
    source_dir: str | Path | None = None,
    source_zip: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
    scenario: str = "default",
) -> Path:
    """Build a WFLW manifest from 98-point annotations.

    If ``annotation_file`` is not supplied, the source waterfall is used:
    explicit source args, standard cache, then configured download URL.
    """
    cleanup: contextlib.AbstractContextManager[Path] | None = None
    if annotation_file is None:
        resolved = resolve_dataset_source(
            WFLW_SOURCE,
            cache_dir=cache_dir,
            source_dir=source_dir,
            source_zip=source_zip,
            download_url=download_url,
            force_download=force_download,
            no_download=no_download,
        )
        cleanup = _source_root(resolved)
        root = cleanup.__enter__()
        annotations = _find_wflw_annotation(root)
        inferred_image_root = _find_wflw_image_root(root)
    else:
        annotations = Path(annotation_file)
        inferred_image_root = annotations.parent
    try:
        if not annotations.is_file():
            raise FileNotFoundError(f"WFLW annotation file not found: {annotations}")
        root = inferred_image_root if image_root is None else Path(image_root)
        samples = []
        for index, line in enumerate(annotations.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 197:
                raise ValueError(f"WFLW line {index + 1} has too few fields")
            points = np.asarray([float(value) for value in parts[:196]], dtype="float32")
            image_rel = parts[-1]
            samples.append(
                {
                    "sample_id": Path(image_rel).with_suffix("").as_posix(),
                    "dataset": "wflw",
                    "condition": scenario,
                    "image": str((root / image_rel).resolve()),
                    "source_schema": "2d_98",
                    "points": normalize_landmarks(points.reshape(98, 2), source_schema="2d_98"),
                }
            )
        return _write_manifest_and_audit(samples, Path(output_dir), "wflw", scenario)
    finally:
        if cleanup is not None:
            cleanup.__exit__(None, None, None)


def build_cofw_manifest(
    source_json: str | Path | None,
    output_dir: str | Path,
    *,
    image_root: str | Path | None = None,
    source_dir: str | Path | None = None,
    source_zip: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    download_url: str | None = None,
    force_download: bool = False,
    no_download: bool = False,
    scenario: str = "default",
) -> Path:
    """Build a COFW manifest from a simple JSON export.

    Expected input shape is ``{"samples": [{"sample_id", "landmarks", "image",
    "conditions"}]}`` or a bare list with the same item shape.
    """
    cleanup: contextlib.AbstractContextManager[Path] | None = None
    if source_json is None:
        resolved = resolve_dataset_source(
            COFW_SOURCE,
            cache_dir=cache_dir,
            source_dir=source_dir,
            source_zip=source_zip,
            download_url=download_url,
            force_download=force_download,
            no_download=no_download,
        )
        if resolved.is_file() and not is_archive(resolved):
            source = resolved
            root = source.parent
        else:
            cleanup = _source_root(resolved)
            root = cleanup.__enter__()
            source = _find_cofw_json(root)
    else:
        source = Path(source_json)
        root = source.parent
    try:
        if not source.is_file():
            raise FileNotFoundError(f"COFW JSON not found: {source}")
        image_base = root if image_root is None else Path(image_root)
        payload = json.loads(source.read_text(encoding="utf-8"))
        entries = payload.get("samples", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError("COFW JSON must contain a list or a 'samples' list")
        samples = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"COFW entry {index + 1} must be an object")
            raw_points = entry.get("ground_truth", entry.get("landmarks"))
            if raw_points is None:
                raise ValueError(f"COFW entry {index + 1} missing landmarks")
            points = np.asarray(raw_points, dtype="float32")
            conditions = dict(entry.get("conditions", {}))
            image_value = str(entry.get("image", ""))
            image_path = Path(image_value)
            if image_value and not image_path.is_absolute():
                image_value = str((image_base / image_path).resolve())
            samples.append(
                {
                    "sample_id": str(entry.get("sample_id") or entry.get("id") or index),
                    "dataset": "cofw",
                    "condition": str(conditions.get("scenario", scenario)),
                    "image": image_value,
                    "points": normalize_landmarks(points),
                }
            )
        return _write_manifest_and_audit(samples, Path(output_dir), "cofw", scenario)
    finally:
        if cleanup is not None:
            cleanup.__exit__(None, None, None)


@contextlib.contextmanager
def _source_root(source: Path) -> T.Iterator[Path]:
    """Yield an extracted root directory for a source archive or directory."""
    if source.is_dir():
        yield source
    else:
        with extract_archive_to_temp(source) as root:
            yield root


def _find_wflw_annotation(root: Path) -> Path:
    """Find the best WFLW 98-point annotation file inside ``root``."""
    candidates = [path for path in root.rglob("*.txt") if "98pt" in path.name.lower()]
    if not candidates:
        raise FileNotFoundError(
            f"No WFLW 98-point annotation file found under {root}. "
            "Pass --wflw-annotations to point at it explicitly."
        )
    return sorted(candidates, key=lambda p: ("test" not in p.name.lower(), len(p.parts), p.name))[0]


def _find_wflw_image_root(root: Path) -> Path:
    """Return likely WFLW image root for relative annotation image paths."""
    for name in ("WFLW_images", "images", "Images"):
        matches = [path for path in root.rglob(name) if path.is_dir()]
        if matches:
            return sorted(matches, key=lambda p: len(p.parts))[0]
    return root


def _find_cofw_json(root: Path) -> Path:
    """Find a COFW JSON export inside ``root``."""
    candidates = sorted(root.rglob("*.json"), key=lambda p: ("cofw" not in p.name.lower(), len(p.parts), p.name))
    if not candidates:
        raise FileNotFoundError(
            f"No COFW JSON export found under {root}. Pass --cofw-json to point at it explicitly."
        )
    return candidates[0]


def _matching_image(path: Path) -> Path | None:
    """Return the matching image for a landmark path, if present."""
    for ext in IMAGE_EXTS:
        candidate = path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _safe_filename(value: str) -> str:
    """Return a readable filename-safe sample identifier."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe.strip("._") or "sample"


def _entry_path(value: str, output_dir: Path) -> Path:
    """Return a manifest path as absolute or relative to ``output_dir``."""
    path = Path(value)
    return path if path.is_absolute() else output_dir / path


def _build_audit(
    manifest_samples: T.Sequence[dict[str, T.Any]],
    output_dir: Path,
    dataset: str,
    scenario: str,
) -> dict[str, T.Any]:
    """Build a dataset audit payload for landmark manifests."""
    condition_counts: dict[str, int] = {}
    shape_counts: dict[str, int] = {}
    source_schema_counts: dict[str, int] = {}
    missing_images: list[str] = []
    missing_landmarks: list[str] = []
    invalid_landmarks: list[dict[str, T.Any]] = []
    sample_ids = [str(sample.get("sample_id", "")) for sample in manifest_samples]
    duplicate_ids = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})

    for sample in manifest_samples:
        condition = str(sample.get("condition", scenario))
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
        source_schema = str(sample.get("source_schema", "2d_68"))
        source_schema_counts[source_schema] = source_schema_counts.get(source_schema, 0) + 1
        image = str(sample.get("image", ""))
        if image and not _entry_path(image, output_dir).is_file():
            missing_images.append(image)
        landmarks = str(sample.get("landmarks", ""))
        landmark_path = _entry_path(landmarks, output_dir)
        if not landmark_path.is_file():
            missing_landmarks.append(landmarks)
            continue
        try:
            shape = tuple(np.load(str(landmark_path)).shape)
        except (OSError, ValueError) as err:
            invalid_landmarks.append({"sample_id": sample.get("sample_id", ""), "error": str(err)})
            continue
        shape_key = "x".join(str(part) for part in shape)
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        if shape != (68, 2):
            invalid_landmarks.append({"sample_id": sample.get("sample_id", ""), "shape": shape_key})

    return {
        "schema_version": 1,
        "dataset": dataset,
        "total_entries": len(manifest_samples),
        "condition_counts": condition_counts,
        "count_per_dataset": {dataset: len(manifest_samples)},
        "count_per_source_schema": source_schema_counts,
        "landmark_shape_counts": shape_counts,
        "missing_images": missing_images,
        "missing_landmarks": missing_landmarks,
        "invalid_landmarks": invalid_landmarks,
        "duplicate_sample_ids": duplicate_ids,
        "supported_datasets": SUPPORTED_DATASETS,
    }


def _write_source_notes(output_dir: Path) -> None:
    """Write generated source/licensing notes next to generated manifests."""
    notes = output_dir / "SOURCE_NOTES.md"
    if notes.is_file():
        return
    notes.write_text(
        "# Landmark quality dataset source notes\n\n"
        "This directory was populated by `tools/landmarks/build_quality_dataset.py`.\n\n"
        "The builder resolves sources from explicit CLI paths, `.fs_cache/landmark_quality`, "
        "or configured download URLs. Review upstream dataset terms before use or redistribution.\n\n"
        "Do not commit generated images, annotations, or manifests unless licensing has been reviewed.\n",
        encoding="utf-8",
    )


def _write_manifest_and_audit(
    samples: T.Sequence[dict[str, T.Any]],
    output_dir: Path,
    dataset: str,
    scenario: str,
) -> Path:
    """Write manifest, audit, and any inline landmark arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir = output_dir / "landmarks"
    manifest_samples: list[dict[str, T.Any]] = []
    for sample in samples:
        entry = dict(sample)
        points = entry.pop("points", None)
        if points is not None:
            landmarks_dir.mkdir(parents=True, exist_ok=True)
            landmarks_path = landmarks_dir / f"{_safe_filename(entry['sample_id'])}.npy"
            np.save(str(landmarks_path), np.asarray(points, dtype="float32"))
            entry["landmarks"] = str(landmarks_path.relative_to(output_dir))
        manifest_samples.append(entry)

    manifest = output_dir / "manifest.json"
    audit = output_dir / "dataset_audit.json"
    manifest.write_text(
        json.dumps(
            {"dataset": dataset, "samples": manifest_samples},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            _build_audit(manifest_samples, output_dir, dataset, scenario),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_source_notes(output_dir)
    return manifest
