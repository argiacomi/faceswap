#!/usr/bin/env python3
"""Landmark quality dataset manifest helpers."""

from __future__ import annotations

import json
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.schema import normalize_landmarks

SUPPORTED_DATASETS = ("wflw", "cofw", "directory")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


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
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, T.Any]] = []
    for landmarks in sorted(src.glob("*.npy")):
        image = next(
            (
                src / f"{landmarks.stem}{ext}"
                for ext in (".png", ".jpg", ".jpeg")
                if (src / f"{landmarks.stem}{ext}").is_file()
            ),
            None,
        )
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
    manifest = out / "manifest.json"
    audit = out / "dataset_audit.json"
    manifest.write_text(
        json.dumps({"dataset": dataset_name, "samples": samples}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "total_entries": len(samples),
                "condition_counts": {scenario: len(samples)},
                "supported_datasets": SUPPORTED_DATASETS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


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
    annotation_file: str | Path,
    output_dir: str | Path,
    *,
    image_root: str | Path | None = None,
    scenario: str = "default",
) -> Path:
    """Build a WFLW manifest from an annotation text file.

    WFLW annotations provide 98 landmarks. They are converted to the canonical
    68-point schema before being written to the generated dataset folder.
    """
    annotations = Path(annotation_file)
    root = annotations.parent if image_root is None else Path(image_root)
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
                "points": normalize_landmarks(points.reshape(98, 2), source_schema="2d_98"),
            }
        )
    return _write_manifest_and_audit(samples, Path(output_dir), "wflw", scenario)


def build_cofw_manifest(
    source_json: str | Path,
    output_dir: str | Path,
    *,
    scenario: str = "default",
) -> Path:
    """Build a COFW manifest from a simple JSON export.

    Expected input shape is ``{"samples": [{"sample_id", "landmarks", "image",
    "conditions"}]}`` or a bare list with the same item shape.
    """
    source = Path(source_json)
    payload = json.loads(source.read_text(encoding="utf-8"))
    entries = payload.get("samples", payload)
    if not isinstance(entries, list):
        raise ValueError("COFW JSON must contain a list or a 'samples' list")
    samples = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"COFW entry {index + 1} must be an object")
        points = np.asarray(entry.get("ground_truth", entry.get("landmarks")), dtype="float32")
        conditions = dict(entry.get("conditions", {}))
        samples.append(
            {
                "sample_id": str(entry.get("sample_id") or entry.get("id") or index),
                "dataset": "cofw",
                "condition": str(conditions.get("scenario", scenario)),
                "image": str(entry.get("image", "")),
                "points": normalize_landmarks(points),
            }
        )
    return _write_manifest_and_audit(samples, Path(output_dir), "cofw", scenario)


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
    condition_counts: dict[str, int] = {}
    for sample in manifest_samples:
        condition = str(sample.get("condition", scenario))
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
    audit.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "total_entries": len(manifest_samples),
                "condition_counts": condition_counts,
                "supported_datasets": SUPPORTED_DATASETS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
