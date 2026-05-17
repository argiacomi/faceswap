#!/usr/bin/env python3
"""COFW-68 source materialization helpers.

COFW has two commonly referenced landmark layouts:

* Native COFW uses 29 landmarks.
* COFW-68 is a separate benchmark that adds 68-point annotations for the COFW
  test split.

The 68-point ensemble pipeline consumes COFW-68, not native COFW-29. This module
materializes the JSON export expected by ``build_cofw_manifest`` from the two
upstream source archives:

* Caltech COFW color images: ``COFW_test_color.mat``
* Ghiasi/Fowlkes COFW-68 annotations: ``COFW68_Data/test_annotations/*.mat``
"""

from __future__ import annotations

import json
import logging
import shutil
import typing as T
from pathlib import Path

import cv2
import numpy as np

from lib.landmarks.datasets.sources import (
    DEFAULT_CACHE_DIR,
    download,
    extract_archive_to_cache,
)

logger = logging.getLogger(__name__)

COFW_COLOR_URL = "https://data.caltech.edu/records/bc0bf-nc666/files/COFW_color.zip?download=1"
COFW68_ANNOTATIONS_URL = "https://github.com/golnazghiasi/cofw68-benchmark/archive/master.zip"
COFW68_JSON_NAME = "cofw_68.json"


def _load_mat(path: Path) -> dict[str, T.Any]:
    """Load a MATLAB ``.mat`` file using available readers."""
    try:
        from scipy.io import loadmat as scipy_loadmat

        return dict(scipy_loadmat(str(path)))
    except NotImplementedError:
        pass
    except ValueError as err:
        # scipy raises ValueError for some v7.3/HDF5 files.
        if "Unknown mat file type" not in str(err):
            raise
    try:
        from hdf5storage import loadmat as hdf5_loadmat
    except ImportError as err:  # pragma: no cover - depends on optional local env
        raise ImportError(
            f"Unable to read MATLAB v7.3 file {path}. Install hdf5storage, or provide "
            "a prebuilt cofw_68.json export."
        ) from err
    return dict(hdf5_loadmat(str(path)))


def _find_file(root: Path, name: str) -> Path:
    """Return the first matching file under ``root``."""
    matches = sorted(root.rglob(name), key=lambda item: (len(item.parts), str(item)))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {root}")
    return matches[0]


def _annotation_files(root: Path) -> list[Path]:
    """Return sorted COFW-68 annotation mat files."""
    files = sorted(
        root.rglob("*_points.mat"),
        key=lambda item: (
            int(item.stem.split("_", 1)[0]) if item.stem.split("_", 1)[0].isdigit() else item.name
        ),
    )
    if not files:
        raise FileNotFoundError(f"COFW-68 annotation '*_points.mat' files not found under {root}")
    return files


def _cofw_images(images_mat: Path) -> list[np.ndarray]:
    """Return COFW test images from ``COFW_test_color.mat``."""
    payload = _load_mat(images_mat)
    raw = payload.get("IsT")
    if raw is None:
        raise ValueError(f"COFW image mat missing 'IsT': {images_mat}")
    images: list[np.ndarray] = []
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        iterable = raw.reshape(-1)
    else:
        iterable = np.asarray(raw)
    for item in iterable:
        image = np.asarray(item)
        if image.size == 0:
            continue
        image = np.squeeze(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.ndim != 3:
            raise ValueError(f"unexpected COFW image shape in {images_mat}: {image.shape}")
        # hdf5 readers may return channel-first arrays for MATLAB cell contents.
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] != 3:
            raise ValueError(f"unexpected COFW color image shape in {images_mat}: {image.shape}")
        images.append(np.asarray(image, dtype="uint8"))
    if not images:
        raise ValueError(f"no COFW test images found in {images_mat}")
    return images


def _points_and_occlusion(annotation_mat: Path) -> tuple[np.ndarray, list[int]]:
    """Return 68-point landmarks and occlusion flags from one annotation file."""
    payload = _load_mat(annotation_mat)
    points = np.asarray(payload.get("Points"), dtype="float32")
    if points.shape != (68, 2):
        raise ValueError(f"{annotation_mat} Points must have shape (68, 2), got {points.shape}")
    occlusion = payload.get("Occ")
    if occlusion is None:
        return points, []
    occ = np.asarray(occlusion).reshape(-1).astype("int32")
    return points, [int(value) for value in occ.tolist()]


def _visibility_from_occlusion(occlusion: T.Sequence[int]) -> list[bool]:
    """Return COFW visibility flags from COFW occlusion flags.

    COFW-68 ``Occ`` uses truthy values for occluded landmarks. Faceswap's
    manifest/evaluation path expects visibility semantics instead, so invert
    the flags before writing the materialized JSON. This lets geometry metrics
    evaluate visible hulls rather than treating occluded COFW points as visible
    mask geometry.
    """
    return [not bool(value) for value in occlusion]


def _cofw_bbox_xywh_to_ltrb(bbox: T.Sequence[float]) -> list[float]:
    """Convert a COFW-68 benchmark bbox from ``x, y, width, height`` to ltrb.

    The upstream ``cofw68_test_bboxes.mat`` stores boxes as xywh. Generic bbox
    heuristics can misread rows such as ``[47, 121, 124, 127]`` as already-ltrb
    because ``124 > 47`` and ``127 > 121``, yielding an impossible 77x6 crop.
    Materialize the bbox as explicit ltrb here so downstream manifest loading is
    unambiguous and prediction caches are built from the correct face crop.
    """
    if len(bbox) < 4:
        raise ValueError(f"COFW bbox must contain at least 4 values, got {bbox!r}")
    x, y, width, height = (float(value) for value in bbox[:4])
    if width <= 0 or height <= 0:
        raise ValueError(f"COFW xywh bbox must have positive width/height, got {bbox!r}")
    return [x, y, x + width, y + height]


def _cofw68_json_current(path: Path) -> bool:
    """Return whether an existing COFW-68 JSON has normalized bbox metadata."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        return False
    saw_benchmark_bbox = False
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        if metadata.get("face_bbox_source") != "cofw68_benchmark":
            continue
        saw_benchmark_bbox = True
        if metadata.get("face_bbox_format") != "ltrb":
            return False
        bbox = metadata.get("face_bbox")
        if not isinstance(bbox, list | tuple) or len(bbox) < 4:
            return False
        left, top, right, bottom = (float(value) for value in bbox[:4])
        if right <= left or bottom <= top:
            return False
    return saw_benchmark_bbox


def _load_bboxes(annotation_root: Path, count: int) -> list[list[float] | None]:
    """Load optional COFW-68 benchmark bboxes.

    Returned rows are raw upstream xywh values. Conversion to ltrb happens when
    materializing per-sample metadata so the raw source value can be preserved.
    """
    matches = sorted(annotation_root.rglob("cofw68_test_bboxes.mat"))
    if not matches:
        return [None] * count
    payload = _load_mat(matches[0])
    raw = payload.get("bboxes")
    if raw is None:
        return [None] * count
    bboxes = np.asarray(raw, dtype="float32")
    if bboxes.ndim != 2 or bboxes.shape[1] < 4:
        raise ValueError(f"COFW-68 bboxes must have shape (N, 4+), got {bboxes.shape}")
    values = [[float(item) for item in row[:4]] for row in bboxes[:count]]
    return values + [None] * max(0, count - len(values))


def build_cofw68_json_from_sources(
    *,
    color_root: str | Path,
    annotation_root: str | Path,
    output_json: str | Path,
    image_output_dir: str | Path | None = None,
) -> Path:
    """Build a ``cofw_68.json`` export from extracted COFW-68 sources."""
    color_root = Path(color_root)
    annotation_root = Path(annotation_root)
    output_json = Path(output_json)
    image_dir = (
        Path(image_output_dir) if image_output_dir is not None else output_json.parent / "images"
    )
    image_dir.mkdir(parents=True, exist_ok=True)

    images = _cofw_images(_find_file(color_root, "COFW_test_color.mat"))
    annotations = _annotation_files(annotation_root)
    if len(images) < len(annotations):
        raise ValueError(
            f"COFW image count ({len(images)}) is smaller than COFW-68 annotation count ({len(annotations)})"
        )
    bboxes = _load_bboxes(annotation_root, len(annotations))
    samples: list[dict[str, T.Any]] = []
    for index, annotation in enumerate(annotations, start=1):
        points, occlusion = _points_and_occlusion(annotation)
        image = images[index - 1]
        image_path = image_dir / f"{index}.png"
        # COFW mats are RGB. OpenCV writes BGR.
        cv2.imwrite(str(image_path), image[:, :, ::-1])
        metadata: dict[str, T.Any] = {
            "image_id": f"{index}.png",
            "source_dataset": "cofw_68",
            "annotation_file": annotation.name,
        }
        entry: dict[str, T.Any] = {
            "sample_id": f"cofw68/{index:04d}",
            "image": str(image_path.resolve()),
            "landmarks": points.astype("float32").tolist(),
            "conditions": {"occlusion": bool(occlusion and any(occlusion))},
            "metadata": metadata,
        }
        if occlusion:
            visibility = _visibility_from_occlusion(occlusion)
            metadata["occlusion"] = occlusion
            metadata["visibility"] = visibility
            entry["visibility"] = visibility
        if bboxes[index - 1] is not None:
            raw_bbox = [float(value) for value in bboxes[index - 1][:4]]
            bbox = _cofw_bbox_xywh_to_ltrb(raw_bbox)
            metadata["face_bbox"] = bbox
            metadata["face_bbox_raw"] = raw_bbox
            metadata["face_bbox_format"] = "ltrb"
            metadata["face_bbox_raw_format"] = "xywh"
            metadata["face_bbox_source"] = "cofw68_benchmark"
            entry["face_bbox"] = bbox
        samples.append(entry)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"dataset": "cofw_68", "samples": samples}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote COFW-68 JSON export: %s entries=%d", output_json, len(samples))
    return output_json


def resolve_cofw68_json(
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    no_download: bool = False,
) -> Path:
    """Resolve or build the cached COFW-68 JSON export."""
    cache_root = Path(cache_dir) / "cofw"
    output_json = cache_root / COFW68_JSON_NAME
    if output_json.is_file() and not force_download:
        if _cofw68_json_current(output_json):
            logger.info("Using cached COFW-68 JSON export: %s", output_json)
            return output_json
        logger.info("Rebuilding stale COFW-68 JSON export with normalized bbox metadata: %s", output_json)
    if no_download:
        raise FileNotFoundError(
            f"COFW-68 JSON export not found or stale in {cache_root}. Download disabled by --no-download. "
            "Provide --cofw-json or rebuild .fs_cache/landmark_quality/cofw/cofw_68.json with normalized bboxes."
        )
    color_archive = download(
        COFW_COLOR_URL,
        cache_root / "COFW_color.zip",
        force=force_download,
        label="COFW color images",
    )
    annotations_archive = download(
        COFW68_ANNOTATIONS_URL,
        cache_root / "cofw68-benchmark-master.zip",
        force=force_download,
        label="COFW-68 annotations",
    )
    color_root = extract_archive_to_cache(
        color_archive,
        cache_root / "color" / "extracted",
        force=force_download,
        label="COFW color archive",
    )
    annotation_root = extract_archive_to_cache(
        annotations_archive,
        cache_root / "cofw68-benchmark" / "extracted",
        force=force_download,
        label="COFW-68 annotation archive",
    )
    if output_json.exists():
        output_json.unlink()
    image_output_dir = cache_root / "images"
    if force_download and image_output_dir.exists():
        shutil.rmtree(image_output_dir)
    return build_cofw68_json_from_sources(
        color_root=color_root,
        annotation_root=annotation_root,
        output_json=output_json,
        image_output_dir=image_output_dir,
    )
