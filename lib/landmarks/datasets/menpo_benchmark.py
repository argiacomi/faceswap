#!/usr/bin/env python3
"""MenpoBenchmark-style dataset manifest helpers.

Supports the MenpoBenchmark Google Drive packages for Menpo2D and MultiPIE.
The parser is intentionally permissive because the packages are folder-based
benchmark dumps rather than a single stable JSON schema.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.core.schema import normalize_landmarks
from lib.landmarks.datasets import (
    DEFAULT_INTEROCULAR_NORMALIZER_SOURCE,
    _explicit_scenario_groups,
    _filter_samples,
    _normalize_condition_label,
    _write_manifest_and_audit,
)
from lib.landmarks.datasets.sources import (
    DEFAULT_CACHE_DIR,
    DatasetSourceSpec,
    extract_archive_to_temp,
    resolve_dataset_source,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ANNO_EXTS = (".pts", ".txt", ".npy", ".npz", ".json")
PROFILE_TOKENS = ("profile", "39")
SEMIFRONTAL_TOKENS = ("semifrontal", "semi_frontal", "semi-frontal", "68")
LEFT_TOKENS = ("left", "_l_", "-l-", "yaw_left")
RIGHT_TOKENS = ("right", "_r_", "-r-", "yaw_right")


def _source_root(source: Path) -> contextlib.AbstractContextManager[Path]:
    @contextlib.contextmanager
    def _ctx() -> T.Iterator[Path]:
        if source.is_dir():
            yield source
        else:
            with extract_archive_to_temp(source) as root:
                yield root

    return _ctx()


def _read_pts(path: Path) -> np.ndarray:
    values: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(("version:", "n_points:", "{", "}")):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                values.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    if not values:
        raise ValueError(f"no landmark rows found in {path}")
    return T.cast(np.ndarray, np.asarray(values, dtype="float32"))


def _read_txt(path: Path) -> np.ndarray:
    vals: list[float] = []
    for token in path.read_text(encoding="utf-8", errors="ignore").replace(",", " ").split():
        with contextlib.suppress(ValueError):
            vals.append(float(token))
    arr = np.asarray(vals, dtype="float32")
    if arr.size % 3 == 0 and arr.size // 3 in {39, 68}:
        return T.cast(np.ndarray, arr.reshape((-1, 3))[:, :2].astype("float32", copy=False))
    if arr.size % 2 == 0 and arr.size // 2 in {39, 68}:
        return T.cast(np.ndarray, arr.reshape((-1, 2)).astype("float32", copy=False))
    raise ValueError(f"unsupported txt landmark shape in {path}: {arr.size} values")


def _read_json(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("landmarks", "points", "pts", "ground_truth"):
            if key in payload:
                payload = payload[key]
                break
    arr = np.asarray(payload, dtype="float32")
    if arr.ndim == 1:
        if arr.size % 3 == 0:
            arr = arr.reshape((-1, 3))[:, :2]
        elif arr.size % 2 == 0:
            arr = arr.reshape((-1, 2))
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return T.cast(np.ndarray, arr[:, :2].astype("float32", copy=False))
    raise ValueError(f"unsupported json landmark shape in {path}: {arr.shape}")


def _read_landmarks(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".pts":
        return _read_pts(path)
    if suffix == ".txt":
        return _read_txt(path)
    if suffix == ".npy":
        arr = np.load(path).astype("float32")
        if arr.ndim == 1:
            arr = arr.reshape((-1, 2))
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return T.cast(np.ndarray, arr[:, :2])
    if suffix == ".npz":
        data = np.load(path)
        for key in ("landmarks", "points", "pts", "arr_0"):
            if key in data:
                arr = np.asarray(data[key], dtype="float32")
                if arr.ndim == 1:
                    arr = arr.reshape((-1, 2))
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return T.cast(np.ndarray, arr[:, :2])
    if suffix == ".json":
        return _read_json(path)
    raise ValueError(f"unsupported landmark annotation: {path}")


def _image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for image in root.rglob("*"):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
            index.setdefault(image.stem.lower(), []).append(image)
    return index


def _matching_image(annotation: Path, images: dict[str, list[Path]]) -> Path | None:
    stem = annotation.stem.lower()
    candidates = images.get(stem)
    if candidates:
        return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]

    # Common Menpo-style annotations sometimes include suffixes like
    # image_001_lms or image_001_68.
    reduced = re_sub_suffix(stem)
    candidates = images.get(reduced)
    if candidates:
        return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]
    return None


def re_sub_suffix(stem: str) -> str:
    for suffix in ("_lms", "_landmarks", "_pts", "_68", "_39"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _bbox_diag(points: np.ndarray) -> float:
    lt = np.min(points, axis=0)
    rb = np.max(points, axis=0)
    diag = float(np.linalg.norm(rb - lt))
    if math.isfinite(diag) and diag > 0:
        return diag
    raise ValueError("invalid bbox normalizer")


def _normalizer(points: np.ndarray, sample_id: str) -> tuple[float, str]:
    if points.shape == (68, 2):
        canonical = normalize_landmarks(points, source_schema="2d_68")
        value = float(np.linalg.norm(canonical[36] - canonical[45]))
        if math.isfinite(value) and value > 0:
            return value, DEFAULT_INTEROCULAR_NORMALIZER_SOURCE
    return _bbox_diag(points), "landmark_bbox_diagonal"


def _pose_group(path: Path, points: np.ndarray) -> str:
    text = path.as_posix().lower()
    if points.shape[0] == 39 or any(tok in text for tok in PROFILE_TOKENS):
        return "profile"
    if any(tok in text for tok in SEMIFRONTAL_TOKENS):
        return "semifrontal"
    return "semifrontal" if points.shape[0] == 68 else "profile"


def _yaw_side(path: Path) -> str | None:
    text = path.as_posix().lower()
    if any(tok in text for tok in LEFT_TOKENS):
        return "left"
    if any(tok in text for tok in RIGHT_TOKENS):
        return "right"
    return None


def _conditions(
    dataset: str, annotation: Path, points: np.ndarray, scenario: str
) -> tuple[str, ...]:
    labels: list[str] = []
    pose = _pose_group(annotation, points)
    labels.append(pose)
    if pose == "profile":
        labels.extend(("self_occlusion", "single_eye_visible"))
    side = _yaw_side(annotation)
    if side and pose == "profile":
        labels.append(f"profile_{side}")
    elif side:
        labels.append(f"large_yaw_{side}")
    return tuple(dict.fromkeys(_normalize_condition_label(x) for x in labels if x)) or (scenario,)


def _find_annotations(root: Path) -> list[Path]:
    annotations = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in ANNO_EXTS
        and p.name not in {"manifest.json", "dataset_audit.json", "metrics.json"}
        and not p.name.startswith(".")
    ]
    return sorted(annotations)


def _points_for_manifest(points: np.ndarray) -> tuple[np.ndarray, str]:
    if points.shape == (68, 2):
        return normalize_landmarks(points, source_schema="2d_68"), "2d_68"
    if points.shape == (39, 2):
        return T.cast(np.ndarray, np.ascontiguousarray(points, dtype="float32")), "2d_39"
    raise ValueError(f"expected 68 or 39 landmarks, got {points.shape}")


def build_menpo_benchmark_manifest(
    *,
    dataset_name: str,
    source_spec: DatasetSourceSpec,
    output_dir: str | Path,
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
    include_39pt_profile: bool = False,
) -> Path:
    """Build a canonical manifest from a MenpoBenchmark-style package."""
    resolved = resolve_dataset_source(
        source_spec,
        cache_dir=cache_dir,
        source_dir=source_dir,
        source_zip=source_zip,
        download_url=download_url,
        force_download=force_download,
        no_download=no_download,
    )
    scenario_groups = _explicit_scenario_groups(scenarios)
    with _source_root(resolved) as root:
        images = _image_index(root)
        samples: list[dict[str, T.Any]] = []
        for annotation in _find_annotations(root):
            try:
                raw_points = _read_landmarks(annotation)
                points, source_schema = _points_for_manifest(raw_points)
            except Exception:
                continue
            if source_schema == "2d_39" and not include_39pt_profile:
                continue
            image = _matching_image(annotation, images)
            if image is None:
                continue

            rel = annotation.relative_to(root).with_suffix("").as_posix()
            sample_id = rel
            normalizer, normalizer_source = _normalizer(points, sample_id)
            labels = _conditions(dataset_name, annotation, points, scenario)
            metadata: dict[str, T.Any] = {
                "source_landmark_count": int(points.shape[0]),
                "source_annotation": str(annotation.resolve()),
                "normalizer_source": normalizer_source,
                "menpo_benchmark_pose_group": _pose_group(annotation, points),
            }
            side = _yaw_side(annotation)
            if side:
                metadata["yaw_side"] = side

            samples.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset_name,
                    "condition": labels[0],
                    "conditions": labels,
                    "image": str(image.resolve()),
                    "source_schema": source_schema,
                    "source": {"dataset": dataset_name, "source_id": rel},
                    "metadata": metadata,
                    "normalizer": normalizer,
                    "points": points,
                }
            )

        if not samples:
            raise FileNotFoundError(
                f"No usable {dataset_name} MenpoBenchmark image/landmark pairs found under {root}. "
                "Expected 68-point or 39-point .pts/.txt/.npy/.npz/.json annotations with matching images."
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
