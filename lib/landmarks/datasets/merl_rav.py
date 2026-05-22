#!/usr/bin/env python3
"""MERL-RAV dataset manifest builder.

MERL-RAV provides 68-point reannotations over AFLW images. Manifests are built
by translating MERL-RAV 68-point annotations into AFLW release-2 cropped-image
coordinates using a per-image crop origin recovered from 5 robust anchors
derived from the 68 landmarks compared against the AFLW 5-point keypoints in
``aflw_*_keypoints.mat``.

MERL-RAV label semantics:

* positive ``x y``: visible landmark
* negative ``-x -y``: externally occluded, estimated at ``abs(x), abs(y)``
* ``-1 -1``: self-occluded, location not estimated

For the AFLW release-2 crop manifest path all negative coordinates are treated
as invalid: they are masked out instead of forced into the crop with ``abs()``.
The per-landmark validity is preserved alongside the translated landmarks so
downstream consumers can score only the source-valid, in-crop positions.
"""

from __future__ import annotations

import contextlib
import logging
import re
import typing as T
from pathlib import Path

import numpy as np

from lib.landmarks.datasets import (
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

logger = logging.getLogger(__name__)

MERL_RAV_LABELS_URL = (
    "https://github.com/abhi1kumar/MERL-RAV_dataset/archive/refs/heads/master.zip"
)

DEFAULT_AFLW_RELEASE2_DIR = Path(
    ".fs_cache/landmark_quality/aflw/extracted/data/aflw/aflw_release-2"
)

MERL_RAV_SOURCE = DatasetSourceSpec(
    dataset="MERL-RAV",
    cache_subdir="merl-rav",
    canonical_archive="MERL-RAV_dataset-master.zip",
    cache_aliases=("merl-rav.zip", "MERL-RAV.zip", "MERL_RAV.zip"),
    extracted_aliases=("merl_rav_organized", "MERL-RAV_dataset-master", "MERL-RAV", "MERL_RAV"),
    url=MERL_RAV_LABELS_URL,
    manual_hint=(
        "MERL-RAV labels default to the MERL-RAV GitHub archive. Manifests are "
        "produced by translating 68-point reannotations into AFLW release-2 "
        "cropped image coordinates; provide the AFLW release-2 cropped dataset "
        "under --aflw-release2-dir or at "
        f"{DEFAULT_AFLW_RELEASE2_DIR}."
    ),
)

AFLW_RELEASE2_SPLITS: tuple[tuple[str, str, str], ...] = (
    ("train", "aflw_train_images.txt", "aflw_train_keypoints.mat"),
    ("test", "aflw_test_images.txt", "aflw_test_keypoints.mat"),
)

_AFLW_SOURCE_STEM_RE = re.compile(r"(image\d+)", re.IGNORECASE)


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
    return sorted(path for path in root.rglob("*.pts") if path.is_file())


def _parse_pts_signed(path: Path) -> np.ndarray:
    """Parse a MERL-RAV ``.pts`` file preserving signed coordinates."""
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
    return np.asarray(rows, dtype=np.float64)  # type: ignore[no-any-return]


def _visibility_for_crop(
    signed_xy: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Return per-landmark visibility labels plus source points (NaN where invalid).

    For the AFLW release-2 crop manifest path, negative coordinates (both the
    ``-1 -1`` self-occluded sentinel and externally-occluded ``-x -y`` pairs)
    are treated as invalid/missing. ``abs()`` is not applied: invalid positions
    are masked instead so the per-image crop origin is estimated from only the
    source-valid landmarks.
    """
    visibility: list[str] = []
    points = np.full(signed_xy.shape, np.nan, dtype=np.float64)
    for idx, (x_value, y_value) in enumerate(signed_xy):
        if x_value == -1 and y_value == -1:
            visibility.append("self_occluded")
        elif x_value < 0 or y_value < 0:
            visibility.append("externally_occluded")
        else:
            visibility.append("visible")
            points[idx] = (x_value, y_value)
    return visibility, points


def _source_valid_xy(points_xy: np.ndarray) -> np.ndarray:
    """Return a boolean mask of finite, non-negative landmark positions."""
    mask: np.ndarray = (
        np.isfinite(points_xy).all(axis=1) & (points_xy[:, 0] >= 0) & (points_xy[:, 1] >= 0)
    )
    return mask


def _robust_center(points_xy: np.ndarray) -> np.ndarray:
    """Return the median of valid points, or NaN when no point is valid."""
    valid = _source_valid_xy(points_xy)
    pts = points_xy[valid]
    if len(pts) == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)  # type: ignore[no-any-return]
    return np.asarray(np.median(pts, axis=0), dtype=np.float64)  # type: ignore[no-any-return]


def _landmarks68_to_5anchors_xy(pts68_xy: np.ndarray) -> np.ndarray:
    """Derive the 5-point AFLW-style anchors from 68 landmarks in xy order."""
    stacked: np.ndarray = np.vstack(
        [
            _robust_center(pts68_xy[36:42]),
            _robust_center(pts68_xy[42:48]),
            pts68_xy[30],
            pts68_xy[48],
            pts68_xy[54],
        ]
    ).astype(np.float64)
    return stacked


def _estimate_crop_origin_xy(
    src5_xy: np.ndarray,
    dst5_xy: np.ndarray,
    *,
    min_anchors: int = 3,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Estimate an integer ``(x1, y1)`` crop origin from 5-anchor correspondences."""
    valid = (
        np.isfinite(src5_xy).all(axis=1)
        & np.isfinite(dst5_xy).all(axis=1)
        & (src5_xy[:, 0] >= 0)
        & (src5_xy[:, 1] >= 0)
    )
    used = int(valid.sum())
    if used < min_anchors:
        raise RuntimeError(f"Need at least {min_anchors} valid anchors, got {used}")
    offsets_xy = src5_xy[valid] - dst5_xy[valid]
    origin = np.round(np.median(offsets_xy, axis=0)).astype(int)
    residuals = offsets_xy - origin.astype(np.float64)
    return origin, used, residuals


def _aflw_source_stem(stem: str) -> str | None:
    """Return the AFLW ``imageNNNNN`` token within a MERL-RAV label stem."""
    match = _AFLW_SOURCE_STEM_RE.search(stem)
    return match.group(1).lower() if match else None


def _read_image_size(path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` for an image without decoding pixel data."""
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return height, width


def _load_aflw_release2_split(
    images_txt: Path,
    mat_path: Path,
) -> list[tuple[str, np.ndarray, tuple[int, int]]]:
    """Load AFLW release-2 split rows of ``(relative_path, gt5_xy, (h, w))``."""
    from scipy.io import loadmat

    raw_lines = images_txt.read_text(encoding="utf-8").splitlines()
    lines = [line.strip().replace("\\", "/") for line in raw_lines if line.strip()]
    mat = loadmat(str(mat_path))
    if "gt" not in mat or "hw" not in mat:
        raise ValueError(f"AFLW release-2 .mat missing 'gt'/'hw' keys: {mat_path}")
    gt = np.asarray(mat["gt"])
    hw = np.asarray(mat["hw"])
    if gt.shape[0] != len(lines) or hw.shape[0] != len(lines):
        raise ValueError(
            f"AFLW release-2 row mismatch: {images_txt.name} has {len(lines)} rows but "
            f"gt={gt.shape[0]}, hw={hw.shape[0]} in {mat_path.name}"
        )
    rows: list[tuple[str, np.ndarray, tuple[int, int]]] = []
    for idx, relative in enumerate(lines):
        gt_yx = np.asarray(gt[idx], dtype=np.float64)
        if gt_yx.shape != (5, 2):
            raise ValueError(
                f"AFLW release-2 gt row {idx} has shape {gt_yx.shape}, expected (5, 2): {mat_path}"
            )
        gt5_xy = gt_yx[:, ::-1].copy()
        height = int(hw[idx, 0])
        width = int(hw[idx, 1])
        rows.append((relative, gt5_xy, (height, width)))
    return rows


def _build_aflw_release2_index(
    release2_dir: Path,
    *,
    splits: T.Sequence[str] = ("train", "test"),
) -> tuple[dict[str, list[dict[str, T.Any]]], Path, dict[str, int]]:
    """Build a stem-keyed AFLW release-2 crop index from the keypoint splits."""
    crops_root = release2_dir / "output"
    if not crops_root.is_dir():
        raise FileNotFoundError(
            f"AFLW release-2 cropped image directory not found: {crops_root}. Expected "
            f"output/ alongside aflw_train_keypoints.mat in {release2_dir}."
        )
    requested = tuple(splits)
    index: dict[str, list[dict[str, T.Any]]] = {}
    counts: dict[str, int] = {name: 0 for name in requested}
    loaded_any = False
    for name, images_basename, mat_basename in AFLW_RELEASE2_SPLITS:
        if name not in requested:
            continue
        images_txt = release2_dir / images_basename
        mat_path = release2_dir / mat_basename
        if not images_txt.is_file() or not mat_path.is_file():
            logger.info(
                "AFLW release-2 split '%s' not present in %s; skipping", name, release2_dir
            )
            continue
        for row_idx, (relative, gt5_xy, hw) in enumerate(
            _load_aflw_release2_split(images_txt, mat_path)
        ):
            stem = Path(relative).stem
            source_stem = stem.split("_", 1)[0].lower()
            index.setdefault(source_stem, []).append(
                {
                    "split": name,
                    "row": row_idx,
                    "relative": relative,
                    "absolute": crops_root / relative,
                    "gt5_xy": gt5_xy,
                    "hw": hw,
                    "stem": stem,
                }
            )
            counts[name] += 1
            loaded_any = True
    if not loaded_any:
        raise FileNotFoundError(
            f"No AFLW release-2 keypoint splits found in {release2_dir} for splits={requested}"
        )
    return index, crops_root, counts


def _select_best_crop_candidate(
    src5_xy: np.ndarray,
    candidates: T.Sequence[dict[str, T.Any]],
    *,
    min_anchors: int,
) -> tuple[dict[str, T.Any] | None, np.ndarray, np.ndarray, int]:
    """Choose the AFLW crop whose 5-point keypoints best match the MERL-RAV anchors."""
    best_candidate: dict[str, T.Any] | None = None
    best_origin: np.ndarray = np.zeros(2, dtype=int)
    best_residuals: np.ndarray = np.zeros((0, 2), dtype=np.float64)
    best_used = 0
    best_score = float("inf")
    for candidate in candidates:
        try:
            origin, used, residuals = _estimate_crop_origin_xy(
                src5_xy, candidate["gt5_xy"], min_anchors=min_anchors
            )
        except RuntimeError:
            continue
        score = float(np.median(np.abs(residuals))) if residuals.size else float("inf")
        if score < best_score:
            best_candidate = candidate
            best_origin = origin
            best_residuals = residuals
            best_used = used
            best_score = score
    return best_candidate, best_origin, best_residuals, best_used


def _translate_to_crop(
    src_points: np.ndarray,
    origin_xy: np.ndarray,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate landmarks into crop coords and return the in-crop validity mask."""
    height, width = image_hw
    src_valid = _source_valid_xy(src_points)
    translated = np.full(src_points.shape, np.nan, dtype=np.float64)
    if src_valid.any():
        translated[src_valid] = src_points[src_valid] - origin_xy.astype(np.float64)
    in_crop = src_valid & (
        np.isfinite(translated).all(axis=1)
        & (translated[:, 0] >= 0)
        & (translated[:, 1] >= 0)
        & (translated[:, 0] < float(width))
        & (translated[:, 1] < float(height))
    )
    return translated, src_valid, in_crop


def _bbox_from_valid(points: np.ndarray, mask: np.ndarray, hw: tuple[int, int]) -> list[float]:
    """Return a landmark bbox from in-crop points, falling back to crop extents."""
    height, width = hw
    if not mask.any():
        return [0.0, 0.0, float(width), float(height)]
    valid = points[mask]
    left, top = np.min(valid, axis=0)
    right, bottom = np.max(valid, axis=0)
    return [float(left), float(top), float(right), float(bottom)]


def _sample_from_aflw_crop(
    *,
    annotation: Path,
    label_root: Path,
    crop_entry: dict[str, T.Any],
    src_points: np.ndarray,
    origin_xy: np.ndarray,
    residuals: np.ndarray,
    used_anchors: int,
    visibility: list[str],
    image_hw: tuple[int, int],
) -> dict[str, T.Any]:
    """Build a manifest sample for one MERL-RAV/AFLW release-2 crop pair."""
    translated, src_valid_mask, in_crop_mask = _translate_to_crop(src_points, origin_xy, image_hw)
    relative_annotation = annotation.relative_to(label_root)
    condition_labels = _labels_from_path(relative_annotation)
    if any(value == "externally_occluded" for value in visibility):
        condition_labels = tuple(dict.fromkeys((*condition_labels, "occlusion")))
    label_id = relative_annotation.with_suffix("").as_posix().replace("/", "_")
    sample_id = f"{label_id}__{crop_entry['stem']}"
    residual_abs = np.abs(residuals) if residuals.size else np.zeros((0, 2), dtype=np.float64)
    metadata: dict[str, T.Any] = {
        "image_id": crop_entry["relative"],
        "annotation_file": relative_annotation.as_posix(),
        "visibility": list(visibility),
        "self_occluded_count": sum(1 for value in visibility if value == "self_occluded"),
        "externally_occluded_count": sum(
            1 for value in visibility if value == "externally_occluded"
        ),
        "aflw_release2_split": crop_entry["split"],
        "aflw_release2_row": int(crop_entry["row"]),
        "crop_height": int(image_hw[0]),
        "crop_width": int(image_hw[1]),
        "crop_origin_xy": [int(origin_xy[0]), int(origin_xy[1])],
        "anchor_used_count": used_anchors,
        "anchor_residual_median_xy": [
            float(np.median(residual_abs[:, 0])) if residual_abs.size else 0.0,
            float(np.median(residual_abs[:, 1])) if residual_abs.size else 0.0,
        ],
        "anchor_residual_max_xy": [
            float(np.max(residual_abs[:, 0])) if residual_abs.size else 0.0,
            float(np.max(residual_abs[:, 1])) if residual_abs.size else 0.0,
        ],
        "landmark_source_valid_mask": [bool(value) for value in src_valid_mask.tolist()],
        "landmark_in_crop_mask": [bool(value) for value in in_crop_mask.tolist()],
        "landmark_source_valid_count": int(src_valid_mask.sum()),
        "landmark_in_crop_count": int(in_crop_mask.sum()),
        "face_bbox": _bbox_from_valid(translated, in_crop_mask, image_hw),
        "face_bbox_source": "merl_rav_aflw_release2_crop_translated_landmarks",
        "aflw_image_source": "aflw_release2_cropped",
    }
    # Saved landmarks must remain finite; invalid positions are zeroed and the
    # validity is preserved in the per-landmark mask above.
    finite_landmarks = np.where(np.isfinite(translated), translated, 0.0).astype("float32")
    return {
        "sample_id": sample_id,
        "dataset": "merl-rav",
        "condition": condition_labels[0],
        "conditions": condition_labels,
        "image": str(crop_entry["absolute"].resolve()),
        "source_schema": "2d_68",
        "source": {"dataset": "merl-rav-aflw-release2", "source_id": sample_id},
        "metadata": metadata,
        "points": finite_landmarks,
    }


def _build_samples_aflw_release2(
    label_root: Path,
    release2_dir: Path,
    *,
    splits: T.Sequence[str] = ("train", "test"),
    min_anchors: int = 3,
    validate_hw: bool = True,
    hw_tolerance: int = 0,
) -> tuple[list[dict[str, T.Any]], dict[str, T.Any]]:
    """Translate MERL-RAV labels into AFLW release-2 crop samples plus an audit."""
    index, _crops_root, split_counts = _build_aflw_release2_index(release2_dir, splits=splits)
    samples: list[dict[str, T.Any]] = []
    stats: dict[str, T.Any] = {
        "labels": 0,
        "matched": 0,
        "skipped_no_candidate": 0,
        "skipped_no_origin": 0,
        "skipped_missing_image": 0,
        "skipped_bad_hw": 0,
        "skipped_all_outside_crop": 0,
        "total_source_valid_landmarks": 0,
        "total_in_crop_landmarks": 0,
        "residual_medians": [],
        "aflw_release2_split_counts": split_counts,
    }
    for annotation in _label_files(label_root):
        stats["labels"] += 1
        source_stem = _aflw_source_stem(annotation.stem)
        candidates = index.get(source_stem, []) if source_stem else []
        if not candidates:
            stats["skipped_no_candidate"] += 1
            continue
        signed = _parse_pts_signed(annotation)
        visibility, src_points = _visibility_for_crop(signed)
        src5_xy = _landmarks68_to_5anchors_xy(src_points)
        chosen, origin, residuals, used = _select_best_crop_candidate(
            src5_xy, candidates, min_anchors=min_anchors
        )
        if chosen is None:
            stats["skipped_no_origin"] += 1
            continue
        if not chosen["absolute"].is_file():
            stats["skipped_missing_image"] += 1
            continue
        expected_h, expected_w = chosen["hw"]
        image_hw: tuple[int, int] = (expected_h, expected_w)
        if validate_hw:
            try:
                actual_hw = _read_image_size(chosen["absolute"])
            except OSError:
                stats["skipped_missing_image"] += 1
                continue
            if (
                abs(actual_hw[0] - expected_h) > hw_tolerance
                or abs(actual_hw[1] - expected_w) > hw_tolerance
            ):
                logger.warning(
                    "AFLW release-2 hw mismatch for %s: mat=%s actual=%s",
                    chosen["absolute"],
                    (expected_h, expected_w),
                    actual_hw,
                )
                stats["skipped_bad_hw"] += 1
                continue
            image_hw = actual_hw
        sample = _sample_from_aflw_crop(
            annotation=annotation,
            label_root=label_root,
            crop_entry=chosen,
            src_points=src_points,
            origin_xy=origin,
            residuals=residuals,
            used_anchors=used,
            visibility=visibility,
            image_hw=image_hw,
        )
        if sample["metadata"]["landmark_in_crop_count"] == 0:
            stats["skipped_all_outside_crop"] += 1
            continue
        samples.append(sample)
        stats["matched"] += 1
        stats["total_source_valid_landmarks"] += int(
            sample["metadata"]["landmark_source_valid_count"]
        )
        stats["total_in_crop_landmarks"] += int(sample["metadata"]["landmark_in_crop_count"])
        stats["residual_medians"].append(sample["metadata"]["anchor_residual_median_xy"])
    return samples, stats


def _log_aflw_release2_audit(stats: dict[str, T.Any]) -> None:
    """Emit a structured audit log for the AFLW release-2 crop builder."""
    summary = {
        key: value
        for key, value in stats.items()
        if key
        not in {
            "residual_medians",
            "aflw_release2_split_counts",
            "total_source_valid_landmarks",
            "total_in_crop_landmarks",
        }
    }
    summary["split_counts"] = stats.get("aflw_release2_split_counts", {})
    summary["total_source_valid_landmarks"] = stats.get("total_source_valid_landmarks", 0)
    summary["total_in_crop_landmarks"] = stats.get("total_in_crop_landmarks", 0)
    residuals = stats.get("residual_medians") or []
    if residuals:
        arr = np.asarray(residuals, dtype=np.float64)
        summary["residual_median_xy"] = [
            float(np.median(arr[:, 0])),
            float(np.median(arr[:, 1])),
        ]
        summary["residual_p95_xy"] = [
            float(np.percentile(arr[:, 0], 95)),
            float(np.percentile(arr[:, 1], 95)),
        ]
    logger.info("MERL-RAV AFLW release-2 audit: %s", summary)


def build_merl_rav_manifest(
    output_dir: str | Path,
    *,
    aflw_release2_dir: str | Path | None = None,
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
    splits: T.Sequence[str] = ("train", "test"),
    min_anchors: int = 3,
    validate_hw: bool = True,
) -> Path:
    """Build a MERL-RAV manifest aligned with AFLW release-2 cropped images.

    MERL-RAV 68-point reannotations are matched to AFLW release-2 crops via the
    stem ``imageNNNNN``. Per-image crop origins ``(x1, y1)`` are recovered by
    comparing 5 robust anchors derived from the 68-point annotation against the
    AFLW 5-point keypoints in ``mat['gt']`` and translating every valid
    landmark by the same global origin. Negative MERL-RAV coordinates are
    treated as invalid (no ``abs()`` translation); per-landmark validity is
    written into the sample metadata so downstream scoring can mask invalid
    positions instead of evaluating against forced coordinates. Saved
    ``.npy`` landmarks remain finite (invalid positions are zeroed).
    """
    release2_dir = (
        Path(aflw_release2_dir) if aflw_release2_dir is not None else DEFAULT_AFLW_RELEASE2_DIR
    )
    if not release2_dir.is_dir():
        raise FileNotFoundError(
            f"AFLW release-2 directory not found: {release2_dir}. Expected output/ "
            f"and aflw_train_keypoints.mat/aflw_train_images.txt inside it. Pass "
            f"--aflw-release2-dir or stage the dataset at "
            f"{DEFAULT_AFLW_RELEASE2_DIR}."
        )
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
        scenario_groups = _explicit_scenario_groups(scenarios)
        all_samples: list[dict[str, T.Any]] = []
        aggregate: dict[str, T.Any] = {
            "labels": 0,
            "matched": 0,
            "skipped_no_candidate": 0,
            "skipped_no_origin": 0,
            "skipped_missing_image": 0,
            "skipped_bad_hw": 0,
            "skipped_all_outside_crop": 0,
            "total_source_valid_landmarks": 0,
            "total_in_crop_landmarks": 0,
            "residual_medians": [],
            "aflw_release2_split_counts": {},
        }
        for label_root in _find_label_roots(root):
            samples, stats = _build_samples_aflw_release2(
                label_root,
                release2_dir,
                splits=splits,
                min_anchors=min_anchors,
                validate_hw=validate_hw,
            )
            all_samples.extend(samples)
            for key, value in stats.items():
                if key == "residual_medians":
                    aggregate[key].extend(value)
                elif key == "aflw_release2_split_counts":
                    for split_name, count in value.items():
                        aggregate[key][split_name] = aggregate[key].get(split_name, 0) + count
                else:
                    aggregate[key] += value
        _log_aflw_release2_audit(aggregate)
        if not all_samples:
            detail = " ".join(
                f"{key}={value}"
                for key, value in aggregate.items()
                if key not in {"residual_medians", "aflw_release2_split_counts"}
            )
            raise FileNotFoundError(
                f"No MERL-RAV/AFLW release-2 crop pairs produced from {root}. {detail}"
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
    finally:
        if cleanup is not None:
            cleanup.__exit__(None, None, None)
