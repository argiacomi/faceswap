#!/usr/bin/env python3
"""MERL-RAV dataset manifest builder.

MERL-RAV provides 68-point reannotations over AFLW images. Manifests are built
by translating MERL-RAV 68-point annotations into AFLW release-2 cropped-image
coordinates using a per-image crop origin recovered from 5 robust anchors
derived from the 68 landmarks compared against the AFLW 5-point keypoints in
``aflw_*_keypoints.mat``.

MERL-RAV label semantics:

* positive ``x y``: visible landmark
* negative ``-x -y``: externally occluded landmark estimated at ``abs(x), abs(y)``
* ``-1 -1``: self-occluded landmark with no estimated location

For the AFLW release-2 crop manifest path, externally occluded points are kept
as coordinate-valid estimated locations. Only ``-1 -1`` points are treated as
coordinate-invalid. Coordinate validity and scoring visibility are tracked
separately so downstream evaluation can mask landmarks according to policy
without discarding useful crop/bbox/normalizer geometry.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import tempfile
import typing as T
import zipfile
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
    download,
    is_archive,
    resolve_dataset_source,
    safe_zip_extractall,
)

logger = logging.getLogger(__name__)

MERL_RAV_LABELS_URL = (
    "https://github.com/abhi1kumar/MERL-RAV_dataset/archive/refs/heads/master.zip"
)

DEFAULT_AFLW_RELEASE2_DIR = Path(
    ".fs_cache/landmark_quality/aflw/extracted/data/aflw/aflw_release-2"
)

AFLW_GOOGLE_DRIVE_FILE_ID = "1uSx5hTxkxm48a3No0xm26DeJKpIooqrx"
AFLW_GOOGLE_DRIVE_VIEW_URL = (
    "https://drive.google.com/file/d/1uSx5hTxkxm48a3No0xm26DeJKpIooqrx/view"
)
AFLW_GOOGLE_DRIVE_DIRECT_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1uSx5hTxkxm48a3No0xm26DeJKpIooqrx&export=download&authuser=0"
)
AFLW_ARCHIVE_NAME = "AFLW.zip"
AFLW_CACHE_SUBDIR = "aflw"
DEFAULT_AFLW_DIR = Path(".fs_cache/landmark_quality/aflw/aflw")
AFLW_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


MERL_RAV_SOURCE = DatasetSourceSpec(
    dataset="MERL-RAV",
    cache_subdir="merl-rav",
    canonical_archive="MERL-RAV_dataset-master.zip",
    cache_aliases=("merl-rav.zip", "MERL-RAV.zip", "MERL_RAV.zip"),
    extracted_aliases=("merl_rav_organized", "MERL-RAV_dataset-master", "MERL-RAV", "MERL_RAV"),
    url=MERL_RAV_LABELS_URL,
    manual_hint=(
        "MERL-RAV labels default to the MERL-RAV GitHub archive. Native AFLW mode "
        "matches MERL-RAV 68-point annotations directly to AFLW flickr images by "
        "imageNNNNN and uses "
        f"{DEFAULT_AFLW_DIR} by default. Provide --aflw-image-root for a custom "
        "native AFLW root, or use --merl-rav-coordinate-space=aflw-release2 with "
        "--aflw-release2-dir for the older cropped release-2 translation path."
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
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return labels, coordinate-valid points, and default score visibility.

    MERL-RAV signed-coordinate semantics:

    * positive ``x y``: visible landmark with directly usable coordinate
    * negative ``-x -y``: externally occluded landmark estimated at ``abs(x), abs(y)``
    * ``-1 -1``: self-occluded landmark with no usable coordinate

    Externally occluded landmarks are coordinate-valid, but excluded from the
    default scoring visibility mask.
    """
    visibility: list[str] = []
    points = np.full(signed_xy.shape, np.nan, dtype=np.float64)
    score_visible = np.zeros((signed_xy.shape[0],), dtype=bool)

    for idx, (x_value, y_value) in enumerate(signed_xy):
        if x_value == -1 and y_value == -1:
            visibility.append("self_occluded")
            continue

        if x_value < 0 or y_value < 0:
            visibility.append("externally_occluded")
            points[idx] = (abs(x_value), abs(y_value))
            score_visible[idx] = False
            continue

        visibility.append("visible")
        points[idx] = (x_value, y_value)
        score_visible[idx] = True

    return visibility, points, score_visible


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
    score_visible: np.ndarray,
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

    coordinate_valid_in_crop = in_crop_mask
    score_visibility = coordinate_valid_in_crop & score_visible

    face_bbox = _bbox_from_valid(translated, coordinate_valid_in_crop, image_hw)
    if not all(np.isfinite(value) for value in face_bbox):
        raise ValueError(f"MERL-RAV sample {sample_id} has non-finite face_bbox={face_bbox!r}")

    left, top, right, bottom = face_bbox
    bbox_width = float(right - left)
    bbox_height = float(bottom - top)
    normalizer = max(bbox_width, bbox_height)

    if not np.isfinite(normalizer) or normalizer <= 0.0:
        normalizer = float(max(image_hw[1], image_hw[0]))

    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError(
            f"MERL-RAV sample {sample_id} has invalid fallback normalizer={normalizer!r}"
        )

    top_level_visibility = [bool(value) for value in score_visibility.tolist()]
    coordinate_valid_mask = [bool(value) for value in coordinate_valid_in_crop.tolist()]
    source_valid_mask = [bool(value) for value in src_valid_mask.tolist()]

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
        "landmark_source_valid_mask": source_valid_mask,
        "landmark_in_crop_mask": coordinate_valid_mask,
        "landmark_coordinate_valid_mask": coordinate_valid_mask,
        "landmark_score_visibility_mask": top_level_visibility,
        "landmark_source_valid_count": int(src_valid_mask.sum()),
        "landmark_in_crop_count": int(coordinate_valid_in_crop.sum()),
        "landmark_score_visible_count": int(score_visibility.sum()),
        "face_bbox": face_bbox,
        "face_bbox_source": "merl_rav_aflw_release2_crop_translated_landmarks",
        "normalizer_source": "merl_rav_coordinate_valid_landmark_bbox_max_side",
        "aflw_image_source": "aflw_release2_cropped",
    }

    finite_landmarks = np.where(np.isfinite(translated), translated, 0.0).astype("float32")

    return {
        "sample_id": sample_id,
        "dataset": "merl-rav",
        "condition": condition_labels[0],
        "conditions": condition_labels,
        "image": str(crop_entry["absolute"].resolve()),
        "source_schema": "2d_68",
        "source": {"dataset": "merl-rav-aflw-release2", "source_id": sample_id},
        "normalizer": float(normalizer),
        "visibility": top_level_visibility,
        "metadata": metadata,
        "points": finite_landmarks,
    }


def _aflw_cache_root(cache_dir: str | Path) -> Path:
    """Return the AFLW cache root that stores AFLW.zip and extracted AFLW data."""
    return Path(cache_dir) / AFLW_CACHE_SUBDIR


def _find_aflw_native_root(root: Path) -> Path:
    """Return the AFLW root containing flickr/."""
    root = Path(root)
    if (root / "flickr").is_dir():
        return root

    nested = root / "aflw"
    if (nested / "flickr").is_dir():
        return nested

    candidates = sorted(path.parent for path in root.rglob("flickr") if path.is_dir())
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"AFLW root not found below {root}. Expected a directory containing flickr/."
    )


def _validate_aflw_archive(archive: Path) -> Path:
    """Return a valid AFLW zip archive or raise a clear cache/download error."""
    if not archive.is_file():
        raise FileNotFoundError(f"AFLW archive not found: {archive}")
    if not zipfile.is_zipfile(archive):
        raise ValueError(
            f"Invalid AFLW zip archive: {archive}. Remove the cached file and retry, "
            "or provide a valid AFLW.zip / --aflw-image-root."
        )
    return archive


def _download_aflw_archive(
    cache_dir: str | Path,
    *,
    force_download: bool,
) -> Path:
    """Download AFLW.zip into .fs_cache/landmark_quality/aflw."""
    cache_root = _aflw_cache_root(cache_dir)
    archive = cache_root / AFLW_ARCHIVE_NAME

    try:
        candidate = download(
            None,
            archive,
            force=force_download,
            google_drive_file_id=AFLW_GOOGLE_DRIVE_FILE_ID,
            label="AFLW archive",
        )
        return _validate_aflw_archive(candidate)
    except Exception as err:
        logger.warning(
            "AFLW Google Drive id download failed or produced an invalid zip, "
            "retrying direct download URL: %s",
            err,
        )
        if archive.exists():
            archive.unlink()

        candidate = download(
            AFLW_GOOGLE_DRIVE_DIRECT_URL,
            archive,
            force=True,
            google_drive_file_id=None,
            label="AFLW archive",
        )
        return _validate_aflw_archive(candidate)


def _extract_aflw_archive_to_native_root(
    archive: Path,
    cache_dir: str | Path,
    *,
    force_extract: bool,
) -> Path:
    """Extract AFLW.zip so the usable root is .fs_cache/landmark_quality/aflw/aflw."""
    cache_root = _aflw_cache_root(cache_dir)
    aflw_root = cache_root / "aflw"

    if not force_extract and (aflw_root / "flickr").is_dir():
        return aflw_root

    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="aflw.", suffix=".part", dir=cache_root))
    moved_tmp = False

    try:
        archive = _validate_aflw_archive(archive)
        with zipfile.ZipFile(archive, "r") as zf:
            safe_zip_extractall(zf, tmp_dir)

        extracted_root = _find_aflw_native_root(tmp_dir)

        if aflw_root.exists():
            if aflw_root.is_dir():
                shutil.rmtree(aflw_root)
            else:
                aflw_root.unlink()

        if extracted_root == tmp_dir:
            os.replace(tmp_dir, aflw_root)
            moved_tmp = True
        else:
            os.replace(extracted_root, aflw_root)

        if not (aflw_root / "flickr").is_dir():
            raise FileNotFoundError(f"Extracted AFLW root missing flickr/: {aflw_root}")

        return aflw_root
    finally:
        if not moved_tmp and tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def resolve_aflw_native_root(
    aflw_image_root: str | Path | None,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    no_download: bool = False,
) -> Path:
    """Resolve the native AFLW image root used by MERL-RAV image matching."""
    if aflw_image_root is not None:
        return _find_aflw_native_root(Path(aflw_image_root))

    cache_root = _aflw_cache_root(cache_dir)
    aflw_root = cache_root / "aflw"

    if not force_download and (aflw_root / "flickr").is_dir():
        return aflw_root

    archive = cache_root / AFLW_ARCHIVE_NAME
    if not archive.is_file() or force_download:
        if no_download:
            raise FileNotFoundError(
                f"AFLW native source not found at {aflw_root} and download is disabled. "
                f"Place {AFLW_ARCHIVE_NAME} in {cache_root} or pass --aflw-image-root. "
                f"Google Drive source: {AFLW_GOOGLE_DRIVE_VIEW_URL}"
            )
        archive = _download_aflw_archive(cache_dir, force_download=force_download)

    return _extract_aflw_archive_to_native_root(
        archive,
        cache_dir,
        force_extract=force_download,
    )


def _build_aflw_native_image_index(aflw_root: Path) -> dict[str, Path]:
    """Build an imageNNNNN keyed index over AFLW flickr images."""
    flickr_root = aflw_root / "flickr"
    if not flickr_root.is_dir():
        raise FileNotFoundError(f"AFLW flickr directory not found: {flickr_root}")

    index: dict[str, Path] = {}
    duplicates = 0

    for image in sorted(flickr_root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in AFLW_IMAGE_EXTS:
            continue

        stem = _aflw_source_stem(image.stem)
        if stem is None:
            continue

        if stem in index:
            duplicates += 1
            continue

        index[stem] = image

    if not index:
        raise FileNotFoundError(f"No AFLW images found below {flickr_root}")

    if duplicates:
        logger.warning("Ignored %d duplicate AFLW image stems below %s", duplicates, flickr_root)

    return index


def _in_image_mask(points_xy: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Return coordinate-valid landmarks that fall inside the native AFLW image."""
    height, width = image_hw
    source_valid = _source_valid_xy(points_xy)
    mask: np.ndarray = source_valid & (
        np.isfinite(points_xy).all(axis=1)
        & (points_xy[:, 0] >= 0)
        & (points_xy[:, 1] >= 0)
        & (points_xy[:, 0] < float(width))
        & (points_xy[:, 1] < float(height))
    )
    return mask


def _relative_to_or_absolute(path: Path, root: Path) -> str:
    """Return a stable relative path where possible."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _sample_from_native_aflw_image(
    *,
    annotation: Path,
    label_root: Path,
    image_path: Path,
    aflw_root: Path,
    src_points: np.ndarray,
    visibility: list[str],
    score_visible: np.ndarray,
    image_hw: tuple[int, int],
) -> dict[str, T.Any]:
    """Build a manifest sample for one MERL-RAV annotation matched to one AFLW image."""
    relative_annotation = annotation.relative_to(label_root)
    condition_labels = _labels_from_path(relative_annotation)
    if any(value == "externally_occluded" for value in visibility):
        condition_labels = tuple(dict.fromkeys((*condition_labels, "occlusion")))

    label_id = relative_annotation.with_suffix("").as_posix().replace("/", "_")
    sample_id = f"{label_id}__{image_path.stem}"

    source_valid_mask = _source_valid_xy(src_points)
    coordinate_valid_in_image = _in_image_mask(src_points, image_hw)
    score_visibility = coordinate_valid_in_image & score_visible

    face_bbox = _bbox_from_valid(src_points, coordinate_valid_in_image, image_hw)
    if not all(np.isfinite(value) for value in face_bbox):
        raise ValueError(f"MERL-RAV sample {sample_id} has non-finite face_bbox={face_bbox!r}")

    left, top, right, bottom = face_bbox
    bbox_width = float(right - left)
    bbox_height = float(bottom - top)
    normalizer = max(bbox_width, bbox_height)

    if not np.isfinite(normalizer) or normalizer <= 0.0:
        normalizer = float(max(image_hw[1], image_hw[0]))

    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError(
            f"MERL-RAV sample {sample_id} has invalid fallback normalizer={normalizer!r}"
        )

    top_level_visibility = [bool(value) for value in score_visibility.tolist()]
    coordinate_valid_mask = [bool(value) for value in coordinate_valid_in_image.tolist()]
    source_valid = [bool(value) for value in source_valid_mask.tolist()]

    metadata: dict[str, T.Any] = {
        "image_id": _relative_to_or_absolute(image_path, aflw_root),
        "annotation_file": relative_annotation.as_posix(),
        "visibility": list(visibility),
        "self_occluded_count": sum(1 for value in visibility if value == "self_occluded"),
        "externally_occluded_count": sum(
            1 for value in visibility if value == "externally_occluded"
        ),
        "image_height": int(image_hw[0]),
        "image_width": int(image_hw[1]),
        "landmark_source_valid_mask": source_valid,
        "landmark_in_image_mask": coordinate_valid_mask,
        "landmark_coordinate_valid_mask": coordinate_valid_mask,
        "landmark_score_visibility_mask": top_level_visibility,
        "landmark_source_valid_count": int(source_valid_mask.sum()),
        "landmark_in_image_count": int(coordinate_valid_in_image.sum()),
        "landmark_score_visible_count": int(score_visibility.sum()),
        "face_bbox": face_bbox,
        "face_bbox_source": "merl_rav_native_aflw_image_landmarks",
        "normalizer_source": "merl_rav_coordinate_valid_landmark_bbox_max_side",
        "aflw_image_source": "aflw_native",
    }

    finite_landmarks = np.where(np.isfinite(src_points), src_points, 0.0).astype("float32")

    return {
        "sample_id": sample_id,
        "dataset": "merl-rav",
        "condition": condition_labels[0],
        "conditions": condition_labels,
        "image": str(image_path.resolve()),
        "source_schema": "2d_68",
        "source": {"dataset": "merl-rav-aflw-native", "source_id": sample_id},
        "normalizer": float(normalizer),
        "visibility": top_level_visibility,
        "metadata": metadata,
        "points": finite_landmarks,
    }


def _split_from_annotation_path(path: Path) -> str | None:
    """Infer train/test from MERL-RAV label path parts."""
    parts = {part.lower().replace("-", "_") for part in path.parts}
    if "trainset" in parts:
        return "train"
    if "testset" in parts:
        return "test"
    return None


def _build_samples_native_aflw(
    label_root: Path,
    aflw_root: Path,
    *,
    splits: T.Sequence[str] = ("train", "test"),
) -> tuple[list[dict[str, T.Any]], dict[str, T.Any]]:
    """Build MERL-RAV samples by directly matching annotations to native AFLW images."""
    image_index = _build_aflw_native_image_index(aflw_root)
    requested = {split.lower() for split in splits}

    samples: list[dict[str, T.Any]] = []
    stats: dict[str, T.Any] = {
        "labels": 0,
        "matched": 0,
        "skipped_split": 0,
        "skipped_no_image": 0,
        "skipped_missing_image": 0,
        "skipped_bad_image": 0,
        "skipped_all_outside_image": 0,
        "skipped_no_score_visible": 0,
        "total_source_valid_landmarks": 0,
        "total_in_image_landmarks": 0,
        "total_score_visible_landmarks": 0,
        "aflw_native_image_count": len(image_index),
    }

    for annotation in _label_files(label_root):
        stats["labels"] += 1

        split_name = _split_from_annotation_path(annotation.relative_to(label_root))
        if split_name is not None and requested and split_name not in requested:
            stats["skipped_split"] += 1
            continue

        source_stem = _aflw_source_stem(annotation.stem)
        image_path = image_index.get(source_stem or "")
        if image_path is None:
            stats["skipped_no_image"] += 1
            continue

        if not image_path.is_file():
            stats["skipped_missing_image"] += 1
            continue

        try:
            image_hw = _read_image_size(image_path)
        except OSError:
            stats["skipped_bad_image"] += 1
            continue

        signed = _parse_pts_signed(annotation)
        visibility, src_points, score_visible = _visibility_for_crop(signed)

        sample = _sample_from_native_aflw_image(
            annotation=annotation,
            label_root=label_root,
            image_path=image_path,
            aflw_root=aflw_root,
            src_points=src_points,
            visibility=visibility,
            score_visible=score_visible,
            image_hw=image_hw,
        )

        if sample["metadata"]["landmark_in_image_count"] == 0:
            stats["skipped_all_outside_image"] += 1
            continue

        if not any(sample["visibility"]):
            stats["skipped_no_score_visible"] += 1
            continue

        samples.append(sample)
        stats["matched"] += 1
        stats["total_source_valid_landmarks"] += int(
            sample["metadata"]["landmark_source_valid_count"]
        )
        stats["total_in_image_landmarks"] += int(sample["metadata"]["landmark_in_image_count"])
        stats["total_score_visible_landmarks"] += int(
            sample["metadata"]["landmark_score_visible_count"]
        )

    return samples, stats


def _log_native_aflw_audit(stats: dict[str, T.Any]) -> None:
    """Emit a structured audit log for the native AFLW image builder."""
    logger.info("MERL-RAV native AFLW audit: %s", stats)


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
        "skipped_no_score_visible": 0,
        "total_source_valid_landmarks": 0,
        "total_in_crop_landmarks": 0,
        "total_score_visible_landmarks": 0,
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
        visibility, src_points, score_visible = _visibility_for_crop(signed)
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
            score_visible=score_visible,
            image_hw=image_hw,
        )

        if sample["metadata"]["landmark_in_crop_count"] == 0:
            stats["skipped_all_outside_crop"] += 1
            continue
        if not any(sample["visibility"]):
            stats["skipped_no_score_visible"] += 1
            continue

        samples.append(sample)
        stats["matched"] += 1
        stats["total_source_valid_landmarks"] += int(
            sample["metadata"]["landmark_source_valid_count"]
        )
        stats["total_in_crop_landmarks"] += int(sample["metadata"]["landmark_in_crop_count"])
        stats["total_score_visible_landmarks"] += int(
            sample["metadata"]["landmark_score_visible_count"]
        )
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
            "total_score_visible_landmarks",
        }
    }
    summary["split_counts"] = stats.get("aflw_release2_split_counts", {})
    summary["total_source_valid_landmarks"] = stats.get("total_source_valid_landmarks", 0)
    summary["total_in_crop_landmarks"] = stats.get("total_in_crop_landmarks", 0)
    summary["total_score_visible_landmarks"] = stats.get("total_score_visible_landmarks", 0)

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
    aflw_image_root: str | Path | None = None,
    coordinate_space: str | None = None,
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
    """Build a MERL-RAV manifest aligned with AFLW images.

    Native AFLW mode matches MERL-RAV annotations directly to AFLW flickr
    images by imageNNNNN and uses MERL-RAV coordinates directly in image
    space. AFLW CSV and AFLW .pts files are intentionally ignored.

    AFLW release-2 mode keeps the older cropped-image translation path.
    """
    selected_space = (coordinate_space or "").strip().lower().replace("_", "-")
    if not selected_space:
        selected_space = "aflw-release2" if aflw_release2_dir is not None else "native-aflw"

    if selected_space not in {"native-aflw", "aflw-release2"}:
        raise ValueError(
            "coordinate_space must be one of {'native-aflw', 'aflw-release2'}, "
            f"got {coordinate_space!r}"
        )

    release2_dir: Path | None = None
    aflw_root: Path | None = None

    if selected_space == "aflw-release2":
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
    else:
        aflw_root = resolve_aflw_native_root(
            aflw_image_root,
            cache_dir=cache_dir,
            force_download=force_download,
            no_download=no_download,
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
        aggregate: dict[str, T.Any] = {}

        for label_root in _find_label_roots(root):
            if selected_space == "aflw-release2":
                assert release2_dir is not None
                samples, stats = _build_samples_aflw_release2(
                    label_root,
                    release2_dir,
                    splits=splits,
                    min_anchors=min_anchors,
                    validate_hw=validate_hw,
                )
            else:
                assert aflw_root is not None
                samples, stats = _build_samples_native_aflw(
                    label_root,
                    aflw_root,
                    splits=splits,
                )

            all_samples.extend(samples)

            for key, value in stats.items():
                if isinstance(value, list):
                    aggregate.setdefault(key, []).extend(value)
                elif isinstance(value, dict):
                    target = aggregate.setdefault(key, {})
                    for nested_key, nested_value in value.items():
                        target[nested_key] = target.get(nested_key, 0) + nested_value
                elif isinstance(value, (int, float)):
                    aggregate[key] = aggregate.get(key, 0) + value
                else:
                    aggregate[key] = value

        if selected_space == "aflw-release2":
            _log_aflw_release2_audit(aggregate)
            error_label = "MERL-RAV/AFLW release-2 crop pairs"
        else:
            _log_native_aflw_audit(aggregate)
            error_label = "MERL-RAV/native AFLW image pairs"

        if not all_samples:
            detail = " ".join(
                f"{key}={value}"
                for key, value in aggregate.items()
                if key not in {"residual_medians", "aflw_release2_split_counts"}
            )
            raise FileNotFoundError(f"No {error_label} produced from {root}. {detail}")

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
