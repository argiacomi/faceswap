#!/usr/bin/env python3
"""Visual audit overlays for landmark quality datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from lib.landmarks.datasets.polish import entry_path, manifest_entries, rewrite_manifest

logger = logging.getLogger(__name__)
LANDMARK_REGIONS: tuple[tuple[str, range, tuple[int, int, int]], ...] = (
    ("jaw", range(0, 17), (255, 128, 0)),
    ("brow", range(17, 27), (0, 180, 255)),
    ("nose", range(27, 36), (0, 255, 255)),
    ("left_eye", range(36, 42), (0, 255, 0)),
    ("right_eye", range(42, 48), (255, 0, 255)),
    ("mouth", range(48, 68), (255, 255, 0)),
)


def _load_cv2():
    """Import OpenCV lazily so non-overlay builds do not require it at import time."""
    try:
        import cv2  # pylint:disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - environment dependent
        logger.warning("OpenCV unavailable; skipping landmark indexed/region overlays")
        return None
    return cv2


def _image_for_entry(entry: dict, output_dir: Path):
    cv2 = _load_cv2()
    if cv2 is None:
        return None
    image_value = str(entry.get("image", ""))
    if not image_value:
        return None
    image_path = entry_path(image_value, output_dir)
    if not image_path.is_file():
        return None
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def _landmarks_for_entry(entry: dict, output_dir: Path) -> np.ndarray | None:
    landmark_value = str(entry.get("landmarks", ""))
    if not landmark_value:
        return None
    landmark_path = entry_path(landmark_value, output_dir)
    if not landmark_path.is_file():
        return None
    points = np.load(str(landmark_path)).astype("float32")
    if points.ndim != 2 or points.shape[1] < 2:
        return None
    return points


def _overlay_dir(entry: dict, output_dir: Path) -> Path:
    landmark_path = entry_path(str(entry.get("landmarks", "")), output_dir)
    return landmark_path.parent / "overlays"


def _draw_indexed(image, landmarks: np.ndarray):
    cv2 = _load_cv2()
    if cv2 is None:
        return None
    overlay = image.copy()
    for index, point in enumerate(landmarks):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        cv2.circle(overlay, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            str(index),
            (x + 2, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _draw_regions(image, landmarks: np.ndarray):
    cv2 = _load_cv2()
    if cv2 is None:
        return None
    overlay = image.copy()
    legend_y = 14
    for name, indices, color in LANDMARK_REGIONS:
        points = []
        for index in indices:
            if index >= len(landmarks):
                continue
            x, y = int(round(float(landmarks[index, 0]))), int(round(float(landmarks[index, 1])))
            points.append((x, y))
            cv2.circle(overlay, (x, y), 2, color, -1)
        if len(points) > 1:
            cv2.polylines(overlay, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
        cv2.putText(overlay, name, (4, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        legend_y += 13
    return overlay


def _write(path: Path, image) -> bool:
    cv2 = _load_cv2()
    if cv2 is None or image is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image))


def write_indexed_region_overlays(output_dir: str | Path) -> list[Path]:
    """Write landmarks_indexed.png and landmarks_regions.png for readable samples."""
    root = Path(output_dir)
    payload, entries = manifest_entries(root)
    written: list[Path] = []
    changed = False
    for entry in entries:
        image = _image_for_entry(entry, root)
        landmarks = _landmarks_for_entry(entry, root)
        if image is None or landmarks is None:
            continue
        overlay_dir = _overlay_dir(entry, root)
        indexed = overlay_dir / "landmarks_indexed.png"
        regions = overlay_dir / "landmarks_regions.png"
        overlays = entry.setdefault("metadata", {}).setdefault("overlays", {})
        if _write(indexed, _draw_indexed(image, landmarks)):
            overlays["indexed"] = indexed.relative_to(root).as_posix()
            written.append(indexed)
            changed = True
        if _write(regions, _draw_regions(image, landmarks)):
            overlays["regions"] = regions.relative_to(root).as_posix()
            written.append(regions)
            changed = True
    if changed:
        rewrite_manifest(root, payload, entries)
    logger.info("Wrote %d landmark indexed/region overlay files under %s", len(written), root)
    return written
