#!/usr/bin/env python3
"""Write detector-derived face bboxes into a landmark manifest."""

from __future__ import annotations

import argparse
import json
import logging
import typing as T
from pathlib import Path

import cv2
import numpy as np

from lib.landmarks.progress import progress_iter

logger = logging.getLogger(__name__)


class DetectionError(RuntimeError):
    """Raised when detector output cannot be used."""


def _read_manifest(path: Path) -> tuple[dict[str, T.Any], list[dict[str, T.Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "samples" if "samples" in payload else "scenarios"
    entries = payload.get(key, [])
    if not isinstance(entries, list):
        raise ValueError("manifest samples/scenarios must be a list")
    return payload, [entry for entry in entries if isinstance(entry, dict)], key


def _write_manifest(path: Path, payload: dict[str, T.Any], key: str, entries: list[dict[str, T.Any]]) -> None:
    payload[key] = entries
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sample_id(entry: dict[str, T.Any]) -> str:
    return str(entry.get("sample_id") or entry.get("id") or entry.get("name") or "")


def _entry_image(entry: dict[str, T.Any], manifest: Path) -> Path:
    image = Path(str(entry.get("image", "")))
    return image if image.is_absolute() else manifest.parent / image


def _landmark_bbox(entry: dict[str, T.Any], manifest: Path) -> list[float]:
    landmarks = Path(str(entry.get("landmarks", "")))
    path = landmarks if landmarks.is_absolute() else manifest.parent / landmarks
    points = np.load(str(path)).astype("float32")
    finite = points[np.all(np.isfinite(points[:, :2]), axis=1), :2]
    if finite.size == 0:
        raise ValueError(f"no finite landmarks for {_sample_id(entry)}")
    left, top = np.min(finite, axis=0)
    right, bottom = np.max(finite, axis=0)
    return [float(left), float(top), float(right), float(bottom)]


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image has invalid dimensions")
    scale = size / float(max(height, width))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, image.shape[2]), dtype="float32")
    pad_left = int((size - resized_w) // 2)
    pad_top = int((size - resized_h) // 2)
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized.astype("float32")
    return canvas, scale, pad_left, pad_top


def _box_to_frame(box: np.ndarray, *, scale: float, pad_left: int, pad_top: int, image_shape: tuple[int, int]) -> list[float]:
    height, width = image_shape
    left = (float(box[0]) - pad_left) / scale
    top = (float(box[1]) - pad_top) / scale
    right = (float(box[2]) - pad_left) / scale
    bottom = (float(box[3]) - pad_top) / scale
    left = max(0.0, min(left, float(width - 1)))
    top = max(0.0, min(top, float(height - 1)))
    right = max(left + 1.0, min(right, float(width)))
    bottom = max(top + 1.0, min(bottom, float(height)))
    return [left, top, right, bottom]


def _iou(a: T.Sequence[float], b: T.Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a[:4])
    bx1, by1, bx2, by2 = (float(value) for value in b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _build_detector(name: str) -> T.Any:
    key = name.strip().lower().replace("_", "-")
    if key not in {"cv2-dnn", "cv2dnn"}:
        raise ValueError(f"unsupported detector '{name}'. Supported detectors: cv2-dnn")
    from plugins.extract.detect.cv2_dnn import CV2DNNDetect

    detector = CV2DNNDetect()
    detector.model = detector.load_model()
    return detector


def _select_index(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    selection: str,
    target: list[float] | None,
    min_iou: float,
) -> tuple[int, float | None]:
    if selection == "largest":
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        return int(np.argmax(areas)), None
    if selection == "gt-iou":
        if target is None:
            raise ValueError("gt-iou selection requires landmarks")
        ious = np.asarray([_iou(box, target) for box in boxes], dtype="float32")
        index = int(np.argmax(ious))
        match_iou = float(ious[index])
        if match_iou < min_iou:
            raise DetectionError(f"best IoU {match_iou:.4f} below minimum {min_iou:.4f}")
        return index, match_iou
    return int(np.argmax(scores)), None


def _detect_one(
    detector: T.Any,
    image_path: Path,
    *,
    selection: str,
    target: list[float] | None,
    min_iou: float,
) -> tuple[list[float], float, float | None]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    canvas, scale, pad_left, pad_top = _letterbox(image, int(detector.input_size))
    batch = np.asarray([canvas], dtype="float32")
    raw = detector.process(detector.pre_process(batch))[0]
    detections = raw[raw[:, 2] >= float(detector.confidence)]
    if not len(detections):
        raise DetectionError(f"no detection in {image_path}")
    scores = detections[:, 2]
    detector_boxes = detections[:, 3:7] * float(detector.input_size)
    frame_boxes = np.asarray(
        [
            _box_to_frame(
                box,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                image_shape=image.shape[:2],
            )
            for box in detector_boxes
        ],
        dtype="float32",
    )
    index, match_iou = _select_index(
        frame_boxes,
        scores,
        selection=selection,
        target=target,
        min_iou=min_iou,
    )
    return frame_boxes[index].astype("float32").tolist(), float(scores[index]), match_iou


def apply_detector_bboxes(args: argparse.Namespace) -> dict[str, T.Any]:
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output_manifest).expanduser().resolve() if args.output_manifest else manifest
    payload, entries, key = _read_manifest(manifest)
    detector = _build_detector(args.detector)
    written = missing = 0
    records: list[dict[str, T.Any]] = []
    show_progress = getattr(args, "log_level", "INFO") != "ERROR"
    for entry in progress_iter(entries, label="Detector bboxes", enabled=show_progress):
        sid = _sample_id(entry)
        metadata = entry.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            entry["metadata"] = metadata
        try:
            target = _landmark_bbox(entry, manifest) if args.selection == "gt-iou" else None
            bbox, confidence, match_iou = _detect_one(
                detector,
                _entry_image(entry, manifest),
                selection=args.selection,
                target=target,
                min_iou=args.min_iou,
            )
            source = f"faceswap_detector/{args.detector}"
        except Exception as err:
            missing += 1
            if args.on_missing == "error":
                raise
            if args.on_missing == "gt":
                bbox = _landmark_bbox(entry, manifest)
                confidence = 0.0
                match_iou = 1.0
                source = "gt_landmarks_detector_fallback"
            else:
                logger.debug("No detector bbox for %s: %s", sid, err)
                records.append({"sample_id": sid, "status": "missing", "error": str(err)})
                continue
        metadata["face_bbox"] = bbox
        metadata["face_bbox_source"] = source
        metadata["face_bbox_detector_selection"] = args.selection
        metadata["face_bbox_confidence"] = confidence
        if match_iou is not None:
            metadata["face_bbox_gt_iou"] = match_iou
        records.append(
            {
                "sample_id": sid,
                "status": "ok",
                "bbox": bbox,
                "confidence": confidence,
                "source": source,
                "gt_iou": match_iou,
            }
        )
        written += 1
    _write_manifest(output, payload, key, entries)
    summary = {
        "manifest": str(output),
        "detector": args.detector,
        "selection": args.selection,
        "min_iou": args.min_iou,
        "on_missing": args.on_missing,
        "written": written,
        "missing": missing,
        "total": len(entries),
        "records": records,
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--detector", default="cv2-dnn")
    parser.add_argument("--selection", choices=("confidence", "largest", "gt-iou"), default="gt-iou")
    parser.add_argument("--min-iou", type=float, default=0.25)
    parser.add_argument("--on-missing", choices=("error", "skip", "gt"), default="error")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    if args.min_iou < 0.0 or args.min_iou > 1.0:
        raise SystemExit("--min-iou must be between 0 and 1")
    summary = apply_detector_bboxes(args)
    print(f"Detector bboxes: written={summary['written']} missing={summary['missing']} total={summary['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
