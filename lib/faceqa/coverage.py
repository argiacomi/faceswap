#!/usr/bin/env python3
"""Faceset coverage metric computation."""

from __future__ import annotations

import logging
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lib.align import AlignedFace
from lib.align.faceset_qa import FaceQAFile, FaceQARecord, build_index
from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.serializer import get_serializer
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

POSE_YAW_THRESHOLDS = (15.0, 30.0, 60.0)
ROLL_RISK_THRESHOLDS = (10.0, 25.0)
BLUR_THRESHOLDS = (6.0, 3.0, 1.0)
RESOLUTION_THRESHOLDS = (256, 128, 64)
OCCLUSION_THRESHOLDS = (0.05, 0.20, 0.50)
LIGHTING_THRESHOLDS = (50.0, 200.0, 235.0)
DISTANCE_THRESHOLDS = (0.10, 0.20, 0.40)


@dataclass
class FacesetCoverageReport:
    """Aggregated coverage metrics for a faceset."""

    total_faces: int = 0
    usable_faces: int = 0
    bucket_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    bucket_percentages: dict[str, dict[str, float]] = field(default_factory=dict)
    metric_summary: dict[str, dict[str, float | None]] = field(default_factory=dict)
    duplicate_ratio: float | None = None
    identity_outlier_ratio: float | None = None
    mask_qa_distribution: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    sidecar_used: bool = False

    def coverage_dict(self) -> dict[str, dict[str, T.Any]]:
        """Return per-dimension counts and percentages."""
        return {
            dimension: {
                "counts": counts,
                "percentages": self.bucket_percentages.get(dimension, {}),
            }
            for dimension, counts in self.bucket_counts.items()
        }


def load_alignments_records(path: str | Path) -> list[tuple[str, int, FileAlignments]]:
    """Load face records from an alignments file without mutating it."""
    alignments_path = Path(path)
    raw = get_serializer("compressed").load(str(alignments_path))
    data = raw.get("__data__", raw)
    records: list[tuple[str, int, FileAlignments]] = []
    for frame, value in data.items():
        entry = AlignmentsEntry.from_dict(value)
        records.extend((frame, idx, face) for idx, face in enumerate(entry.faces))
    return records


def records_from_alignments(
    path: str | Path,
    *,
    qa_file: FaceQAFile | None = None,
) -> list[FaceQARecord]:
    """Return FaceQA records derived from alignments and optional sidecar data."""
    sidecar_index = build_index(qa_file) if qa_file is not None else {}
    records: list[FaceQARecord] = []
    for frame, idx, face in load_alignments_records(path):
        record = _record_from_alignment(frame, idx, face)
        sidecar_record = sidecar_index.get((frame, idx))
        records.append(_merge_records(record, sidecar_record))
    if qa_file is not None and not records:
        return list(qa_file.faces)
    return records


def _merge_records(base: FaceQARecord, override: FaceQARecord | None) -> FaceQARecord:
    """Fill missing sidecar values with alignment-derived values."""
    if override is None:
        return base
    merged = FaceQARecord.from_dict(override.to_dict())
    for key, value in base.to_dict().items():
        current = getattr(merged, key)
        if current is None or current == []:
            setattr(merged, key, value)
    return merged


def _record_from_alignment(frame: str, idx: int, face: FileAlignments) -> FaceQARecord:
    """Build one record from the fields stored in an alignments face."""
    record = FaceQARecord(
        frame=frame,
        face_index=idx,
        resolution=[int(face.w), int(face.h)],
        mask_qa_ref=_mask_ref(face),
    )
    if face.identity:
        record.identity_model = ",".join(sorted(face.identity))
    _populate_aligned_metrics(record, face)
    _populate_thumb_metrics(record, face)
    _populate_pose_metadata(record, face.metadata)
    _populate_metadata(record, face.metadata)
    return record


def _populate_aligned_metrics(record: FaceQARecord, face: FileAlignments) -> None:
    """Populate pose and mean-face distance from landmarks."""
    try:
        aligned = AlignedFace(face.landmarks_xy)
    except Exception:  # pylint:disable=broad-except
        logger.debug("Could not derive aligned metrics for %s:%s", record.frame, record.face_index)
        return
    record.yaw = float(aligned.pose.yaw)
    record.pitch = float(aligned.pose.pitch)
    record.roll = float(aligned.pose.roll)
    record.average_distance = float(aligned.average_distance)


def _populate_thumb_metrics(record: FaceQARecord, face: FileAlignments) -> None:
    """Populate cheap image quality metrics from stored alignments thumbnails."""
    image = _decode_thumb(face.thumb)
    if image is None:
        return
    record.blur_score = _estimate_blur(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    record.exposure_mean = float(np.mean(gray))
    if image.ndim == 3:
        record.black_pixel_ratio = float(np.ndarray.all(image == [0, 0, 0], axis=2).mean())


def _populate_metadata(record: FaceQARecord, metadata: dict[str, T.Any]) -> None:
    """Copy known FaceQA fields from per-face metadata when present."""
    payload = metadata.get("faceqa", metadata)
    if not isinstance(payload, dict):
        return
    for key in FaceQARecord.__dataclass_fields__:  # pylint:disable=no-member
        if getattr(record, key) in (None, []) and key in payload:
            setattr(record, key, payload[key])


def _populate_pose_metadata(record: FaceQARecord, metadata: dict[str, T.Any]) -> None:
    """Prefer native persisted pose metadata over landmark-derived pose."""
    pose = _extract_pose_metadata(metadata)
    if pose is None:
        return
    record.yaw = _float_or_none(pose.get("yaw"))
    record.pitch = _float_or_none(pose.get("pitch"))
    record.roll = _float_or_none(pose.get("roll"))
    record.pose_source = str(pose.get("source") or "metadata")
    record.pose_model = None if pose.get("model") is None else str(pose.get("model"))
    delta = pose.get("delta")
    if isinstance(delta, dict):
        record.pose_delta_yaw = _float_or_none(delta.get("yaw"))
        record.pose_delta_pitch = _float_or_none(delta.get("pitch"))
        record.pose_delta_roll = _float_or_none(delta.get("roll"))


def _extract_pose_metadata(metadata: dict[str, T.Any]) -> dict[str, T.Any] | None:
    """Return native pose metadata from known alignment metadata envelopes."""
    for namespace in ("spiga", "faceqa"):
        payload = metadata.get(namespace)
        if isinstance(payload, dict) and isinstance(payload.get("pose"), dict):
            return T.cast(dict[str, T.Any], payload["pose"])
    pose = metadata.get("pose")
    if isinstance(pose, dict):
        return T.cast(dict[str, T.Any], pose)
    ensemble = metadata.get("landmark_ensemble")
    if isinstance(ensemble, dict):
        adapter_pose = ensemble.get("adapter_pose")
        if isinstance(adapter_pose, dict):
            spiga_pose = adapter_pose.get("spiga")
            if isinstance(spiga_pose, dict):
                return T.cast(dict[str, T.Any], spiga_pose)
    return None


def _float_or_none(value: T.Any) -> float | None:
    """Return a float for numeric metadata values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decode_thumb(thumb: np.ndarray | None) -> np.ndarray | None:
    """Decode an alignments thumbnail if present."""
    if thumb is None:
        return None
    arr = np.asarray(thumb, dtype=np.uint8)
    if arr.ndim != 3:
        decoded = cv2.imdecode(arr.reshape(-1), cv2.IMREAD_COLOR)
        return T.cast(np.ndarray | None, decoded)
    return arr


def _estimate_blur(image: np.ndarray) -> float:
    """Return the same normalized Laplacian blur score used by Sort."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur_map = T.cast(np.ndarray, cv2.Laplacian(gray, cv2.CV_32F))
    return float(np.var(blur_map) / np.sqrt(gray.shape[0] * gray.shape[1]))


def _mask_ref(face: FileAlignments) -> str | None:
    """Return a compact mask QA key from available mask metadata."""
    if not face.mask:
        return None
    return ",".join(sorted(face.mask))


def _pose_bucket(record: FaceQARecord) -> str:
    if record.yaw is None:
        return "unknown"
    abs_yaw = abs(record.yaw)
    if abs_yaw <= POSE_YAW_THRESHOLDS[0]:
        return "frontal"
    if abs_yaw <= POSE_YAW_THRESHOLDS[1]:
        return "slight"
    if abs_yaw <= POSE_YAW_THRESHOLDS[2]:
        return "profile"
    return "extreme"


def _roll_bucket(record: FaceQARecord) -> str:
    if record.roll is None:
        return "unknown"
    abs_roll = abs(record.roll)
    if abs_roll <= ROLL_RISK_THRESHOLDS[0]:
        return "stable"
    if abs_roll <= ROLL_RISK_THRESHOLDS[1]:
        return "tilted"
    return "risky"


def _blur_bucket(record: FaceQARecord) -> str:
    if record.blur_score is None:
        return "unknown"
    if record.blur_score >= BLUR_THRESHOLDS[0]:
        return "sharp"
    if record.blur_score >= BLUR_THRESHOLDS[1]:
        return "acceptable"
    if record.blur_score >= BLUR_THRESHOLDS[2]:
        return "blurry"
    return "unusable"


def _resolution_bucket(record: FaceQARecord) -> str:
    if len(record.resolution) < 2:
        return "unknown"
    try:
        side = min(int(record.resolution[0]), int(record.resolution[1]))
    except (TypeError, ValueError):
        return "unknown"
    if side >= RESOLUTION_THRESHOLDS[0]:
        return "high"
    if side >= RESOLUTION_THRESHOLDS[1]:
        return "medium"
    if side >= RESOLUTION_THRESHOLDS[2]:
        return "low"
    return "tiny"


def _distance_bucket(record: FaceQARecord) -> str:
    if record.average_distance is None:
        return "unknown"
    if record.average_distance <= DISTANCE_THRESHOLDS[0]:
        return "typical"
    if record.average_distance <= DISTANCE_THRESHOLDS[1]:
        return "elevated"
    if record.average_distance <= DISTANCE_THRESHOLDS[2]:
        return "high"
    return "extreme"


def _occlusion_bucket(record: FaceQARecord) -> str:
    if record.occlusion_score is None:
        return "unknown"
    if record.occlusion_score < OCCLUSION_THRESHOLDS[0]:
        return "none"
    if record.occlusion_score < OCCLUSION_THRESHOLDS[1]:
        return "mild"
    if record.occlusion_score < OCCLUSION_THRESHOLDS[2]:
        return "moderate"
    return "severe"


def _lighting_bucket(record: FaceQARecord) -> str:
    if record.exposure_mean is None:
        return "unknown"
    if record.exposure_mean < LIGHTING_THRESHOLDS[0]:
        return "dark"
    if record.exposure_mean < LIGHTING_THRESHOLDS[1]:
        return "normal"
    if record.exposure_mean < LIGHTING_THRESHOLDS[2]:
        return "bright"
    return "overexposed"


def _expression_bucket(record: FaceQARecord) -> str:
    return str(record.expression_bucket) if record.expression_bucket is not None else "unknown"


def _duplicate_bucket(record: FaceQARecord) -> str:
    recommendation = record.duplicate_keep_recommendation
    if recommendation is not None:
        return str(recommendation)
    if record.duplicate_cluster is not None or record.duplicate_cluster_id is not None:
        return "clustered"
    return "unknown"


def _identity_bucket(record: FaceQARecord) -> str:
    if record.identity_final_decision is not None:
        return str(record.identity_final_decision)
    if record.identity_quality_flag is not None:
        return str(record.identity_quality_flag)
    if record.identity_model is not None or record.identity_score is not None:
        return "available"
    return "unknown"


def _mask_qa_bucket(record: FaceQARecord) -> str:
    return record.mask_qa_ref if record.mask_qa_ref is not None else "unknown"


_DIMENSION_BUCKETS: dict[str, list[str]] = {
    "pose": ["frontal", "slight", "profile", "extreme", "unknown"],
    "roll": ["stable", "tilted", "risky", "unknown"],
    "blur": ["sharp", "acceptable", "blurry", "unusable", "unknown"],
    "resolution": ["high", "medium", "low", "tiny", "unknown"],
    "misalignment": ["typical", "elevated", "high", "extreme", "unknown"],
    "occlusion": ["none", "mild", "moderate", "severe", "unknown"],
    "lighting": ["dark", "normal", "bright", "overexposed", "unknown"],
    "duplicates": ["keep", "prune_candidate", "review", "clustered", "unknown"],
    "identity": [
        "inlier",
        "primary",
        "borderline",
        "outlier",
        "reject",
        "review",
        "available",
        "unknown",
    ],
    "mask_qa": ["unknown"],
    "expression": [],
}

_BUCKET_FNS = {
    "pose": _pose_bucket,
    "roll": _roll_bucket,
    "blur": _blur_bucket,
    "resolution": _resolution_bucket,
    "misalignment": _distance_bucket,
    "occlusion": _occlusion_bucket,
    "lighting": _lighting_bucket,
    "duplicates": _duplicate_bucket,
    "identity": _identity_bucket,
    "mask_qa": _mask_qa_bucket,
    "expression": _expression_bucket,
}


def _count_buckets(records: list[FaceQARecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for dimension, bucket_fn in _BUCKET_FNS.items():
        dimension_counts = {bucket: 0 for bucket in _DIMENSION_BUCKETS.get(dimension, [])}
        for record in records:
            try:
                bucket = bucket_fn(record)
            except Exception:  # pylint:disable=broad-except
                bucket = "unknown"
            dimension_counts[bucket] = dimension_counts.get(bucket, 0) + 1
        counts[dimension] = dimension_counts
    return counts


def _percentages(counts: dict[str, dict[str, int]], total: int) -> dict[str, dict[str, float]]:
    if total == 0:
        return {dimension: {key: 0.0 for key in buckets} for dimension, buckets in counts.items()}
    return {
        dimension: {key: round(value / total * 100, 2) for key, value in buckets.items()}
        for dimension, buckets in counts.items()
    }


def _compute_duplicate_ratio(records: list[FaceQARecord]) -> float | None:
    if not any(record.duplicate_keep_recommendation is not None for record in records):
        return None
    return sum(
        record.duplicate_keep_recommendation == "prune_candidate" for record in records
    ) / len(records)


def _is_identity_outlier(record: FaceQARecord) -> bool:
    if record.identity_final_decision is not None:
        return record.identity_final_decision in ("outlier", "reject")
    if record.identity_quality_flag is not None:
        return record.identity_quality_flag in ("outlier", "reject")
    return any(flag in ("identity_outlier", "outlier", "reject") for flag in record.quality_flags)


def _compute_identity_outlier_ratio(records: list[FaceQARecord]) -> float | None:
    has_identity_meta = any(
        record.identity_quality_flag is not None
        or record.identity_final_decision is not None
        or bool(record.quality_flags)
        for record in records
    )
    if not has_identity_meta:
        return None
    return sum(_is_identity_outlier(record) for record in records) / len(records)


def _mask_qa_distribution(records: list[FaceQARecord]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for record in records:
        key = _mask_qa_bucket(record)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _count_usable(
    records: list[FaceQARecord], exclude_duplicates: bool, exclude_outliers: bool
) -> int:
    usable = 0
    for record in records:
        if exclude_duplicates and record.duplicate_keep_recommendation == "prune_candidate":
            continue
        if exclude_outliers and _is_identity_outlier(record):
            continue
        usable += 1
    return usable


def _metric_summary(records: list[FaceQARecord]) -> dict[str, dict[str, float | None]]:
    fields_ = {
        "yaw": [record.yaw for record in records],
        "pitch": [record.pitch for record in records],
        "roll": [record.roll for record in records],
        "blur_score": [record.blur_score for record in records],
        "average_distance": [record.average_distance for record in records],
        "exposure_mean": [record.exposure_mean for record in records],
    }
    return {name: _summarize(values) for name, values in fields_.items()}


def _summarize(values: list[float | None]) -> dict[str, float | None]:
    numeric = np.array([value for value in values if value is not None], dtype="float32")
    if numeric.size == 0:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(np.min(numeric)),
        "median": float(np.median(numeric)),
        "max": float(np.max(numeric)),
    }


def compute_coverage(
    records: T.Sequence[FaceQARecord],
    *,
    exclude_duplicates: bool = False,
    exclude_outliers: bool = False,
    sidecar_used: bool = False,
) -> FacesetCoverageReport:
    """Compute coverage metrics from FaceQA records."""
    records_list = list(records)
    total = len(records_list)
    counts = _count_buckets(records_list)
    return FacesetCoverageReport(
        total_faces=total,
        usable_faces=_count_usable(records_list, exclude_duplicates, exclude_outliers),
        bucket_counts=counts,
        bucket_percentages=_percentages(counts, total),
        metric_summary=_metric_summary(records_list),
        duplicate_ratio=_compute_duplicate_ratio(records_list) if total else None,
        identity_outlier_ratio=_compute_identity_outlier_ratio(records_list) if total else None,
        mask_qa_distribution=_mask_qa_distribution(records_list),
        source_counts={
            "alignments": total,
            "sidecar": sum(
                any(
                    value not in (None, [], "")
                    for key, value in record.to_dict().items()
                    if key not in ("frame", "face_index")
                )
                for record in records_list
            )
            if sidecar_used
            else 0,
        },
        sidecar_used=sidecar_used,
    )


__all__ = get_module_objects(__name__)
