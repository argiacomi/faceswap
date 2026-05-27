#!/usr/bin/env python3
"""Faceset coverage metric computation."""

from __future__ import annotations

import logging
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lib.align import AlignedFace, DetectedFace
from lib.align.faceset_qa import FaceQAFile, FaceQARecord, build_index
from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.faceqa.expression import (
    EXPRESSION_BUCKETS,
    bucket_for_features,
    compute_expression_features,
)
from lib.serializer import get_serializer
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

if T.TYPE_CHECKING:
    from collections.abc import Callable

POSE_YAW_THRESHOLDS = (15.0, 30.0, 60.0)
POSE_PITCH_THRESHOLDS = (15.0, 30.0)
ROLL_RISK_THRESHOLDS = (10.0, 25.0)
BLUR_THRESHOLDS = (6.0, 3.0, 1.0)
RESOLUTION_THRESHOLDS = (256, 128, 64)
OCCLUSION_THRESHOLDS = (0.05, 0.20, 0.50)
LIGHTING_THRESHOLDS = (50.0, 200.0, 235.0)
DISTANCE_THRESHOLDS = (0.10, 0.20, 0.40)
POSE_LIMITS = {"yaw": (-120.0, 120.0), "pitch": (-90.0, 90.0), "roll": (-90.0, 90.0)}
POSE_HIGH_CONFIDENCE_DELTA = 10.0
POSE_MEDIUM_CONFIDENCE_DELTA = 25.0
SPIGA_BACKFILL_INPUT_SIZE = 256


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
    joint_pose_coverage: dict[str, T.Any] = field(default_factory=dict)
    expression_coverage: dict[str, T.Any] = field(default_factory=dict)
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
    pose_backfiller: Callable[[FaceQARecord, FileAlignments], dict[str, T.Any] | None]
    | None = None,
) -> list[FaceQARecord]:
    """Return FaceQA records derived from alignments and optional sidecar data."""
    sidecar_index = build_index(qa_file) if qa_file is not None else {}
    records: list[FaceQARecord] = []
    for frame, idx, face in load_alignments_records(path):
        record = _record_from_alignment(frame, idx, face)
        sidecar_record = sidecar_index.get((frame, idx))
        record = _merge_records(record, sidecar_record)
        if pose_backfiller is not None and not _valid_spiga_pose(record):
            _apply_backfilled_pose(record, pose_backfiller(record, face))
        _select_pose(record)
        _populate_expression(record, face)
        records.append(record)
    if qa_file is not None and not records:
        records = list(qa_file.faces)
        for record in records:
            _select_pose(record)
        return records
    return records


def _merge_records(base: FaceQARecord, override: FaceQARecord | None) -> FaceQARecord:
    """Fill missing sidecar values with alignment-derived values."""
    if override is None:
        return base
    merged = T.cast(FaceQARecord, FaceQARecord.from_dict(override.to_dict()))
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


def _populate_expression(record: FaceQARecord, face: FileAlignments) -> None:
    """Derive expression features and bucket from face landmarks."""
    features = compute_expression_features(face.landmarks_xy)
    if features is None:
        record.mouth_openness = None
        record.mouth_width_ratio = None
        record.smile_proxy = None
        record.eye_closure = None
        record.brow_raise_proxy = None
        record.expression_asymmetry = None
        record.expression_bucket = None
        return
    record.mouth_openness = features["mouth_openness"]
    record.mouth_width_ratio = features["mouth_width_ratio"]
    record.smile_proxy = features["smile_proxy"]
    record.eye_closure = features["eye_closure"]
    record.brow_raise_proxy = features["brow_raise_proxy"]
    record.expression_asymmetry = features["expression_asymmetry"]
    record.expression_bucket = bucket_for_features(features)


def _populate_aligned_metrics(record: FaceQARecord, face: FileAlignments) -> None:
    """Populate pose and mean-face distance from landmarks."""
    try:
        aligned = AlignedFace(face.landmarks_xy)
    except Exception:  # pylint:disable=broad-except
        logger.debug("Could not derive aligned metrics for %s:%s", record.frame, record.face_index)
        return
    record.alignment_yaw = float(aligned.pose.yaw)
    record.alignment_pitch = float(aligned.pose.pitch)
    record.alignment_roll = float(aligned.pose.roll)
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
    """Populate native SPIGA pose metadata when present."""
    pose = _extract_pose_metadata(metadata)
    if pose is None:
        return
    record.spiga_yaw = _float_or_none(pose.get("yaw"))
    record.spiga_pitch = _float_or_none(pose.get("pitch"))
    record.spiga_roll = _float_or_none(pose.get("roll"))
    record.spiga_pose_source = str(pose.get("source") or "spiga")
    record.spiga_pose_model = None if pose.get("model") is None else str(pose.get("model"))
    delta = pose.get("delta")
    if isinstance(delta, dict):
        record.pose_delta_yaw = _float_or_none(delta.get("yaw"))
        record.pose_delta_pitch = _float_or_none(delta.get("pitch"))
        record.pose_delta_roll = _float_or_none(delta.get("roll"))
        _set_pose_max_delta(record)


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


def _pose_values(record: FaceQARecord, prefix: str) -> dict[str, float | None]:
    """Return yaw/pitch/roll values for one candidate prefix."""
    return {
        "yaw": getattr(record, f"{prefix}_yaw"),
        "pitch": getattr(record, f"{prefix}_pitch"),
        "roll": getattr(record, f"{prefix}_roll"),
    }


def _valid_pose_values(values: dict[str, float | None]) -> bool:
    """Return whether pose values are numeric, finite, and in sane ranges."""
    for key, value in values.items():
        if value is None:
            return False
        if not np.isfinite(value):
            return False
        low, high = POSE_LIMITS[key]
        if not low <= value <= high:
            return False
    return True


def _valid_spiga_pose(record: FaceQARecord) -> bool:
    """Return whether the record already has a valid SPIGA pose candidate."""
    if _valid_pose_values(_pose_values(record, "spiga")):
        return True
    if record.pose_source in ("spiga", "spiga_backfill"):
        return _valid_pose_values({"yaw": record.yaw, "pitch": record.pitch, "roll": record.roll})
    return False


def _apply_backfilled_pose(record: FaceQARecord, pose: dict[str, T.Any] | None) -> None:
    """Store a backfilled SPIGA pose candidate on a record."""
    if pose is None:
        return
    record.spiga_yaw = _float_or_none(pose.get("yaw"))
    record.spiga_pitch = _float_or_none(pose.get("pitch"))
    record.spiga_roll = _float_or_none(pose.get("roll"))
    record.spiga_pose_source = str(pose.get("source") or "spiga_backfill")
    record.spiga_pose_model = None if pose.get("model") is None else str(pose.get("model"))


def _set_pose_deltas(record: FaceQARecord) -> None:
    """Populate SPIGA-minus-alignment pose deltas when both candidates exist."""
    spiga = _pose_values(record, "spiga")
    alignment = _pose_values(record, "alignment")
    if not _valid_pose_values(spiga) or not _valid_pose_values(alignment):
        return
    record.pose_delta_yaw = spiga["yaw"] - alignment["yaw"]  # type:ignore[operator]
    record.pose_delta_pitch = spiga["pitch"] - alignment["pitch"]  # type:ignore[operator]
    record.pose_delta_roll = spiga["roll"] - alignment["roll"]  # type:ignore[operator]
    _set_pose_max_delta(record)


def _set_pose_max_delta(record: FaceQARecord) -> None:
    """Populate maximum absolute pose disagreement."""
    deltas = [record.pose_delta_yaw, record.pose_delta_pitch, record.pose_delta_roll]
    numeric = [abs(delta) for delta in deltas if delta is not None]
    record.pose_max_abs_delta = max(numeric) if numeric else None


def _pose_confidence(record: FaceQARecord, source: str) -> str:
    """Return confidence for the selected pose source."""
    if source == "alignment":
        return "fallback"
    if record.pose_max_abs_delta is None:
        return "unknown"
    if record.pose_max_abs_delta <= POSE_HIGH_CONFIDENCE_DELTA:
        return "high"
    if record.pose_max_abs_delta <= POSE_MEDIUM_CONFIDENCE_DELTA:
        return "medium"
    return "low"


def _select_pose(record: FaceQARecord) -> None:
    """Select the coverage pose using SPIGA-first deterministic rules."""
    # Backward compatibility: previous sidecars stored selected SPIGA pose directly.
    if record.spiga_yaw is None and record.pose_source in ("spiga", "spiga_backfill"):
        record.spiga_yaw = record.yaw
        record.spiga_pitch = record.pitch
        record.spiga_roll = record.roll
        record.spiga_pose_source = record.pose_source
        record.spiga_pose_model = record.pose_model

    _set_pose_deltas(record)
    spiga = _pose_values(record, "spiga")
    if _valid_pose_values(spiga):
        record.yaw = spiga["yaw"]
        record.pitch = spiga["pitch"]
        record.roll = spiga["roll"]
        record.pose_source = record.spiga_pose_source or "spiga"
        record.pose_model = record.spiga_pose_model or "spiga"
        record.pose_confidence = _pose_confidence(record, record.pose_source)
        return

    alignment = _pose_values(record, "alignment")
    if _valid_pose_values(alignment):
        record.yaw = alignment["yaw"]
        record.pitch = alignment["pitch"]
        record.roll = alignment["roll"]
        record.pose_source = "alignment"
        record.pose_model = "aligned_face"
        record.pose_confidence = "fallback"
        return

    record.yaw = None
    record.pitch = None
    record.roll = None
    record.pose_source = "unknown"
    record.pose_model = None
    record.pose_confidence = "unknown"


def _decode_thumb(thumb: np.ndarray | None) -> np.ndarray | None:
    """Decode an alignments thumbnail if present."""
    if thumb is None:
        return None
    arr: np.ndarray = np.asarray(thumb, dtype=np.uint8)
    if arr.ndim != 3:
        decoded = cv2.imdecode(arr.reshape(-1), cv2.IMREAD_COLOR)
        return T.cast(np.ndarray | None, decoded)
    return arr


def _estimate_blur(image: np.ndarray) -> float:
    """Return the same normalized Laplacian blur score used by Sort."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur_map = T.cast(np.ndarray, cv2.Laplacian(gray, cv2.CV_32F))
    return float(np.var(blur_map) / np.sqrt(gray.shape[0] * gray.shape[1]))


class SpigaPoseBackfiller:
    """Backfill native SPIGA pose from source frames and stored alignments."""

    def __init__(self, frames_loader: T.Any) -> None:
        self._frames_loader = frames_loader
        self._plugin: T.Any = None
        self._disabled_reason: str | None = None
        self._last_frame: str | None = None
        self._last_image: np.ndarray | None = None

    def __call__(self, record: FaceQARecord, face: FileAlignments) -> dict[str, T.Any] | None:
        """Return SPIGA pose metadata for a face, or ``None`` when unavailable."""
        if self._disabled_reason is not None:
            return None
        aligned_face = self._aligned_face(record, face, self._input_size())
        if aligned_face is None:
            return None
        plugin = self._load_plugin()
        if plugin is None:
            return None
        if aligned_face.shape[:2] != (plugin.input_size, plugin.input_size):
            aligned_face = cv2.resize(
                aligned_face,
                (plugin.input_size, plugin.input_size),
                interpolation=cv2.INTER_CUBIC,
            )
        feed = aligned_face.astype("float32", copy=False) / 255.0
        try:
            raw = plugin.process(feed[None])
            plugin.post_process(raw)
        except Exception as err:  # pylint:disable=broad-except
            self._disabled_reason = str(err)
            logger.warning("SPIGA pose backfill failed; falling back to alignment pose: %s", err)
            return None
        metadata = getattr(plugin, "last_debug_metadata", [])
        if not metadata:
            return None
        pose = metadata[0].get("pose") if isinstance(metadata[0], dict) else None
        if not isinstance(pose, dict):
            return None
        pose = dict(pose)
        pose["source"] = "spiga_backfill"
        pose.setdefault("model", "spiga")
        return pose

    def _aligned_face(
        self,
        record: FaceQARecord,
        face: FileAlignments,
        size: int,
    ) -> np.ndarray | None:
        """Load the source frame and reconstruct the aligned face crop for SPIGA."""
        image = self._load_frame(record.frame)
        if image is None:
            return None
        try:
            face_obj = DetectedFace()
            face_obj.from_alignment(face, image=image)
            face_obj.load_aligned(image, size=size, centering="head")
        except Exception as err:  # pylint:disable=broad-except
            logger.warning(
                "Could not reconstruct aligned face for SPIGA pose backfill "
                "(frame: %s, face: %s): %s",
                record.frame,
                record.face_index,
                err,
            )
            return None
        return T.cast(np.ndarray, face_obj.aligned.face)

    def _load_frame(self, frame: str) -> np.ndarray | None:
        """Load a source frame by alignment frame name."""
        if frame == self._last_frame and self._last_image is not None:
            return self._last_image
        try:
            image = T.cast(np.ndarray, self._frames_loader.load_image(frame))
        except Exception as err:  # pylint:disable=broad-except
            logger.warning(
                "Could not load source frame for SPIGA pose backfill '%s': %s", frame, err
            )
            return None
        self._last_frame = frame
        self._last_image = image
        return image

    def _input_size(self) -> int:
        """Return the SPIGA input size without forcing model loading."""
        return (
            SPIGA_BACKFILL_INPUT_SIZE
            if self._plugin is None
            else int(getattr(self._plugin, "input_size", SPIGA_BACKFILL_INPUT_SIZE))
        )

    def _load_plugin(self) -> T.Any | None:
        """Load SPIGA lazily only when a frame-backed crop is needed."""
        if self._plugin is not None:
            return self._plugin
        try:
            from plugins.extract.align.spiga import SPIGA  # pylint:disable=import-outside-toplevel

            plugin = SPIGA()
            plugin.model = plugin.load_model()
        except Exception as err:  # pylint:disable=broad-except
            self._disabled_reason = str(err)
            logger.warning(
                "SPIGA pose backfill unavailable; falling back to alignment pose: %s", err
            )
            return None
        self._plugin = plugin
        return self._plugin


def _mask_ref(face: FileAlignments) -> str | None:
    """Return a compact mask QA key from available mask metadata."""
    if not face.mask:
        return None
    return ",".join(sorted(face.mask))


def _pose_bucket(record: FaceQARecord) -> str:
    """Return a signed yaw bucket. Negative yaw is left, positive is right."""
    if record.yaw is None:
        return "unknown"
    yaw = record.yaw
    abs_yaw = abs(yaw)
    if abs_yaw <= POSE_YAW_THRESHOLDS[0]:
        return "frontal"
    side = "left" if yaw < 0 else "right"
    if abs_yaw <= POSE_YAW_THRESHOLDS[1]:
        return f"{side}_slight"
    if abs_yaw <= POSE_YAW_THRESHOLDS[2]:
        return f"{side}_profile"
    return f"{side}_extreme"


def _pitch_bucket(record: FaceQARecord) -> str:
    """Return a signed pitch bucket. Negative pitch is down, positive is up."""
    if record.pitch is None:
        return "unknown"
    pitch = record.pitch
    abs_pitch = abs(pitch)
    if abs_pitch <= POSE_PITCH_THRESHOLDS[0]:
        return "neutral"
    direction = "down" if pitch < 0 else "up"
    if abs_pitch <= POSE_PITCH_THRESHOLDS[1]:
        return direction
    return f"{direction}_extreme"


def _pose_source_bucket(record: FaceQARecord) -> str:
    return record.pose_source if record.pose_source is not None else "unknown"


def _pose_confidence_bucket(record: FaceQARecord) -> str:
    return record.pose_confidence if record.pose_confidence is not None else "unknown"


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


YAW_BUCKETS: tuple[str, ...] = (
    "left_extreme",
    "left_profile",
    "left_slight",
    "frontal",
    "right_slight",
    "right_profile",
    "right_extreme",
)
PITCH_BUCKETS: tuple[str, ...] = (
    "down_extreme",
    "down",
    "neutral",
    "up",
    "up_extreme",
)

_DIMENSION_BUCKETS: dict[str, list[str]] = {
    "pose": [*YAW_BUCKETS, "unknown"],
    "pitch": [*PITCH_BUCKETS, "unknown"],
    "pose_sources": ["spiga", "spiga_backfill", "alignment", "unknown"],
    "pose_confidence": ["high", "medium", "low", "fallback", "unknown"],
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
    "expression": [*EXPRESSION_BUCKETS, "unknown"],
}

_BUCKET_FNS = {
    "pose": _pose_bucket,
    "pitch": _pitch_bucket,
    "pose_sources": _pose_source_bucket,
    "pose_confidence": _pose_confidence_bucket,
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
    return float(
        sum(record.duplicate_keep_recommendation == "prune_candidate" for record in records)
        / len(records)
    )


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
        "pose_max_abs_delta": [record.pose_max_abs_delta for record in records],
        "blur_score": [record.blur_score for record in records],
        "average_distance": [record.average_distance for record in records],
        "exposure_mean": [record.exposure_mean for record in records],
        "mouth_openness": [record.mouth_openness for record in records],
        "mouth_width_ratio": [record.mouth_width_ratio for record in records],
        "smile_proxy": [record.smile_proxy for record in records],
        "eye_closure": [record.eye_closure for record in records],
        "brow_raise_proxy": [record.brow_raise_proxy for record in records],
        "expression_asymmetry": [record.expression_asymmetry for record in records],
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


def _expression_coverage(records: list[FaceQARecord]) -> dict[str, T.Any]:
    """Return expression bucket coverage with entropy and occupancy metrics."""
    counts: dict[str, int] = {bucket: 0 for bucket in EXPRESSION_BUCKETS}
    unknown = 0
    for record in records:
        bucket = _expression_bucket(record)
        if bucket == "unknown":
            unknown += 1
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
    classified = sum(counts.values())
    total_bins = len(EXPRESSION_BUCKETS)
    occupied = sum(1 for value in counts.values() if value > 0)
    percentages = (
        {bucket: round(value / classified * 100, 2) for bucket, value in counts.items()}
        if classified
        else {bucket: 0.0 for bucket in counts}
    )
    if classified:
        probabilities = np.array(
            [value / classified for value in counts.values() if value > 0],
            dtype="float64",
        )
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    else:
        entropy = 0.0
    bin_coverage_pct = round(occupied / total_bins * 100, 2) if total_bins else 0.0
    missing_bins = [bucket for bucket, value in counts.items() if value == 0]
    return {
        "counts": counts,
        "percentages": percentages,
        "total_bins": total_bins,
        "occupied_expression_bins": occupied,
        "empty_expression_bins": total_bins - occupied,
        "expression_bin_coverage_pct": bin_coverage_pct,
        "expression_entropy": round(entropy, 4),
        "classified_faces": classified,
        "unknown_faces": unknown,
        "missing_bins": missing_bins,
    }


def _joint_cell_label(yaw_bucket: str, pitch_bucket: str) -> str:
    """Return a stable joint pose cell label."""
    return f"{yaw_bucket}+{pitch_bucket}"


def _joint_pose_coverage(records: list[FaceQARecord]) -> dict[str, T.Any]:
    """Return yaw x pitch joint cell coverage with entropy and occupancy."""
    cells = [_joint_cell_label(y, p) for y in YAW_BUCKETS for p in PITCH_BUCKETS]
    counts: dict[str, int] = {cell: 0 for cell in cells}
    unknown = 0
    for record in records:
        yaw_bucket = _pose_bucket(record)
        pitch_bucket = _pitch_bucket(record)
        if yaw_bucket == "unknown" or pitch_bucket == "unknown":
            unknown += 1
            continue
        counts[_joint_cell_label(yaw_bucket, pitch_bucket)] += 1
    classified = sum(counts.values())
    total_cells = len(cells)
    occupied_cells = sum(1 for value in counts.values() if value > 0)
    empty_cells = total_cells - occupied_cells
    percentages = (
        {cell: round(value / classified * 100, 2) for cell, value in counts.items()}
        if classified
        else {cell: 0.0 for cell in counts}
    )
    if classified:
        probabilities = np.array(
            [value / classified for value in counts.values() if value > 0],
            dtype="float64",
        )
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    else:
        entropy = 0.0
    bin_coverage_pct = round(occupied_cells / total_cells * 100, 2) if total_cells else 0.0
    missing_cells = [cell for cell, value in counts.items() if value == 0]
    return {
        "counts": counts,
        "percentages": percentages,
        "total_cells": total_cells,
        "occupied_pose_cells": occupied_cells,
        "empty_pose_cells": empty_cells,
        "pose_bin_coverage_pct": bin_coverage_pct,
        "pose_entropy": round(entropy, 4),
        "classified_faces": classified,
        "unknown_faces": unknown,
        "missing_cells": missing_cells,
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
        joint_pose_coverage=_joint_pose_coverage(records_list),
        expression_coverage=_expression_coverage(records_list),
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
