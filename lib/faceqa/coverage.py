#!/usr/bin/env python3
"""Faceset coverage metric computation."""

from __future__ import annotations

import importlib
import logging
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lib.align import AlignedFace, DetectedFace
from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.faceqa.buckets import (
    blur_bucket as _blur_bucket,
)
from lib.faceqa.buckets import (
    expression_bucket as _expression_bucket,
)
from lib.faceqa.buckets import (
    identity_bucket as _identity_bucket,
)
from lib.faceqa.buckets import (
    is_identity_outlier as _is_identity_outlier,
)
from lib.faceqa.buckets import (
    lighting_bucket as _lighting_bucket,
)
from lib.faceqa.buckets import (
    mask_qa_bucket as _mask_qa_bucket,
)
from lib.faceqa.buckets import (
    misalignment_bucket as _misalignment_bucket,
)
from lib.faceqa.buckets import (
    occlusion_bucket as _occlusion_bucket,
)
from lib.faceqa.buckets import (
    pitch_bucket as _pitch_bucket,
)
from lib.faceqa.buckets import (
    pose_bucket as _pose_bucket,
)
from lib.faceqa.buckets import (
    pose_confidence_bucket as _pose_confidence_bucket,
)
from lib.faceqa.buckets import (
    pose_source_bucket as _pose_source_bucket,
)
from lib.faceqa.buckets import (
    resolution_bucket as _resolution_bucket,
)
from lib.faceqa.expression import EXPRESSION_BUCKETS, compute_expression_features
from lib.faceqa.expression import bucket_for_features as expression_bucket_for_features
from lib.faceqa.lighting import LIGHTING_BUCKETS, compute_lighting_features
from lib.faceqa.record import FaceQARecord
from lib.serializer import get_serializer
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

if T.TYPE_CHECKING:
    from collections.abc import Callable

# Bucket-classification thresholds live in ``lib.faceqa.buckets`` so the
# bucket helpers and their thresholds stay co-located. The validation
# thresholds below (``POSE_LIMITS`` etc.) are NOT bucket-classification
# values — they gate ``_valid_pose_values`` / ``_pose_confidence`` — and
# remain in coverage.
ROLL_RISK_THRESHOLDS = (10.0, 25.0)
POSE_LIMITS = {"yaw": (-120.0, 120.0), "pitch": (-90.0, 90.0), "roll": (-90.0, 90.0)}

# Bucket labels emitted by ``_pose_bucket`` / ``_pitch_bucket``. Coverage,
# readiness, and pruning all read these labels (in particular scoring.py needs
# them as module-level constants to size joint pose grids without re-deriving
# them from the threshold tuples).
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
POSE_HIGH_CONFIDENCE_DELTA = 10.0
POSE_MEDIUM_CONFIDENCE_DELTA = 25.0
DEFAULT_IDENTITY_BACKFILL_MODEL = "arcface"
MIN_IDENTITY_QUALITY_FACES = 3
IDENTITY_REJECT_THRESHOLD = 0.15
IDENTITY_OUTLIER_THRESHOLD = 0.25
IDENTITY_BORDERLINE_THRESHOLD = 0.35


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
    identity_embedding_coverage_ratio: float | None = None
    identity_decision_coverage_ratio: float | None = None
    identity_unknown_ratio: float | None = None
    mask_qa_distribution: dict[str, int] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    joint_pose_coverage: dict[str, T.Any] = field(default_factory=dict)
    expression_coverage: dict[str, T.Any] = field(default_factory=dict)
    lighting_coverage: dict[str, T.Any] = field(default_factory=dict)
    image_metrics_provenance: dict[str, int] = field(default_factory=dict)
    """Number of records grouped by image_metrics provenance tag."""

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
    _, entries = load_alignments_envelope(path)
    records: list[tuple[str, int, FileAlignments]] = []
    for frame, entry in entries.items():
        records.extend((frame, idx, face) for idx, face in enumerate(entry.faces))
    return records


def load_alignments_envelope(
    path: str | Path,
) -> tuple[dict[str, T.Any], dict[str, AlignmentsEntry]]:
    """Return ``(raw_envelope, entries_by_frame)`` for a FaceQA enrichment pass.

    The raw envelope retains any ``__meta__`` block so callers can mutate the
    returned ``entries`` (for example to backfill SPIGA pose or identity
    quality into ``face.metadata['faceqa']``) and persist them back via
    :func:`save_alignments_envelope` without losing alignments-file metadata.
    """
    serializer = get_serializer("compressed")
    raw = serializer.load(str(path))
    raw_dict: dict[str, T.Any] = raw if isinstance(raw, dict) else {"__data__": raw}
    data_block = raw_dict.get("__data__")
    data: dict[str, T.Any] = data_block if isinstance(data_block, dict) else raw_dict
    entries: dict[str, AlignmentsEntry] = {
        frame: AlignmentsEntry.from_dict(value) for frame, value in data.items()
    }
    return raw_dict, entries


def save_alignments_envelope(
    path: str | Path,
    raw: dict[str, T.Any],
    entries: dict[str, AlignmentsEntry],
) -> None:
    """Persist mutated FaceQA-enriched entries back into the alignments file."""
    serializer = get_serializer("compressed")
    new_data = {frame: entry.to_dict() for frame, entry in entries.items()}
    if isinstance(raw.get("__data__"), dict):
        raw["__data__"] = new_data
        serializer.save(str(path), raw)
    else:
        serializer.save(str(path), new_data)


def _write_faceqa_metadata(face: FileAlignments, key: str, value: T.Any) -> None:
    """Write ``face.metadata['faceqa'][key] = value`` (create envelope if needed)."""
    envelope = face.metadata.get("faceqa")
    if not isinstance(envelope, dict):
        envelope = {}
        face.metadata["faceqa"] = envelope
    envelope[key] = value


def records_from_alignments(
    source: str | Path | dict[str, AlignmentsEntry],
    *,
    pose_backfiller: Callable[[FaceQARecord, FileAlignments], dict[str, T.Any] | None]
    | None = None,
    metrics_backfiller: Callable[[FileAlignments, str, int], dict[str, T.Any] | None]
    | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> list[FaceQARecord]:
    """Return FaceQA records derived from alignments + embedded ``face.metadata['faceqa']``.

    ``source`` accepts either an alignments file path or a pre-loaded
    ``entries`` dict (as returned by :func:`load_alignments_envelope`) so the
    FaceQA dispatcher can backfill in-memory and persist once at the end.

    ``metrics_backfiller`` reconstructs the aligned crop from the source frame
    so blur / lighting / black-pixel metrics carry the authoritative
    ``frame_aligned_crop`` provenance. When absent, FaceQA falls back to the
    stored thumbnail (``thumbnail_fallback`` provenance).

    ``progress_callback`` (when supplied) is invoked with ``1`` after each
    face is processed so the FaceQA dispatcher can drive a ``tqdm`` bar the
    GUI process wrapper consumes.
    """
    if isinstance(source, dict):
        entries = source
    else:
        _, entries = load_alignments_envelope(source)
    records: list[FaceQARecord] = []
    for frame, entry in entries.items():
        for idx, face in enumerate(entry.faces):
            record = _record_from_alignment(frame, idx, face)
            _populate_image_metrics(record, face, metrics_backfiller=metrics_backfiller)
            if pose_backfiller is not None and not _valid_spiga_pose(record):
                pose = pose_backfiller(record, face)
                if pose is not None:
                    _apply_backfilled_pose(record, pose)
                    _write_faceqa_metadata(face, "pose", pose)
            _select_pose(record)
            _populate_expression(record, face)
            records.append(record)
            if progress_callback is not None:
                progress_callback(1)
    return records


def _record_from_alignment(frame: str, idx: int, face: FileAlignments) -> FaceQARecord:
    """Build one record from the fields stored in an alignments face."""
    record = FaceQARecord(
        frame=frame,
        face_index=idx,
        resolution=[int(face.w), int(face.h)],
        mask_qa_ref=_mask_ref(face),
    )
    if face.identity:
        # Initialize ``identity_model`` from the alignment payload itself
        # (comma-joined identity keys). ``_populate_metadata`` below may
        # intentionally overwrite this with a stored, more specific
        # ``identity_model`` value if the metadata envelope carries one
        # (issue #192 P3 — small explanatory comment).
        record.identity_model = ",".join(sorted(face.identity))
    _populate_aligned_metrics(record, face)
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
    record.expression_bucket = expression_bucket_for_features(features)


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


# Provenance tags written to face.metadata["faceqa"]["image_metrics"]["provenance"].
# Frame-derived metrics are authoritative; thumbnail-derived metrics carry
# reduced trust and are surfaced separately in coverage so readiness can
# warn / down-weight them.
IMAGE_METRICS_PROVENANCE_FRAME = "frame_aligned_crop"
IMAGE_METRICS_PROVENANCE_THUMBNAIL = "thumbnail_fallback"
# NOTE: a "faces_fallback" provenance was previously declared as a constant but
# never had a backfill path; readers can no longer rely on it. Add the
# constant + implementation together when extracted-face fallback is wired in.

# These top-level record fields are populated exclusively by _populate_image_metrics.
# Generic faceqa metadata loading must not copy stale top-level metric values first,
# otherwise an authoritative frame-derived payload can be blocked by old fallback data.
_IMAGE_METRIC_RECORD_FIELDS: frozenset[str] = frozenset(
    {
        "blur_score",
        "blur_fft_score",
        "black_pixel_ratio",
        "color_gray",
        "color_luma",
        "color_green",
        "color_orange",
        "mean_luminance",
        "luminance_variance",
        "contrast",
        "left_right_ratio",
        "top_bottom_ratio",
        "saturation",
        "color_warmth",
        "image_metrics_provenance",
    }
)


def _compute_image_metrics(image: np.ndarray) -> dict[str, T.Any] | None:
    """Compute blur / black-pixel / lighting metrics from a BGR uint8 crop."""
    if image is None or image.size == 0:
        return None
    metrics: dict[str, T.Any] = {"blur_score": _estimate_blur(image)}
    if image.ndim == 3:
        # ``image.any(axis=2)`` reduces to a single ``(H, W)`` bool array
        # along the channel axis, then the inverted ``~`` flag-true means
        # every channel was zero. This avoids the previous
        # ``image == [0, 0, 0]`` allocation of a full ``(H, W, 3)`` bool
        # grid before reducing (issue #192 P2).
        metrics["black_pixel_ratio"] = float((~image.any(axis=2)).mean())
    features = compute_lighting_features(image)
    if features is not None:
        for key in (
            "mean_luminance",
            "luminance_variance",
            "contrast",
            "left_right_ratio",
            "top_bottom_ratio",
            "saturation",
            "color_warmth",
        ):
            metrics[key] = features[key]
    return metrics


def _apply_image_metrics(
    record: FaceQARecord,
    payload: dict[str, T.Any],
    *,
    overwrite: bool = False,
) -> None:
    """Copy metric values from an image_metrics payload onto a record.

    ``overwrite=True`` is reserved for authoritative frame-derived metrics. It
    lets a new frame crop replace stale top-level/fallback values that may have
    been persisted by older FaceQA runs.
    """
    for key, value in payload.items():
        if key == "provenance":
            if value is not None and (overwrite or record.image_metrics_provenance is None):
                record.image_metrics_provenance = str(value)
            continue
        if not hasattr(record, key):
            continue
        if overwrite or getattr(record, key) in (None, []):
            setattr(record, key, value)


def _populate_image_metrics(
    record: FaceQARecord,
    face: FileAlignments,
    *,
    metrics_backfiller: Callable[[FileAlignments, str, int], dict[str, T.Any] | None]
    | None = None,
) -> None:
    """Resolve image metrics for one face, preferring frame-derived values.

    Frame-derived metrics are authoritative. Existing thumbnail-fallback metrics
    from a prior run must NOT block a frame upgrade when ``metrics_backfiller``
    is available; the metric source policy is frames > thumbnail, irrespective
    of which one ran first chronologically.

    Resolution order:

    1. Existing ``face.metadata['faceqa']['image_metrics']`` IF its provenance
       is already ``frame_aligned_crop`` — that's the authoritative source and
       there's nothing to upgrade.
    2. ``metrics_backfiller`` (frame-derived; provenance ``frame_aligned_crop``).
       When this succeeds it overwrites any stale thumbnail-fallback block.
    3. Existing non-frame metadata as a fallback (preserves whatever a prior
       thumbnail run wrote when the backfiller is unavailable or fails).
    4. Decoded alignments thumbnail (provenance ``thumbnail_fallback``).
    """
    existing = face.metadata.get("faceqa", {}).get("image_metrics")
    existing_provenance = existing.get("provenance") if isinstance(existing, dict) else None

    if (
        isinstance(existing, dict)
        and existing_provenance == IMAGE_METRICS_PROVENANCE_FRAME
        and len(existing) > 1
    ):
        _apply_image_metrics(record, existing, overwrite=True)
        return

    if metrics_backfiller is not None:
        payload = metrics_backfiller(face, record.frame, record.face_index)
        if isinstance(payload, dict) and len(payload) > 0:
            payload.setdefault("provenance", IMAGE_METRICS_PROVENANCE_FRAME)
            _write_faceqa_metadata(face, "image_metrics", payload)
            _apply_image_metrics(record, payload, overwrite=True)
            return

    if isinstance(existing, dict) and len(existing) > 1:
        _apply_image_metrics(record, existing)
        return

    image = _decode_thumb(face.thumb)
    if image is None:
        return
    metrics = _compute_image_metrics(image)
    if metrics is None:
        return
    metrics["provenance"] = IMAGE_METRICS_PROVENANCE_THUMBNAIL
    _write_faceqa_metadata(face, "image_metrics", metrics)
    _apply_image_metrics(record, metrics)


_IDENTITY_ENVELOPE_FIELDS: tuple[tuple[str, str], ...] = (
    ("model", "identity_model"),
    ("score", "identity_score"),
    ("quality_flag", "identity_quality_flag"),
    ("final_decision", "identity_final_decision"),
    ("primary_model", "identity_primary_model"),
    ("verifier_model", "identity_verifier_model"),
    ("primary_score", "identity_primary_score"),
    ("verifier_score", "identity_verifier_score"),
    ("agreement", "identity_agreement"),
    ("decision_reason", "identity_decision_reason"),
    ("verifier_trigger", "identity_verifier_trigger"),
    ("threshold_used", "identity_threshold_used"),
    ("consensus", "identity_consensus"),
    ("cluster", "identity_cluster"),
)


# Frozen set of every ``FaceQARecord`` field — used by
# ``_populate_metadata`` to avoid walking every dataclass field per face
# when most metadata payloads carry only a handful of keys
# (issue #192 P2).
_RECORD_FIELDS: frozenset[str] = frozenset(
    FaceQARecord.__dataclass_fields__  # pylint:disable=no-member
)


def _populate_metadata(record: FaceQARecord, metadata: dict[str, T.Any]) -> None:
    """Copy known FaceQA fields from per-face metadata when present.

    Iterates the payload keys (typically a small handful) rather than
    every dataclass field per face — the previous shape paid
    ``len(FaceQARecord.fields)`` Python iterations per record (issue
    #192 P2).
    """
    payload = metadata.get("faceqa", metadata)
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if key not in _RECORD_FIELDS or key in _IMAGE_METRIC_RECORD_FIELDS:
            continue
        if getattr(record, key) in (None, []):
            setattr(record, key, value)
    identity_payload = payload.get("identity")
    if isinstance(identity_payload, dict):
        for src_key, dst_key in _IDENTITY_ENVELOPE_FIELDS:
            if getattr(record, dst_key) in (None, []) and src_key in identity_payload:
                setattr(record, dst_key, identity_payload[src_key])
    # NOTE: ``image_metrics`` is intentionally NOT walked here — _populate_image_metrics
    # is the single owner of those fields. Without that exclusion a persisted
    # thumbnail_fallback block would race with a frame backfill and lock in
    # stale values before the frame upgrade ran.


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
        """Return SPIGA pose metadata for a face, or ``None`` when unavailable.

        The crop fed to SPIGA mirrors the normal extractor path: a
        detector-style square ROI carved out of the RAW source frame, not a
        head-normalized aligned crop. See :meth:`_detector_style_crop` for
        the exact reproduction of :class:`lib.infer.align.Aligner`'s ROI
        math (issue #190).
        """
        if self._disabled_reason is not None:
            return None
        plugin = self._load_plugin()
        if plugin is None:
            return None
        detector_crop = self._detector_style_crop(record, face, plugin)
        if detector_crop is None:
            return None
        feed = detector_crop.astype("float32", copy=False) / 255.0
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

    def _detector_style_crop(
        self,
        record: FaceQARecord,
        face: FileAlignments,
        plugin: T.Any,
    ) -> np.ndarray | None:
        """Reproduce SPIGA's detector-bbox crop path against the raw frame.

        Mirrors :class:`lib.infer.align.Aligner`'s pre-process path so the
        backfilled SPIGA input matches what the normal extractor would have
        produced (issue #190). The fix replaces the previous head-normalized
        aligned crop (which removed / distorted pose cues) with:

        1. A detector bbox derived from the stored alignment rectangle
           ``(face.x, face.y, face.x + face.w, face.y + face.h)``.
        2. ``plugin.pre_process(bbox)`` builds the square ROI using SPIGA's
           own ``target_dist`` / centering math — calling the plugin
           preserves any future SPIGA-config tuning automatically.
        3. ROI clamped to frame bounds; the destination region inside the
           SPIGA-input canvas is computed from the clamp delta so partial
           off-frame faces map correctly without stretching.
        4. Raw frame crop is resized into the destination box on a
           ``(input_size, input_size, 3)`` zero canvas. Padding is zeros,
           matching the extractor's :meth:`_crop_and_resize` behaviour.
        5. Colour order respects ``plugin.is_rgb``; OpenCV frames are BGR
           by default which is what SPIGA expects.
        """
        image = self._load_frame(record.frame)
        if image is None:
            return None
        height, width = image.shape[:2]

        # Detector-style bbox from the stored alignment rectangle.
        bbox = np.array(
            [
                [
                    int(face.x),
                    int(face.y),
                    int(face.x + face.w),
                    int(face.y + face.h),
                ]
            ],
            dtype=np.int32,
        )
        try:
            roi = plugin.pre_process(bbox)
        except Exception as err:  # pylint:disable=broad-except
            logger.warning(
                "SPIGA pre_process failed during backfill (frame: %s, face: %s): %s",
                record.frame,
                record.face_index,
                err,
            )
            return None
        if roi is None or len(roi) == 0:
            return None
        original = np.asarray(roi[0], dtype=np.int32)
        if original.shape != (4,):
            return None
        side = int(original[2] - original[0])
        if side <= 0 or int(original[3] - original[1]) != side:
            # SPIGA's pre_process is meant to return a square; if it didn't
            # we can't safely reconstruct the crop.
            return None

        # Clamp the ROI to the frame so we never read out-of-bounds pixels.
        # Match :meth:`lib.infer.align.Aligner._clamp_roi`: every coord is
        # clipped to ``dim - 1`` (not ``dim``). The destination-box math
        # downstream depends on this so partial off-frame faces produce the
        # same destination shape the extractor would (issue #190 follow-up).
        clamp_l = int(np.clip(int(original[0]), 0, width - 1))
        clamp_t = int(np.clip(int(original[1]), 0, height - 1))
        clamp_r = int(np.clip(int(original[2]), 0, width - 1))
        clamp_b = int(np.clip(int(original[3]), 0, height - 1))
        if clamp_r <= clamp_l or clamp_b <= clamp_t:
            return None

        input_size = int(plugin.input_size)
        scale = input_size / side

        # Map the clamped ROI back into the SPIGA-input canvas, mirroring
        # ``lib.infer.align.Aligner._get_destinations`` so partial off-frame
        # faces produce the same destination box the extractor would.
        dest_l = int(round((clamp_l - int(original[0])) * scale))
        dest_t = int(round((clamp_t - int(original[1])) * scale))
        dest_r = int(round((clamp_r - int(original[0])) * scale))
        dest_b = int(round((clamp_b - int(original[1])) * scale))
        dest_l = max(0, min(input_size, dest_l))
        dest_t = max(0, min(input_size, dest_t))
        dest_r = max(0, min(input_size, dest_r))
        dest_b = max(0, min(input_size, dest_b))
        if dest_r <= dest_l or dest_b <= dest_t:
            return None

        src = image[clamp_t:clamp_b, clamp_l:clamp_r]
        if src.size == 0:
            return None
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        canvas = np.zeros((input_size, input_size, 3), dtype=image.dtype)
        resized = cv2.resize(src, (dest_r - dest_l, dest_b - dest_t), interpolation=interpolation)
        canvas[dest_t:dest_b, dest_l:dest_r] = resized

        # OpenCV produces BGR; SPIGA's ``is_rgb`` is False so this is already
        # correct. Honour the plugin's preference for future-proofing.
        if getattr(plugin, "is_rgb", False):
            canvas = canvas[..., ::-1].copy()
        return T.cast(np.ndarray, canvas)

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


# ---------------------------------------------------------------------------
# Identity backfilling — analogous to SpigaPoseBackfiller above.
#
# Identity vectors live on FileAlignments.identity (dict[str, np.ndarray]).
# When the pruning/redundancy pipeline asks "does this face have an identity
# guardrail?", we want a yes wherever possible. The identity backfiller picks
# the majority identity model already present on the faceset (same shape as
# the redundancy layer's has_identity check), loads that plugin lazily, and
# fills in any missing vectors from the source frames.
# ---------------------------------------------------------------------------


# Map of identity ``storage_name`` (the key used inside ``face.identity``) to
# ``(module_path, class_name)`` for the in-tree identity plugins. New plugins
# can be registered here; the value must be a no-argument constructor.
_IDENTITY_PLUGIN_REGISTRY: dict[str, tuple[str, str]] = {
    "insightface": ("plugins.extract.identity.insightface", "InsightFace"),
    "adaface": ("plugins.extract.identity.adaface", "AdaFace"),
    "arcface": ("plugins.extract.identity.arcface", "ArcFace"),
    "vggface2": ("plugins.extract.identity.vggface2", "VGGFace2"),
}


@dataclass
class IdentityBackfillReport:
    """Summary of an identity backfill pass."""

    model: str | None = None
    total_faces: int = 0
    already_present: int = 0
    backfilled: int = 0
    skipped_no_frame: int = 0
    skipped_failed: int = 0
    disabled_reason: str | None = None
    persisted: bool = False

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "model": self.model,
            "total_faces": self.total_faces,
            "already_present": self.already_present,
            "backfilled": self.backfilled,
            "skipped_no_frame": self.skipped_no_frame,
            "skipped_failed": self.skipped_failed,
            "disabled_reason": self.disabled_reason,
            "persisted": self.persisted,
        }


@dataclass
class IdentityCoverageStatus:
    """Summary of existing identity embedding coverage in an alignments file."""

    model: str | None = None
    total_faces: int = 0
    available_faces: int = 0
    missing_faces: int = 0
    model_counts: dict[str, int] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Return whether every face has a vector for the selected model."""
        return self.total_faces == self.available_faces

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "model": self.model,
            "total_faces": self.total_faces,
            "available_faces": self.available_faces,
            "missing_faces": self.missing_faces,
            "model_counts": self.model_counts,
            "complete": self.complete,
        }


@dataclass
class IdentityQualityReport:
    """Summary of identity quality decisions derived from embeddings."""

    model: str | None = None
    total_faces: int = 0
    vectors_available: int = 0
    classified: int = 0
    inlier: int = 0
    borderline: int = 0
    outlier: int = 0
    reject: int = 0
    updated: bool = False
    disabled_reason: str | None = None

    def to_dict(self) -> dict[str, T.Any]:
        return {
            "model": self.model,
            "total_faces": self.total_faces,
            "vectors_available": self.vectors_available,
            "classified": self.classified,
            "inlier": self.inlier,
            "borderline": self.borderline,
            "outlier": self.outlier,
            "reject": self.reject,
            "updated": self.updated,
            "disabled_reason": self.disabled_reason,
        }


# Tie-break order when two identity models have the same coverage count.
# ArcFace stays the stable FaceQA default for equal coverage; ``insightface``
# / ``adaface`` / ``vggface2`` follow. Unknown models fall back to
# deterministic alphabetical ordering — see ``_identity_model_priority``.
IDENTITY_MODEL_PRIORITY: tuple[str, ...] = (
    "arcface",
    "insightface",
    "adaface",
    "vggface2",
)


def _identity_model_priority(model: str) -> int:
    """Return the ranked position of ``model`` in :data:`IDENTITY_MODEL_PRIORITY`.

    Lower index == higher priority. Unknown models all share the
    ``len(IDENTITY_MODEL_PRIORITY)`` rank so the
    :func:`majority_identity_model` tie-break still has a deterministic
    fallback (alphabetical) below the known models.
    """
    try:
        return IDENTITY_MODEL_PRIORITY.index(model)
    except ValueError:
        return len(IDENTITY_MODEL_PRIORITY)


def majority_identity_model(
    faces: T.Iterable[FileAlignments],
    *,
    candidates: T.Iterable[str] | None = None,
) -> tuple[str | None, dict[str, int]]:
    """Pick the identity model with the broadest coverage on the faceset.

    Mirrors how the redundancy layer thinks about identity: the model that
    most faces already use is the most natural backfill target. Returns the
    chosen model name (or ``None`` when no vectors exist) plus the per-model
    counts for logging.
    """
    counts: dict[str, int] = {}
    allowed = set(candidates) if candidates is not None else None
    for face in faces:
        for key, vec in face.identity.items():
            if allowed is not None and key not in allowed:
                continue
            if vec is None:
                continue
            # Skip the ``np.asarray(vec).ravel()`` allocation for the
            # common path where ``vec`` is already a sized ndarray; fall
            # back to a single ``len`` check otherwise (issue #192 P2).
            try:
                size = vec.size  # type: ignore[union-attr]
            except AttributeError:
                size = len(vec)
            if size == 0:
                continue
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None, counts
    # Tie-break order (issue follow-up):
    #   1. Higher coverage count wins.
    #   2. Earlier ``IDENTITY_MODEL_PRIORITY`` index wins on equal coverage —
    #      negated so ``max`` picks the smaller index (= higher priority).
    #   3. Alphabetical model name as the final deterministic fallback for
    #      unknown models that share the lowest priority rank.
    winning = max(
        counts.items(),
        key=lambda item: (item[1], -_identity_model_priority(item[0]), item[0]),
    )[0]
    return winning, counts


class IdentityBackfiller:
    """Backfill missing identity embeddings from source frames.

    Mirrors :class:`SpigaPoseBackfiller`: lazy plugin load, last-frame image
    caching, mutates the face object in place (writes the new vector into
    ``face.identity``), and self-disables on the first hard error.
    """

    def __init__(self, frames_loader: T.Any, *, model: str | None = None) -> None:
        self._frames_loader = frames_loader
        self._model_name = model
        self._plugin: T.Any = None
        self._disabled_reason: str | None = None
        self._last_frame: str | None = None
        self._last_image: np.ndarray | None = None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def set_model(self, model: str) -> None:
        """Set the identity model name to backfill against."""
        self._model_name = model
        self._plugin = None

    def __call__(
        self,
        face: FileAlignments,
        frame: str,
        face_index: int,
    ) -> np.ndarray | None:
        """Compute and store the embedding for ``face``.

        Returns the new vector (also written into ``face.identity[model_name]``)
        or ``None`` if the backfill could not produce one.
        """
        if self._disabled_reason is not None or self._model_name is None:
            return None
        plugin = self._load_plugin(self._model_name)
        if plugin is None:
            return None
        aligned = self._aligned_face(face, frame, face_index, plugin.input_size, plugin.centering)
        if aligned is None:
            return None
        feed = aligned[None]  # add batch dim; identity plugins expect aligned BGR crops
        try:
            pre_process = getattr(plugin, "pre_process", None)
            if callable(pre_process):
                feed = pre_process(feed)
            raw = plugin.process(feed)
            normalised = plugin.post_process(raw)
        except Exception as err:  # pylint:disable=broad-except
            self._disabled_reason = str(err)
            logger.warning("Identity backfill failed; disabling further attempts: %s", err)
            return None
        vector: np.ndarray = np.asarray(normalised, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            return None
        face.identity[self._model_name] = vector
        return vector

    def _aligned_face(
        self,
        face: FileAlignments,
        frame: str,
        face_index: int,
        size: int,
        centering: str,
    ) -> np.ndarray | None:
        image = self._load_frame(frame)
        if image is None:
            return None
        try:
            face_obj = DetectedFace()
            face_obj.from_alignment(face, image=image)
            face_obj.load_aligned(image, size=size, centering=centering)
        except Exception as err:  # pylint:disable=broad-except
            logger.warning(
                "Could not reconstruct aligned face for identity backfill "
                "(frame: %s, face: %s): %s",
                frame,
                face_index,
                err,
            )
            return None
        aligned = T.cast(np.ndarray, face_obj.aligned.face)
        if aligned.shape[:2] != (size, size):
            aligned = cv2.resize(aligned, (size, size), interpolation=cv2.INTER_CUBIC)
        return aligned

    def _load_frame(self, frame: str) -> np.ndarray | None:
        if frame == self._last_frame and self._last_image is not None:
            return self._last_image
        try:
            image = T.cast(np.ndarray, self._frames_loader.load_image(frame))
        except Exception as err:  # pylint:disable=broad-except
            logger.warning(
                "Could not load source frame for identity backfill '%s': %s", frame, err
            )
            return None
        self._last_frame = frame
        self._last_image = image
        return image

    def _load_plugin(self, model_name: str) -> T.Any | None:
        if self._plugin is not None:
            return self._plugin
        spec = _IDENTITY_PLUGIN_REGISTRY.get(model_name)
        if spec is None:
            self._disabled_reason = f"identity backfill unsupported for model '{model_name}'"
            logger.warning(self._disabled_reason)
            return None
        module_path, class_name = spec
        try:
            module = importlib.import_module(module_path)
            plugin = getattr(module, class_name)()
            plugin.model = plugin.load_model()
        except Exception as err:  # pylint:disable=broad-except
            self._disabled_reason = f"identity backfill plugin '{model_name}' unavailable: {err}"
            logger.warning(self._disabled_reason)
            return None
        self._plugin = plugin
        return plugin


class FrameImageMetricsBackfiller:
    """Compute blur / black-pixel / lighting from the SOURCE FRAME crop.

    Mirrors :class:`SpigaPoseBackfiller`: lazy caching of the last loaded
    frame so consecutive faces in one frame reuse the decode; self-disable on
    the first hard error so per-frame loops do not cascade; writes the
    resulting metrics block (with provenance ``frame_aligned_crop``) into
    ``face.metadata['faceqa']['image_metrics']``.
    """

    DEFAULT_INPUT_SIZE = 256

    def __init__(self, frames_loader: T.Any, *, input_size: int = DEFAULT_INPUT_SIZE) -> None:
        self._frames_loader = frames_loader
        self._input_size = int(input_size)
        self._disabled_reason: str | None = None
        self._last_frame: str | None = None
        self._last_image: np.ndarray | None = None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def __call__(
        self,
        face: FileAlignments,
        frame: str,
        face_index: int,
    ) -> dict[str, T.Any] | None:
        if self._disabled_reason is not None:
            return None
        image = self._load_frame(frame)
        if image is None:
            return None
        try:
            face_obj = DetectedFace()
            face_obj.from_alignment(face, image=image)
            face_obj.load_aligned(image, size=self._input_size, centering="face")
            aligned = T.cast(np.ndarray, face_obj.aligned.face)
        except Exception as err:  # pylint:disable=broad-except
            logger.warning(
                "Could not reconstruct aligned face for image-metrics backfill "
                "(frame: %s, face: %s): %s",
                frame,
                face_index,
                err,
            )
            return None
        metrics = _compute_image_metrics(aligned)
        if metrics is None:
            return None
        metrics["provenance"] = IMAGE_METRICS_PROVENANCE_FRAME
        return metrics

    def _load_frame(self, frame: str) -> np.ndarray | None:
        if frame == self._last_frame and self._last_image is not None:
            return self._last_image
        try:
            image = T.cast(np.ndarray, self._frames_loader.load_image(frame))
        except Exception as err:  # pylint:disable=broad-except
            logger.warning(
                "Could not load source frame for image-metrics backfill '%s': %s",
                frame,
                err,
            )
            self._disabled_reason = str(err)
            return None
        self._last_frame = frame
        self._last_image = image
        return image


def backfill_identity(
    alignments_path: str | Path,
    *,
    frames_loader: T.Any,
    model: str | None = None,
    save: bool = True,
    progress_callback: Callable[[int], None] | None = None,
) -> IdentityBackfillReport:
    """Backfill missing identity embeddings across an entire alignments file.

    Selects the majority identity model (unless ``model`` overrides it), runs
    :class:`IdentityBackfiller` against each face missing a vector under the
    chosen key, and — when ``save`` is true and at least one face was filled —
    writes the mutated alignments back to ``alignments_path``.
    """
    path = Path(alignments_path)
    serializer = get_serializer("compressed")
    raw = serializer.load(str(path))
    data_block = raw.get("__data__")
    data: dict[str, T.Any] = data_block if isinstance(data_block, dict) else raw

    entries: dict[str, AlignmentsEntry] = {
        frame: AlignmentsEntry.from_dict(value) for frame, value in data.items()
    }
    all_faces: list[FileAlignments] = [face for entry in entries.values() for face in entry.faces]
    report = IdentityBackfillReport(total_faces=len(all_faces))

    if model is None:
        chosen, counts = majority_identity_model(all_faces, candidates=_IDENTITY_PLUGIN_REGISTRY)
        if chosen is None:
            unsupported, unsupported_counts = majority_identity_model(all_faces)
            if unsupported is not None:
                logger.info(
                    "Existing identity model '%s' is unsupported for backfill "
                    "(coverage: %s). Falling back to '%s'.",
                    unsupported,
                    unsupported_counts.get(unsupported, 0),
                    DEFAULT_IDENTITY_BACKFILL_MODEL,
                )
            chosen = DEFAULT_IDENTITY_BACKFILL_MODEL
            counts = {}
        _log_identity_model_choice(chosen, counts, note=" (backfill)")
        report.model = chosen
    else:
        if model not in _IDENTITY_PLUGIN_REGISTRY:
            report.model = model
            report.disabled_reason = f"identity backfill unsupported for model '{model}'"
            return report
        report.model = model

    backfiller = IdentityBackfiller(frames_loader, model=report.model)
    any_backfilled = False
    for frame, entry in entries.items():
        for idx, face in enumerate(entry.faces):
            existing = face.identity.get(report.model)
            if existing is not None and np.asarray(existing).ravel().size > 0:
                report.already_present += 1
                if progress_callback is not None:
                    progress_callback(1)
                continue
            prior_image = backfiller._last_image  # noqa: SLF001
            vector = backfiller(face, frame, idx)
            if vector is None:
                if backfiller.disabled_reason is not None:
                    report.disabled_reason = backfiller.disabled_reason
                    break
                if backfiller._last_image is None and prior_image is None:  # noqa: SLF001
                    report.skipped_no_frame += 1
                else:
                    report.skipped_failed += 1
                if progress_callback is not None:
                    progress_callback(1)
                continue
            report.backfilled += 1
            any_backfilled = True
            if progress_callback is not None:
                progress_callback(1)
        if report.disabled_reason is not None:
            break

    if any_backfilled and save and report.disabled_reason is None:
        new_data = {frame: entry.to_dict() for frame, entry in entries.items()}
        if isinstance(data_block, dict):
            raw["__data__"] = new_data
            serializer.save(str(path), raw)
        else:
            serializer.save(str(path), new_data)
        report.persisted = True
    return report


def _format_identity_counts(counts: dict[str, int]) -> str:
    """Render ``counts`` as a stable ``name=N`` summary for logging.

    Higher coverage models come first; alphabetical fallback on ties so the
    log line is deterministic across runs.
    """
    if not counts:
        return "(none)"
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{name}={n}" for name, n in ordered)


def _log_identity_model_choice(
    chosen: str,
    counts: dict[str, int],
    *,
    note: str = "",
) -> None:
    """Emit the per-run identity-model selection diagnostic.

    Shows EVERY supported model that any face carried (via
    ``_format_identity_counts``) plus the winner. When two or more models
    tied at the top coverage count, the priority tie-break path is named
    explicitly so a user reading the log can audit why ``arcface`` beat
    ``insightface`` (or vice versa).
    """
    if not counts:
        if chosen:
            logger.info("Identity model selection: '%s' (no stored vectors%s)", chosen, note)
        return
    summary = _format_identity_counts(counts)
    chosen_count = counts.get(chosen, 0)
    tied = [name for name, n in counts.items() if n == chosen_count]
    if len(tied) > 1:
        logger.info(
            "Identity model selection: %d candidate(s) tied at coverage=%d "
            "(%s); priority tie-break chose '%s'%s",
            len(tied),
            chosen_count,
            ", ".join(sorted(tied)),
            chosen,
            note,
        )
    else:
        logger.info(
            "Identity model selection: '%s' wins on coverage (%s)%s",
            chosen,
            summary,
            note,
        )


def _identity_model_for_faces(
    faces: T.Iterable[FileAlignments],
    *,
    model: str | None = None,
) -> tuple[str | None, dict[str, int]]:
    """Return the identity model to use for coverage/backfill decisions."""
    faces_list = list(faces)
    if model is not None:
        all_counts = majority_identity_model(faces_list)[1]
        logger.info(
            "Identity model selection: explicit override '%s' (stored coverage: %s)",
            model,
            _format_identity_counts(all_counts),
        )
        return model, all_counts

    chosen, counts = majority_identity_model(faces_list, candidates=_IDENTITY_PLUGIN_REGISTRY)
    if chosen is not None:
        _log_identity_model_choice(chosen, counts)
        return chosen, counts

    unsupported, unsupported_counts = majority_identity_model(faces_list)
    if unsupported is not None:
        logger.info(
            "Identity model selection: stored models %s are unsupported for "
            "backfill; using default '%s'.",
            _format_identity_counts(unsupported_counts),
            DEFAULT_IDENTITY_BACKFILL_MODEL,
        )
        return DEFAULT_IDENTITY_BACKFILL_MODEL, unsupported_counts

    logger.info(
        "Identity model selection: no stored identity vectors; using default '%s'.",
        DEFAULT_IDENTITY_BACKFILL_MODEL,
    )
    return DEFAULT_IDENTITY_BACKFILL_MODEL, {}


def identity_coverage_status(
    alignments_path: str | Path,
    *,
    model: str | None = None,
) -> IdentityCoverageStatus:
    """Return current embedding coverage for the selected identity model."""
    faces = [face for _frame, _idx, face in load_alignments_records(alignments_path)]
    selected_model, counts = _identity_model_for_faces(faces, model=model)
    if selected_model is None:
        return IdentityCoverageStatus(total_faces=len(faces), model_counts=counts)

    def _has_vector(face: FileAlignments) -> bool:
        # Bind ``face.identity.get(selected_model)`` once per face so the
        # subsequent ``.size`` check doesn't dictate a second lookup +
        # ``np.asarray().ravel()`` allocation (issue #192 P3).
        vec = face.identity.get(selected_model)
        if vec is None:
            return False
        try:
            return bool(vec.size > 0)  # type: ignore[union-attr]
        except AttributeError:
            return len(vec) > 0

    available = sum(_has_vector(face) for face in faces)
    total = len(faces)
    return IdentityCoverageStatus(
        model=selected_model,
        total_faces=total,
        available_faces=available,
        missing_faces=max(0, total - available),
        model_counts=counts,
    )


def _normalise_identity_vector(vector: T.Any) -> np.ndarray | None:
    """Return a finite, unit-length identity vector or ``None``."""
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return None
    normalised: np.ndarray = arr / norm
    return normalised


def _identity_decision(score: float) -> tuple[str, str]:
    """Return ``(quality_flag, final_decision)`` for a centroid similarity."""
    if score < IDENTITY_REJECT_THRESHOLD:
        return "reject", "reject"
    if score < IDENTITY_OUTLIER_THRESHOLD:
        return "outlier", "outlier"
    if score < IDENTITY_BORDERLINE_THRESHOLD:
        return "borderline", "review"
    return "inlier", "inlier"


def _set_record_value(record: FaceQARecord, field_name: str, value: T.Any) -> bool:
    """Set a FaceQARecord attribute when present and changed."""
    if not hasattr(record, field_name):
        return False
    if getattr(record, field_name) == value:
        return False
    setattr(record, field_name, value)
    return True


def compute_identity_quality(
    records: list[FaceQARecord],
    source: str | Path | dict[str, AlignmentsEntry],
    *,
    model: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> IdentityQualityReport:
    """Derive identity availability/outlier decisions before coverage.

    Reads identity embeddings from ``face.identity[model]`` on each alignment,
    classifies each face against the faceset centroid, mutates the matching
    FaceQA record so :func:`compute_coverage` can report identity buckets and
    ``identity_outlier_ratio``, and persists the decision into
    ``face.metadata['faceqa']['identity']`` so subsequent FaceQA passes can
    read the decision back without re-running the centroid math.

    ``source`` may be an alignments file path or a pre-loaded ``entries`` dict;
    the dict variant lets the FaceQA dispatcher avoid a redundant disk read.
    """
    if isinstance(source, dict):
        entries = source
    else:
        _, entries = load_alignments_envelope(source)

    faces_by_key: dict[tuple[str, int], FileAlignments] = {}
    for frame, entry in entries.items():
        for idx, face in enumerate(entry.faces):
            faces_by_key[(frame, idx)] = face

    selected_model, _counts = _identity_model_for_faces(faces_by_key.values(), model=model)
    report = IdentityQualityReport(total_faces=len(records), model=selected_model)

    def _tick_all_records() -> None:
        """Tick the progress callback once per record so an early-return path
        still drains the GUI tqdm bar to ``len(records)`` (P2 follow-up to
        #187). Without this every disabled-classifier path could leave the
        bar short of its declared total."""
        if progress_callback is None:
            return
        for _r in records:
            progress_callback(1)

    if selected_model is None:
        report.disabled_reason = "no identity model available"
        _tick_all_records()
        return report

    vectors_by_key: dict[tuple[str, int], np.ndarray] = {}
    for key, face in faces_by_key.items():
        vector = _normalise_identity_vector(face.identity.get(selected_model))
        if vector is not None:
            vectors_by_key[key] = vector
    report.vectors_available = len(vectors_by_key)
    if len(vectors_by_key) < MIN_IDENTITY_QUALITY_FACES:
        report.disabled_reason = (
            f"not enough identity vectors to classify outliers "
            f"({len(vectors_by_key)}/{MIN_IDENTITY_QUALITY_FACES})"
        )
        _tick_all_records()
        return report

    matrix = np.stack(list(vectors_by_key.values())).astype(np.float32, copy=False)
    centroid = matrix.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm <= 0.0 or not np.all(np.isfinite(centroid)):
        report.disabled_reason = "identity centroid is invalid"
        _tick_all_records()
        return report
    centroid = centroid / centroid_norm

    classified_keys = set(vectors_by_key)
    # The dispatcher binds this tqdm to ``len(records)`` so the bar tracks
    # the FACE count — including unclassified records that have no vector.
    # Walk records in their declared order, tick per record, and only run
    # the dot-product / decision math when a vector exists. This guarantees
    # the bar reaches 100% even when some faces lack a usable embedding.
    for record in records:
        if progress_callback is not None:
            progress_callback(1)
        key = (record.frame, record.face_index)
        if key not in classified_keys:
            continue
        vector = vectors_by_key[key]
        face = faces_by_key.get(key)
        score = float(np.dot(vector, centroid))
        quality_flag, final_decision = _identity_decision(score)
        report.classified += 1
        if quality_flag == "reject":
            report.reject += 1
        elif quality_flag == "outlier":
            report.outlier += 1
        elif quality_flag == "borderline":
            report.borderline += 1
        else:
            report.inlier += 1

        changed = False
        changed |= _set_record_value(record, "identity_model", selected_model)
        changed |= _set_record_value(record, "identity_score", score)
        changed |= _set_record_value(record, "identity_quality_flag", quality_flag)
        changed |= _set_record_value(record, "identity_final_decision", final_decision)
        report.updated = report.updated or changed

        if face is not None:
            _write_faceqa_metadata(
                face,
                "identity",
                {
                    "model": selected_model,
                    "score": score,
                    "quality_flag": quality_flag,
                    "final_decision": final_decision,
                },
            )

    return report


def _compute_identity_unknown_ratio(
    counts: dict[str, dict[str, int]],
    total: int,
) -> float | None:
    if total == 0:
        return None
    return counts.get("identity", {}).get("unknown", 0) / total


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


# Fields summarised by ``_metric_summary``. Kept in one place so the
# single-pass record walk below stays in sync with the dictionary order
# (issue #192 P2 high — reduce ``records × metrics`` Python iterations to
# a single ``records`` walk per metric pass).
_METRIC_SUMMARY_FIELDS: tuple[str, ...] = (
    "yaw",
    "pitch",
    "roll",
    "pose_max_abs_delta",
    "blur_score",
    "average_distance",
    "mean_luminance",
    "luminance_variance",
    "contrast",
    "left_right_ratio",
    "top_bottom_ratio",
    "saturation",
    "color_warmth",
    "mouth_openness",
    "mouth_width_ratio",
    "smile_proxy",
    "eye_closure",
    "brow_raise_proxy",
    "expression_asymmetry",
)


def _metric_summary(records: list[FaceQARecord]) -> dict[str, dict[str, float | None]]:
    """Return min / median / max per metric field across ``records``.

    Walks the records ONCE and pushes each metric's value into its own
    buffer. The previous shape paid ``len(records) × 19`` Python
    iterations (one list comprehension per field) plus a separate
    ``_summarize`` allocation per metric; the single-pass form below
    keeps the total walks linear in ``len(records)``.
    """
    buffers: dict[str, list[float | None]] = {
        metric_field: [] for metric_field in _METRIC_SUMMARY_FIELDS
    }

    for record in records:
        for metric_field in _METRIC_SUMMARY_FIELDS:
            buffers[metric_field].append(getattr(record, metric_field))

    return {name: _summarize(values) for name, values in buffers.items()}


def _summarize(values: list[float | None]) -> dict[str, float | None]:
    numeric = np.array([value for value in values if value is not None], dtype="float32")
    if numeric.size == 0:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(np.min(numeric)),
        "median": float(np.median(numeric)),
        "max": float(np.max(numeric)),
    }


def _bucketed_coverage(
    records: list[FaceQARecord],
    *,
    buckets: T.Sequence[str],
    bucket_fn: T.Callable[[FaceQARecord], str],
    prefix: str,
) -> dict[str, T.Any]:
    """Return counts / percentages / entropy / occupancy for one bucket
    dimension.

    Extracted helper consumed by both ``_lighting_coverage`` and
    ``_expression_coverage`` (issue #192 P2 high — the two functions
    duplicated count / unknown / classified / occupied / entropy /
    percentage / missing-bin logic). ``prefix`` shapes the dimension-
    specific output keys (``f"occupied_{prefix}_bins"`` etc.) so the
    legacy JSON layout is preserved.
    """
    counts: dict[str, int] = {bucket: 0 for bucket in buckets}
    unknown = 0
    for record in records:
        bucket = bucket_fn(record)
        if bucket == "unknown":
            unknown += 1
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
    classified = sum(counts.values())
    total_bins = len(buckets)
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
        f"occupied_{prefix}_bins": occupied,
        f"empty_{prefix}_bins": total_bins - occupied,
        f"{prefix}_bin_coverage_pct": bin_coverage_pct,
        f"{prefix}_entropy": round(entropy, 4),
        "classified_faces": classified,
        "unknown_faces": unknown,
        "missing_bins": missing_bins,
    }


def _lighting_coverage(records: list[FaceQARecord]) -> dict[str, T.Any]:
    """Return lighting bucket coverage with entropy and occupancy metrics."""
    return _bucketed_coverage(
        records,
        buckets=LIGHTING_BUCKETS,
        bucket_fn=_lighting_bucket,
        prefix="lighting",
    )


def _expression_coverage(records: list[FaceQARecord]) -> dict[str, T.Any]:
    """Return expression bucket coverage with entropy and occupancy metrics."""
    return _bucketed_coverage(
        records,
        buckets=EXPRESSION_BUCKETS,
        bucket_fn=_expression_bucket,
        prefix="expression",
    )


# The per-record bucket classifiers (``_pose_bucket`` etc.) used to live
# here; they were promoted to ``lib.faceqa.buckets`` in issue #192
# Priority 1 to remove the private-API coupling between ``coverage`` and
# ``redundancy``. The names are aliased into this module via the import
# block at the top of the file so the existing internal call sites (the
# ``_BUCKET_DIMENSIONS`` dispatch table below, ``_metric_summary``, etc.)
# continue to work unchanged.


def _mask_ref(face: FileAlignments) -> str | None:
    """Return a stable mask QA reference for a face.

    The FaceQA quality model only needs to know *whether* any mask is on file
    and which keys are available; downstream consumers (mask_qa_distribution)
    treat ``None`` as missing.
    """
    if not face.mask:
        return None
    return ",".join(sorted(face.mask))


_BUCKET_DIMENSIONS: tuple[tuple[str, T.Callable[[FaceQARecord], str]], ...] = (
    ("pose", _pose_bucket),
    ("pitch", _pitch_bucket),
    ("lighting", _lighting_bucket),
    ("expression", _expression_bucket),
    ("blur", _blur_bucket),
    ("resolution", _resolution_bucket),
    ("misalignment", _misalignment_bucket),
    ("occlusion", _occlusion_bucket),
    ("identity", _identity_bucket),
    ("mask_qa", _mask_qa_bucket),
    ("pose_sources", _pose_source_bucket),
    ("pose_confidence", _pose_confidence_bucket),
)


# Canonical label sets per bucket dimension. Pre-populating counts so empty
# buckets surface as 0 (not absent) lets readiness highlight underrepresented
# slots without having to know which buckets are theoretically possible.
_BUCKET_LABELS: dict[str, tuple[str, ...]] = {
    "pose": tuple(YAW_BUCKETS) + ("unknown",),
    "pitch": tuple(PITCH_BUCKETS) + ("unknown",),
    "lighting": tuple(LIGHTING_BUCKETS) + ("unknown",),
    "expression": tuple(EXPRESSION_BUCKETS) + ("unknown",),
    "blur": ("good", "ok", "mediocre", "unusable", "unknown"),
    "resolution": ("good", "ok", "low", "tiny", "unknown"),
    "misalignment": ("low", "medium", "high", "extreme", "unknown"),
    "occlusion": ("minimal", "mild", "moderate", "severe", "unknown"),
    "identity": ("inlier", "borderline", "outlier", "reject", "review", "unknown"),
    "mask_qa": ("present", "missing"),
    "pose_sources": ("spiga", "spiga_backfill", "alignment", "unknown"),
    "pose_confidence": ("high", "medium", "low", "fallback", "unknown"),
}


def _count_buckets(records: list[FaceQARecord]) -> dict[str, dict[str, int]]:
    """Return ``{dimension: {bucket: count}}`` for every coverage dimension.

    Every canonical bucket per dimension is initialised to 0 so empty
    buckets are visible to readiness/redundancy without them having to
    know the full label set up front.
    """
    counts: dict[str, dict[str, int]] = {
        dim: {bucket: 0 for bucket in _BUCKET_LABELS.get(dim, ())}
        for dim, _fn in _BUCKET_DIMENSIONS
    }
    for record in records:
        for dim, fn in _BUCKET_DIMENSIONS:
            label = fn(record)
            counts[dim][label] = counts[dim].get(label, 0) + 1
    return counts


def _percentages(
    counts: dict[str, dict[str, int]],
    total: int,
) -> dict[str, dict[str, float]]:
    """Convert raw counts into bucket percentages (rounded to 2dp)."""
    if total <= 0:
        return {dim: {bucket: 0.0 for bucket in dim_counts} for dim, dim_counts in counts.items()}
    return {
        dim: {bucket: round(count * 100.0 / total, 2) for bucket, count in dim_counts.items()}
        for dim, dim_counts in counts.items()
    }


def _compute_duplicate_ratio(records: list[FaceQARecord]) -> float | None:
    """Return the ratio of records flagged as duplicate prune candidates.

    Returns ``None`` when no record carries any duplicate annotation so the
    quality signal_coverage and confidence layers can distinguish "no
    duplicate evidence" from "evidence + zero duplicates".
    """
    if not records:
        return None
    annotated = sum(
        1
        for record in records
        if record.duplicate_keep_recommendation is not None
        or record.duplicate_representative is not None
        or record.duplicate_cluster is not None
        or record.duplicate_cluster_id is not None
    )
    if annotated == 0:
        return None
    flagged = sum(
        1
        for record in records
        if record.duplicate_keep_recommendation in ("prune_candidate", "review")
        or record.duplicate_representative is False
    )
    return flagged / len(records)


def _compute_identity_outlier_ratio(records: list[FaceQARecord]) -> float | None:
    """Return the identity-outlier ratio, or ``None`` when no records carry an
    identity classification (distinguishes absence of evidence from a
    zero-outlier finding)."""
    if not records:
        return None
    annotated = sum(
        1
        for r in records
        if r.identity_quality_flag is not None or r.identity_final_decision is not None
    )
    if annotated == 0:
        return None
    return sum(1 for r in records if _is_identity_outlier(r)) / len(records)


def _compute_identity_embedding_coverage_ratio(
    records: list[FaceQARecord],
) -> float | None:
    if not records:
        return None
    return sum(1 for r in records if r.identity_model) / len(records)


def _compute_identity_decision_coverage_ratio(
    records: list[FaceQARecord],
) -> float | None:
    if not records:
        return None
    return sum(
        1
        for r in records
        if r.identity_quality_flag is not None or r.identity_final_decision is not None
    ) / len(records)


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


def _summarise_image_metrics_provenance(
    records: T.Sequence[FaceQARecord],
    entries: dict[str, AlignmentsEntry] | None,
) -> dict[str, int]:
    """Return ``{provenance: count}`` for image_metrics across the faceset."""
    from collections import Counter as _Counter

    if entries is None:
        # Without entries we cannot inspect face.metadata; bail with empty dict.
        return {}
    counter: _Counter[str] = _Counter()
    for entry in entries.values():
        for face in entry.faces:
            envelope = face.metadata.get("faceqa", {}).get("image_metrics")
            if isinstance(envelope, dict):
                tag = str(envelope.get("provenance") or "unknown")
            else:
                tag = "missing"
            counter[tag] += 1
    if not counter:
        # Records present but no metadata block at all.
        return {"missing": len(records)}
    return dict(counter)


def compute_coverage(
    records: T.Sequence[FaceQARecord],
    *,
    exclude_duplicates: bool = False,
    exclude_outliers: bool = False,
    entries: dict[str, AlignmentsEntry] | None = None,
) -> FacesetCoverageReport:
    """Compute coverage metrics from FaceQA records.

    ``entries`` (when supplied by the dispatcher) lets the coverage report
    surface the per-face image_metrics provenance summary so readiness can
    flag thumbnail-fallback coverage.
    """
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
        identity_embedding_coverage_ratio=_compute_identity_embedding_coverage_ratio(records_list),
        identity_decision_coverage_ratio=_compute_identity_decision_coverage_ratio(records_list),
        identity_unknown_ratio=_compute_identity_unknown_ratio(counts, total),
        mask_qa_distribution=_mask_qa_distribution(records_list),
        joint_pose_coverage=_joint_pose_coverage(records_list),
        expression_coverage=_expression_coverage(records_list),
        lighting_coverage=_lighting_coverage(records_list),
        source_counts={"alignments": total},
        image_metrics_provenance=_summarise_image_metrics_provenance(records_list, entries),
    )


__all__ = get_module_objects(__name__)
