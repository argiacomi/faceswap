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
from lib.align.faceset_qa import FaceQARecord
from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.faceqa.expression import EXPRESSION_BUCKETS, compute_expression_features
from lib.faceqa.expression import bucket_for_features as expression_bucket_for_features
from lib.faceqa.lighting import LIGHTING_BUCKETS, LightingFeatures, compute_lighting_features
from lib.faceqa.lighting import bucket_for_features as lighting_bucket_for_features
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
DISTANCE_THRESHOLDS = (0.10, 0.20, 0.40)
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
SPIGA_BACKFILL_INPUT_SIZE = 256

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
) -> list[FaceQARecord]:
    """Return FaceQA records derived from alignments + embedded ``face.metadata['faceqa']``.

    ``source`` accepts either an alignments file path or a pre-loaded
    ``entries`` dict (as returned by :func:`load_alignments_envelope`) so the
    FaceQA dispatcher can backfill in-memory and persist once at the end.

    ``metrics_backfiller`` reconstructs the aligned crop from the source frame
    so blur / lighting / black-pixel metrics carry the authoritative
    ``frame_aligned_crop`` provenance. When absent, FaceQA falls back to the
    stored thumbnail (``thumbnail_fallback`` provenance).
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
IMAGE_METRICS_PROVENANCE_FACES = "faces_fallback"


def _compute_image_metrics(image: np.ndarray) -> dict[str, T.Any] | None:
    """Compute blur / black-pixel / lighting metrics from a BGR uint8 crop."""
    if image is None or image.size == 0:
        return None
    metrics: dict[str, T.Any] = {"blur_score": _estimate_blur(image)}
    if image.ndim == 3:
        metrics["black_pixel_ratio"] = float(np.all(image == [0, 0, 0], axis=2).mean())
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


def _apply_image_metrics(record: FaceQARecord, payload: dict[str, T.Any]) -> None:
    """Copy metric values from an image_metrics payload onto a record."""
    for key, value in payload.items():
        if key == "provenance":
            continue
        if hasattr(record, key) and getattr(record, key) in (None, []):
            setattr(record, key, value)


def _populate_image_metrics(
    record: FaceQARecord,
    face: FileAlignments,
    *,
    metrics_backfiller: Callable[[FileAlignments, str, int], dict[str, T.Any] | None]
    | None = None,
) -> None:
    """Resolve image metrics for one face, preferring frame-derived values.

    Priority order:

    1. ``face.metadata['faceqa']['image_metrics']`` if previously written.
    2. ``metrics_backfiller`` (frame-derived; sets provenance ``frame_aligned_crop``).
    3. Stored alignments thumbnail (provenance ``thumbnail_fallback``).
    """
    existing = face.metadata.get("faceqa", {}).get("image_metrics")
    if isinstance(existing, dict) and len(existing) > 0:
        _apply_image_metrics(record, existing)
        return

    if metrics_backfiller is not None:
        payload = metrics_backfiller(face, record.frame, record.face_index)
        if isinstance(payload, dict) and len(payload) > 0:
            payload.setdefault("provenance", IMAGE_METRICS_PROVENANCE_FRAME)
            _write_faceqa_metadata(face, "image_metrics", payload)
            _apply_image_metrics(record, payload)
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


def _populate_metadata(record: FaceQARecord, metadata: dict[str, T.Any]) -> None:
    """Copy known FaceQA fields from per-face metadata when present."""
    payload = metadata.get("faceqa", metadata)
    if not isinstance(payload, dict):
        return
    for key in FaceQARecord.__dataclass_fields__:  # pylint:disable=no-member
        if getattr(record, key) in (None, []) and key in payload:
            setattr(record, key, payload[key])
    identity_payload = payload.get("identity")
    if isinstance(identity_payload, dict):
        for src_key, dst_key in _IDENTITY_ENVELOPE_FIELDS:
            if getattr(record, dst_key) in (None, []) and src_key in identity_payload:
                setattr(record, dst_key, identity_payload[src_key])
    image_metrics = payload.get("image_metrics")
    if isinstance(image_metrics, dict):
        for key, value in image_metrics.items():
            if hasattr(record, key) and getattr(record, key) in (None, []):
                setattr(record, key, value)


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
            arr = np.asarray(vec).ravel()
            if arr.size == 0:
                continue
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None, counts
    winning = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
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
        logger.info("Identity backfill model selected: '%s' (coverage: %s)", chosen, counts)
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
                continue
            report.backfilled += 1
            any_backfilled = True
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


def _identity_model_for_faces(
    faces: T.Iterable[FileAlignments],
    *,
    model: str | None = None,
) -> tuple[str | None, dict[str, int]]:
    """Return the identity model to use for coverage/backfill decisions."""
    faces_list = list(faces)
    if model is not None:
        return model, majority_identity_model(faces_list)[1]

    chosen, counts = majority_identity_model(faces_list, candidates=_IDENTITY_PLUGIN_REGISTRY)
    if chosen is not None:
        return chosen, counts

    unsupported, unsupported_counts = majority_identity_model(faces_list)
    if unsupported is not None:
        logger.info(
            "Existing identity model '%s' is unsupported for backfill. "
            "Using default supported model '%s' for FaceQA identity coverage.",
            unsupported,
            DEFAULT_IDENTITY_BACKFILL_MODEL,
        )
        return DEFAULT_IDENTITY_BACKFILL_MODEL, unsupported_counts

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

    available = sum(
        (
            face.identity.get(selected_model) is not None
            and np.asarray(face.identity.get(selected_model)).ravel().size > 0
        )
        for face in faces
    )
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
    if selected_model is None:
        report.disabled_reason = "no identity model available"
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
        return report

    matrix = np.stack(list(vectors_by_key.values())).astype(np.float32, copy=False)
    centroid = matrix.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm <= 0.0 or not np.all(np.isfinite(centroid)):
        report.disabled_reason = "identity centroid is invalid"
        return report
    centroid = centroid / centroid_norm

    records_by_key = {(record.frame, record.face_index): record for record in records}
    for key, vector in vectors_by_key.items():
        record = records_by_key.get(key)
        face = faces_by_key.get(key)
        if record is None:
            continue

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


def _metric_summary(records: list[FaceQARecord]) -> dict[str, dict[str, float | None]]:
    fields_ = {
        "yaw": [record.yaw for record in records],
        "pitch": [record.pitch for record in records],
        "roll": [record.roll for record in records],
        "pose_max_abs_delta": [record.pose_max_abs_delta for record in records],
        "blur_score": [record.blur_score for record in records],
        "average_distance": [record.average_distance for record in records],
        "mean_luminance": [record.mean_luminance for record in records],
        "luminance_variance": [record.luminance_variance for record in records],
        "contrast": [record.contrast for record in records],
        "left_right_ratio": [record.left_right_ratio for record in records],
        "top_bottom_ratio": [record.top_bottom_ratio for record in records],
        "saturation": [record.saturation for record in records],
        "color_warmth": [record.color_warmth for record in records],
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


def _lighting_coverage(records: list[FaceQARecord]) -> dict[str, T.Any]:
    """Return lighting bucket coverage with entropy and occupancy metrics."""
    counts: dict[str, int] = {bucket: 0 for bucket in LIGHTING_BUCKETS}
    unknown = 0
    for record in records:
        bucket = _lighting_bucket(record)
        if bucket == "unknown":
            unknown += 1
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
    classified = sum(counts.values())
    total_bins = len(LIGHTING_BUCKETS)
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
        "occupied_lighting_bins": occupied,
        "empty_lighting_bins": total_bins - occupied,
        "lighting_bin_coverage_pct": bin_coverage_pct,
        "lighting_entropy": round(entropy, 4),
        "classified_faces": classified,
        "unknown_faces": unknown,
        "missing_bins": missing_bins,
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


def _pose_bucket(record: FaceQARecord) -> str:
    """Return the yaw bucket label for a record (``unknown`` if missing)."""
    yaw = record.yaw
    if yaw is None or not np.isfinite(yaw):
        return "unknown"
    mag = abs(float(yaw))
    low, mid, high = POSE_YAW_THRESHOLDS
    if mag <= low:
        return "frontal"
    sign = "left" if yaw < 0 else "right"
    if mag <= mid:
        return f"{sign}_slight"
    if mag <= high:
        return f"{sign}_profile"
    return f"{sign}_extreme"


def _pitch_bucket(record: FaceQARecord) -> str:
    """Return the pitch bucket label for a record (``unknown`` if missing)."""
    pitch = record.pitch
    if pitch is None or not np.isfinite(pitch):
        return "unknown"
    mag = abs(float(pitch))
    low, high = POSE_PITCH_THRESHOLDS
    if mag <= low:
        return "neutral"
    sign = "down" if pitch < 0 else "up"
    if mag <= high:
        return sign
    return f"{sign}_extreme"


def _lighting_bucket(record: FaceQARecord) -> str:
    """Return the lighting bucket label for a record (``unknown`` if missing).

    Only the fields actually consulted by :func:`lighting_bucket_for_features`
    are required (mean_luminance, contrast, left_right_ratio,
    top_bottom_ratio, color_warmth); the optional ``luminance_variance`` /
    ``saturation`` slots default to ``0.0`` so legacy records that only carry
    the bucket-discriminating features still classify.
    """
    required = (
        record.mean_luminance,
        record.left_right_ratio,
        record.top_bottom_ratio,
        record.contrast,
        record.color_warmth,
    )
    if any(value is None for value in required):
        return "unknown"
    features: LightingFeatures = {
        "mean_luminance": float(T.cast(float, record.mean_luminance)),
        "luminance_variance": float(record.luminance_variance or 0.0),
        "contrast": float(T.cast(float, record.contrast)),
        "left_right_ratio": float(T.cast(float, record.left_right_ratio)),
        "top_bottom_ratio": float(T.cast(float, record.top_bottom_ratio)),
        "saturation": float(record.saturation or 0.0),
        "color_warmth": float(T.cast(float, record.color_warmth)),
    }
    label = lighting_bucket_for_features(features)
    return label if label else "unknown"


def _mask_ref(face: FileAlignments) -> str | None:
    """Return a stable mask QA reference for a face.

    The FaceQA quality model only needs to know *whether* any mask is on file
    and which keys are available; downstream consumers (mask_qa_distribution)
    treat ``None`` as missing.
    """
    if not face.mask:
        return None
    return ",".join(sorted(face.mask))


def _mask_qa_bucket(record: FaceQARecord) -> str:
    """Return the mask-availability bucket label for a face."""
    if not record.mask_qa_ref:
        return "missing"
    return "present"


def _is_identity_outlier(record: FaceQARecord) -> bool:
    """Return whether a record is classified as an identity outlier/reject."""
    flag = record.identity_quality_flag
    decision = record.identity_final_decision
    if flag in ("outlier", "reject"):
        return True
    return decision in ("outlier", "reject")


def _expression_bucket(record: FaceQARecord) -> str:
    """Return the expression bucket label (``unknown`` when missing)."""
    return record.expression_bucket or "unknown"


def _blur_bucket(record: FaceQARecord) -> str:
    """Return the blur bucket label for a record."""
    score = record.blur_score
    if score is None or not np.isfinite(score):
        return "unknown"
    good, ok_, mediocre = BLUR_THRESHOLDS
    if score >= good:
        return "good"
    if score >= ok_:
        return "ok"
    if score >= mediocre:
        return "mediocre"
    return "unusable"


def _resolution_bucket(record: FaceQARecord) -> str:
    """Return the resolution bucket label for a record."""
    if not record.resolution:
        return "unknown"
    width = record.resolution[0] if record.resolution else 0
    height = record.resolution[1] if len(record.resolution) > 1 else width
    min_side = min(int(width), int(height))
    good, ok_, low = RESOLUTION_THRESHOLDS
    if min_side >= good:
        return "good"
    if min_side >= ok_:
        return "ok"
    if min_side >= low:
        return "low"
    return "tiny"


def _misalignment_bucket(record: FaceQARecord) -> str:
    """Return the landmark-misalignment bucket label for a record."""
    distance = record.average_distance
    if distance is None or not np.isfinite(distance):
        return "unknown"
    low, medium, high = DISTANCE_THRESHOLDS
    if distance <= low:
        return "low"
    if distance <= medium:
        return "medium"
    if distance <= high:
        return "high"
    return "extreme"


def _occlusion_bucket(record: FaceQARecord) -> str:
    """Return the occlusion bucket label for a record."""
    score = record.occlusion_score
    if score is None or not np.isfinite(score):
        return "unknown"
    minimal, mild, moderate = OCCLUSION_THRESHOLDS
    if score <= minimal:
        return "minimal"
    if score <= mild:
        return "mild"
    if score <= moderate:
        return "moderate"
    return "severe"


def _identity_bucket(record: FaceQARecord) -> str:
    """Return the identity-classification bucket label for a record."""
    flag = record.identity_quality_flag or record.identity_final_decision
    if flag in ("inlier", "borderline", "outlier", "reject", "review"):
        return str(flag)
    return "unknown"


def _pose_source_bucket(record: FaceQARecord) -> str:
    return record.pose_source or "unknown"


def _pose_confidence_bucket(record: FaceQARecord) -> str:
    return record.pose_confidence or "unknown"


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
    summary: dict[str, int] = {}
    if entries is None:
        # Without entries we cannot inspect face.metadata; bail with empty dict.
        return summary
    for entry in entries.values():
        for face in entry.faces:
            envelope = face.metadata.get("faceqa", {}).get("image_metrics")
            if isinstance(envelope, dict):
                tag = str(envelope.get("provenance") or "unknown")
            else:
                tag = "missing"
            summary[tag] = summary.get(tag, 0) + 1
    if not summary:
        # Records present but no metadata block at all.
        summary = {"missing": len(records)}
    return summary


def compute_coverage(
    records: T.Sequence[FaceQARecord],
    *,
    exclude_duplicates: bool = False,
    exclude_outliers: bool = False,
    sidecar_used: bool = False,
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
        image_metrics_provenance=_summarise_image_metrics_provenance(records_list, entries),
    )


__all__ = get_module_objects(__name__)
