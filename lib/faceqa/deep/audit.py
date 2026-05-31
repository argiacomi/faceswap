#!/usr/bin/env python3
"""Deep-audit orchestration for the FaceQA ``--deep-analysis deca`` mode.

Reconstructs aligned crops from the source frames, runs them through the DECA
encoder to obtain FLAME pose / expression / lighting coefficients, computes
coverage metrics over those coefficients (see :mod:`lib.faceqa.deep.metrics`),
and assembles the ``deep_audit`` report block - including a landmark-vs-DECA
comparison and a DECA-derived readiness sub-score.

The encoder is injected, so this module is fully testable with a lightweight
synthetic encoder; nothing here imports torch directly.
"""

from __future__ import annotations

import logging
import typing as T
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from lib.align import DetectedFace
from lib.faceqa.deep import metrics as metrics_mod
from lib.faceqa.deep.deca_encoder import (
    DECA_INPUT_SIZE,
    DecaCoefficients,
    DecaEncoder,
    decode_parameters,
)
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

DEEP_AUDIT_SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZE = 16

# Fixed nominal per-group scales so the combined latent (expression + pose +
# lighting) is binned in roughly comparable units. DECA FLAME expression
# coefficients are ~unit; pose is axis-angle radians (small); SH lighting
# spans a few units. These are deliberate approximations and a tuning target
# during real-weights validation (issue #156 checklist).
EXPRESSION_SCALE = 1.0
POSE_SCALE = 0.5
LIGHT_SCALE = 5.0

# Readiness weighting across the DECA coverage signals (sums to 1.0).
_READINESS_WEIGHTS = {
    "expression_entropy": 0.15,
    "expression_occupied": 0.15,
    "pose_entropy": 0.10,
    "pose_occupied": 0.15,
    "lighting_entropy": 0.10,
    "lighting_occupied": 0.15,
    "latent_entropy": 0.10,
    "cluster_balance": 0.05,
    "cluster_dispersion": 0.05,
}
_READINESS_OCCUPIED_FLOOR = 0.35
_READINESS_DISPERSION_FLOOR = 0.05
_READINESS_LOW_COVERAGE_CAP = 49.0
_READINESS_DISPERSION_REFERENCE = 0.25


@dataclass
class DeepAuditReport:
    """Machine-readable DECA deep-audit result (attached to ``deep_audit``)."""

    mode: str = "deca"
    schema_version: int = DEEP_AUDIT_SCHEMA_VERSION
    status: str = "ok"
    weights_path: str = ""
    device: str = "cpu"
    device_auto_selected: bool = False
    faces_total: int = 0
    faces_encoded: int = 0
    faces_skipped: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    missing_keys_count: int = 0
    unexpected_keys_count: int = 0
    matched_key_ratio: float | None = None
    expression_space_coverage: dict[str, T.Any] = field(default_factory=dict)
    pose_space_coverage: dict[str, T.Any] = field(default_factory=dict)
    lighting_space_coverage: dict[str, T.Any] = field(default_factory=dict)
    latent_entropy: dict[str, T.Any] = field(default_factory=dict)
    cluster_coverage: dict[str, T.Any] = field(default_factory=dict)
    deca_readiness: dict[str, T.Any] = field(default_factory=dict)
    landmark_vs_deca: dict[str, T.Any] = field(default_factory=dict)
    pruning_signals: list[dict[str, T.Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, T.Any]:
        """Return a stable JSON-serializable representation."""
        return asdict(self)


class DecaCropExtractor:
    """Reconstruct DECA-ready RGB crops from source frames.

    Mirrors ``lib.faceqa.coverage.FrameImageMetricsBackfiller``: a lazy
    one-frame decode cache so consecutive faces in a frame reuse the decode,
    and self-disable on the first hard frame-load error so a broken loader
    does not spam per-face warnings.
    """

    def __init__(self, frames_loader: T.Any, *, input_size: int = DECA_INPUT_SIZE) -> None:
        self._frames_loader = frames_loader
        self._input_size = int(input_size)
        self._disabled_reason: str | None = None
        self._last_frame: str | None = None
        self._last_image: np.ndarray | None = None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def crop(self, face: T.Any, frame: str, face_index: int) -> np.ndarray | None:
        """Return an ``(input_size, input_size, 3)`` RGB crop, or ``None``.

        ``None`` is returned (with the reason logged once per failure mode)
        when the frame cannot be loaded or the aligned crop cannot be
        reconstructed; the caller tallies the skip.
        """
        if self._disabled_reason is not None:
            return None
        image = self._load_frame(frame)
        if image is None:
            return None
        try:
            face_obj = DetectedFace()
            face_obj.from_alignment(face, image=image)
            face_obj.load_aligned(image, size=self._input_size, centering="face")
            aligned_bgr = T.cast(np.ndarray, face_obj.aligned.face)
        except Exception as err:  # pylint:disable=broad-except
            logger.warning(
                "Could not reconstruct aligned crop for DECA deep audit (frame: %s, face: %s): %s",
                frame,
                face_index,
                err,
            )
            return None
        if aligned_bgr.ndim != 3 or aligned_bgr.shape[2] != 3:
            return None
        return T.cast(np.ndarray, cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB))

    def _load_frame(self, frame: str) -> np.ndarray | None:
        if frame == self._last_frame and self._last_image is not None:
            return self._last_image
        try:
            image = T.cast(np.ndarray, self._frames_loader.load_image(frame))
        except Exception as err:  # pylint:disable=broad-except
            logger.warning("Could not load source frame for DECA deep audit '%s': %s", frame, err)
            self._disabled_reason = str(err)
            return None
        self._last_frame = frame
        self._last_image = image
        return image


def _encode_faceset(
    entries: dict[str, T.Any],
    *,
    extractor: DecaCropExtractor,
    encoder: DecaEncoder,
    batch_size: int,
    progress_callback: Callable[[int], None] | None,
) -> tuple[DecaCoefficients | None, list[dict[str, T.Any]], int, int, dict[str, int]]:
    """Encode every face crop and return decoded coefficients + skip tallies."""
    encoded_refs: list[dict[str, T.Any]] = []
    pending: list[np.ndarray] = []
    skipped = 0
    skip_reasons: dict[str, int] = {}
    params: list[np.ndarray] = []

    def _flush() -> None:
        if not pending:
            return
        batch = np.stack(pending, axis=0)
        params.append(np.asarray(encoder.encode(batch), dtype=np.float32))
        pending.clear()

    total = 0
    for frame, entry in entries.items():
        for face_index, face in enumerate(entry.faces):
            total += 1
            crop = extractor.crop(face, frame, face_index)
            if progress_callback is not None:
                progress_callback(1)
            if crop is None:
                skipped += 1
                reason = extractor.disabled_reason or "crop_failed"
                key = "frame_loader_disabled" if extractor.disabled_reason else reason
                skip_reasons[key] = skip_reasons.get(key, 0) + 1
                continue
            encoded_refs.append({"frame": frame, "face_index": face_index})
            pending.append(crop)
            if len(pending) >= batch_size:
                _flush()
    _flush()

    if not params:
        return None, [], total, skipped, skip_reasons
    parameters = np.concatenate(params, axis=0)
    return decode_parameters(parameters), encoded_refs, total, skipped, skip_reasons


def _combined_latent(coefficients: DecaCoefficients) -> np.ndarray:
    """Return the per-group-scaled concatenation used for latent metrics."""
    return T.cast(
        np.ndarray,
        np.concatenate(
            (
                coefficients.expression / EXPRESSION_SCALE,
                coefficients.pose / POSE_SCALE,
                coefficients.light / LIGHT_SCALE,
            ),
            axis=1,
        ),
    )


def _space_cell_labels(
    features: np.ndarray,
    *,
    center: float,
    scale: float,
    n_bins: int = metrics_mod.DEFAULT_BINS,
    n_dims: int = metrics_mod.DEFAULT_SPACE_DIMS,
    clip: float = metrics_mod.DEFAULT_CLIP,
) -> list[str | None]:
    """Return coarse per-face cell labels for pruning diversity checks."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return []
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.any():
        return [None] * int(matrix.shape[0])

    dims = metrics_mod._top_variance_dims(matrix[finite], n_dims)  # pylint:disable=protected-access
    bins = max(1, int(n_bins))
    labels: list[str | None] = [None] * int(matrix.shape[0])
    for row_index, row in enumerate(matrix):
        if not finite[row_index]:
            continue
        parts: list[str] = []
        for dim in dims:
            mapped = metrics_mod._center_scale_clip(  # pylint:disable=protected-access
                np.asarray([row[dim]], dtype=np.float64), center, scale, clip
            )
            idx = metrics_mod._bin_indices(  # pylint:disable=protected-access
                mapped, bins, clip
            )[0]
            parts.append(f"{int(dim)}:{int(idx)}")
        labels[row_index] = "|".join(parts)
    return labels


def _latent_cluster_labels(latent: np.ndarray) -> list[int | None]:
    """Return deterministic per-face latent cluster labels for pruning."""
    matrix = np.asarray(latent, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return []
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.any():
        return [None] * int(matrix.shape[0])
    labels: list[int | None] = [None] * int(matrix.shape[0])
    valid = matrix[finite]
    effective_clusters = int(min(metrics_mod.DEFAULT_CLUSTERS, valid.shape[0]))
    if effective_clusters <= 1:
        valid_labels = np.zeros(valid.shape[0], dtype=np.intp)
    else:
        valid_labels = metrics_mod._kmeans_labels(  # pylint:disable=protected-access
            valid,
            effective_clusters,
            seed=metrics_mod.DEFAULT_SEED,
        )
    valid_iter = iter(int(label) for label in valid_labels)
    for row_index, is_finite in enumerate(finite):
        if is_finite:
            labels[row_index] = next(valid_iter)
    return labels


def _pruning_signals(
    refs: list[dict[str, T.Any]],
    coefficients: DecaCoefficients,
    latent: np.ndarray,
) -> list[dict[str, T.Any]]:
    """Build per-face DECA signatures consumed by pruning."""
    expression_cells = _space_cell_labels(
        coefficients.expression, center=0.0, scale=EXPRESSION_SCALE
    )
    pose_cells = _space_cell_labels(coefficients.pose, center=0.0, scale=POSE_SCALE)
    lighting_cells = _space_cell_labels(coefficients.light, center=0.0, scale=LIGHT_SCALE)
    latent_clusters = _latent_cluster_labels(latent)
    signals: list[dict[str, T.Any]] = []
    for index, ref in enumerate(refs):
        signal = {
            "frame": ref["frame"],
            "face_index": int(ref["face_index"]),
            "encoded_index": index,
            "expression_cell": expression_cells[index],
            "pose_cell": pose_cells[index],
            "lighting_cell": lighting_cells[index],
            "latent_cluster": latent_clusters[index],
        }
        signal["signature"] = [
            signal["expression_cell"],
            signal["pose_cell"],
            signal["lighting_cell"],
            signal["latent_cluster"],
        ]
        signals.append(signal)
    return signals


def _readiness_from_metrics(
    expression: dict[str, T.Any],
    pose: dict[str, T.Any],
    lighting: dict[str, T.Any],
    latent: dict[str, T.Any],
    cluster: dict[str, T.Any],
) -> dict[str, T.Any]:
    """Derive a 0-100 DECA readiness sub-score from the coverage metrics."""
    expression_entropy = float(expression.get("entropy_coverage", 0.0) or 0.0)
    expression_occupied = float(expression.get("occupied_coverage", 0.0) or 0.0)
    pose_entropy = float(pose.get("entropy_coverage", 0.0) or 0.0)
    pose_occupied = float(pose.get("occupied_coverage", 0.0) or 0.0)
    lighting_entropy = float(lighting.get("entropy_coverage", 0.0) or 0.0)
    lighting_occupied = float(lighting.get("occupied_coverage", 0.0) or 0.0)
    mean_dispersion = float(cluster.get("mean_dispersion", 0.0) or 0.0)
    components = {
        "expression_entropy": expression_entropy,
        "expression_occupied": expression_occupied,
        "pose_entropy": pose_entropy,
        "pose_occupied": pose_occupied,
        "lighting_entropy": lighting_entropy,
        "lighting_occupied": lighting_occupied,
        "latent_entropy": float(latent.get("entropy", 0.0) or 0.0),
        "cluster_balance": float(cluster.get("balance", 0.0) or 0.0),
        "cluster_dispersion": min(1.0, mean_dispersion / _READINESS_DISPERSION_REFERENCE),
    }
    score = sum(components[name] * weight for name, weight in _READINESS_WEIGHTS.items())
    notes: list[str] = []
    if expression_entropy < 0.5 or expression_occupied < _READINESS_OCCUPIED_FLOOR:
        notes.append("Low DECA expression-space coverage: faceset spans few expressions.")
    if pose_entropy < 0.5 or pose_occupied < _READINESS_OCCUPIED_FLOOR:
        notes.append("Low DECA pose-space coverage: faceset spans few head/jaw poses.")
    if lighting_entropy < 0.5 or lighting_occupied < _READINESS_OCCUPIED_FLOOR:
        notes.append("Low DECA lighting-space coverage: limited illumination variety.")
    if components["latent_entropy"] < 0.5:
        notes.append("Low DECA latent entropy: faces cluster into few latent modes.")
    if mean_dispersion < _READINESS_DISPERSION_FLOOR:
        notes.append("Low DECA cluster dispersion: latent clusters have little absolute spread.")
    gate_reasons: list[str] = []
    if pose_occupied < _READINESS_OCCUPIED_FLOOR:
        gate_reasons.append("pose occupied coverage")
    if lighting_occupied < _READINESS_OCCUPIED_FLOOR:
        gate_reasons.append("lighting occupied coverage")
    if mean_dispersion < _READINESS_DISPERSION_FLOOR:
        gate_reasons.append("cluster dispersion")
    if gate_reasons:
        score = min(score, _READINESS_LOW_COVERAGE_CAP / 100.0)
        notes.append(
            "DECA readiness capped below pass because "
            f"{', '.join(gate_reasons)} is below the validation floor."
        )
    return {
        "score": round(score * 100.0, 2),
        "weights": dict(_READINESS_WEIGHTS),
        "components": {name: round(value, 4) for name, value in components.items()},
        "gates": {
            "occupied_floor": _READINESS_OCCUPIED_FLOOR,
            "dispersion_floor": _READINESS_DISPERSION_FLOOR,
            "low_coverage_cap": _READINESS_LOW_COVERAGE_CAP,
            "applied": bool(gate_reasons),
            "reasons": gate_reasons,
        },
        "mean_dispersion": round(mean_dispersion, 4),
        "notes": notes,
    }


def _landmark_component(readiness_scores: dict[str, T.Any], name: str) -> dict[str, float]:
    """Pull a landmark-based component's coverage figures for comparison."""
    components = readiness_scores.get("components", {}) if readiness_scores else {}
    component = components.get(name, {}) or {}
    return {
        "entropy_coverage": round(float(component.get("entropy_coverage", 0.0) or 0.0), 4),
        "occupied_coverage": round(float(component.get("occupied_coverage", 0.0) or 0.0), 4),
        "classified_faces": int(component.get("classified_faces", 0) or 0),
    }


def _comparison(
    readiness_scores: dict[str, T.Any],
    expression_space: dict[str, object],
    lighting_space: dict[str, object],
) -> dict[str, object]:
    """Build the landmark-vs-DECA comparison block.

    Both sides report ``entropy_coverage`` / ``occupied_coverage`` on the same
    ``[0, 1]`` scale (landmark/thumbnail bucket coverage vs DECA
    coefficient-space coverage), so they are directly comparable even though
    they measure different signals.
    """
    notes = [
        "Landmark/thumbnail coverage is bucket-occupancy over hand-crafted "
        "features; DECA coverage is occupancy over learned 3D coefficient "
        "space. Large gaps highlight variety the landmark heuristics miss.",
    ]
    return {
        "expression": {
            "landmark": _landmark_component(readiness_scores, "expression"),
            "deca": {
                "entropy_coverage": expression_space.get("entropy_coverage", 0.0),
                "occupied_coverage": expression_space.get("occupied_coverage", 0.0),
                "samples": expression_space.get("samples", 0),
            },
        },
        "lighting": {
            "landmark": _landmark_component(readiness_scores, "lighting"),
            "deca": {
                "entropy_coverage": lighting_space.get("entropy_coverage", 0.0),
                "occupied_coverage": lighting_space.get("occupied_coverage", 0.0),
                "samples": lighting_space.get("samples", 0),
            },
        },
        "notes": notes,
    }


def run_deep_audit(
    entries: dict[str, T.Any],
    *,
    encoder: DecaEncoder,
    frames_loader: T.Any,
    readiness_scores: dict[str, T.Any] | None = None,
    weights_path: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: Callable[[int], None] | None = None,
) -> DeepAuditReport:
    """Run the DECA deep audit over a faceset and return the report block.

    Parameters
    ----------
    entries
        Alignments entries (``{frame: AlignmentsEntry}``) - the same in-memory
        envelope the coverage pass enriched.
    encoder
        Anything satisfying :class:`~lib.faceqa.deep.deca_encoder.DecaEncoder`.
    frames_loader
        Source-frame loader exposing ``load_image(frame)``.
    readiness_scores
        The landmark-based ``readiness_scores`` dict, used for the
        landmark-vs-DECA comparison. Optional.
    weights_path
        Path the encoder weights were loaded from (recorded for provenance).
    batch_size
        Faces per encoder forward pass.
    progress_callback
        Called with ``1`` per face processed (for the CLI progress bar).
    """
    extractor = DecaCropExtractor(frames_loader)
    coefficients, encoded_refs, total, skipped, skip_reasons = _encode_faceset(
        entries,
        extractor=extractor,
        encoder=encoder,
        batch_size=max(1, int(batch_size)),
        progress_callback=progress_callback,
    )

    matched_key_ratio = getattr(encoder, "matched_key_ratio", None)
    report = DeepAuditReport(
        weights_path=weights_path,
        device=str(getattr(encoder, "device", "cpu")),
        device_auto_selected=bool(getattr(encoder, "device_auto_selected", False)),
        faces_total=total,
        faces_skipped=skipped,
        skipped_reasons=skip_reasons,
        missing_keys_count=int(getattr(encoder, "missing_keys_count", 0) or 0),
        unexpected_keys_count=int(getattr(encoder, "unexpected_keys_count", 0) or 0),
        matched_key_ratio=(
            None if matched_key_ratio is None else round(float(matched_key_ratio), 4)
        ),
    )
    scores = readiness_scores or {}

    if coefficients is None or len(coefficients) == 0:
        report.status = "no_faces_encoded"
        report.notes.append(
            "DECA deep audit produced no coefficients; check frame availability "
            "and the source frames directory."
        )
        return report

    report.faces_encoded = len(coefficients)
    expression_space = metrics_mod.space_coverage(
        coefficients.expression, center=0.0, scale=EXPRESSION_SCALE
    )
    pose_space = metrics_mod.space_coverage(coefficients.pose, center=0.0, scale=POSE_SCALE)
    lighting_space = metrics_mod.space_coverage(coefficients.light, center=0.0, scale=LIGHT_SCALE)
    latent = _combined_latent(coefficients)
    latent_entropy = metrics_mod.latent_entropy(latent)
    cluster = metrics_mod.cluster_coverage(latent)

    report.expression_space_coverage = expression_space
    report.pose_space_coverage = pose_space
    report.lighting_space_coverage = lighting_space
    report.latent_entropy = latent_entropy
    report.cluster_coverage = cluster
    report.deca_readiness = _readiness_from_metrics(
        expression_space, pose_space, lighting_space, latent_entropy, cluster
    )
    report.landmark_vs_deca = _comparison(scores, expression_space, lighting_space)
    report.pruning_signals = _pruning_signals(encoded_refs, coefficients, latent)
    if skipped:
        report.notes.append(
            f"{skipped}/{total} faces skipped during DECA encoding "
            "(missing frames or crop failures)."
        )
    return report


def build_encoder(*, device: str = "auto") -> DecaEncoder:
    """Load the real DECA encoder from cached research weights.

    Thin indirection over :func:`lib.faceqa.deep.weights.load_deca_encoder`
    so the orchestrator depends only on :mod:`lib.faceqa.deep.audit`. Imported
    lazily to keep ``torch`` off the default coverage path.
    """
    from lib.faceqa.deep.weights import load_deca_encoder

    return load_deca_encoder(device=device)


__all__ = get_module_objects(__name__)
