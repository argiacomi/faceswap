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
    "expression": 0.3,
    "pose": 0.2,
    "lighting": 0.25,
    "latent_entropy": 0.15,
    "cluster_balance": 0.1,
}


@dataclass
class DeepAuditReport:
    """Machine-readable DECA deep-audit result (attached to ``deep_audit``)."""

    mode: str = "deca"
    schema_version: int = DEEP_AUDIT_SCHEMA_VERSION
    status: str = "ok"
    weights_path: str = ""
    faces_total: int = 0
    faces_encoded: int = 0
    faces_skipped: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    expression_space_coverage: dict[str, object] = field(default_factory=dict)
    pose_space_coverage: dict[str, object] = field(default_factory=dict)
    lighting_space_coverage: dict[str, object] = field(default_factory=dict)
    latent_entropy: dict[str, object] = field(default_factory=dict)
    cluster_coverage: dict[str, object] = field(default_factory=dict)
    deca_readiness: dict[str, object] = field(default_factory=dict)
    landmark_vs_deca: dict[str, object] = field(default_factory=dict)
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
        return cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

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
) -> tuple[DecaCoefficients | None, int, int, dict[str, int]]:
    """Encode every face crop and return decoded coefficients + skip tallies."""
    crops: list[np.ndarray] = []
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
            crops.append(crop)
            pending.append(crop)
            if len(pending) >= batch_size:
                _flush()
    _flush()

    if not params:
        return None, total, skipped, skip_reasons
    parameters = np.concatenate(params, axis=0)
    return decode_parameters(parameters), total, skipped, skip_reasons


def _combined_latent(coefficients: DecaCoefficients) -> np.ndarray:
    """Return the per-group-scaled concatenation used for latent metrics."""
    return np.concatenate(
        (
            coefficients.expression / EXPRESSION_SCALE,
            coefficients.pose / POSE_SCALE,
            coefficients.light / LIGHT_SCALE,
        ),
        axis=1,
    )


def _readiness_from_metrics(
    expression: dict[str, object],
    pose: dict[str, object],
    lighting: dict[str, object],
    latent: dict[str, object],
    cluster: dict[str, object],
) -> dict[str, object]:
    """Derive a 0-100 DECA readiness sub-score from the coverage metrics."""
    components = {
        "expression": float(expression.get("entropy_coverage", 0.0) or 0.0),
        "pose": float(pose.get("entropy_coverage", 0.0) or 0.0),
        "lighting": float(lighting.get("entropy_coverage", 0.0) or 0.0),
        "latent_entropy": float(latent.get("entropy", 0.0) or 0.0),
        "cluster_balance": float(cluster.get("balance", 0.0) or 0.0),
    }
    score = sum(components[name] * weight for name, weight in _READINESS_WEIGHTS.items())
    notes: list[str] = []
    if components["expression"] < 0.5:
        notes.append("Low DECA expression-space coverage: faceset spans few expressions.")
    if components["pose"] < 0.5:
        notes.append("Low DECA pose-space coverage: faceset spans few head/jaw poses.")
    if components["lighting"] < 0.5:
        notes.append("Low DECA lighting-space coverage: limited illumination variety.")
    if components["latent_entropy"] < 0.5:
        notes.append("Low DECA latent entropy: faces cluster into few latent modes.")
    return {
        "score": round(score * 100.0, 2),
        "weights": dict(_READINESS_WEIGHTS),
        "components": {name: round(value, 4) for name, value in components.items()},
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
    coefficients, total, skipped, skip_reasons = _encode_faceset(
        entries,
        extractor=extractor,
        encoder=encoder,
        batch_size=max(1, int(batch_size)),
        progress_callback=progress_callback,
    )

    report = DeepAuditReport(
        weights_path=weights_path,
        faces_total=total,
        faces_skipped=skipped,
        skipped_reasons=skip_reasons,
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
    if skipped:
        report.notes.append(
            f"{skipped}/{total} faces skipped during DECA encoding "
            "(missing frames or crop failures)."
        )
    return report


def build_encoder(*, device: str = "cpu") -> DecaEncoder:
    """Load the real DECA encoder from cached research weights.

    Thin indirection over :func:`lib.faceqa.deep.weights.load_deca_encoder`
    so the orchestrator depends only on :mod:`lib.faceqa.deep.audit`. Imported
    lazily to keep ``torch`` off the default coverage path.
    """
    from lib.faceqa.deep.weights import load_deca_encoder

    return load_deca_encoder(device=device)


__all__ = get_module_objects(__name__)
