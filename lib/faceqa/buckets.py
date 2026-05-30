#!/usr/bin/env python3
"""Public bucket-classifier helpers for FaceQA records.

These functions translate a :class:`~lib.faceqa.record.FaceQARecord` into a
single bucket label per coverage dimension (pose, pitch, lighting,
expression, blur, resolution, misalignment, occlusion, identity, mask_qa,
pose_sources, pose_confidence). They are imported by:

* ``lib.faceqa.coverage`` — the canonical ``compute_coverage`` consumer.
* ``lib.faceqa.redundancy`` — for the bucket views inside
  ``_features_for`` and the per-dimension ``_bucket_of`` dispatch used by
  effective-coverage computation.

Splitting the helpers out of ``coverage.py`` resolves the private-bucket
coupling between the two modules called out in issue #192 (Priority 1):
``redundancy.py`` used to import ``_lighting_bucket``, ``_pitch_bucket``,
and ``_pose_bucket`` inside function bodies — including inside the O(N²)
redundancy decisions — which made the cross-module call relationship
implicit and risked accidental private-API drift.
"""

from __future__ import annotations

import typing as T

import numpy as np

from lib.faceqa.lighting import LightingFeatures
from lib.faceqa.lighting import bucket_for_features as lighting_bucket_for_features
from lib.faceqa.record import FaceQARecord
from lib.utils import get_module_objects

# Threshold tuples used by the bucket classifiers. These move out of
# ``coverage.py`` together with the helpers so the public API surface is
# self-contained: a caller that wants to classify a record (or tweak the
# thresholds in tests) does not need to import anything from
# ``coverage``.
POSE_YAW_THRESHOLDS: tuple[float, float, float] = (15.0, 30.0, 60.0)
POSE_PITCH_THRESHOLDS: tuple[float, float] = (15.0, 30.0)
BLUR_THRESHOLDS: tuple[float, float, float] = (6.0, 3.0, 1.0)
RESOLUTION_THRESHOLDS: tuple[int, int, int] = (256, 128, 64)
OCCLUSION_THRESHOLDS: tuple[float, float, float] = (0.05, 0.20, 0.50)
DISTANCE_THRESHOLDS: tuple[float, float, float] = (0.10, 0.20, 0.40)


def pose_bucket(record: FaceQARecord) -> str:
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


def pitch_bucket(record: FaceQARecord) -> str:
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


def lighting_bucket(record: FaceQARecord) -> str:
    """Return the lighting bucket label for a record (``unknown`` if missing).

    Only the fields actually consulted by
    :func:`lib.faceqa.lighting.bucket_for_features` are required
    (``mean_luminance``, ``contrast``, ``left_right_ratio``,
    ``top_bottom_ratio``, ``color_warmth``); the optional
    ``luminance_variance`` / ``saturation`` slots default to ``0.0`` so
    legacy records that only carry the bucket-discriminating features
    still classify.
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


def expression_bucket(record: FaceQARecord) -> str:
    """Return the expression bucket label (``unknown`` when missing)."""
    return record.expression_bucket or "unknown"


def blur_bucket(record: FaceQARecord) -> str:
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


def resolution_bucket(record: FaceQARecord) -> str:
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


def misalignment_bucket(record: FaceQARecord) -> str:
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


def occlusion_bucket(record: FaceQARecord) -> str:
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


def identity_bucket(record: FaceQARecord) -> str:
    """Return the identity-classification bucket label for a record."""
    flag = record.identity_quality_flag or record.identity_final_decision
    if flag in ("inlier", "borderline", "outlier", "reject", "review"):
        return str(flag)
    return "unknown"


def mask_qa_bucket(record: FaceQARecord) -> str:
    """Return the mask-availability bucket label for a face."""
    if not record.mask_qa_ref:
        return "missing"
    return "present"


def pose_source_bucket(record: FaceQARecord) -> str:
    """Return the pose-source provenance label for a record."""
    return record.pose_source or "unknown"


def pose_confidence_bucket(record: FaceQARecord) -> str:
    """Return the pose-confidence bucket label for a record."""
    return record.pose_confidence or "unknown"


def is_identity_outlier(record: FaceQARecord) -> bool:
    """Return whether a record is classified as an identity outlier/reject."""
    flag = record.identity_quality_flag
    decision = record.identity_final_decision
    if flag in ("outlier", "reject"):
        return True
    return decision in ("outlier", "reject")


__all__ = get_module_objects(__name__)
