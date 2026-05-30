#!/usr/bin/env python3
"""Landmark-derived expression features and bucketing for FaceQA."""

from __future__ import annotations

import logging
import typing as T

import numpy as np

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


# 68-point Multi-PIE landmark indices.
RIGHT_EYE = (36, 37, 38, 39, 40, 41)
LEFT_EYE = (42, 43, 44, 45, 46, 47)
# Pre-built int32 index arrays so the expression-feature pass can index
# into ``landmarks`` without paying a fresh ``list(RIGHT_EYE)`` / ``np.asarray``
# allocation per face (issue #192 P2).
_RIGHT_EYE_IDX = np.array(RIGHT_EYE, dtype=np.int32)
_LEFT_EYE_IDX = np.array(LEFT_EYE, dtype=np.int32)
RIGHT_BROW = (17, 18, 19, 20, 21)
LEFT_BROW = (22, 23, 24, 25, 26)
MOUTH_OUTER_LEFT_CORNER = 48
MOUTH_OUTER_RIGHT_CORNER = 54
MOUTH_OUTER_TOP = (50, 51, 52)
MOUTH_OUTER_BOTTOM = (56, 57, 58)
MOUTH_INNER_TOP = (61, 62, 63)
MOUTH_INNER_BOTTOM = (65, 66, 67)

# Reference EAR for a fully-open eye; used to normalize closure into [0, 1].
EAR_OPEN_REFERENCE = 0.30

EXPRESSION_BUCKETS: tuple[str, ...] = (
    "neutral",
    "slight_open",
    "talking_open",
    "smile",
    "eyes_closed",
    "expressive",
)

# Bucket selection thresholds. All feature values are normalized by interocular
# distance, so these are dimensionless and stable across face sizes.
EYES_CLOSED_THRESHOLD = 0.6
TALKING_OPEN_THRESHOLD = 0.35
SLIGHT_OPEN_THRESHOLD = 0.15
SMILE_PROXY_THRESHOLD = 0.05
SMILE_MAX_OPENNESS = 0.20
EXPRESSIVE_ASYMMETRY_THRESHOLD = 0.15
EXPRESSIVE_BROW_RAISE_THRESHOLD = 0.55


class ExpressionFeatures(T.TypedDict):
    """Landmark-derived expression feature scalars."""

    mouth_openness: float
    mouth_width_ratio: float
    smile_proxy: float
    eye_closure: float
    brow_raise_proxy: float
    expression_asymmetry: float


def _eye_aspect_ratio(landmarks: np.ndarray, indices: T.Sequence[int]) -> float:
    """Return the eye aspect ratio for one eye."""
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in indices)
    vertical = float(np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
    horizontal = float(np.linalg.norm(p1 - p4))
    if horizontal <= 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


def _interocular_distance(landmarks: np.ndarray) -> float:
    """Return the distance between right and left eye centers."""
    right = landmarks[_RIGHT_EYE_IDX].mean(axis=0)
    left = landmarks[_LEFT_EYE_IDX].mean(axis=0)
    return float(np.linalg.norm(left - right))


def compute_expression_features(landmarks: np.ndarray | None) -> ExpressionFeatures | None:
    """Return scalar expression features derived from 68-point landmarks.

    Returns ``None`` when landmarks are missing, the wrong shape, or degenerate
    (zero interocular distance).
    """
    if landmarks is None:
        return None
    points = np.asarray(landmarks, dtype="float32")
    if points.ndim != 2 or points.shape[0] < 68 or points.shape[1] != 2:
        return None
    interocular = _interocular_distance(points)
    if interocular <= 1e-6:
        return None

    inner_top = points[list(MOUTH_INNER_TOP)].mean(axis=0)
    inner_bottom = points[list(MOUTH_INNER_BOTTOM)].mean(axis=0)
    mouth_openness = float(np.linalg.norm(inner_bottom - inner_top)) / interocular

    left_corner = points[MOUTH_OUTER_LEFT_CORNER]
    right_corner = points[MOUTH_OUTER_RIGHT_CORNER]
    mouth_width = float(np.linalg.norm(right_corner - left_corner))
    mouth_width_ratio = mouth_width / interocular

    outer_top = points[list(MOUTH_OUTER_TOP)].mean(axis=0)
    outer_bottom = points[list(MOUTH_OUTER_BOTTOM)].mean(axis=0)
    mouth_center_y = float((outer_top[1] + outer_bottom[1]) / 2.0)
    corner_y = float((left_corner[1] + right_corner[1]) / 2.0)
    # Smaller y = higher on image. Positive value means corners are raised.
    smile_proxy = (mouth_center_y - corner_y) / interocular

    right_ear = _eye_aspect_ratio(points, RIGHT_EYE)
    left_ear = _eye_aspect_ratio(points, LEFT_EYE)
    mean_ear = (right_ear + left_ear) / 2.0
    eye_closure = max(0.0, 1.0 - mean_ear / EAR_OPEN_REFERENCE)

    right_brow_y = float(points[list(RIGHT_BROW)].mean(axis=0)[1])
    left_brow_y = float(points[list(LEFT_BROW)].mean(axis=0)[1])
    brow_y = (right_brow_y + left_brow_y) / 2.0
    right_eye_top_y = float(points[[37, 38]].mean(axis=0)[1])
    left_eye_top_y = float(points[[43, 44]].mean(axis=0)[1])
    eye_top_y = (right_eye_top_y + left_eye_top_y) / 2.0
    # Positive value means brows sit higher than eye tops.
    brow_raise_proxy = max(0.0, (eye_top_y - brow_y) / interocular)

    ear_asymmetry = abs(left_ear - right_ear)
    corner_height_asymmetry = abs(left_corner[1] - right_corner[1]) / interocular
    expression_asymmetry = float(ear_asymmetry + corner_height_asymmetry)

    return {
        "mouth_openness": round(mouth_openness, 4),
        "mouth_width_ratio": round(mouth_width_ratio, 4),
        "smile_proxy": round(smile_proxy, 4),
        "eye_closure": round(eye_closure, 4),
        "brow_raise_proxy": round(brow_raise_proxy, 4),
        "expression_asymmetry": round(expression_asymmetry, 4),
    }


def bucket_for_features(features: ExpressionFeatures | None) -> str:
    """Return the expression bucket for a feature dict, or ``unknown``."""
    if features is None:
        return "unknown"
    if features["eye_closure"] >= EYES_CLOSED_THRESHOLD:
        return "eyes_closed"
    if features["mouth_openness"] >= TALKING_OPEN_THRESHOLD:
        return "talking_open"
    if (
        features["smile_proxy"] >= SMILE_PROXY_THRESHOLD
        and features["mouth_openness"] < SMILE_MAX_OPENNESS
    ):
        return "smile"
    if features["mouth_openness"] >= SLIGHT_OPEN_THRESHOLD:
        return "slight_open"
    if (
        features["expression_asymmetry"] >= EXPRESSIVE_ASYMMETRY_THRESHOLD
        or features["brow_raise_proxy"] >= EXPRESSIVE_BROW_RAISE_THRESHOLD
    ):
        return "expressive"
    return "neutral"


__all__ = get_module_objects(__name__)
