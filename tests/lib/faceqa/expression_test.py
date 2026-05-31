#!/usr/bin/env python3
"""Tests for landmark-derived FaceQA expression features."""

from __future__ import annotations

import numpy as np

from lib.faceqa.expression import (
    EXPRESSION_BUCKETS,
    bucket_for_features,
    compute_expression_features,
)


def _neutral_landmarks() -> np.ndarray:
    """Return a synthetic 68-point face with eyes open and mouth closed."""
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]

    # Right eye corners and lids (open).
    points[36] = (30.0, 40.0)  # outer corner
    points[39] = (40.0, 40.0)  # inner corner
    points[37] = (33.0, 37.0)
    points[38] = (37.0, 37.0)
    points[40] = (37.0, 43.0)
    points[41] = (33.0, 43.0)

    # Left eye corners and lids.
    points[42] = (60.0, 40.0)
    points[45] = (70.0, 40.0)
    points[43] = (63.0, 37.0)
    points[44] = (67.0, 37.0)
    points[46] = (67.0, 43.0)
    points[47] = (63.0, 43.0)

    # Brows sit just above eyes.
    points[17] = (29.0, 35.0)
    points[18] = (32.0, 33.0)
    points[19] = (35.0, 32.0)
    points[20] = (38.0, 33.0)
    points[21] = (41.0, 35.0)
    points[22] = (59.0, 35.0)
    points[23] = (62.0, 33.0)
    points[24] = (65.0, 32.0)
    points[25] = (68.0, 33.0)
    points[26] = (71.0, 35.0)

    # Mouth: corners level with mouth center, lips closed.
    points[48] = (42.0, 75.0)  # left corner
    points[54] = (58.0, 75.0)  # right corner
    points[50] = (46.0, 73.0)  # outer top
    points[51] = (50.0, 73.0)
    points[52] = (54.0, 73.0)
    points[56] = (54.0, 77.0)  # outer bottom
    points[57] = (50.0, 77.0)
    points[58] = (46.0, 77.0)
    points[61] = (46.0, 75.0)  # inner top
    points[62] = (50.0, 75.0)
    points[63] = (54.0, 75.0)
    points[65] = (54.0, 75.0)  # inner bottom (same as top → closed)
    points[66] = (50.0, 75.0)
    points[67] = (46.0, 75.0)
    return points


def test_compute_expression_features_neutral_returns_low_values() -> None:
    features = compute_expression_features(_neutral_landmarks())

    assert features is not None
    assert features["mouth_openness"] < 0.05
    assert abs(features["smile_proxy"]) < 0.05
    assert features["eye_closure"] < 0.2
    assert features["expression_asymmetry"] < 0.05
    assert bucket_for_features(features) == "neutral"


def test_compute_expression_features_returns_none_for_invalid_landmarks() -> None:
    assert compute_expression_features(None) is None
    assert compute_expression_features(np.zeros((10, 2), dtype="float32")) is None
    # Degenerate: all points collapsed onto one location → interocular = 0.
    assert compute_expression_features(np.zeros((68, 2), dtype="float32")) is None


def test_bucket_for_features_classifies_open_mouth_as_talking() -> None:
    points = _neutral_landmarks()
    # Open the inner lips by 8 pixels (interocular is ~30 → openness ~0.27).
    for top in (61, 62, 63):
        points[top, 1] -= 6.0
    for bottom in (65, 66, 67):
        points[bottom, 1] += 6.0

    features = compute_expression_features(points)

    assert features is not None
    assert features["mouth_openness"] > 0.35
    assert bucket_for_features(features) == "talking_open"


def test_bucket_for_features_classifies_smile_when_corners_raised() -> None:
    points = _neutral_landmarks()
    # Raise the mouth corners by 4 pixels (smaller y) without opening the mouth.
    points[48, 1] -= 3.0
    points[54, 1] -= 3.0

    features = compute_expression_features(points)

    assert features is not None
    assert features["smile_proxy"] >= 0.05
    assert features["mouth_openness"] < 0.20
    assert bucket_for_features(features) == "smile"


def test_bucket_for_features_classifies_eyes_closed() -> None:
    points = _neutral_landmarks()
    # Collapse upper and lower eyelids so EAR → 0.
    for upper in (37, 38, 43, 44):
        points[upper, 1] = 40.0
    for lower in (40, 41, 46, 47):
        points[lower, 1] = 40.0

    features = compute_expression_features(points)

    assert features is not None
    assert features["eye_closure"] >= 0.6
    assert bucket_for_features(features) == "eyes_closed"


def test_bucket_for_features_classifies_expressive_on_raised_brows() -> None:
    points = _neutral_landmarks()
    # Raise both brows substantially.
    for brow in range(17, 27):
        points[brow, 1] -= 18.0

    features = compute_expression_features(points)

    assert features is not None
    assert features["brow_raise_proxy"] >= 0.55
    assert bucket_for_features(features) == "expressive"


def test_bucket_for_features_classifies_expressive_on_asymmetry() -> None:
    points = _neutral_landmarks()
    # Drop one mouth corner significantly (smirk-like asymmetry).
    points[48, 1] += 6.0

    features = compute_expression_features(points)

    assert features is not None
    assert features["expression_asymmetry"] >= 0.15
    assert bucket_for_features(features) == "expressive"


def test_bucket_for_features_unknown_for_missing() -> None:
    assert bucket_for_features(None) == "unknown"


def test_expression_buckets_constant_is_stable() -> None:
    assert EXPRESSION_BUCKETS == (
        "neutral",
        "slight_open",
        "talking_open",
        "smile",
        "eyes_closed",
        "expressive",
    )
