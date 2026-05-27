#!/usr/bin/env python3
"""Tests for thumbnail-derived FaceQA lighting features."""

from __future__ import annotations

import numpy as np

from lib.faceqa.lighting import (
    LIGHTING_BUCKETS,
    bucket_for_features,
    compute_lighting_features,
)


def _flat_gray_image(value: int = 128, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_compute_lighting_features_flat_image_is_neutral() -> None:
    features = compute_lighting_features(_flat_gray_image(128))

    assert features is not None
    assert abs(features["mean_luminance"] - 128.0) < 1.0
    assert features["contrast"] < 1.0
    assert features["luminance_variance"] < 1.0
    assert abs(features["left_right_ratio"] - 1.0) < 1e-3
    assert abs(features["top_bottom_ratio"] - 1.0) < 1e-3
    assert abs(features["color_warmth"]) < 1.0
    assert bucket_for_features(features) == "flat_frontal"


def test_compute_lighting_features_handles_invalid_inputs() -> None:
    assert compute_lighting_features(None) is None
    assert compute_lighting_features(np.zeros((1, 1, 3), dtype=np.uint8)) is None


def test_bucket_for_features_classifies_dark_and_overexposed() -> None:
    dark = compute_lighting_features(_flat_gray_image(30))
    bright = compute_lighting_features(_flat_gray_image(240))

    assert bucket_for_features(dark) == "dark"
    assert bucket_for_features(bright) == "overexposed"


def test_bucket_for_features_classifies_side_lit() -> None:
    image = np.full((32, 32, 3), 90, dtype=np.uint8)
    image[:, 16:] = 200  # bright right half

    features = compute_lighting_features(image)

    assert features is not None
    assert features["left_right_ratio"] < 0.7
    assert bucket_for_features(features) == "side_lit"


def test_bucket_for_features_classifies_top_lit() -> None:
    image = np.full((32, 32, 3), 90, dtype=np.uint8)
    image[:16, :] = 200  # bright top half

    features = compute_lighting_features(image)

    assert features is not None
    assert features["top_bottom_ratio"] > 1.3
    assert bucket_for_features(features) == "top_lit"


def test_bucket_for_features_classifies_high_contrast() -> None:
    # Checkerboard of black and white pixels: large per-pixel variance but
    # equal means in every quadrant so side/top buckets do not trigger.
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    grid = np.indices((32, 32)).sum(axis=0) % 2
    image[grid == 0] = 255

    features = compute_lighting_features(image)

    assert features is not None
    assert features["contrast"] >= 60.0
    assert abs(features["left_right_ratio"] - 1.0) < 0.05
    assert abs(features["top_bottom_ratio"] - 1.0) < 0.05
    assert bucket_for_features(features) == "high_contrast"


def test_bucket_for_features_classifies_warm_and_cool() -> None:
    warm_image = np.zeros((32, 32, 3), dtype=np.uint8)
    warm_image[:, :, 2] = 160  # red
    warm_image[:, :, 1] = 110  # green
    warm_image[:, :, 0] = 90  # blue

    cool_image = np.zeros((32, 32, 3), dtype=np.uint8)
    cool_image[:, :, 0] = 160  # blue
    cool_image[:, :, 1] = 110
    cool_image[:, :, 2] = 90

    warm = compute_lighting_features(warm_image)
    cool = compute_lighting_features(cool_image)

    assert warm is not None and cool is not None
    assert warm["color_warmth"] > 20.0
    assert cool["color_warmth"] < -20.0
    assert bucket_for_features(warm) == "warm"
    assert bucket_for_features(cool) == "cool"


def test_bucket_for_features_unknown_for_none() -> None:
    assert bucket_for_features(None) == "unknown"


def test_lighting_buckets_constant_is_stable() -> None:
    assert LIGHTING_BUCKETS == (
        "dark",
        "overexposed",
        "side_lit",
        "top_lit",
        "high_contrast",
        "warm",
        "cool",
        "flat_frontal",
    )
