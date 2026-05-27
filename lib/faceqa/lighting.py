#!/usr/bin/env python3
"""Thumbnail-derived lighting features and bucketing for FaceQA."""

from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


LIGHTING_BUCKETS: tuple[str, ...] = (
    "dark",
    "overexposed",
    "side_lit",
    "top_lit",
    "high_contrast",
    "warm",
    "cool",
    "flat_frontal",
)

# Thresholds operate on 0-255 image data.
DARK_LUMINANCE_THRESHOLD = 60.0
OVEREXPOSED_LUMINANCE_THRESHOLD = 220.0
HIGH_CONTRAST_STD_THRESHOLD = 60.0
SIDE_RATIO_THRESHOLD = 0.30  # |left/right - 1| in this range or wider = side-lit.
TOP_RATIO_THRESHOLD = 0.30
WARMTH_THRESHOLD = 20.0  # signed R - B mean.


class LightingFeatures(T.TypedDict):
    """Thumbnail-derived lighting feature scalars."""

    mean_luminance: float
    luminance_variance: float
    contrast: float
    left_right_ratio: float
    top_bottom_ratio: float
    saturation: float
    color_warmth: float


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator`` clamped to avoid division by zero."""
    if numerator < 1.0 and denominator < 1.0:
        return 1.0
    return max(numerator, 1.0) / max(denominator, 1.0)


def compute_lighting_features(image: np.ndarray | None) -> LightingFeatures | None:
    """Return scalar lighting features derived from a thumbnail image.

    The input is expected to be a BGR uint8 image (the format used by stored
    alignments thumbnails). Returns ``None`` when the image is missing or has
    a shape that cannot be split into halves.
    """
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim not in (2, 3) or array.shape[0] < 2 or array.shape[1] < 2:
        return None
    color = array if array.ndim == 3 else cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype("float32", copy=False)

    mean_luminance = float(gray_f.mean())
    luminance_variance = float(gray_f.var())
    contrast = float(gray_f.std())

    height, width = gray_f.shape
    left = float(gray_f[:, : width // 2].mean())
    right = float(gray_f[:, width // 2 :].mean())
    top = float(gray_f[: height // 2, :].mean())
    bottom = float(gray_f[height // 2 :, :].mean())
    left_right_ratio = _safe_ratio(left, right)
    top_bottom_ratio = _safe_ratio(top, bottom)

    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:, :, 1].astype("float32").mean())

    blue = color[:, :, 0].astype("float32")
    red = color[:, :, 2].astype("float32")
    color_warmth = float(red.mean() - blue.mean())

    return {
        "mean_luminance": round(mean_luminance, 4),
        "luminance_variance": round(luminance_variance, 4),
        "contrast": round(contrast, 4),
        "left_right_ratio": round(left_right_ratio, 4),
        "top_bottom_ratio": round(top_bottom_ratio, 4),
        "saturation": round(saturation, 4),
        "color_warmth": round(color_warmth, 4),
    }


def bucket_for_features(features: LightingFeatures | None) -> str:
    """Return the lighting bucket for a feature dict, or ``unknown``."""
    if features is None:
        return "unknown"
    luminance = features["mean_luminance"]
    if luminance >= OVEREXPOSED_LUMINANCE_THRESHOLD:
        return "overexposed"
    if luminance <= DARK_LUMINANCE_THRESHOLD:
        return "dark"
    if abs(features["left_right_ratio"] - 1.0) >= SIDE_RATIO_THRESHOLD:
        return "side_lit"
    if abs(features["top_bottom_ratio"] - 1.0) >= TOP_RATIO_THRESHOLD:
        return "top_lit"
    if features["contrast"] >= HIGH_CONTRAST_STD_THRESHOLD:
        return "high_contrast"
    warmth = features["color_warmth"]
    if warmth >= WARMTH_THRESHOLD:
        return "warm"
    if warmth <= -WARMTH_THRESHOLD:
        return "cool"
    return "flat_frontal"


__all__ = get_module_objects(__name__)
