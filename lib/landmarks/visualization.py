#!/usr/bin/env python3
"""Debug visualization helpers with an optional OpenCV fast path."""

from __future__ import annotations

import typing as T

import numpy as np

from lib.landmarks.schema import LandmarkPrediction, to_canonical_68

Color = tuple[int, int, int]


def _coerce_image(image: np.ndarray, *, copy: bool) -> np.ndarray:
    array = np.array(image, copy=copy)
    if array.ndim not in (2, 3):
        raise ValueError(f"debug images must be 2D or 3D arrays, got shape {array.shape}")
    return array


def _paint_numpy(image: np.ndarray, x_val: int, y_val: int, color: Color, radius: int) -> None:
    height, width = image.shape[:2]
    x_min = max(0, x_val - radius)
    x_max = min(width, x_val + radius + 1)
    y_min = max(0, y_val - radius)
    y_max = min(height, y_val + radius + 1)
    if image.ndim == 2:
        image[y_min:y_max, x_min:x_max] = color[0]
    else:
        image[y_min:y_max, x_min:x_max, : min(3, image.shape[2])] = color[: image.shape[2]]


def draw_landmarks(
    image: np.ndarray,
    points: LandmarkPrediction | np.ndarray,
    *,
    color: Color = (0, 255, 0),
    radius: int = 1,
    copy: bool = True,
) -> np.ndarray:
    """Draw landmarks on an image using OpenCV when available, otherwise numpy writes."""
    if radius < 0:
        raise ValueError("radius cannot be negative")
    output = _coerce_image(image, copy=copy)
    landmark_points = (
        points.canonical_68().points
        if isinstance(points, LandmarkPrediction)
        else to_canonical_68(points)
    )
    try:
        import cv2  # type:ignore[import-not-found]
    except ImportError:
        cv2 = None

    for x_float, y_float in landmark_points[:, :2]:
        x_val = int(round(float(x_float)))
        y_val = int(round(float(y_float)))
        if cv2 is not None:
            cv2.circle(output, (x_val, y_val), radius, color, thickness=-1)
        else:
            _paint_numpy(output, x_val, y_val, color, radius)
    return output


def make_debug_overlay(
    image: np.ndarray,
    predictions: T.Mapping[str, LandmarkPrediction | np.ndarray],
    *,
    colors: T.Sequence[Color] | None = None,
) -> np.ndarray:
    """Overlay multiple named predictions for debugging."""
    palette = colors or (
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    )
    output = _coerce_image(image, copy=True)
    for idx, points in enumerate(predictions.values()):
        output = draw_landmarks(
            output,
            points,
            color=palette[idx % len(palette)],
            radius=1,
            copy=False,
        )
    return output
