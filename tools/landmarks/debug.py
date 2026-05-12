#!/usr/bin/env python3
"""Small debug helpers for standalone landmark experiments."""

from __future__ import annotations

import numpy as np

from lib.landmarks.schema import LandmarkPrediction
from lib.landmarks.visualization import make_debug_overlay


def overlay_predictions(
    image: np.ndarray,
    predictions: dict[str, LandmarkPrediction | np.ndarray],
) -> np.ndarray:
    """Return an image with landmark predictions overlaid."""
    return make_debug_overlay(image, predictions)
