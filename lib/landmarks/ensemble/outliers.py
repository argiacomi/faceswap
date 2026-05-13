#!/usr/bin/env python3
"""Compatibility wrappers for canonical landmark outlier rejection."""

from __future__ import annotations

from lib.landmarks.rejection import (
    PerLandmarkOutlierResult,
    pairwise_disagreement,
    reject_outliers,
    weighted_median,
)

__all__ = [
    "PerLandmarkOutlierResult",
    "pairwise_disagreement",
    "reject_outliers",
    "weighted_median",
]
