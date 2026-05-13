#!/usr/bin/env python3
"""Standalone landmark ensemble helpers."""

import typing as T

__all__ = ("predict_landmarks_68",)


def predict_landmarks_68(*args: T.Any, **kwargs: T.Any):
    """Predict fused canonical 68-point landmarks in frame-pixel space."""
    from lib.landmarks.predict import predict_landmarks_68 as _predict_landmarks_68

    return _predict_landmarks_68(*args, **kwargs)
