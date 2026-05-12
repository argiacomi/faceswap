#!/usr/bin/env python3
"""Tests for landmark coordinate transforms."""

import numpy as np
import pytest

from lib.landmarks.coordinates import (
    frame_to_normalized_crop,
    normalized_crop_to_frame,
    roi_to_matrix,
)


def test_roi_to_matrix_maps_unit_crop_to_frame_pixels() -> None:
    """Normalized crop coordinates map to original frame pixels."""
    matrix = roi_to_matrix(np.array([10, 20, 110, 120], dtype="int32"))
    points = np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]], dtype="float32")

    transformed = normalized_crop_to_frame(points, matrix)

    np.testing.assert_allclose(
        transformed,
        np.array([[10.0, 20.0], [60.0, 45.0], [110.0, 120.0]], dtype="float32"),
    )


def test_frame_to_normalized_crop_is_inverse_transform() -> None:
    """Frame-space points can be mapped back to Faceswap plugin output space."""
    matrix = roi_to_matrix(np.array([10, 20, 110, 120], dtype="int32"))
    points = np.array([[10.0, 20.0], [60.0, 45.0], [110.0, 120.0]], dtype="float32")

    transformed = frame_to_normalized_crop(points, matrix)

    np.testing.assert_allclose(
        transformed,
        np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]], dtype="float32"),
        rtol=1e-6,
        atol=1e-6,
    )


def test_batched_coordinate_transforms() -> None:
    """Batched points use the matching matrix for each face."""
    matrices = roi_to_matrix(np.array([[0, 0, 10, 10], [10, 20, 30, 40]], dtype="int32"))
    points = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.25, 0.5], [1.0, 0.0]],
        ],
        dtype="float32",
    )

    transformed = normalized_crop_to_frame(points, matrices)

    np.testing.assert_allclose(
        transformed,
        np.array(
            [
                [[0.0, 0.0], [10.0, 10.0]],
                [[15.0, 30.0], [30.0, 20.0]],
            ],
            dtype="float32",
        ),
    )


def test_roi_to_matrix_requires_square_roi() -> None:
    """Aligner crop matrices intentionally mirror Faceswap's square ROI contract."""
    with pytest.raises(ValueError, match="square"):
        roi_to_matrix(np.array([0, 0, 10, 12], dtype="int32"))
