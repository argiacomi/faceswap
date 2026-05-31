#!/usr/bin/env python3
"""Unit tests for the shared bbox→square-ROI helper extracted in #195.

Verifies the rounding / centring / dtype behaviour the previous three
duplicate copies (ensemble, SPIGA, HRNet) preserved, AND that the
helper produces byte-identical output to each of those legacy
expressions on the same input.
"""

from __future__ import annotations

import numpy as np

from lib.align.aligned_utils import bbox_to_square_roi


def _legacy(batch: np.ndarray, scale: float) -> np.ndarray:
    """The inline expression that lived in the three aligner plugins
    before #195. Reproduced here so the test pins the exact contract."""
    heights = batch[:, 3] - batch[:, 1]
    widths = batch[:, 2] - batch[:, 0]
    ctr_x = np.rint((batch[:, 0] + batch[:, 2]) * 0.5).astype("int32")
    ctr_y = np.rint((batch[:, 1] + batch[:, 3]) * 0.5).astype("int32")
    side = np.maximum(widths, heights) * scale
    half = np.rint(side * 0.5).astype("int32")
    retval = np.empty((batch.shape[0], 4), dtype=np.int32)
    retval[:, 0] = ctr_x - half
    retval[:, 1] = ctr_y - half
    retval[:, 2] = ctr_x + half
    retval[:, 3] = ctr_y + half
    return retval  # type: ignore[no-any-return]


def test_helper_matches_legacy_inline_expression_for_int_input() -> None:
    """Integer bboxes — the most common shape the aligners receive."""
    batch = np.array(
        [
            [10, 20, 110, 130],
            [50, 60, 140, 170],
            [0, 0, 200, 200],
        ],
        dtype=np.int32,
    )
    for scale in (1.0, 1.25, 1.3, 1.5):
        np.testing.assert_array_equal(
            bbox_to_square_roi(batch, scale),
            _legacy(batch, scale),
        )


def test_helper_matches_legacy_for_float_input() -> None:
    """Float bboxes — the aligners also receive sub-pixel detector boxes."""
    batch = np.array(
        [
            [10.4, 20.6, 110.2, 130.7],
            [50.1, 60.9, 140.4, 170.3],
        ],
        dtype=np.float32,
    )
    for scale in (1.0, 1.25, 1.3, 1.5):
        np.testing.assert_array_equal(
            bbox_to_square_roi(batch, scale),
            _legacy(batch, scale),
        )


def test_output_is_int32_with_expected_shape() -> None:
    batch = np.array([[0, 0, 100, 100]], dtype=np.int32)
    roi = bbox_to_square_roi(batch, 1.25)
    assert roi.dtype == np.int32
    assert roi.shape == (1, 4)


def test_square_output_is_actually_square() -> None:
    """The returned ROI must be square — side length should match in
    both x and y (modulo rounding by 1 px which is inherent to the
    original expression's ``rint(half * 2)``)."""
    batch = np.array([[10, 20, 90, 200]], dtype=np.int32)
    roi = bbox_to_square_roi(batch, 1.0)
    width = int(roi[0, 2] - roi[0, 0])
    height = int(roi[0, 3] - roi[0, 1])
    assert width == height


def test_centring_uses_input_bbox_centre() -> None:
    """The output ROI must be centred on the centre of the input bbox
    (rounded to int) — bumping the input centre by 10 px shifts the
    output ROI by 10 px."""
    base = np.array([[10, 20, 110, 120]], dtype=np.int32)
    shifted = base + np.array([10, 10, 10, 10], dtype=np.int32)

    roi_base = bbox_to_square_roi(base, 1.25)
    roi_shifted = bbox_to_square_roi(shifted, 1.25)

    np.testing.assert_array_equal(
        roi_shifted - roi_base,
        np.array([[10, 10, 10, 10]], dtype=np.int32),
    )


def test_helper_handles_empty_batch() -> None:
    """Empty input must produce an empty (0, 4) int32 output without
    raising — defends against the aligners receiving a zero-face batch."""
    batch = np.empty((0, 4), dtype=np.int32)  # type: ignore[var-annotated]
    roi = bbox_to_square_roi(batch, 1.25)
    assert roi.shape == (0, 4)
    assert roi.dtype == np.int32
