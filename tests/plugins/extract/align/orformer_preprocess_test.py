#!/usr/bin/env python3
"""Unit tests for ORFormer.pre_process() ROI geometry.

Covers:
  - Square output for square and rectangular detector boxes
  - max(width, height) side selection for non-square inputs
  - Configured crop_scale controls the expansion factor
  - Parametrised assertions for benchmark scales 1.20 / 1.30 / 1.40 / 1.50
  - Output dtype is np.int32
  - Rounding stability for odd-sized detector boxes

Import note
-----------
plugins.extract.base creates a circular import with lib.infer when imported in
isolation.  Boot lib.infer.align first so the full chain initialises before any
plugins.extract import.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.infer.align import Align  # noqa: F401  (import for side-effect)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _orformer(target_dist: float = 1.20):
    """Return an ORFormer instance without loading weights."""
    from plugins.extract.align.orformer import ORFormer

    plugin = object.__new__(ORFormer)
    plugin._target_dist = target_dist
    return plugin


# ---------------------------------------------------------------------------
# Square output
# ---------------------------------------------------------------------------


def test_pre_process_output_is_square_from_square_box() -> None:
    """A square detector box produces a square ROI."""
    plugin = _orformer()
    bbox = np.array([[10, 20, 110, 120]], dtype=np.int32)  # 100×100
    roi = plugin.pre_process(bbox)
    widths = roi[:, 2] - roi[:, 0]
    heights = roi[:, 3] - roi[:, 1]
    np.testing.assert_array_equal(widths, heights)


def test_pre_process_output_is_square_from_rect_box() -> None:
    """A rectangular detector box also produces a square ROI."""
    plugin = _orformer()
    bboxes = np.array(
        [
            [10, 20, 110, 80],  # 100 wide, 60 tall
            [0, 0, 50, 90],
        ],  # 50 wide, 90 tall
        dtype=np.int32,
    )
    roi = plugin.pre_process(bboxes)
    widths = roi[:, 2] - roi[:, 0]
    heights = roi[:, 3] - roi[:, 1]
    np.testing.assert_array_equal(widths, heights)


# ---------------------------------------------------------------------------
# max(width, height) side selection
# ---------------------------------------------------------------------------


def test_pre_process_uses_max_dimension() -> None:
    """Side length is driven by the longer dimension, not the shorter one."""
    plugin = _orformer(target_dist=1.0)
    bbox = np.array([[0, 0, 100, 200]], dtype=np.int32)  # 100×200; height dominates
    roi = plugin.pre_process(bbox)
    side = roi[0, 2] - roi[0, 0]
    # max(100, 200) * 1.0 = 200, rounded, so side == 200
    assert side == 200


# ---------------------------------------------------------------------------
# Known-value golden test
# ---------------------------------------------------------------------------


def test_pre_process_golden_default_scale() -> None:
    """Verify exact ROI coordinates for a concrete box at scale 1.20.

    bbox = [[10, 20, 110, 220]]
      width=100, height=200, center=(60, 120)
      side = 200 * 1.20 = 240, half = 120
      expected = [60-120, 120-120, 60+120, 120+120] = [-60, 0, 180, 240]
    """
    plugin = _orformer(target_dist=1.20)
    bbox = np.array([[10, 20, 110, 220]], dtype=np.int32)
    roi = plugin.pre_process(bbox)
    expected = np.array([[-60, 0, 180, 240]], dtype=np.int32)
    np.testing.assert_array_equal(roi, expected)


# ---------------------------------------------------------------------------
# Configured scale drives ROI size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale,expected_side",
    [
        (1.20, 240),
        (1.30, 260),
        (1.40, 280),
        (1.50, 300),
    ],
)
def test_pre_process_scale_controls_roi_size(scale: float, expected_side: int) -> None:
    """crop_scale config value directly controls the ROI side length.

    bbox = [[0, 0, 0, 200]] — height=200, width=0, center=(0, 100)
    side = 200 * scale, half = side / 2 (rounded)
    """
    plugin = _orformer(target_dist=scale)
    bbox = np.array([[0, 0, 0, 200]], dtype=np.int32)
    roi = plugin.pre_process(bbox)
    actual_side = roi[0, 2] - roi[0, 0]
    assert actual_side == expected_side


# ---------------------------------------------------------------------------
# Output dtype
# ---------------------------------------------------------------------------


def test_pre_process_output_dtype_is_int32() -> None:
    """pre_process must return np.int32 coordinates."""
    plugin = _orformer()
    bbox = np.array([[5, 10, 55, 60]], dtype=np.int32)
    roi = plugin.pre_process(bbox)
    assert roi.dtype == np.int32


# ---------------------------------------------------------------------------
# Rounding stability
# ---------------------------------------------------------------------------


def test_pre_process_rounding_stability_odd_sizes() -> None:
    """Odd-sized detector boxes round consistently without off-by-one errors."""
    plugin = _orformer(target_dist=1.20)
    # width=101, height=201 → max=201, side=201*1.2=241.2 → half=rint(120.6)=121
    bbox = np.array([[10, 10, 111, 211]], dtype=np.int32)
    roi = plugin.pre_process(bbox)
    # Output must still be square
    widths = roi[:, 2] - roi[:, 0]
    heights = roi[:, 3] - roi[:, 1]
    np.testing.assert_array_equal(widths, heights)
    # dtype preserved
    assert roi.dtype == np.int32


# ---------------------------------------------------------------------------
# Batch dimension
# ---------------------------------------------------------------------------


def test_pre_process_batch_shape_preserved() -> None:
    """Output shape matches input batch dimension."""
    plugin = _orformer()
    bboxes = np.array([[0, 0, 100, 100], [10, 10, 60, 80], [5, 5, 55, 55]], dtype=np.int32)
    roi = plugin.pre_process(bboxes)
    assert roi.shape == (3, 4)
