#!/usr/bin/env python3
"""Targeted DARK decoder regressions."""

from __future__ import annotations

import numpy as np

from plugins.extract.align.dark_decoder import Dark


def test_dark_decoder_flat_heatmaps_stay_finite_and_in_bounds() -> None:
    """Flat heatmaps should fall back safely instead of producing large coordinates."""
    decoder = Dark(68, 64)
    heatmap = np.zeros((2, 68, 64, 64), dtype=np.float32)

    coords = decoder(heatmap)

    assert coords.dtype == np.float32
    assert coords.shape == (2, 68, 2)
    assert np.all(np.isfinite(coords))
    assert float(coords.min()) >= 0.0
    assert float(coords.max()) <= 63.0


def test_dark_decoder_noisy_heatmaps_stay_finite_and_in_bounds() -> None:
    """Weak noisy heatmaps should remain bounded in heatmap space."""
    decoder = Dark(68, 64)
    rng = np.random.default_rng(123)
    heatmap = rng.uniform(0.0, 1e-4, (2, 68, 64, 64)).astype(np.float32)

    coords = decoder(heatmap)

    assert coords.dtype == np.float32
    assert np.all(np.isfinite(coords))
    assert float(coords.min()) >= 0.0
    assert float(coords.max()) <= 63.0


def test_dark_taylor_falls_back_for_near_singular_hessian() -> None:
    """A near-singular Hessian should keep the original argmax coordinate."""
    decoder = Dark(1, 64, min_hessian_det=1e-6)
    heatmap = np.zeros((1, 1, 64, 64), dtype=np.float32)
    x_coord = 30
    y_coord = 30
    heatmap[0, 0, y_coord, x_coord + 1] = 1.0
    heatmap[0, 0, y_coord, x_coord - 1] = -1.0
    heatmap[0, 0, y_coord, x_coord + 2] = 2e-4
    heatmap[0, 0, y_coord - 2, x_coord] = 2e-4
    heatmap[0, 0, y_coord + 2, x_coord] = 2e-4
    coords = np.array([[[x_coord, y_coord]]], dtype=np.float32)

    refined = decoder.taylor(heatmap, coords.copy())

    np.testing.assert_array_equal(refined, coords)


def test_dark_taylor_clamps_large_offsets_to_local_refinement() -> None:
    """Large but finite Taylor steps should be clipped to local sub-pixel motion."""
    decoder = Dark(1, 64, min_hessian_det=1e-12, max_offset=1.0)
    heatmap = np.zeros((1, 1, 64, 64), dtype=np.float32)
    x_coord = 30
    y_coord = 30
    heatmap[0, 0, y_coord, x_coord + 1] = 1.0
    heatmap[0, 0, y_coord, x_coord - 1] = -1.0
    heatmap[0, 0, y_coord, x_coord + 2] = 2e-4
    heatmap[0, 0, y_coord - 2, x_coord] = 2e-4
    heatmap[0, 0, y_coord + 2, x_coord] = 2e-4
    coords = np.array([[[x_coord, y_coord]]], dtype=np.float32)

    refined = decoder.taylor(heatmap, coords.copy())

    assert refined.dtype == np.float32
    assert refined[0, 0, 0] == np.float32(x_coord - 1.0)
    assert refined[0, 0, 1] == np.float32(y_coord)
