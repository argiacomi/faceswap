#!/usr/bin/env python3
"""Tests for the FaceQA deep-audit coverage metrics.

These exercise the pure-numpy / scikit-learn metric core with synthetic
coefficient matrices - no torch and no DECA weights required.
"""

from __future__ import annotations

import typing as T

import numpy as np
import pytest

from lib.faceqa.deep import metrics as m


def _diverse(rng: np.random.Generator, n: int = 300, d: int = 50) -> np.ndarray:
    return T.cast(np.ndarray, rng.normal(size=(n, d)))


def _collapsed(rng: np.random.Generator, n: int = 300, d: int = 50) -> np.ndarray:
    point = rng.normal(size=(1, d))
    return T.cast(np.ndarray, np.tile(point, (n, 1)) + rng.normal(scale=1e-3, size=(n, d)))


# ---------------------------------------------------------------------------
# space_coverage
# ---------------------------------------------------------------------------


def test_space_coverage_diverse_beats_collapsed() -> None:
    """A faceset spanning the space must out-cover a collapsed one.

    This is the core property: absolute binning (not per-faceset z-scoring)
    means a collapsed coefficient lands in a single bin.
    """
    rng = np.random.default_rng(0)
    diverse = m.space_coverage(_diverse(rng))
    collapsed = m.space_coverage(_collapsed(rng))
    assert diverse["occupied_coverage"] > collapsed["occupied_coverage"]
    assert diverse["entropy_coverage"] > collapsed["entropy_coverage"]
    assert collapsed["entropy_coverage"] == 0.0
    assert collapsed["occupied_coverage"] <= 1.0 / m.DEFAULT_BINS + 1e-9


def test_space_coverage_empty_is_zeroed() -> None:
    for empty in (None, np.empty((0, 0)), np.empty((0, 50))):
        report = m.space_coverage(empty)
        assert report["samples"] == 0
        assert report["occupied_coverage"] == 0.0
        assert report["per_dimension"] == []


def test_space_coverage_drops_non_finite_rows() -> None:
    data = np.array([[1.0, 2.0], [np.nan, 3.0], [0.5, 0.5], [np.inf, 0.0]])
    assert m.space_coverage(data)["samples"] == 2


def test_space_coverage_scale_widens_bins() -> None:
    """A larger reference scale compresses values into fewer bins."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(300, 20))
    tight = m.space_coverage(data, scale=1.0)
    wide = m.space_coverage(data, scale=10.0)
    assert tight["occupied_coverage"] >= wide["occupied_coverage"]


# ---------------------------------------------------------------------------
# latent_entropy
# ---------------------------------------------------------------------------


def test_latent_entropy_collapsed_single_cell() -> None:
    rng = np.random.default_rng(2)
    collapsed = m.latent_entropy(_collapsed(rng))
    assert collapsed["occupied_cells"] == 1
    assert collapsed["entropy"] == 0.0


def test_latent_entropy_diverse_high() -> None:
    rng = np.random.default_rng(3)
    diverse = m.latent_entropy(_diverse(rng))
    assert diverse["occupied_cells"] > 1
    assert diverse["entropy"] > 0.5


def test_latent_entropy_empty() -> None:
    report = m.latent_entropy(None)
    assert report["samples"] == 0
    assert report["entropy"] == 0.0


# ---------------------------------------------------------------------------
# cluster_coverage
# ---------------------------------------------------------------------------


def test_cluster_coverage_dispersion_separates_collapse() -> None:
    """``mean_dispersion`` (absolute) is the collapse signal, not ``balance``."""
    rng = np.random.default_rng(4)
    diverse = m.cluster_coverage(_diverse(rng))
    collapsed = m.cluster_coverage(_collapsed(rng))
    assert diverse["mean_dispersion"] > collapsed["mean_dispersion"]
    assert collapsed["mean_dispersion"] < 0.05
    assert diverse["mean_dispersion"] > 0.5


def test_cluster_coverage_is_deterministic() -> None:
    rng = np.random.default_rng(5)
    data = _diverse(rng)
    assert m.cluster_coverage(data) == m.cluster_coverage(data)


def test_cluster_coverage_handles_fewer_samples_than_clusters() -> None:
    rng = np.random.default_rng(6)
    data = rng.normal(size=(3, 50))
    report = m.cluster_coverage(data, n_clusters=8)
    assert report["occupied_clusters"] <= 3
    assert sum(report["cluster_sizes"]) == 3


def test_cluster_coverage_single_sample() -> None:
    report = m.cluster_coverage(np.zeros((1, 50)))
    assert report["occupied_clusters"] == 1
    assert report["cluster_sizes"] == [1]


def test_cluster_coverage_empty() -> None:
    report = m.cluster_coverage(None)
    assert report["samples"] == 0
    assert report["occupied_clusters"] == 0
    assert report["cluster_sizes"] == []


# ---------------------------------------------------------------------------
# _normalized_entropy primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([10, 0, 0, 0], 0.0),  # single occupied bin
        ([5, 5], 1.0),  # perfectly uniform over 2
        ([0, 0, 0], 0.0),  # empty
    ],
)
def test_normalized_entropy(counts: list[int], expected: float) -> None:
    assert m._normalized_entropy(np.array(counts, dtype=float)) == pytest.approx(expected)
