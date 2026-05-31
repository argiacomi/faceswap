#!/usr/bin/env python3
"""Pure-numpy coverage metrics over DECA coefficient vectors.

These functions are the measurable core of the FaceQA deep audit (issue
#156). They take stacked per-face coefficient vectors (expression, lighting,
or a combined latent) and report how well the faceset *spans* that space:

* :func:`space_coverage` - per-dimension occupancy + entropy over a binned,
  z-scored coefficient space. Used for expression-space and lighting-space
  coverage.
* :func:`latent_entropy` - normalized Shannon entropy of a joint coarse
  binning of the dominant latent dimensions.
* :func:`cluster_coverage` - balance of a deterministic k-means partition of
  the latent (how evenly samples spread across discovered clusters).

Everything is deterministic (seeded) and depends only on numpy plus
scikit-learn (a pinned base dependency), so the deep audit metrics are fully
unit-testable without the DECA model or torch.
"""

from __future__ import annotations

import logging
import typing as T

import numpy as np
from sklearn.cluster import KMeans

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

# Default binning / clustering parameters. Chosen to stay stable for small
# facesets (a few hundred faces) without exploding the joint cell count.
DEFAULT_BINS = 8
DEFAULT_SPACE_DIMS = 4
DEFAULT_LATENT_BINS = 4
DEFAULT_LATENT_DIMS = 6
DEFAULT_CLUSTERS = 8
DEFAULT_CLIP = 3.0
DEFAULT_SEED = 1337


def _as_matrix(features: np.ndarray | None) -> np.ndarray:
    """Return ``features`` as a finite ``(n_samples, n_dims)`` float64 matrix.

    Rows containing non-finite values are dropped so a single bad encode does
    not poison the whole metric.
    """
    if features is None:
        return T.cast(np.ndarray, np.empty((0, 0), dtype=np.float64))
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.size == 0:
        return T.cast(np.ndarray, np.empty((0, 0), dtype=np.float64))
    finite_rows = np.isfinite(matrix).all(axis=1)
    return T.cast(np.ndarray, matrix[finite_rows])


def _normalized_entropy(counts: np.ndarray) -> float:
    """Return the Shannon entropy of ``counts`` normalized to ``[0, 1]``.

    ``0.0`` means all mass sits in one bin; ``1.0`` means a perfectly uniform
    spread across the occupied support. A single occupied bin returns ``0.0``.
    """
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts[counts > 0] / total
    if probs.size <= 1:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / float(np.log(probs.size))


def _top_variance_dims(matrix: np.ndarray, n_dims: int) -> np.ndarray:
    """Return the indices of the ``n_dims`` highest-variance columns.

    Indices are returned in descending variance order so the most expressive
    coefficients drive the binning.
    """
    variances = matrix.var(axis=0)
    count = int(min(n_dims, matrix.shape[1]))
    if count <= 0:
        return T.cast(np.ndarray, np.empty(0, dtype=np.intp))
    # ``argsort`` is ascending; take the tail and reverse for descending order.
    return T.cast(np.ndarray, np.argsort(variances)[-count:][::-1])


def _center_scale_clip(
    column: np.ndarray,
    center: float,
    scale: float,
    clip: float,
) -> np.ndarray:
    """Map ``column`` into the fixed reference range ``[-clip, clip]``.

    Coverage is measured against an *absolute* reference scale - NOT the
    faceset's own standard deviation. Per-sample z-scoring would normalize
    away the very spread we are trying to quantify, making a faceset
    collapsed onto a single expression look identical to a diverse one. With
    a fixed ``center`` / ``scale``, a collapsed coefficient lands in a single
    bin (low coverage) while a faceset that genuinely spans the space fills
    many bins (high coverage).
    """
    safe_scale = scale if abs(scale) > 1e-9 else 1.0
    return T.cast(np.ndarray, np.clip((column - center) / safe_scale, -clip, clip))


def _bin_indices(values: np.ndarray, n_bins: int, clip: float) -> np.ndarray:
    """Map clipped z-scores in ``[-clip, clip]`` to integer bins ``[0, n_bins)``."""
    width = (2.0 * clip) / n_bins
    idx = np.floor((values + clip) / width).astype(np.intp)
    return T.cast(np.ndarray, np.clip(idx, 0, n_bins - 1))


def space_coverage(
    features: np.ndarray | None,
    *,
    n_bins: int = DEFAULT_BINS,
    n_dims: int = DEFAULT_SPACE_DIMS,
    clip: float = DEFAULT_CLIP,
    center: float = 0.0,
    scale: float = 1.0,
) -> dict[str, T.Any]:
    """Report occupancy + entropy coverage of a coefficient space.

    The space is reduced to its ``n_dims`` highest-variance coefficients,
    each mapped into the fixed reference range ``[-clip, clip]`` (via
    ``center`` / ``scale``) and binned into ``n_bins``. Per-dimension
    occupancy (occupied bins / ``n_bins``) and normalized entropy are
    averaged across the selected dimensions.

    Coverage is absolute: a faceset collapsed onto one expression occupies a
    single bin per dimension regardless of how it is positioned, while one
    spanning the space fills many bins. ``center`` / ``scale`` set the
    reference frame (DECA FLAME coefficients are ~unit-scaled, so the
    defaults suit expression / pose; the caller pre-scales lighting SH).

    Parameters
    ----------
    features
        ``(n_samples, n_dims)`` coefficient matrix. ``None`` / empty yields a
        zeroed report with ``samples == 0``.

    Returns
    -------
    dict
        ``samples``, ``dimensions_used``, ``bins``, ``occupied_coverage``,
        ``entropy_coverage``, and ``per_dimension`` breakdown.
    """
    matrix = _as_matrix(features)
    samples = int(matrix.shape[0])
    bins = max(1, int(n_bins))
    if samples == 0:
        return {
            "samples": 0,
            "dimensions_used": 0,
            "bins": bins,
            "occupied_coverage": 0.0,
            "entropy_coverage": 0.0,
            "per_dimension": [],
        }

    dims = _top_variance_dims(matrix, n_dims)
    per_dimension: list[dict[str, float | int]] = []
    occupied_ratios: list[float] = []
    entropy_ratios: list[float] = []
    for dim in dims:
        mapped = _center_scale_clip(matrix[:, dim], center, scale, clip)
        idx = _bin_indices(mapped, bins, clip)
        counts = np.bincount(idx, minlength=bins)
        occupied = float((counts > 0).sum()) / bins
        entropy = _normalized_entropy(counts)
        occupied_ratios.append(occupied)
        entropy_ratios.append(entropy)
        per_dimension.append(
            {
                "dimension": int(dim),
                "occupied_bins": int((counts > 0).sum()),
                "occupied_coverage": round(occupied, 4),
                "entropy_coverage": round(entropy, 4),
            }
        )

    return {
        "samples": samples,
        "dimensions_used": int(dims.size),
        "bins": bins,
        "occupied_coverage": round(float(np.mean(occupied_ratios)), 4),
        "entropy_coverage": round(float(np.mean(entropy_ratios)), 4),
        "per_dimension": per_dimension,
    }


def latent_entropy(
    features: np.ndarray | None,
    *,
    n_bins: int = DEFAULT_LATENT_BINS,
    n_dims: int = DEFAULT_LATENT_DIMS,
    clip: float = DEFAULT_CLIP,
    center: float = 0.0,
    scale: float = 1.0,
) -> dict[str, T.Any]:
    """Report normalized Shannon entropy of a joint coarse binning.

    The latent is reduced to its ``n_dims`` highest-variance dimensions and
    each is coarsely binned into ``n_bins`` over the fixed reference range
    ``[-clip, clip]``; the per-sample tuple of bin indices defines a joint
    cell. Entropy of the cell-occupancy distribution measures how evenly the
    faceset fills the dominant latent volume. Because binning is absolute, a
    collapsed faceset lands in a single cell (entropy ``0.0``).
    """
    matrix = _as_matrix(features)
    samples = int(matrix.shape[0])
    bins = max(1, int(n_bins))
    if samples == 0:
        return {
            "samples": 0,
            "dimensions_used": 0,
            "bins": bins,
            "occupied_cells": 0,
            "entropy": 0.0,
        }

    dims = _top_variance_dims(matrix, n_dims)
    # Encode each sample's per-dim bin tuple as a single mixed-radix cell id.
    cell_ids: np.ndarray = np.zeros(samples, dtype=np.int64)
    for dim in dims:
        mapped = _center_scale_clip(matrix[:, dim], center, scale, clip)
        idx: np.ndarray = _bin_indices(mapped, bins, clip).astype(np.int64)
        cell_ids = cell_ids * bins + idx

    _, counts = np.unique(cell_ids, return_counts=True)
    return {
        "samples": samples,
        "dimensions_used": int(dims.size),
        "bins": bins,
        "occupied_cells": int(counts.size),
        "entropy": round(_normalized_entropy(counts), 4),
    }


def _kmeans_labels(
    matrix: np.ndarray,
    n_clusters: int,
    *,
    seed: int,
) -> np.ndarray:
    """Return deterministic k-means labels for ``matrix``.

    Standardizes columns first so no single high-variance coefficient
    dominates the Euclidean distance, then defers to
    :class:`sklearn.cluster.KMeans` (a pinned base dependency) with a fixed
    ``random_state`` for reproducible audits.
    """
    std = matrix.std(axis=0)
    std[std <= 1e-9] = 1.0
    scaled = (matrix - matrix.mean(axis=0)) / std
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return T.cast(np.ndarray, model.fit_predict(scaled).astype(np.intp))


def cluster_coverage(
    features: np.ndarray | None,
    *,
    n_clusters: int = DEFAULT_CLUSTERS,
    seed: int = DEFAULT_SEED,
) -> dict[str, T.Any]:
    """Report how evenly the faceset spreads across latent clusters.

    Runs a deterministic k-means over the latent and measures the *balance*
    of the partition: the normalized entropy of the cluster-size
    distribution. ``balance`` describes partition *structure* (is whatever
    spread exists multi-modal and even, or dominated by one mode); it is
    computed on standardized features, so on its own it cannot distinguish a
    genuinely diverse faceset from a near-collapsed one that k-means still
    splits into even noise clusters. ``mean_dispersion`` - the mean
    per-dimension standard deviation in *absolute* coefficient space -
    supplies that missing magnitude signal: it falls toward ``0.0`` when the
    faceset collapses onto a single point.
    """
    matrix = _as_matrix(features)
    samples = int(matrix.shape[0])
    requested = max(1, int(n_clusters))
    if samples == 0:
        return {
            "samples": 0,
            "requested_clusters": requested,
            "occupied_clusters": 0,
            "balance": 0.0,
            "min_cluster_fraction": 0.0,
            "mean_dispersion": 0.0,
            "cluster_sizes": [],
        }

    mean_dispersion = round(float(matrix.std(axis=0).mean()), 4)
    effective = int(min(requested, samples))
    if effective <= 1:
        # A single cluster (or single sample) is fully occupied but carries no
        # spread information.
        return {
            "samples": samples,
            "requested_clusters": requested,
            "occupied_clusters": 1,
            "balance": 0.0,
            "min_cluster_fraction": 1.0,
            "mean_dispersion": mean_dispersion,
            "cluster_sizes": [samples],
        }

    labels = _kmeans_labels(matrix, effective, seed=seed)
    sizes = np.bincount(labels, minlength=effective)
    occupied = int((sizes > 0).sum())
    balance = _normalized_entropy(sizes)
    min_fraction = float(sizes[sizes > 0].min()) / samples if occupied else 0.0
    return {
        "samples": samples,
        "requested_clusters": requested,
        "occupied_clusters": occupied,
        "balance": round(balance, 4),
        "min_cluster_fraction": round(min_fraction, 4),
        "mean_dispersion": mean_dispersion,
        "cluster_sizes": [int(size) for size in sizes],
    }


__all__ = get_module_objects(__name__)
