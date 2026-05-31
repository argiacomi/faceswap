#!/usr/bin/env python3
"""Regression tests for the P2 ``lib/align`` performance changes (issue #191)."""

from __future__ import annotations

import numpy as np


def test_none_sentinel_cache_does_not_recompute_zero_values() -> None:
    """Cached ``average_distance`` / ``relative_eye_mouth_position`` must trust
    a stored zero. Previously the cache defaults were ``0.0`` with truthiness
    guards, so a perfectly-aligned face (legitimate ``0.0``) was recomputed
    on every property access.
    """
    from lib.align.aligned_face import _FaceCache

    cache = _FaceCache()
    assert cache.average_distance is None
    assert cache.relative_eye_mouth_position is None

    cache.average_distance = 0.0
    cache.relative_eye_mouth_position = 0.0

    # The property guards use ``is None`` now, so a stored 0.0 short-circuits
    # the recomputation. The cache itself is the contract we're pinning here:
    # storing 0.0 is distinguishable from never-computed (None).
    assert cache.average_distance == 0.0
    assert cache.relative_eye_mouth_position == 0.0
    assert cache.average_distance is not None
    assert cache.relative_eye_mouth_position is not None


def test_centerings_constant_matches_centering_type() -> None:
    """``_CENTERINGS`` is the hoisted equivalent of
    ``T.get_args(T.Literal['legacy','face','head'])``. Keep them aligned so
    ``_padding_from_coverage`` stays in sync with ``CenteringType``.
    """
    import typing as T

    from lib.align.aligned_face import _CENTERINGS

    assert T.get_args(T.Literal["legacy", "face", "head"]) == _CENTERINGS


def test_mask_stored_mask_cache_invalidates_on_replace_mask() -> None:
    """``Mask.stored_mask`` must drop its cache when the underlying ``_mask``
    is rewritten (``add`` -> ``replace_mask`` -> ``self._mask = ...``).
    """
    from lib.align.aligned_mask import Mask

    mask = Mask()
    base = np.zeros((32, 32), dtype=np.uint8)  # type: ignore[var-annotated]
    base[10:20, 10:20] = 200
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mask.add(base, identity)

    first = mask.stored_mask
    second = mask.stored_mask
    assert first is second, "second access should hit the cache"

    # Replace with a different mask; cache must invalidate.
    different = np.full((32, 32), 50, dtype=np.uint8)  # type: ignore[var-annotated]
    mask.replace_mask(different)
    third = mask.stored_mask
    assert third is not first, "cache must be dropped after replace_mask"
    assert int(third.mean()) != int(first.mean())


def test_solve_pnp_returns_correct_shape() -> None:
    """``Batch3D.solve_pnp`` must still return ``(2, N, 3, 1)`` after the
    preallocated buffer refactor.
    """
    from lib.align.pose import _CORE_LMS, Batch3D

    rng = np.random.default_rng(0)
    n = 4
    landmarks = rng.uniform(0.2, 0.8, (n, 68, 2)).astype(np.float32)

    result = Batch3D.solve_pnp(landmarks)

    expected_shape = (2, n, 3, 1)
    assert result.shape == expected_shape
    assert result.dtype == np.float32
    del _CORE_LMS  # quiet unused import
