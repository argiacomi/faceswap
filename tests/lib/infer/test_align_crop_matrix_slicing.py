#!/usr/bin/env python3
"""Tests for align runner crop-matrix slicing across re-feed chunks."""

from __future__ import annotations

import typing as T
from types import MethodType, SimpleNamespace

import numpy as np
import numpy.typing as npt

from lib.infer.align import Align, ReAlign, ReFeed
from lib.infer.objects import ExtractBatch
from plugins.extract.base import ExtractPlugin


class _Plugin:
    """Fake plugin that records geometry handed to it before each predict chunk."""

    name = "ensemble"
    input_size = 256
    batch_size = 2

    def __init__(self) -> None:
        self.calls: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]] = []
        self.last_debug_metadata: list[dict[str, T.Any]] = []

    def set_crop_matrices(
        self,
        matrices: npt.NDArray[np.float32],
        *,
        detector_bboxes: npt.NDArray[np.float32] | None = None,
    ) -> None:
        self.calls.append(
            (
                matrices.copy(),
                None if detector_bboxes is None else detector_bboxes.copy(),
            )
        )


def _handler(plugin: _Plugin, total_feeds: int) -> Align:
    handler = object.__new__(Align)
    handler.plugin = T.cast(ExtractPlugin, plugin)
    handler._re_feed = T.cast(ReFeed, SimpleNamespace(total_feeds=total_feeds))
    return handler


def test_detector_bboxes_follow_face_major_refeed_rows() -> None:
    plugin = _Plugin()
    handler = _handler(plugin, total_feeds=3)
    batch = T.cast(
        ExtractBatch,
        SimpleNamespace(
            bboxes=np.array([[10, 20, 30, 40], [100, 200, 300, 400]], dtype=np.int32),
        ),
    )
    matrices: npt.NDArray[np.float32] = np.arange(
        6 * 3 * 3,
        dtype=np.float32,
    ).reshape(6, 3, 3)

    bboxes = handler._detector_bboxes_for_matrices(
        batch,
        matrices[2:4],
        row_indices=np.arange(2, 4, dtype=np.intp),
        source_matrix_count=6,
    )

    np.testing.assert_allclose(
        bboxes,
        np.array([[10, 20, 30, 40], [100, 200, 300, 400]], dtype=np.float32),
    )


def test_get_predictions_sets_exact_matrix_and_bbox_slice_per_refeed_chunk() -> None:
    plugin = _Plugin()
    handler = _handler(plugin, total_feeds=3)
    batch = T.cast(
        ExtractBatch,
        SimpleNamespace(
            matrices=np.arange(6 * 3 * 3, dtype=np.float32).reshape(6, 3, 3),
            bboxes=np.array([[10, 20, 30, 40], [100, 200, 300, 400]], dtype=np.int32),
        ),
    )
    feed: npt.NDArray[np.float32] = np.arange(
        6 * 4,
        dtype=np.float32,
    ).reshape(6, 4)

    def predict(self: Align, chunk: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        del self
        plugin.last_debug_metadata = [{"feed0": float(row[0])} for row in chunk]
        out = np.repeat(chunk[:, :2, None], 68, axis=2).swapaxes(1, 2).astype(np.float32)
        return T.cast(npt.NDArray[np.float32], out)

    handler._predict = MethodType(predict, handler)  # type: ignore[method-assign]

    result = handler._get_predictions(is_final=True, feed=feed, batch=batch)

    assert result.shape == (6, 68, 2)
    assert len(plugin.calls) == 3

    np.testing.assert_allclose(plugin.calls[0][0], batch.matrices[0:2])
    np.testing.assert_allclose(plugin.calls[1][0], batch.matrices[2:4])
    np.testing.assert_allclose(plugin.calls[2][0], batch.matrices[4:6])

    np.testing.assert_allclose(
        plugin.calls[0][1],
        np.array([[10, 20, 30, 40], [10, 20, 30, 40]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        plugin.calls[1][1],
        np.array([[10, 20, 30, 40], [100, 200, 300, 400]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        plugin.calls[2][1],
        np.array([[100, 200, 300, 400], [100, 200, 300, 400]], dtype=np.float32),
    )

    assert plugin.last_debug_metadata == [
        {"feed0": 0.0},
        {"feed0": 4.0},
        {"feed0": 8.0},
        {"feed0": 12.0},
        {"feed0": 16.0},
        {"feed0": 20.0},
    ]


class _ReFeedDouble:
    """Callable re-feed double for second-pass re-align tests."""

    def __init__(self, matrices: npt.NDArray[np.float32]) -> None:
        self.total_feeds = 3
        self._matrices = matrices

    def __call__(
        self,
        matrices: npt.NDArray[np.float32],
        with_roi: bool = False,
        size: int = 0,
    ) -> npt.NDArray[np.float32]:
        del matrices, with_roi, size
        return self._matrices


class _ReAlignDouble:
    """ReAlign double exposing default and final crop matrices."""

    def __init__(
        self,
        default_crop_matrices: npt.NDArray[np.float32],
        final_matrices: npt.NDArray[np.float32],
    ) -> None:
        self.iterations = 2
        self.default_crop_matrices = default_crop_matrices
        self.matrices = final_matrices
        self.received_matrices: npt.NDArray[np.float32] | None = None

    def get_images(
        self,
        matrices: npt.NDArray[np.float32],
        feeds: int,
    ) -> npt.NDArray[np.float32]:
        self.received_matrices = matrices.copy()
        return np.zeros((matrices.shape[0], 256, 256, 3), dtype=np.float32)


def test_prepare_data_second_pass_updates_batch_matrices_for_refeed_realign() -> None:
    """Second-pass ReAlign matrices must be installed before final prediction."""
    plugin = _Plugin()
    handler = _handler(plugin, total_feeds=3)

    default_crop_matrices: npt.NDArray[np.float32] = np.arange(
        2 * 3 * 3,
        dtype=np.float32,
    ).reshape(2, 3, 3)
    refed_crop_matrices: npt.NDArray[np.float32] = np.arange(
        100,
        100 + 6 * 3 * 3,
        dtype=np.float32,
    ).reshape(6, 3, 3)
    final_realign_matrices: npt.NDArray[np.float32] = np.arange(
        200,
        200 + 6 * 3 * 3,
        dtype=np.float32,
    ).reshape(6, 3, 3)

    re_align = _ReAlignDouble(default_crop_matrices, final_realign_matrices)
    handler._re_feed = T.cast(ReFeed, _ReFeedDouble(refed_crop_matrices))
    handler._re_align = T.cast(ReAlign, re_align)

    stale_first_pass_matrices: npt.NDArray[np.float32] = np.arange(
        300,
        300 + 2 * 3 * 3,
        dtype=np.float32,
    ).reshape(2, 3, 3)
    batch = T.cast(
        ExtractBatch,
        SimpleNamespace(
            bboxes=np.array([[10, 20, 30, 40], [100, 200, 300, 400]], dtype=np.int32),
            matrices=stale_first_pass_matrices,
            data=None,
        ),
    )

    handler._prepare_data(batch, iteration=2)

    assert batch.data is not None
    assert batch.data.shape == (6, 256, 256, 3)
    np.testing.assert_allclose(re_align.received_matrices, refed_crop_matrices)
    np.testing.assert_allclose(batch.matrices, final_realign_matrices)
    np.testing.assert_allclose(plugin.calls[-1][0], final_realign_matrices)
    np.testing.assert_allclose(
        plugin.calls[-1][1],
        np.array(
            [
                [10, 20, 30, 40],
                [10, 20, 30, 40],
                [10, 20, 30, 40],
                [100, 200, 300, 400],
                [100, 200, 300, 400],
                [100, 200, 300, 400],
            ],
            dtype=np.float32,
        ),
    )
