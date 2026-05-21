#!/usr/bin/env python3
"""Tests for align runner crop matrix handshakes."""

from types import SimpleNamespace

import numpy as np

from lib.infer.align import Align
from lib.infer.objects import ExtractBatchAligned


class _MatrixAwarePlugin:
    """Plugin double that records runner-provided crop geometry."""

    input_size = 256

    def __init__(self) -> None:
        self.matrices: np.ndarray | None = None
        self.detector_bboxes: np.ndarray | None = None

    def set_crop_matrices(
        self,
        matrices: np.ndarray,
        *,
        detector_bboxes: np.ndarray | None = None,
    ) -> None:
        """Record the current crop matrices."""
        self.matrices = matrices.copy()
        self.detector_bboxes = None if detector_bboxes is None else detector_bboxes.copy()


class _FakeReAlign:
    """Re-align double that exposes updated crop-to-frame matrices."""

    enabled = True
    iterations = 2

    def __init__(self) -> None:
        self.default_crop_matrices = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]],
            ],
            dtype="float32",
        )
        self._matrices = np.empty((0, 3, 3), dtype="float32")

    @property
    def matrices(self) -> np.ndarray:
        """The matrices produced for the second pass."""
        return self._matrices

    def get_images(self, matrices: np.ndarray, feeds: int) -> np.ndarray:
        """Store a visibly updated matrix batch and return placeholder crops."""
        assert feeds == 1
        self._matrices = matrices.copy()
        self._matrices[:, 0, 2] += 11.0
        return np.zeros((matrices.shape[0], 256, 256, 3), dtype="float32")


def test_prepare_data_passes_realign_second_pass_matrices_to_plugin() -> None:
    """Second pass plugins receive re-align matrices before prediction."""
    plugin = _MatrixAwarePlugin()
    re_align = _FakeReAlign()
    handler = object.__new__(Align)
    handler.plugin = plugin
    handler._re_align = re_align
    handler._re_feed = SimpleNamespace(total_feeds=1)
    batch = SimpleNamespace(
        bboxes=np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype="int32"),
        data=None,
    )

    Align._prepare_data(handler, batch, iteration=2)

    np.testing.assert_allclose(plugin.matrices, re_align.matrices)
    np.testing.assert_allclose(plugin.detector_bboxes, batch.bboxes.astype("float32"))
    assert batch.data.shape == (2, 256, 256, 3)


def test_post_process_persists_plugin_debug_metadata() -> None:
    """Align post-process stores plugin metadata for alignments serialization."""
    handler = object.__new__(Align)
    handler.plugin = SimpleNamespace(
        name="Ensemble",
        last_debug_metadata=[{"selected_candidate": "spiga", "bucket": "profile_left"}],
    )
    handler._overridden = {"post_process": False}
    handler._re_align = SimpleNamespace(enabled=False)
    handler._re_feed = SimpleNamespace(total_feeds=1, merge=lambda landmarks: landmarks)
    handler._filters = lambda batch: None
    handler._landmark_type = None
    batch = SimpleNamespace(
        data=np.zeros((1, 68, 2), dtype="float32"),
        matrices=np.eye(3, dtype="float32")[None],
        aligned=ExtractBatchAligned(),
    )

    Align.post_process(handler, batch)

    assert batch.aligned.metadata == [
        {"landmark_ensemble": {"selected_candidate": "spiga", "bucket": "profile_left"}}
    ]
