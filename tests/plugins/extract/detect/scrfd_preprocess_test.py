#!/usr/bin/env python3
"""Tests for SCRFD detector preprocessing parity across CPU/torch/legacy paths."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from plugins.extract.detect.scrfd import SCRFD


def _build_detector(mode: str = "torch", device: str = "cpu") -> SCRFD:
    """Build a minimal SCRFD instance without invoking the heavyweight constructor."""
    detector = object.__new__(SCRFD)
    detector.input_size = 640
    detector.name = "SCRFD"
    detector._postprocess_mode = mode
    # SCRFD exposes ``device`` as a read-only property backed by ``_torch.device``.
    detector._torch = SimpleNamespace(device=torch.device(device))
    detector._last_profile_events = []
    return detector


class _RecordingModel(nn.Module):
    """Capture the tensor passed to ``forward`` and return well-shaped dummy outputs."""

    def __init__(self, seen: list[torch.Tensor]) -> None:
        super().__init__()
        self._seen = seen

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        self._seen.append(inputs.detach().cpu().clone())
        batch_size = inputs.shape[0]
        score = torch.ones((batch_size, 4, 1), dtype=torch.float32, device=inputs.device)
        boxes = torch.ones((batch_size, 4, 4), dtype=torch.float32, device=inputs.device)
        return [score, score.clone(), score.clone(), boxes, boxes.clone(), boxes.clone()]


def test_pre_process_cpu_matches_legacy_rgb_normalization() -> None:
    """The new ``cv2.dnn.blobFromImages`` CPU path must match the legacy normalization."""
    detector = _build_detector(mode="numpy", device="cpu")
    batch = np.array(
        [[[[0, 10, 20], [30, 40, 50]], [[60, 70, 80], [90, 100, 110]]]], dtype=np.uint8
    )

    result = detector.pre_process(batch)
    expected = ((batch[..., ::-1].astype(np.float32) - 127.5) / 128.0).transpose(0, 3, 1, 2)

    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def test_process_forced_torch_accepts_uint8_bgr_batches() -> None:
    """Forced torch mode must normalize/reorder a compact uint8 BGR batch on-device."""
    detector = _build_detector(mode="torch")
    seen: list[torch.Tensor] = []
    detector.model = _RecordingModel(seen)
    batch = np.array(
        [[[[0, 10, 20], [30, 40, 50]], [[60, 70, 80], [90, 100, 110]]]], dtype=np.uint8
    )

    detector.process(batch)

    assert len(seen) == 1
    expected = ((batch[..., ::-1].astype(np.float32) - 127.5) / 128.0).transpose(0, 3, 1, 2)
    np.testing.assert_allclose(seen[0].numpy(), expected, atol=1e-6, rtol=1e-6)


def test_process_legacy_numpy_path_passes_through_pre_processed_blob() -> None:
    """When ``from_torch`` receives an already-NCHW float blob, it must be forwarded as-is."""
    detector = _build_detector(mode="numpy")
    sentinel = np.empty((1,), dtype=object)
    detector.from_torch = lambda batch: sentinel  # type: ignore[method-assign]

    batch = np.zeros((1, 3, 16, 16), dtype=np.float32)

    assert detector.process(batch) is sentinel


def test_process_torch_path_accepts_legacy_normalized_nchw_input() -> None:
    """Torch path must keep accepting already-normalized float32 NCHW (legacy callers)."""
    detector = _build_detector(mode="torch")
    seen: list[torch.Tensor] = []
    detector.model = _RecordingModel(seen)
    batch = np.zeros((1, 3, 16, 16), dtype=np.float32)
    batch[0, 0, 0, 0] = 0.25  # marker to detect mutation

    detector.process(batch)

    assert len(seen) == 1
    # Same dtype, shape preserved, value unchanged (no normalization re-applied).
    assert seen[0].dtype == torch.float32
    assert seen[0].shape == batch.shape
    assert seen[0][0, 0, 0, 0].item() == 0.25
