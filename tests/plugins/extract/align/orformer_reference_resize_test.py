#!/usr/bin/env python3
"""Reference resize parity test for ORFormer.

Current production path:
    normalise(256×256) → F.interpolate(..., size=(64,64), bilinear) → VQ-VAE

Upstream path:
    cv2.resize(256×256, 64×64, INTER_LINEAR) → normalise → VQ-VAE

Because normalisation is a linear per-channel operation the two paths should
produce nearly identical results, but OpenCV and PyTorch bilinear kernels are
not bit-identical.  This test measures the gap so we can decide whether to
change the production path.

Test strategy
-------------
The test runs entirely without model weights: it only compares the *input*
tensors going into the VQ-VAE encoder, not the model outputs.  This keeps the
test fast, deterministic, and weight-free.

If you later want to compare full landmark outputs you will need the official
weights; at that point see orformer_parity_test.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from lib.infer.align import Align  # noqa: F401  (import for side-effect)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_crop(seed: int = 0, h: int = 256, w: int = 256) -> np.ndarray:
    """Return an (H, W, 3) uint8 synthetic face crop."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)  # type: ignore[no-any-return]


def _faceswap_reference_input(crop_uint8: np.ndarray) -> np.ndarray:
    """Reproduce the current production reference-input path.

    normalise 256×256 → PyTorch bilinear downsample → (1, 3, 64, 64)
    """
    import torch
    from torch.nn import functional as F

    crop_f32 = crop_uint8.astype(np.float32) / 255.0
    norm = (crop_f32 - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)[np.newaxis])  # (1,3,256,256)
    ref = F.interpolate(tensor, size=(64, 64), mode="bilinear", align_corners=False)
    return ref.numpy()  # type: ignore[no-any-return]


def _upstream_reference_input(crop_uint8: np.ndarray) -> np.ndarray:
    """Reproduce the upstream reference-input path.

    cv2.resize to 64×64 → normalise → (1, 3, 64, 64)
    """
    import cv2

    resized = cv2.resize(crop_uint8, (64, 64), interpolation=cv2.INTER_LINEAR)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(norm.transpose(2, 0, 1)[np.newaxis])  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_reference_input_shape_and_dtype(seed: int) -> None:
    """Both paths produce (1, 3, 64, 64) float32 tensors."""
    crop = _make_synthetic_crop(seed)
    fs_ref = _faceswap_reference_input(crop)
    up_ref = _upstream_reference_input(crop)
    assert fs_ref.shape == (1, 3, 64, 64)
    assert up_ref.shape == (1, 3, 64, 64)
    assert fs_ref.dtype == np.float32
    assert up_ref.dtype == np.float32


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_reference_input_max_abs_diff(seed: int) -> None:
    """Max absolute difference between PyTorch and OpenCV reference paths.

    This is a measurement test, not a strict tolerance gate.  The threshold
    here (0.05) is intentionally loose — we are documenting the gap, not
    asserting bit-parity.  Tighten once a decision is made about whether to
    switch the production path to OpenCV.

    Recorded diff for 3 synthetic crops (seed 0/1/42):
      typically < 0.01 for natural-image-like content
    """
    crop = _make_synthetic_crop(seed)
    fs_ref = _faceswap_reference_input(crop)
    up_ref = _upstream_reference_input(crop)
    max_diff = float(np.abs(fs_ref - up_ref).max())
    assert max_diff < 0.05, (
        f"Max abs diff between PyTorch and OpenCV reference paths is {max_diff:.6f}. "
        "If this exceeds 0.05 the two paths diverge significantly and the "
        "production path should be reviewed."
    )


def test_reference_input_mean_abs_diff_reported(capsys) -> None:
    """Report mean abs diff for all seeds; does not assert a hard threshold."""
    diffs = []
    for seed in range(5):
        crop = _make_synthetic_crop(seed)
        fs_ref = _faceswap_reference_input(crop)
        up_ref = _upstream_reference_input(crop)
        diffs.append(float(np.abs(fs_ref - up_ref).mean()))
    mean_diff = np.mean(diffs)
    print(f"\n[resize_parity] mean abs diff across 5 seeds: {mean_diff:.6f}")
    # Not asserting a threshold — this is an observational report.


def test_forward_accepts_external_reference_input() -> None:
    """ORFormerFaceswapModel.forward() accepts an externally built reference_input.

    Uses a tiny stub model (no weights) to verify the signature and that the
    reference_input tensor is routed to the VQ-VAE correctly.
    """
    import unittest.mock as mock

    import torch

    from plugins.extract.align._orformer.model import ORFormerFaceswapModel

    # Build a model instance without loading weights via __new__ + manual init
    # of the two sub-networks.  We mock orformer and hgnet forward so we don't
    # need real weights.
    model = object.__new__(ORFormerFaceswapModel)
    model.orformer = mock.MagicMock()
    model.hgnet = mock.MagicMock()

    fake_heatmaps = torch.zeros(1, 13, 64, 64)
    fake_landmarks = torch.zeros(1, 68, 2)
    model.orformer.return_value = fake_heatmaps
    model.hgnet.return_value = fake_landmarks

    inputs = torch.zeros(1, 3, 256, 256)
    ref = torch.ones(1, 3, 64, 64) * 0.5

    _ = model.forward(inputs, reference_input=ref)

    # Verify the external reference_input was passed to the VQ-VAE encoder
    model.orformer.assert_called_once_with(ref)


def test_forward_without_reference_input_uses_interpolation() -> None:
    """When reference_input is None, F.interpolate is used (production path)."""
    import unittest.mock as mock

    import torch

    from plugins.extract.align._orformer.model import ORFormerFaceswapModel

    model = object.__new__(ORFormerFaceswapModel)
    model.orformer = mock.MagicMock()
    model.hgnet = mock.MagicMock()

    fake_heatmaps = torch.zeros(1, 15, 64, 64)
    fake_landmarks = torch.zeros(1, 98, 2)
    model.orformer.return_value = fake_heatmaps
    model.hgnet.return_value = fake_landmarks

    inputs = torch.rand(1, 3, 256, 256)
    _ = model.forward(inputs)

    # The VQ-VAE was called with the downsampled input, not the original
    called_ref = model.orformer.call_args[0][0]
    assert called_ref.shape == (1, 3, 64, 64)
