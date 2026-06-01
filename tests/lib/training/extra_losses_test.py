#!/usr/bin/env python3
"""Tests for the advanced, optional training-loss components in ``lib.training``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lib.training.data import BatchMeta
from lib.training.extra_losses import (
    BoundaryLoss,
    IdentityLoss,
    RegionWeightedPerceptualLoss,
    compute_boundary_band,
    occlusion_exclusion_weight,
)
from lib.training.loss import LossCollator

_SIZE = 32
_BATCH = 2


def _face_mask(size: int = _SIZE, batch: int = _BATCH, value: float = 1.0) -> torch.Tensor:
    """Build a centered square ``(N, 1, H, W)`` mask."""
    mask = torch.zeros(batch, 1, size, size)
    lo, hi = size // 4, size - size // 4
    mask[:, :, lo:hi, lo:hi] = value
    return mask


def _region_mask(top: int, bottom: int, left: int, right: int) -> torch.Tensor:
    """Build a rectangular ``(N, 1, H, W)`` mask."""
    mask = torch.zeros(_BATCH, 1, _SIZE, _SIZE)
    mask[:, :, top:bottom, left:right] = 1.0
    return mask


class _FakeRecognizer(nn.Module):
    """A tiny deterministic stand-in for a face-recognition model."""

    def __init__(self, embedding_dim: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, embedding_dim, kernel_size=3, padding=1)

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        """Return ``(N, D)`` embeddings."""
        return self.conv(faces).mean(dim=(2, 3))


# ---------------------------------------------------------------------------
# compute_boundary_band
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["inner", "outer", "both"])
def test_boundary_band_is_non_empty(mode: str) -> None:
    """A band derived from a non-trivial mask should be non-empty for every mode."""
    band = compute_boundary_band(_face_mask(), band_pixels=4, mode=mode)
    assert band.shape == (_BATCH, 1, _SIZE, _SIZE)
    assert float(band.sum()) > 0.0
    assert float(band.min()) >= 0.0
    assert float(band.max()) <= 1.0


def test_boundary_band_empty_mask_is_empty() -> None:
    """An empty mask should produce an empty band rather than an error."""
    band = compute_boundary_band(torch.zeros(_BATCH, 1, _SIZE, _SIZE), band_pixels=4, mode="both")
    assert float(band.sum()) == 0.0


def test_boundary_band_modes_differ() -> None:
    """Inner / outer bands should not be identical for a non-trivial mask."""
    mask = _face_mask()
    inner = compute_boundary_band(mask, band_pixels=3, mode="inner")
    outer = compute_boundary_band(mask, band_pixels=3, mode="outer")
    assert not torch.equal(inner, outer)


# ---------------------------------------------------------------------------
# occlusion_exclusion_weight
# ---------------------------------------------------------------------------
def test_occlusion_weight_full_strength_zeros_occluded() -> None:
    """A strength of 1.0 should fully exclude occluded pixels."""
    occ = _region_mask(8, 24, 8, 24)
    weight = occlusion_exclusion_weight(occ, strength=1.0)
    assert float(weight[occ > 0.5].max()) == 0.0
    assert float(weight[occ < 0.5].min()) == 1.0


def test_occlusion_weight_zero_strength_is_noop() -> None:
    """A strength of 0.0 should leave the weight at 1.0 everywhere."""
    weight = occlusion_exclusion_weight(_region_mask(8, 24, 8, 24), strength=0.0)
    assert torch.allclose(weight, torch.ones_like(weight))


# ---------------------------------------------------------------------------
# BoundaryLoss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("function", ["mae", "mse", "ssim"])
def test_boundary_loss_forward_backward(function: str) -> None:
    """The boundary loss should return a finite per-item scalar and backprop."""
    loss_fn = BoundaryLoss(function, band_pixels=4, mode="both", color_order="bgr")
    y_true = torch.rand(_BATCH, 3, _SIZE, _SIZE)
    y_pred = torch.rand(_BATCH, 3, _SIZE, _SIZE, requires_grad=True)
    loss = loss_fn(y_true, y_pred, _face_mask())
    assert loss.shape == (_BATCH,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert y_pred.grad is not None and torch.isfinite(y_pred.grad).all()


def test_boundary_loss_empty_mask_no_nan() -> None:
    """A spatial boundary loss over an empty band must not produce NaNs."""
    loss_fn = BoundaryLoss("mae", band_pixels=4, mode="both", color_order="bgr")
    empty = torch.zeros(_BATCH, 1, _SIZE, _SIZE)
    loss = loss_fn(torch.rand(_BATCH, 3, _SIZE, _SIZE), torch.rand(_BATCH, 3, _SIZE, _SIZE), empty)
    assert torch.isfinite(loss).all()
    assert float(loss.sum()) == 0.0


# ---------------------------------------------------------------------------
# RegionWeightedPerceptualLoss
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("function", ["ssim", "mae"])
def test_region_perceptual_forward_backward(function: str) -> None:
    """Region-weighted perceptual loss should support spatial and non-spatial bases."""
    loss_fn = RegionWeightedPerceptualLoss(function, "bgr", 2.0, 3.0, 2.0, 1.5)
    y_true = torch.rand(_BATCH, 3, _SIZE, _SIZE)
    y_pred = torch.rand(_BATCH, 3, _SIZE, _SIZE, requires_grad=True)
    loss = loss_fn(
        y_true, y_pred, _face_mask(), _region_mask(8, 12, 8, 20), _region_mask(20, 24, 8, 20)
    )
    assert loss.shape == (_BATCH,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert y_pred.grad is not None and torch.isfinite(y_pred.grad).all()


def test_region_perceptual_falls_back_without_masks() -> None:
    """Without a face mask the loss should still produce a finite value."""
    loss_fn = RegionWeightedPerceptualLoss("ssim", "bgr", 2.0, 3.0, 2.0, 1.5)
    y_true = torch.rand(_BATCH, 3, _SIZE, _SIZE)
    y_pred = torch.rand(_BATCH, 3, _SIZE, _SIZE)
    loss = loss_fn(y_true, y_pred, None, None, None)
    assert loss.shape == (_BATCH,)
    assert torch.isfinite(loss).all()


def test_region_perceptual_weight_changes_output() -> None:
    """Increasing a region weight should change the loss value."""
    y_true = torch.rand(_BATCH, 3, _SIZE, _SIZE)
    y_pred = torch.rand(_BATCH, 3, _SIZE, _SIZE)
    mask = _face_mask()
    eye = _region_mask(8, 12, 8, 20)
    base = RegionWeightedPerceptualLoss("ssim", "bgr", 1.0, 1.0, 1.0, 1.0)(
        y_true, y_pred, mask, eye, None
    )
    heavy = RegionWeightedPerceptualLoss("ssim", "bgr", 1.0, 8.0, 1.0, 1.0)(
        y_true, y_pred, mask, eye, None
    )
    assert not torch.allclose(base, heavy)


# ---------------------------------------------------------------------------
# IdentityLoss
# ---------------------------------------------------------------------------
def test_identity_loss_is_zero_for_identical_inputs() -> None:
    """Identical inputs should have a near-zero identity distance."""
    loss_fn = IdentityLoss(_FakeRecognizer(), input_size=16, color_order="bgr")
    image = torch.rand(_BATCH, 3, _SIZE, _SIZE)
    loss = loss_fn(image, image.clone(), _face_mask())
    assert loss.shape == (_BATCH,)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)


@pytest.mark.parametrize("crop", ["face", "mask_bbox"])
def test_identity_loss_forward_backward(crop: str) -> None:
    """Identity loss should backprop into the prediction only."""
    loss_fn = IdentityLoss(_FakeRecognizer(), input_size=16, color_order="bgr", crop=crop)
    y_true = torch.rand(_BATCH, 3, _SIZE, _SIZE)
    y_pred = torch.rand(_BATCH, 3, _SIZE, _SIZE, requires_grad=True)
    loss = loss_fn(y_true, y_pred, _face_mask())
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert y_pred.grad is not None and torch.isfinite(y_pred.grad).all()


def test_identity_loss_keeps_recognizer_frozen_and_unregistered() -> None:
    """The recognizer must be frozen and excluded from the module's parameters."""
    recognizer = _FakeRecognizer()
    loss_fn = IdentityLoss(recognizer, input_size=16, color_order="bgr")
    # No parameters are registered, so nothing is serialized into a checkpoint.
    assert list(loss_fn.parameters()) == []
    assert loss_fn.state_dict() == {}
    # The recognizer's weights are frozen.
    assert all(not p.requires_grad for p in recognizer.parameters())


# ---------------------------------------------------------------------------
# LossCollator integration
# ---------------------------------------------------------------------------
def _collator(**kwargs: object) -> LossCollator:
    """Build a LossCollator with sensible test defaults."""
    return LossCollator(
        ["ssim", "mae", "none", "none"],
        [1.0, 1.0, 0.0, 0.0],
        "bgr",
        True,
        3.0,
        2.0,
        _SIZE,
        **kwargs,  # type: ignore[arg-type]
    )


def _meta(with_occlusion: bool = False) -> BatchMeta:
    """Build a single-output BatchMeta with region masks."""
    kwargs: dict[str, list[torch.Tensor]] = {
        "mask_face": [_face_mask()],
        "mask_eye": [_region_mask(8, 12, 8, 20)],
        "mask_mouth": [_region_mask(20, 24, 8, 20)],
    }
    if with_occlusion:
        kwargs["mask_occlusion"] = [_region_mask(10, 22, 10, 22)]
    return BatchMeta(**kwargs)


def _targets() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build NHWC target / prediction tensors for a single output."""
    y_true = [torch.rand(_BATCH, _SIZE, _SIZE, 3)]
    y_pred = [torch.rand(_BATCH, _SIZE, _SIZE, 3, requires_grad=True)]
    return y_true, y_pred


def test_collator_disabled_matches_base_keys() -> None:
    """With every advanced option disabled, only the base loss keys are present."""
    collator = _collator()
    y_true, y_pred = _targets()
    out = collator(y_true, y_pred, _meta())
    assert set(out.unweighted[0]) == {"ssim", "mae"}


def test_collator_extra_loss_keys_present() -> None:
    """Enabled advanced losses appear in the per-output loss dicts and weighting."""
    collator = _collator(
        boundary_loss=BoundaryLoss("mae", 4, "both", "bgr"),
        boundary_weight=0.1,
        region_perceptual_loss=RegionWeightedPerceptualLoss("ssim", "bgr", 2.0, 2.0, 2.0, 2.0),
        region_perceptual_weight=0.1,
        identity_loss=IdentityLoss(_FakeRecognizer(), 16, "bgr"),
        identity_weight=0.1,
    )
    y_true, y_pred = _targets()
    out = collator(y_true, y_pred, _meta())
    assert {"boundary", "region_perceptual", "identity"} <= set(out.unweighted[0])
    # Weighted values are scaled copies of the unweighted values.
    assert torch.allclose(out.weighted[0]["boundary"], out.unweighted[0]["boundary"] * 0.1)
    out.total.backward()
    assert y_pred[0].grad is not None and torch.isfinite(y_pred[0].grad).all()


def test_collator_identity_start_iteration_gate() -> None:
    """The identity loss should only apply once the start iteration is reached."""
    collator = _collator(
        identity_loss=IdentityLoss(_FakeRecognizer(), 16, "bgr"),
        identity_weight=0.1,
        identity_start_iteration=5,
    )
    y_true, y_pred = _targets()
    collator.set_iteration(0)
    assert "identity" not in collator(y_true, y_pred, _meta()).unweighted[0]
    collator.set_iteration(5)
    assert "identity" in collator(y_true, y_pred, _meta()).unweighted[0]


def test_collator_occlusion_changes_loss() -> None:
    """Enabling occlusion exclusion should change the reconstruction loss value."""
    torch.manual_seed(0)
    y_true, y_pred = _targets()
    meta = _meta(with_occlusion=True)
    baseline = _collator()(y_true, y_pred, meta).unweighted[0]["mae"]
    excluded = _collator(occlusion_strength=1.0)(y_true, y_pred, meta).unweighted[0]["mae"]
    assert not torch.allclose(baseline, excluded)


def test_collator_occlusion_requires_mask() -> None:
    """Occlusion exclusion is a no-op when no occlusion mask is present in the batch."""
    y_true, y_pred = _targets()
    meta = _meta(with_occlusion=False)
    baseline = _collator()(y_true, y_pred, meta).unweighted[0]["mae"]
    no_mask = _collator(occlusion_strength=1.0)(y_true, y_pred, meta).unweighted[0]["mae"]
    assert torch.allclose(baseline, no_mask)
