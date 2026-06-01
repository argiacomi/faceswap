#!/usr/bin/env python3
"""Tests for the advanced-loss configuration items and the mask ordering helper."""

from __future__ import annotations

import pytest

from lib.training.data.data_set import configured_training_masks
from plugins.train import train_config as cfg

_EXPECTED_DEFAULTS = {
    "identity_loss": "none",
    "identity_loss_weight": 0.1,
    "identity_loss_crop": "face",
    "identity_loss_start_iteration": 0,
    "region_perceptual_loss": "none",
    "region_perceptual_loss_weight": 0.1,
    "region_perceptual_face_weight": 1.0,
    "region_perceptual_eye_weight": 1.0,
    "region_perceptual_mouth_weight": 1.0,
    "region_perceptual_skin_weight": 1.0,
    "boundary_loss": "none",
    "boundary_loss_function": "mae",
    "boundary_loss_weight": 0.1,
    "boundary_band_pixels": 8,
    "boundary_band_mode": "both",
    "occlusion_exclusion": "none",
    "occlusion_mask_type": "none",
    "occlusion_exclusion_strength": 1.0,
}


@pytest.mark.parametrize(("name", "expected"), list(_EXPECTED_DEFAULTS.items()))
def test_advanced_loss_config_defaults(name: str, expected: object) -> None:
    """Every advanced-loss option must default to a disabled / neutral value."""
    assert getattr(cfg.Loss, name)() == expected


def test_configured_masks_default_order() -> None:
    """The default configuration stacks the face, eye and mouth masks in order."""
    assert configured_training_masks() == [
        ("extended", "mask_face"),
        ("eye", "mask_eye"),
        ("mouth", "mask_mouth"),
    ]


def test_configured_masks_includes_occlusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabling occlusion exclusion appends an occlusion mask after the existing masks."""
    monkeypatch.setattr(cfg.Loss, "occlusion_exclusion", lambda: "mask")
    monkeypatch.setattr(cfg.Loss, "occlusion_mask_type", lambda: "xseg")
    masks = configured_training_masks()
    assert masks[-1] == ("xseg", "mask_occlusion")
    assert [field for _, field in masks] == [
        "mask_face",
        "mask_eye",
        "mask_mouth",
        "mask_occlusion",
    ]


def test_configured_masks_occlusion_disabled_when_type_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An occlusion mode with no mask type selected must not add a mask channel."""
    monkeypatch.setattr(cfg.Loss, "occlusion_exclusion", lambda: "mask")
    monkeypatch.setattr(cfg.Loss, "occlusion_mask_type", lambda: "none")
    assert all(field != "mask_occlusion" for _, field in configured_training_masks())


def test_boundary_loss_requests_face_mask_without_legacy_mask_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary loss must request a face mask even when legacy mask losses are disabled."""
    monkeypatch.setattr(cfg.Loss, "learn_mask", lambda: False)
    monkeypatch.setattr(cfg.Loss, "penalized_mask_loss", lambda: False)
    monkeypatch.setattr(cfg.Loss, "boundary_loss", lambda: "mask_band")
    fields = [field for _, field in configured_training_masks()]
    assert "mask_face" in fields


def test_identity_mask_bbox_requests_face_mask_without_legacy_mask_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity loss with mask_bbox crop must request a face mask on its own."""
    monkeypatch.setattr(cfg.Loss, "learn_mask", lambda: False)
    monkeypatch.setattr(cfg.Loss, "penalized_mask_loss", lambda: False)
    monkeypatch.setattr(cfg.Loss, "identity_loss", lambda: "arcface")
    monkeypatch.setattr(cfg.Loss, "identity_loss_crop", lambda: "mask_bbox")
    assert "mask_face" in [field for _, field in configured_training_masks()]


def test_region_perceptual_requests_region_masks_without_legacy_multipliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Region-weighted perceptual loss must request its own eye/mouth/face masks."""
    monkeypatch.setattr(cfg.Loss, "learn_mask", lambda: False)
    monkeypatch.setattr(cfg.Loss, "penalized_mask_loss", lambda: False)
    monkeypatch.setattr(cfg.Loss, "region_perceptual_loss", lambda: "ssim")
    monkeypatch.setattr(cfg.Loss, "region_perceptual_eye_weight", lambda: 3.0)
    monkeypatch.setattr(cfg.Loss, "region_perceptual_mouth_weight", lambda: 2.0)
    fields = [field for _, field in configured_training_masks()]
    assert {"mask_face", "mask_eye", "mask_mouth"} <= set(fields)


def test_region_perceptual_neutral_weights_request_no_region_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With neutral region weights, region perceptual loss needs no eye/mouth masks."""
    monkeypatch.setattr(cfg.Loss, "learn_mask", lambda: False)
    monkeypatch.setattr(cfg.Loss, "penalized_mask_loss", lambda: False)
    monkeypatch.setattr(cfg.Loss, "region_perceptual_loss", lambda: "ssim")
    fields = [field for _, field in configured_training_masks()]
    # The face mask is still requested (for skin / face weighting) but eye/mouth are not.
    assert "mask_eye" not in fields
    assert "mask_mouth" not in fields
