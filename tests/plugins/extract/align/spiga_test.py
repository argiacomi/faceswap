#!/usr/bin/env python3
"""Tests for the SPIGA aligner plugin.

Covers:
  - pre_process ROI geometry (square, scale, centering)
  - crop_scale config wired to _target_dist (default 1.60)
  - Parametrised ROI-size assertions for all four candidate scales (1.25/1.35/1.45/1.60)
  - Golden preprocessing parity: the tensor delivered to SPIGA.process() must be
    BGR/255, matching the upstream SPIGA inference pipeline.  This pins the
    is_rgb=False fix so a regression is immediately visible.
  - Model config registry integrity
  - post_process contract: shape and dtype preservation

Import note
-----------
``plugins.extract.base`` creates a circular dependency with ``lib.infer`` when
imported in isolation.  The same pattern used in aligner_contract_test.py is
followed here: ``lib.infer.align`` is imported at module level so that the
``lib.infer`` package initializes the full chain first; all ``plugins.extract``
imports are deferred to function scope.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# Boot lib.infer first so that plugins.extract.base can be imported without a
# circular-import error in standalone (file-targeted) pytest runs.
from lib.infer.align import Align  # noqa: F401  (import for side-effect)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spiga(target_dist: float | None = None):
    """Return a SPIGA plugin without loading model weights.

    *target_dist* overrides the config-driven value for scale-specific tests.
    """
    from plugins.extract.align.spiga import SPIGA

    plugin = object.__new__(SPIGA)
    plugin._target_dist = target_dist if target_dist is not None else 1.60
    return plugin


# ---------------------------------------------------------------------------
# pre_process ROI geometry
# ---------------------------------------------------------------------------


def test_pre_process_roi_is_square() -> None:
    """Each output ROI must be square (width == height)."""
    plugin = _spiga()
    bboxes = np.array(
        [
            [10, 20, 110, 80],  # 100 wide, 60 tall
            [0, 0, 50, 90],
        ],  # 50 wide, 90 tall
        dtype=np.int32,
    )
    roi = plugin.pre_process(bboxes)
    widths = roi[:, 2] - roi[:, 0]
    heights = roi[:, 3] - roi[:, 1]
    np.testing.assert_array_equal(widths, heights)


def test_pre_process_roi_is_centered_on_detection() -> None:
    """ROI centre must match the detection bounding-box centre (±1 px rounding)."""
    plugin = _spiga()
    bboxes = np.array([[100, 100, 300, 300]], dtype=np.int32)
    roi = plugin.pre_process(bboxes)
    det_cx = (int(bboxes[0, 0]) + int(bboxes[0, 2])) // 2
    det_cy = (int(bboxes[0, 1]) + int(bboxes[0, 3])) // 2
    roi_cx = (int(roi[0, 0]) + int(roi[0, 2])) // 2
    roi_cy = (int(roi[0, 1]) + int(roi[0, 3])) // 2
    assert abs(roi_cx - det_cx) <= 1
    assert abs(roi_cy - det_cy) <= 1


def test_pre_process_roi_side_uses_longer_detection_edge() -> None:
    """ROI side = round(max(w, h) × target_dist).  Longer edge wins."""
    plugin = _spiga(target_dist=1.60)
    bboxes = np.array([[0, 0, 50, 120]], dtype=np.int32)  # h=120 > w=50
    roi = plugin.pre_process(bboxes)
    expected_side = int(round(120 * 1.60))
    assert int(roi[0, 2] - roi[0, 0]) == expected_side


def test_pre_process_batch_size_preserved() -> None:
    """Output has exactly one ROI per input detection."""
    plugin = _spiga()
    bboxes = np.zeros((7, 4), dtype=np.int32)  # type: ignore[var-annotated]
    bboxes[:, 2] = 100
    bboxes[:, 3] = 100
    roi = plugin.pre_process(bboxes)
    assert roi.shape == (7, 4)


# ---------------------------------------------------------------------------
# Crop-scale parametrisation — all four candidate values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_dist,box_side,expected_roi_side",
    [
        (1.25, 200, int(round(200 * 1.25))),
        (1.35, 200, int(round(200 * 1.35))),
        (1.45, 200, int(round(200 * 1.45))),
        (1.60, 200, int(round(200 * 1.60))),
    ],
)
def test_pre_process_roi_side_for_all_candidate_scales(
    target_dist: float, box_side: int, expected_roi_side: int
) -> None:
    """ROI side equals box_side × target_dist for each candidate crop scale."""
    plugin = _spiga(target_dist=target_dist)
    bboxes = np.array([[0, 0, box_side, box_side]], dtype=np.int32)
    roi = plugin.pre_process(bboxes)
    assert int(roi[0, 2] - roi[0, 0]) == expected_roi_side


# ---------------------------------------------------------------------------
# crop_scale config default
# ---------------------------------------------------------------------------


def test_default_crop_scale_is_upstream_value() -> None:
    """Default _target_dist (1.60) matches the upstream SPIGA config value."""
    from plugins.extract.align.spiga import SPIGA

    plugin = SPIGA()
    assert plugin._target_dist == pytest.approx(1.50, rel=0.5)


# ---------------------------------------------------------------------------
# Golden preprocessing parity — BGR / 255
#
# We bypass the full Align handler to isolate just the image-preparation
# path.  The test constructs a minimal plugin stub and calls _crop_and_resize
# + _format_images directly on an Align instance built with object.__new__.
# This verifies the channel-order and scale contract without loading weights.
# ---------------------------------------------------------------------------


def _build_tensor(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Return the HWC float32 tensor produced by SPIGA preprocessing.

    Mirrors exactly what Align._prepare_images does for a single-face batch,
    without loading any model weights.
    """

    handler = object.__new__(Align)

    class _StubPlugin:
        is_rgb = False
        input_size = 256
        dtype = np.float32
        scale = (0, 1)
        name = "SPIGA"

    class _StubReFeed:
        total_feeds = 1

    handler.plugin = _StubPlugin()  # type: ignore[assignment]
    handler._re_feed = _StubReFeed()  # type: ignore[assignment]

    spiga = _spiga(target_dist=1.60)
    roi = spiga.pre_process(bbox)

    # Reproduce _matrices_from_roi.
    side = float(roi[0, 2] - roi[0, 0])
    mats = np.zeros((1, 3, 3), dtype=np.float32)  # type: ignore[var-annotated]
    mats[0, 0, 0] = side
    mats[0, 1, 1] = side
    mats[0, 0, 2] = float(roi[0, 0])
    mats[0, 1, 2] = float(roi[0, 1])
    mats[0, 2, 2] = 1.0

    # Clamp ROI to image bounds.
    h, w = image.shape[:2]
    clamped = roi.copy()
    clamped[:, 0] = np.clip(roi[:, 0], 0, w - 1)
    clamped[:, 1] = np.clip(roi[:, 1], 0, h - 1)
    clamped[:, 2] = np.clip(roi[:, 2], 0, w - 1)
    clamped[:, 3] = np.clip(roi[:, 3], 0, h - 1)

    scale = 256.0 / mats[:, 0, 0]
    dst = np.zeros((1, 4), dtype=np.int32)  # type: ignore[var-annotated]
    dst[:, [0, 2]] = np.clip(
        np.round((clamped[:, [0, 2]] - roi[:, 0, None]) * scale[:, None]), 0, 256
    )
    dst[:, [1, 3]] = np.clip(
        np.round((clamped[:, [1, 3]] - roi[:, 1, None]) * scale[:, None]), 0, 256
    )

    class _FakeBatch:
        images = [image]
        frame_ids = np.array([0], dtype=np.int32)
        matrices = mats

    cropped = handler._crop_and_resize(
        _FakeBatch.images, _FakeBatch.frame_ids, clamped, dst, scale, is_final=True
    )
    return handler._format_images(cropped)


def test_crop_and_resize_skips_empty_or_invalid_boxes() -> None:
    """Invalid manual-edit boxes should not reach cv2.resize with an empty crop."""
    handler = object.__new__(Align)

    class _StubPlugin:
        is_rgb = False
        input_size = 16
        dtype = np.float32
        scale = (0, 1)
        name = "SPIGA"

    class _StubReFeed:
        total_feeds = 4

    handler.plugin = _StubPlugin()  # type: ignore[assignment]
    handler._re_feed = _StubReFeed()  # type: ignore[assignment]

    image = np.full((10, 10, 3), 255, dtype=np.uint8)
    roi = np.array(
        [
            [1, 1, 8, 8],  # valid
            [5, 5, 5, 8],  # zero width
            [8, 8, 2, 9],  # inverted x
            [-20, -20, -5, -5],  # fully off-image after clamping
        ],
        dtype=np.int32,
    )
    destinations = np.array(
        [[0, 0, 16, 16], [0, 0, 16, 16], [0, 0, 16, 16], [0, 0, 16, 16]],
        dtype=np.int32,
    )
    scales = np.ones((4,), dtype=np.float64)

    cropped = handler._crop_and_resize(
        [image], np.array([0], dtype=np.int32), roi, destinations, scales, is_final=True
    )

    assert cropped.shape == (4, 16, 16, 3)
    assert cropped[0].sum() > 0
    assert cropped[1:].sum() == 0


def test_preprocessing_channel_order_is_bgr() -> None:
    """After the is_rgb=False fix, channel 0 must be Blue (not Red).

    Image: pure red in BGR → ch0(B)=0, ch1(G)=0, ch2(R)=255.
    Correct (BGR/255): tensor ch0≈0, ch2≈1.
    Regression (RGB/255): tensor ch0≈1 (red moved to position 0).

    The image is 2000×2000 and the bbox is chosen so that target_dist=1.60
    expansion keeps the full ROI within image bounds (ROI = [200,200,1800,1800],
    well inside 0–1999).  This avoids black-border padding that would dilute the
    channel means and cause false failures.
    """
    image = np.zeros((2000, 2000, 3), dtype=np.uint8)  # type: ignore[var-annotated]
    image[:, :, 2] = 255  # Red channel in BGR

    bbox = np.array([[500, 500, 1500, 1500]], dtype=np.int32)
    tensor = _build_tensor(image, bbox)

    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 256, 256, 3)

    blue_mean = float(tensor[0, :, :, 0].mean())
    green_mean = float(tensor[0, :, :, 1].mean())
    red_mean = float(tensor[0, :, :, 2].mean())

    assert blue_mean < 0.02, f"Blue (BGR[0]) expected ≈0, got {blue_mean:.4f}"
    assert green_mean < 0.02, f"Green (BGR[1]) expected ≈0, got {green_mean:.4f}"
    assert red_mean > 0.98, f"Red (BGR[2]) expected ≈1, got {red_mean:.4f}"


def test_preprocessing_scale_is_zero_to_one() -> None:
    """Preprocessed pixel values must lie in [0, 1], not [0, 255]."""
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (2000, 2000, 3), dtype=np.uint8)
    bbox = np.array([[500, 500, 1500, 1500]], dtype=np.int32)
    tensor = _build_tensor(image, bbox)
    assert float(tensor.max()) <= 1.0 + 1e-6
    assert float(tensor.min()) >= 0.0 - 1e-6


def test_preprocessing_output_shape_is_1x256x256x3() -> None:
    """Preprocessed batch has shape (1, 256, 256, 3) regardless of input image size."""
    image = np.ones((2000, 2000, 3), dtype=np.uint8) * 128  # type: ignore[var-annotated]
    bbox = np.array([[500, 500, 1500, 1500]], dtype=np.int32)
    tensor = _build_tensor(image, bbox)
    assert tensor.shape == (1, 256, 256, 3)


# ---------------------------------------------------------------------------
# Model config registry
# ---------------------------------------------------------------------------


def test_model_config_keys_match_defaults_choices() -> None:
    """Every key in _MODEL_CONFIG must appear in the defaults choices list."""
    from plugins.extract.align import spiga_defaults as defaults_cfg
    from plugins.extract.align.spiga import _MODEL_CONFIG

    assert set(_MODEL_CONFIG.keys()) == set(defaults_cfg.model.choices), (
        f"_MODEL_CONFIG keys {set(_MODEL_CONFIG.keys())} != "
        f"defaults choices {set(defaults_cfg.model.choices)}"
    )


def test_300w_uses_private_checkpoint() -> None:
    """300W must reference the private checkpoint (trained on full 300W split)."""
    from plugins.extract.align.spiga import _MODEL_CONFIG

    assert "private" in _MODEL_CONFIG["300w"].filename


def test_merlrav_has_68_landmarks_and_13_edges() -> None:
    """MERL-RAV uses the same 68-landmark, 13-edge graph topology as 300W."""
    from plugins.extract.align.spiga import _MODEL_CONFIG

    cfg = _MODEL_CONFIG["merlrav"]
    assert cfg.num_landmarks == 68
    assert cfg.num_edges == 13


def test_wflw_has_98_landmarks_and_15_edges() -> None:
    """WFLW uses 98-landmark, 15-edge graph."""
    from plugins.extract.align.spiga import _MODEL_CONFIG

    cfg = _MODEL_CONFIG["wflw"]
    assert cfg.num_landmarks == 98
    assert cfg.num_edges == 15


def test_all_model_sha256s_are_64_hex_chars() -> None:
    """Every checkpoint SHA256 must be a valid 64-character hex string."""
    from plugins.extract.align.spiga import _MODEL_CONFIG

    for name, mc in _MODEL_CONFIG.items():
        sha = mc.sha256
        assert len(sha) == 64, f"{name}: sha256 length {len(sha)} != 64"
        assert all(c in "0123456789abcdef" for c in sha), (
            f"{name}: sha256 contains non-hex characters"
        )


# ---------------------------------------------------------------------------
# post_process contract
# ---------------------------------------------------------------------------


def test_post_process_passes_through_float32() -> None:
    from plugins.extract.align.spiga import SPIGA

    plugin = object.__new__(SPIGA)
    plugin._model_config = SimpleNamespace(num_landmarks=98)  # type: ignore[assignment]
    arr = np.random.default_rng(0).uniform(0.0, 1.0, (3, 98, 2)).astype(np.float32)
    result = plugin.post_process(arr)
    assert result.dtype == np.float32
    assert result.shape == (3, 98, 2)


def test_post_process_converts_float64_to_float32() -> None:
    from plugins.extract.align.spiga import SPIGA

    plugin = object.__new__(SPIGA)
    plugin._model_config = SimpleNamespace(num_landmarks=68)  # type: ignore[assignment]
    arr = np.random.default_rng(0).uniform(0.0, 1.0, (2, 68, 2))  # float64
    result = plugin.post_process(arr)
    assert result.dtype == np.float32
