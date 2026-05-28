#!/usr/bin/env python3
"""Tests for :mod:`lib.infer.torch_preprocess`."""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from lib.infer.torch_preprocess import (
    _resize_with_opencv,
    _warp_with_opencv,
    torch_affine_warp,
    torch_batched_crop_align,
    torch_face_crop,
    torch_normalize,
    torch_resize,
)


def _sample_uint8_image(
    *, batch: int = 1, channels: int = 3, height: int = 12, width: int = 10
) -> torch.Tensor:
    """Create a deterministic BCHW uint8 image batch."""
    base = np.arange(batch * channels * height * width, dtype=np.uint16)
    image = (base.reshape(batch, channels, height, width) * 7) % 251
    return torch.from_numpy(image.astype(np.uint8))


def _sample_float_image(
    *, batch: int = 1, channels: int = 3, height: int = 12, width: int = 10
) -> torch.Tensor:
    """Create a deterministic BCHW float32 image batch."""
    image = _sample_uint8_image(
        batch=batch, channels=channels, height=height, width=width
    )
    return image.to(dtype=torch.float32) / 255.0


def _opencv_resize_reference(
    image: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str = "bilinear",
) -> np.ndarray:
    """Return an OpenCV resize reference in BCHW order."""
    result = _resize_with_opencv(image, size, mode=mode)
    return result.detach().cpu().numpy()


def _opencv_warp_reference(
    image: torch.Tensor,
    matrix: torch.Tensor,
    size: tuple[int, int],
) -> np.ndarray:
    """Return an OpenCV warp reference in BCHW order."""
    result = _warp_with_opencv(image, matrix, size)
    return result.detach().cpu().numpy()


def test_torch_resize_matches_opencv_with_tolerance() -> None:
    """Bilinear Torch resize should stay close to the OpenCV baseline."""
    image = _sample_uint8_image(height=11, width=9)
    result = torch_resize(image, (7, 13), mode="bilinear")
    expected = _opencv_resize_reference(image, (7, 13), mode="bilinear")
    assert result.shape == (1, 3, 7, 13)
    np.testing.assert_allclose(result.cpu().numpy(), expected, atol=4.0)


def test_torch_normalize_matches_manual_formula() -> None:
    """Channel normalization should be an exact float32 transform."""
    image = _sample_float_image()
    mean = (0.1, 0.2, 0.3)
    std = (0.5, 0.25, 0.125)
    result = torch_normalize(image, mean, std)
    expected = (image - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(
        1, 3, 1, 1
    )
    assert result.dtype == torch.float32
    assert torch.allclose(result, expected)


def test_torch_affine_warp_matches_opencv_with_tolerance() -> None:
    """Affine warp should align with OpenCV's warpAffine within interpolation tolerance."""
    image = _sample_uint8_image(height=16, width=16)
    matrix = torch.tensor([[0.8, 0.1, 1.5], [-0.05, 0.9, 2.0]], dtype=torch.float32)
    result = torch_affine_warp(image, matrix, (10, 12))
    expected = _opencv_warp_reference(image, matrix.unsqueeze(0), (10, 12))
    assert result.shape == (1, 3, 10, 12)
    np.testing.assert_allclose(result.cpu().numpy(), expected, atol=4.0)


def test_torch_face_crop_matches_opencv_with_tolerance() -> None:
    """Face crop should produce the same crop ordering and geometry as OpenCV warpAffine."""
    image = _sample_uint8_image(height=20, width=18)
    bbox = torch.tensor([3.0, 4.0, 13.0, 16.0], dtype=torch.float32)
    matrix = torch.tensor([[0.8, 0.0, -2.4], [0.0, 1.0, -4.0]], dtype=torch.float32)
    result = torch_face_crop(image, bbox, (12, 8))
    expected = _opencv_warp_reference(image, matrix.unsqueeze(0), (12, 8))
    assert result.shape == (1, 3, 12, 8)
    np.testing.assert_allclose(result.cpu().numpy(), expected, atol=2.0)


def test_torch_batched_crop_align_preserves_shape_and_order() -> None:
    """A single source frame can expand to multiple ordered face crops."""
    image = _sample_uint8_image(height=20, width=18)
    boxes = torch.tensor(
        [[1.0, 2.0, 9.0, 10.0], [6.0, 5.0, 14.0, 15.0]],
        dtype=torch.float32,
    )
    result = torch_batched_crop_align(image, boxes, (8, 8))
    expected_a = cv2.warpAffine(
        image[0].permute(1, 2, 0).numpy(),
        np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -2.0]], dtype=np.float32),
        (8, 8),
        flags=cv2.INTER_LINEAR,
    )
    expected_b = cv2.warpAffine(
        image[0].permute(1, 2, 0).numpy(),
        np.array([[1.0, 0.0, -6.0], [0.0, 0.8, -4.0]], dtype=np.float32),
        (8, 8),
        flags=cv2.INTER_LINEAR,
    )
    assert result.shape == (2, 3, 8, 8)
    np.testing.assert_allclose(
        result[0].permute(1, 2, 0).cpu().numpy(),
        expected_a,
        atol=2.0,
    )
    np.testing.assert_allclose(
        result[1].permute(1, 2, 0).cpu().numpy(),
        expected_b,
        atol=2.0,
    )


def test_resize_falls_back_to_opencv_when_device_path_is_disabled() -> None:
    """Resize should route through the OpenCV fallback when device support is disabled."""
    image = _sample_float_image()
    with (
        patch(
            "lib.infer.torch_preprocess._resize_supported_on_device", return_value=False
        ),
        patch(
            "lib.infer.torch_preprocess._resize_with_opencv", wraps=_resize_with_opencv
        ) as mock_resize,
    ):
        result = torch_resize(image, (5, 7))
    assert result.shape == (1, 3, 5, 7)
    mock_resize.assert_called_once()


def test_affine_warp_falls_back_to_opencv_when_device_path_is_disabled() -> None:
    """Affine warp should route through the OpenCV fallback when device support is disabled."""
    image = _sample_float_image(height=14, width=14)
    matrix = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]], dtype=torch.float32)
    with (
        patch(
            "lib.infer.torch_preprocess._warp_supported_on_device", return_value=False
        ),
        patch(
            "lib.infer.torch_preprocess._warp_with_opencv", wraps=_warp_with_opencv
        ) as mock_warp,
    ):
        result = torch_affine_warp(image, matrix, (8, 8))
    assert result.shape == (1, 3, 8, 8)
    mock_warp.assert_called_once()


@pytest.mark.parametrize(
    ("func", "args", "match"),
    [
        (torch_resize, (torch.zeros((3, 8, 8), dtype=torch.uint8), (4, 4)), "BCHW"),
        (
            torch_normalize,
            (
                torch.zeros((1, 3, 4, 4), dtype=torch.uint8),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
            ),
            "float32",
        ),
        (
            torch_face_crop,
            (
                torch.zeros((1, 3, 8, 8), dtype=torch.uint8),
                torch.tensor([1.0, 1.0, 1.0, 4.0]),
                (4, 4),
            ),
            "positive width and height",
        ),
        (
            torch_batched_crop_align,
            (
                torch.zeros((2, 3, 8, 8), dtype=torch.uint8),
                torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
                (4, 4),
            ),
            "does not match crop box batch size",
        ),
    ],
)
def test_invalid_inputs_raise_clear_errors(func, args, match: str) -> None:
    """Invalid tensor shapes and dtypes should fail with focused messages."""
    with pytest.raises(ValueError, match=match):
        func(*args)
