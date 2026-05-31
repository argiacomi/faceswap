#!/usr/bin/env python3
"""Shared Torch preprocessing helpers for extract inference paths."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from threading import Lock

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


_SUPPORTED_IMAGE_DTYPES = (torch.uint8, torch.float32)
_INTERPOLATION_MODES = {
    "area": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}
_MPS_FALLBACK_LOCK = Lock()
_MPS_FALLBACK_COUNTS: dict[str, int] = {}


def _validate_output_size(size: tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize an ``(height, width)`` output size."""
    if len(size) != 2:
        raise ValueError(f"Expected output_size as (height, width), got {size!r}")
    height, width = size
    if height <= 0 or width <= 0:
        raise ValueError(f"output_size values must be positive, got {size!r}")
    return int(height), int(width)


def _validate_image(image: torch.Tensor, func_name: str) -> torch.Tensor:
    """Validate the canonical BCHW image contract."""
    if image.ndim != 4:
        raise ValueError(f"{func_name} expects a BCHW tensor, got shape {tuple(image.shape)}")
    if image.dtype not in _SUPPORTED_IMAGE_DTYPES:
        raise ValueError(f"{func_name} expects uint8 or float32 input, got {image.dtype}")
    return image


def _validate_boxes(boxes: torch.Tensor, *, allow_many: bool) -> torch.Tensor:
    """Validate crop boxes in ``[x1, y1, x2, y2]`` frame coordinates."""
    if boxes.ndim == 1:
        if boxes.shape[0] != 4:
            raise ValueError(f"Expected crop box with 4 values, got shape {tuple(boxes.shape)}")
        boxes = boxes.unsqueeze(0)
    elif boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected crop boxes with shape (N, 4), got {tuple(boxes.shape)}")

    if not allow_many and boxes.shape[0] != 1:
        raise ValueError("torch_face_crop expects exactly one crop box")

    boxes = boxes.to(dtype=torch.float32)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    if torch.any(widths <= 0) or torch.any(heights <= 0):
        raise ValueError("Crop boxes must have positive width and height")
    return boxes


def _restore_dtype(image: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Restore an intermediate float tensor to the caller's requested dtype."""
    if dtype == torch.float32:
        return image.to(dtype=torch.float32)
    if dtype == torch.uint8:
        return image.round().clamp_(0.0, 255.0).to(dtype=torch.uint8)
    raise ValueError(f"Unsupported dtype restoration requested: {dtype}")


def _resize_supported_on_device(device_type: str, mode: str) -> bool:
    """Return whether this resize mode is allowed to remain on-device."""
    return not (device_type == "mps" and mode == "bicubic")


def _warp_supported_on_device(device_type: str) -> bool:
    """Return whether affine grid sampling should stay on-device."""
    return device_type != "mps"


def get_mps_fallback_ops() -> tuple[str, ...]:
    """Return the sorted names of recorded MPS fallback operations."""
    with _MPS_FALLBACK_LOCK:
        return tuple(sorted(_MPS_FALLBACK_COUNTS))


def reset_mps_fallback_ops() -> None:
    """Clear any recorded MPS fallback operations."""
    with _MPS_FALLBACK_LOCK:
        _MPS_FALLBACK_COUNTS.clear()


def _record_mps_fallback(operation: str) -> None:
    """Record that an MPS preprocessing operation fell back to the CPU/OpenCV path."""
    with _MPS_FALLBACK_LOCK:
        _MPS_FALLBACK_COUNTS[operation] = _MPS_FALLBACK_COUNTS.get(operation, 0) + 1


def _opencv_interpolation(mode: str) -> int:
    """Map a public interpolation mode to the OpenCV enum."""
    if mode not in _INTERPOLATION_MODES:
        supported = ", ".join(sorted(_INTERPOLATION_MODES))
        raise ValueError(f"Unsupported interpolation mode '{mode}'. Expected one of {supported}")
    return _INTERPOLATION_MODES[mode]


def _batch_to_numpy(images: torch.Tensor) -> np.ndarray:
    """Convert BCHW tensors to BHWC NumPy arrays on CPU."""
    return images.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()


def _batch_from_numpy(images: np.ndarray, *, device: torch.device) -> torch.Tensor:
    """Convert BHWC NumPy arrays back to BCHW tensors on the requested device."""
    if images.ndim != 4:
        raise ValueError(f"Expected BHWC NumPy array, got shape {images.shape}")
    return torch.from_numpy(images).permute(0, 3, 1, 2).to(device=device)


def _resize_with_opencv(
    image: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    """Resize a BCHW tensor through OpenCV for compatibility fallbacks."""
    target_height, target_width = _validate_output_size(size)
    interpolation = _opencv_interpolation(mode)
    images = _batch_to_numpy(image)
    resized = np.empty(
        (images.shape[0], target_height, target_width, images.shape[3]),
        dtype=images.dtype,
    )
    for idx, sample in enumerate(images):
        out = cv2.resize(sample, (target_width, target_height), interpolation=interpolation)
        if out.ndim == 2:
            out = out[..., None]
        resized[idx] = out
    return _batch_from_numpy(resized, device=image.device)


def _warp_with_opencv(
    image: torch.Tensor,
    matrices: torch.Tensor,
    output_size: tuple[int, int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Warp a BCHW tensor through OpenCV for compatibility fallbacks."""
    target_height, target_width = _validate_output_size(output_size)
    interpolation = _opencv_interpolation(mode)
    images = _batch_to_numpy(image)
    mats = matrices.detach().cpu().numpy()
    warped = np.empty(
        (images.shape[0], target_height, target_width, images.shape[3]),
        dtype=images.dtype,
    )
    for idx, (sample, mat) in enumerate(zip(images, mats, strict=False)):
        out = cv2.warpAffine(sample, mat, (target_width, target_height), flags=interpolation)
        if out.ndim == 2:
            out = out[..., None]
        warped[idx] = out
    return _batch_from_numpy(warped, device=image.device)


def _normalize_grid_coords(
    coord: torch.Tensor,
    limit: int,
    *,
    align_corners: bool,
) -> torch.Tensor:
    """Convert pixel-space coordinates to ``grid_sample`` normalized coordinates."""
    if align_corners:
        if limit == 1:
            return torch.zeros_like(coord)
        return (2.0 * coord / float(limit - 1)) - 1.0
    return ((coord + 0.5) * 2.0 / float(limit)) - 1.0


@lru_cache(maxsize=64)
def _cached_normalization_stats(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    device_str: str,
    device_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build + cache ``(mean, std)`` tensors for ``torch_normalize``.

    Issue #194 P2 high: the old shape rebuilt both tensors on EVERY
    ``torch_normalize`` call. With this cache an identical ``(mean,
    std, device)`` tuple reuses the same tensors instead — the size of
    the cache is bounded by the number of distinct model normalisation
    triples in the pipeline.

    ``device_str`` + ``device_index`` are the cache key, not the
    ``torch.device`` object itself, because devices are not hashable
    in a way that survives equal-but-distinct instances.
    """
    device = (
        torch.device(device_str)
        if device_index is None
        else torch.device(f"{device_str}:{device_index}")
    )
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    return mean_tensor, std_tensor


@lru_cache(maxsize=32)
def _cached_affine_scaffold(
    device_str: str,
    device_index: int | None,
    output_height: int,
    output_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build + cache the ``(homogeneous_eye, coords)`` scaffolding used
    by :func:`_affine_grid_from_matrices`.

    Issue #194 P2 high: ``torch.eye(3)`` and the ``torch.meshgrid``
    grids depended only on the output resolution + device, but were
    rebuilt per call. Caching the device-resident scaffolds means
    repeated same-size warps reuse the same tensors.

    Returns ``(eye3, coords)`` where:
      * ``eye3`` is a ``(3, 3)`` float32 identity that the caller
        expands to ``(B, 3, 3)`` per call.
      * ``coords`` is the ``(H, W, 3)`` homogeneous-pixel grid that
        the caller expands to ``(B, H, W, 3)`` per call.
    """
    device = (
        torch.device(device_str)
        if device_index is None
        else torch.device(f"{device_str}:{device_index}")
    )
    eye3 = torch.eye(3, dtype=torch.float32, device=device)
    ys, xs = torch.meshgrid(
        torch.arange(output_height, device=device, dtype=torch.float32),
        torch.arange(output_width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    return eye3, coords


def _device_key(device: torch.device) -> tuple[str, int | None]:
    """Return a hashable ``(type, index)`` tuple for a torch device."""
    return device.type, device.index


def _affine_grid_from_matrices(
    matrices: torch.Tensor,
    *,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    align_corners: bool,
) -> torch.Tensor:
    """Build a pixel-accurate ``grid_sample`` grid from OpenCV-style
    warp matrices.

    Issue #194 P2 high: the static scaffolding (identity matrix +
    pixel-coordinate grid) is built ONCE per ``(device, output_size)``
    via :func:`_cached_affine_scaffold` and reused across calls; only
    the per-batch ``matrices`` mix-in + inversion + per-batch expand
    runs every call.
    """
    input_height, input_width = input_size
    output_height, output_width = output_size
    eye3, coords = _cached_affine_scaffold(
        *_device_key(matrices.device), output_height, output_width
    )
    homogeneous = eye3.unsqueeze(0).repeat(matrices.shape[0], 1, 1)
    homogeneous[:, :2] = matrices
    inverse = torch.linalg.inv(homogeneous)[:, :2]
    coords = coords.unsqueeze(0).expand(matrices.shape[0], -1, -1, -1)
    source = torch.einsum("bij,bhwj->bhwi", inverse, coords)
    grid_x = _normalize_grid_coords(source[..., 0], input_width, align_corners=align_corners)
    grid_y = _normalize_grid_coords(source[..., 1], input_height, align_corners=align_corners)
    return torch.stack((grid_x, grid_y), dim=-1)


def _normalize_matrices(
    matrix: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize affine matrices to ``(B, 2, 3)`` float32 tensors."""
    if matrix.ndim == 2:
        if matrix.shape == (2, 3):
            matrix = matrix.unsqueeze(0)
        elif matrix.shape == (3, 3):
            matrix = matrix[:2].unsqueeze(0)
        else:
            raise ValueError(
                f"Expected affine matrix with shape (2, 3) or (3, 3), got {tuple(matrix.shape)}"
            )
    elif matrix.ndim == 3:
        if matrix.shape[1:] == (3, 3):
            matrix = matrix[:, :2]
        elif matrix.shape[1:] != (2, 3):
            raise ValueError(
                f"Expected affine matrix batch with shape (B, 2, 3) or (B, 3, 3), got {tuple(matrix.shape)}"
            )
    else:
        raise ValueError(
            f"Expected affine matrix with shape (2, 3), (3, 3), (B, 2, 3), or (B, 3, 3), got {tuple(matrix.shape)}"
        )

    matrix = matrix.to(device=device, dtype=torch.float32)
    if matrix.shape[0] == 1 and batch_size > 1:
        return matrix.expand(batch_size, -1, -1)
    if matrix.shape[0] != batch_size:
        raise ValueError(
            f"Affine matrix batch size {matrix.shape[0]} does not match image batch size {batch_size}"
        )
    return matrix


def _boxes_to_matrices(boxes: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Convert crop boxes into OpenCV-style affine warp matrices."""
    output_height, output_width = _validate_output_size(output_size)
    matrices = torch.zeros((boxes.shape[0], 2, 3), dtype=torch.float32, device=boxes.device)
    matrices[:, 0, 0] = float(output_width) / (boxes[:, 2] - boxes[:, 0])
    matrices[:, 1, 1] = float(output_height) / (boxes[:, 3] - boxes[:, 1])
    matrices[:, 0, 2] = -boxes[:, 0] * matrices[:, 0, 0]
    matrices[:, 1, 2] = -boxes[:, 1] * matrices[:, 1, 1]
    return matrices


def torch_resize(
    image: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """Resize a BCHW image tensor while preserving the explicit input dtype contract."""
    image = _validate_image(image, "torch_resize")
    target_size = _validate_output_size(size)
    _opencv_interpolation(mode)

    if not _resize_supported_on_device(image.device.type, mode):
        if image.device.type == "mps":
            _record_mps_fallback(f"torch_resize:{mode}")
        logger.debug("Falling back to OpenCV resize for device=%s mode=%s", image.device, mode)
        return _resize_with_opencv(image, target_size, mode=mode)

    restored_dtype = image.dtype
    work = image.to(dtype=torch.float32)
    kwargs: dict[str, bool] = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = align_corners
    resized = F.interpolate(work, size=target_size, mode=mode, **kwargs)
    return _restore_dtype(resized, restored_dtype)


def torch_normalize(
    image: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    """Normalize a float32 BCHW image tensor with per-channel statistics."""
    image = _validate_image(image, "torch_normalize")
    if image.dtype != torch.float32:
        raise ValueError(
            "torch_normalize expects float32 input so callers control dtype conversion"
        )
    if len(mean) != image.shape[1] or len(std) != image.shape[1]:
        raise ValueError(
            f"mean/std length must match channel count {image.shape[1]}, got {len(mean)} and {len(std)}"
        )

    mean_tensor, std_tensor = _cached_normalization_stats(
        tuple(mean), tuple(std), *_device_key(image.device)
    )
    if torch.any(std_tensor == 0):
        raise ValueError("std values must be non-zero")
    return (image - mean_tensor) / std_tensor


def torch_affine_warp(
    image: torch.Tensor,
    matrix: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Warp a BCHW image tensor using OpenCV-style affine matrices."""
    image = _validate_image(image, "torch_affine_warp")
    target_size = _validate_output_size(output_size)
    matrices = _normalize_matrices(matrix, batch_size=image.shape[0], device=image.device)

    if not _warp_supported_on_device(image.device.type):
        if image.device.type == "mps":
            _record_mps_fallback("torch_affine_warp")
        logger.debug("Falling back to OpenCV warp for device=%s", image.device)
        return _warp_with_opencv(image, matrices, target_size)

    restored_dtype = image.dtype
    work = image.to(dtype=torch.float32)
    grid = _affine_grid_from_matrices(
        matrices,
        input_size=(image.shape[2], image.shape[3]),
        output_size=target_size,
        align_corners=False,
    )
    warped = F.grid_sample(
        work,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return _restore_dtype(warped, restored_dtype)


def torch_face_crop(
    image: torch.Tensor,
    bbox: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Crop a single face from a BCHW image tensor using source-frame box coordinates."""
    image = _validate_image(image, "torch_face_crop")
    if image.shape[0] != 1:
        raise ValueError(
            f"torch_face_crop expects a single-image BCHW tensor, got batch size {image.shape[0]}"
        )
    boxes = _validate_boxes(bbox, allow_many=False).to(device=image.device)
    matrices = _boxes_to_matrices(boxes, output_size)
    return torch_affine_warp(image, matrices, output_size)


def torch_batched_crop_align(
    images: torch.Tensor,
    boxes: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Crop and resize one face box per output item while preserving input order."""
    images = _validate_image(images, "torch_batched_crop_align")
    crop_boxes = _validate_boxes(boxes, allow_many=True).to(device=images.device)

    if images.shape[0] == 1 and crop_boxes.shape[0] > 1:
        images = images.expand(crop_boxes.shape[0], -1, -1, -1)
    elif images.shape[0] != crop_boxes.shape[0]:
        raise ValueError(
            f"Image batch size {images.shape[0]} does not match crop box batch size {crop_boxes.shape[0]}"
        )

    matrices = _boxes_to_matrices(crop_boxes, output_size)
    return torch_affine_warp(images, matrices, output_size)


__all__ = get_module_objects(__name__)
