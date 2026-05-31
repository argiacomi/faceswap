#!/usr/bin/env python3
"""Coordinate-space helpers for landmark adapters.

Conventions
-----------
Faceswap align plugins emit landmarks in normalized crop coordinates where
``(0, 0)`` is the top-left of the aligner crop and ``(1, 1)`` is the bottom-right
of that crop. The extract pipeline then applies a per-face affine matrix to move
those landmarks into original-frame pixel coordinates.

The landmark ensemble fuses only canonical 68-point landmarks in original-frame
pixel coordinates. If a plugin still needs to return a Faceswap-compatible
aligner result, frame-space fused landmarks are transformed back through the
inverse crop matrix to normalized crop coordinates before returning.
"""

from __future__ import annotations

import typing as T

import numpy as np

if T.TYPE_CHECKING:
    import numpy.typing as npt


CoordinateSpace = T.Literal["normalized_crop", "frame"]


def roi_to_matrix(
    roi: npt.NDArray[np.integer | np.floating],
) -> npt.NDArray[np.float32]:
    """Return normalized-crop to frame-pixel matrices for square ROI boxes.

    Parameters
    ----------
    roi
        A single ``(left, top, right, bottom)`` ROI or a batch of ROIs in original
        frame pixels.
    """
    boxes = np.asarray(roi, dtype="float32")
    single = boxes.ndim == 1
    if single:
        boxes = boxes[None]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"roi must have shape (4,) or (N, 4), got {boxes.shape}")
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("roi width and height must be greater than zero")
    if not np.allclose(widths, heights):
        raise ValueError("aligner ROI boxes must be square")

    matrices = np.zeros((boxes.shape[0], 3, 3), dtype="float32")
    matrices[:, 0, 0] = widths
    matrices[:, 1, 1] = heights
    matrices[:, 0, 2] = boxes[:, 0]
    matrices[:, 1, 2] = boxes[:, 1]
    matrices[:, 2, 2] = 1.0
    return matrices[0] if single else matrices


def transform_points(
    points: T.Sequence[T.Sequence[float]] | np.ndarray,
    matrices: np.ndarray,
) -> np.ndarray:
    """Apply affine matrices to unbatched or batched ``(x, y)`` points."""
    pts = np.asarray(points, dtype="float32")
    mats = np.asarray(matrices, dtype="float32")
    single_points = pts.ndim == 2
    single_matrix = mats.ndim == 2
    if single_points:
        pts = pts[None]
    if single_matrix:
        mats = mats[None]
    if pts.ndim != 3 or pts.shape[-1] != 2:
        raise ValueError(f"points must have shape (P, 2) or (N, P, 2), got {pts.shape}")
    if mats.ndim != 3 or mats.shape[1:] != (3, 3):
        raise ValueError(f"matrices must have shape (3, 3) or (N, 3, 3), got {mats.shape}")
    if mats.shape[0] == 1 and pts.shape[0] != 1:
        mats = np.repeat(mats, pts.shape[0], axis=0)
    if pts.shape[0] != mats.shape[0]:
        raise ValueError(
            f"points and matrices batch dimensions must match: {pts.shape[0]} != {mats.shape[0]}"
        )

    linear = mats[:, :2, :2]
    translation = mats[:, :2, 2]
    transformed = pts @ linear.transpose(0, 2, 1) + translation[:, None, :]
    transformed = transformed.astype("float32", copy=False)
    return transformed[0] if single_points and single_matrix else transformed  # type: ignore[no-any-return]


def normalized_crop_to_frame(
    points: T.Sequence[T.Sequence[float]] | np.ndarray,
    matrices: np.ndarray,
) -> np.ndarray:
    """Transform normalized crop coordinates into original-frame pixels."""
    return transform_points(points, matrices)


def frame_to_normalized_crop(
    points: T.Sequence[T.Sequence[float]] | np.ndarray,
    matrices: np.ndarray,
) -> np.ndarray:
    """Transform original-frame pixel coordinates into normalized crop coordinates."""
    inverse = np.linalg.inv(np.asarray(matrices, dtype="float32")).astype("float32")
    return transform_points(points, inverse)
