#!/usr/bin/env python3
"""DARK heatmap decoding for heatmap based aligners."""

import logging

import cv2
import numpy as np

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class Dark:
    """Dark heatmap decoding

    https://github.com/ilovepose/DarkPose

    Parameters
    ----------
    num_points
        The number of landmarks output from the model
    size
        The size of the heatmap
    """

    def __init__(
        self,
        num_points: int,
        size: int,
        blur_kernel: int = 11,
        max_offset: float = 1.0,
        min_hessian_det: float = 1e-6,
    ):
        logger.debug(parse_class_init(locals()))
        self._num_points = num_points
        self._size = size
        self._blur_kernel = blur_kernel
        self._border = (blur_kernel - 1) // 2
        self._max_offset = np.float32(max_offset)
        self._min_hessian_det = np.float32(min_hessian_det)

    def get_max_preds(self, batch_heatmaps: np.ndarray) -> np.ndarray:
        """get predictions from score maps

        Parameters
        ----------
        heatmaps
            Heatmap to derive points from ([batch_size, num_joints, height, width])

        Returns
        -------
        coords
            The derived points from the heatmaps (B, N, 2)
        """
        assert isinstance(batch_heatmaps, np.ndarray), "batch_heatmaps should be numpy.ndarray"
        assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

        batch = batch_heatmaps.shape[0]
        heatmaps_reshaped = batch_heatmaps.reshape((batch, self._num_points, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        max_vals = np.amax(heatmaps_reshaped, 2)

        max_vals = max_vals.reshape((batch, self._num_points, 1))
        idx = idx.reshape((batch, self._num_points, 1))

        preds = np.zeros((batch, self._num_points, 2), dtype=np.float32)
        preds[:, :, 0] = idx[..., 0] % self._size
        preds[:, :, 1] = idx[..., 0] // self._size

        pred_mask = np.repeat(np.greater(max_vals, 0.0), 2, axis=2)
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds  # type: ignore[no-any-return]

    def gaussian_blur(self, heatmap: np.ndarray) -> np.ndarray:
        """Perform gaussian blurring on the heatmaps

        Parameters
        ----------
        heatmap
            Batch of heatmaps to blur (N, points, size, size)

        Returns
        -------
        The blurred heatmaps
        """
        batch_size = heatmap.shape[0]
        origin_max = heatmap.reshape(batch_size, self._num_points, -1).max(axis=2)
        padded = np.pad(
            heatmap,
            (
                (0, 0),
                (0, 0),
                (self._border, self._border),
                (self._border, self._border),
            ),
            mode="constant",
        )
        reshaped = padded.reshape(  # pylint:disable=too-many-function-args)
            batch_size * self._num_points,
            self._size + 2 * self._border,
            self._size + 2 * self._border,
        )
        blurred = np.stack(
            [cv2.GaussianBlur(img, (self._blur_kernel, self._blur_kernel), 0) for img in reshaped]
        )
        blurred = blurred.reshape(
            batch_size,
            self._num_points,
            self._size + 2 * self._border,
            self._size + 2 * self._border,
        )
        cropped = blurred[:, :, self._border : -self._border, self._border : -self._border]
        new_max = cropped.reshape(batch_size, self._num_points, -1).max(axis=2)
        scale = origin_max / (new_max + 1e-8)  # avoid division by zero
        scale = scale[:, :, None, None]
        return cropped * scale  # type: ignore[no-any-return]

    def taylor(
        self,
        heatmap: np.ndarray,
        coords: np.ndarray,  # pylint:disable=too-many-locals
    ) -> np.ndarray:
        """Sub-pixel refine the predictions

        Parameters
        ----------
        heatmap
            The processed heatmaps for refinement
        coords
            The coordinates to be refined

        Returns
        -------
        The refined coordinates
        """
        batch = heatmap.shape[0]
        coords = coords.astype(np.float32, copy=True)
        base_coords = coords.copy()
        px = np.clip(coords[..., 0], 2, self._size - 3).astype(np.int32)
        py = np.clip(coords[..., 1], 2, self._size - 3).astype(np.int32)

        flat_idx = np.arange(batch * self._num_points)
        hm = heatmap.reshape(batch * self._num_points, self._size, self._size)
        px_f = px.reshape(-1)
        py_f = py.reshape(-1)

        dx = 0.5 * (hm[flat_idx, py_f, px_f + 1] - hm[flat_idx, py_f, px_f - 1])
        dy = 0.5 * (hm[flat_idx, py_f + 1, px_f] - hm[flat_idx, py_f - 1, px_f])
        dxx = 0.25 * (
            hm[flat_idx, py_f, px_f + 2]
            - 2 * hm[flat_idx, py_f, px_f]
            + hm[flat_idx, py_f, px_f - 2]
        )
        dyy = 0.25 * (
            hm[flat_idx, py_f + 2, px_f]
            - 2 * hm[flat_idx, py_f, px_f]
            + hm[flat_idx, py_f - 2, px_f]
        )
        dxy = 0.25 * (
            hm[flat_idx, py_f + 1, px_f + 1]
            - hm[flat_idx, py_f - 1, px_f + 1]
            - hm[flat_idx, py_f + 1, px_f - 1]
            + hm[flat_idx, py_f - 1, px_f - 1]
        )

        dx = dx.reshape(batch, self._num_points)
        dy = dy.reshape(batch, self._num_points)
        dxx = dxx.reshape(batch, self._num_points)
        dyy = dyy.reshape(batch, self._num_points)
        dxy = dxy.reshape(batch, self._num_points)

        det = dxx * dyy - dxy**2
        stable = (
            np.isfinite(dx)
            & np.isfinite(dy)
            & np.isfinite(dxx)
            & np.isfinite(dyy)
            & np.isfinite(dxy)
            & np.isfinite(det)
            & (np.abs(det) >= self._min_hessian_det)
        )
        offset_x = np.zeros_like(det, dtype=np.float32)
        offset_y = np.zeros_like(det, dtype=np.float32)

        unstable_points = int(stable.size - np.count_nonzero(stable))
        nonfinite_offsets = 0
        clipped_offsets = 0
        if np.any(stable):
            raw_offset_x = -((dyy[stable] * dx[stable] - dxy[stable] * dy[stable]) / det[stable])
            raw_offset_y = -((-dxy[stable] * dx[stable] + dxx[stable] * dy[stable]) / det[stable])
            finite_offsets = np.isfinite(raw_offset_x) & np.isfinite(raw_offset_y)
            nonfinite_offsets = int(finite_offsets.size - np.count_nonzero(finite_offsets))
            if np.any(finite_offsets):
                clipped_offsets = int(
                    np.count_nonzero(
                        (np.abs(raw_offset_x[finite_offsets]) > self._max_offset)
                        | (np.abs(raw_offset_y[finite_offsets]) > self._max_offset)
                    )
                )
                offset_x[stable] = np.where(
                    finite_offsets,
                    np.clip(raw_offset_x, -self._max_offset, self._max_offset),
                    0.0,
                )
                offset_y[stable] = np.where(
                    finite_offsets,
                    np.clip(raw_offset_y, -self._max_offset, self._max_offset),
                    0.0,
                )

        coords[..., 0] = base_coords[..., 0] + offset_x
        coords[..., 1] = base_coords[..., 1] + offset_y
        unclipped = coords.copy()
        np.clip(coords, 0.0, float(self._size - 1), out=coords)
        clipped_coords = int(np.count_nonzero(coords != unclipped))

        if unstable_points or nonfinite_offsets or clipped_offsets or clipped_coords:
            logger.debug(
                "[Dark] bounded Taylor refinement: unstable=%s nonfinite=%s "
                "offset_clipped=%s coord_clipped=%s",
                unstable_points,
                nonfinite_offsets,
                clipped_offsets,
                clipped_coords,
            )

        return coords.astype(np.float32, copy=False)

    def __call__(self, heatmap: np.ndarray) -> np.ndarray:
        coords = self.get_max_preds(heatmap)

        # post-processing
        heatmap = self.gaussian_blur(heatmap)
        heatmap = np.maximum(heatmap, 1e-10)
        heatmap = np.log(heatmap)
        coords = self.taylor(heatmap, coords)
        return coords.astype(np.float32, copy=False)


get_module_objects(__name__)
