#!/usr/bin/env python3
"""The Frame Viewer for Faceswap's Manual Tool."""

from __future__ import annotations

import typing as T

from PIL import Image, ImageTk

from ._base import View
from .bounding_box import BoundingBox
from .extract_box import ExtractBox
from .landmarks import Landmarks, Mesh
from .mask import Mask as _Mask


class Mask(_Mask):
    """Mask editor with sparse face-index cache handling."""

    _CACHE_FIELDS: T.ClassVar[tuple[str, ...]] = (
        "mask",
        "roi_mask",
        "affine_matrix",
        "interpolator",
        "slices",
        "top_left",
        "mask_roi_size",
    )

    def _set_face_meta_data(self, mask: T.Any, face_index: int) -> None:
        """Store cache entries at their absolute face index."""
        masks = self._meta.get("mask", None)
        if masks is not None and face_index < len(masks) and masks[face_index] is not None:
            return
        super()._set_face_meta_data(mask, face_index)
        for field in self._CACHE_FIELDS:
            values = self._meta.get(field)
            if not values:
                continue
            value = values.pop()
            while len(values) <= face_index:
                values.append(None)
            values[face_index] = value

    def _update_mask_image(
        self,
        key: str,
        face_index: int,
        rgb_color: T.Any,
        opacity: float,
    ) -> None:
        """Keep the photo-image cache aligned to absolute face indices."""
        while len(self._tk_faces) < face_index:
            self._tk_faces.append(ImageTk.PhotoImage(Image.new("RGBA", (1, 1))))
        super()._update_mask_image(key, face_index, rgb_color, opacity)


__all__ = ["BoundingBox", "ExtractBox", "Landmarks", "Mask", "Mesh", "View"]
