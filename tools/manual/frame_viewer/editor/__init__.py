#!/usr/bin/env python3
# mypy: disable-error-code="index"
"""The Frame Viewer for Faceswap's Manual Tool."""

from __future__ import annotations

import tkinter as tk
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

    def _drag_start(self, event: tk.Event, control_click: bool = False) -> None:
        """Start a mask stroke and remember the target face for the stroke."""
        super()._drag_start(event, control_click=control_click)
        face_index = self._mouse_location[1]
        if self._drag_data and face_index is not None:
            self._drag_data["face_index"] = face_index

    def _restore_drag_face_index(self) -> None:
        """Restore the active stroke face if cursor hit testing loses it."""
        if self._mouse_location[1] is not None:
            return
        face_index = self._drag_data.get("face_index") if self._drag_data else None
        if face_index is not None:
            self._mouse_location[1] = face_index

    def _paint(self, event: tk.Event) -> None:
        """Paint against the face captured at stroke start."""
        self._restore_drag_face_index()
        super()._paint(event)
        self._restore_drag_face_index()

    def _drag_stop(self, event: tk.Event) -> None:
        """Finish a stroke against the face captured at stroke start."""
        self._restore_drag_face_index()
        super()._drag_stop(event)


__all__ = ["BoundingBox", "ExtractBox", "Landmarks", "Mask", "Mesh", "View"]
