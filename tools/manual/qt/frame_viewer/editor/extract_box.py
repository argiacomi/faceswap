#!/usr/bin/env python3
"""Extract Box editor controllers for the Qt Manual Tool."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF

from tools.manual.qt.frame_viewer.overlays import ManualFrameOverlay

EDITOR_MODE = "ExtractBox"


def is_active(editor_mode: str) -> bool:
    """Return whether ``editor_mode`` selects this editor."""
    return editor_mode == EDITOR_MODE


class ExtractBoxFrameEditorMixin:
    """Frame-view translate/scale/rotate behavior for Extract Box mode."""

    _EXTRACT_ROTATION_BAND_PX = 24.0

    def _begin_extract_drag(self, position: QPointF) -> bool:
        """Try to start an Extract Box translate/scale/rotate drag (#102).

        Hit-test priority (closest action first):

        1. A corner handle → scale uniformly around the bbox centre.
        2. A point inside the bbox body → translate (landmarks + bbox).
        3. A point inside a ``_EXTRACT_ROTATION_BAND_PX`` halo outside the
           bbox → rotate landmarks around the centre.
        4. Otherwise return ``False`` so the view can pan instead.
        """

        if self._extract_mode_provider is None or not self._extract_mode_provider():
            return False
        if self._active_face_provider is None or self._active_bbox_provider is None:
            return False
        face_index = self._active_face_provider()
        if face_index is None:
            return False
        widget_bbox = self._active_bbox_widget_rect()
        source_bbox = self._active_bbox_source_rect()
        if widget_bbox is None or source_bbox is None:
            return False
        source_point = self._widget_to_source(position)
        if source_point is None:
            return False
        QPointF(
            widget_bbox.x() + widget_bbox.width() / 2.0,
            widget_bbox.y() + widget_bbox.height() / 2.0,
        )
        centre_source = QPointF(
            source_bbox.x() + source_bbox.width() / 2.0,
            source_bbox.y() + source_bbox.height() / 2.0,
        )
        handle = ManualFrameOverlay.handle_at(widget_bbox, position, tolerance=2.0)
        # Scale uses only corner handles (nw/ne/sw/se).  Mid-edge handles
        # also map to scale here (uniform scale by closest corner) so the
        # whole 8-handle ring is responsive.
        if handle is not None:
            radius = math.hypot(
                source_point.x() - centre_source.x(),
                source_point.y() - centre_source.y(),
            )
            if radius <= 0.0:
                return False
            self._edit_drag_mode = "extract_scale"
            self._edit_drag_handle = handle
            self._extract_drag_center = centre_source
            self._extract_drag_start_radius = radius
            self._extract_drag_scale = 1.0
            self._edit_drag_source_anchor = source_point
            self._edit_drag_original_bbox = QRectF(source_bbox)
            self._edit_drag_current_bbox = QRectF(source_bbox)
            return True
        if widget_bbox.contains(position):
            self._edit_drag_mode = "extract_translate"
            self._edit_drag_handle = None
            self._edit_drag_source_anchor = source_point
            self._edit_drag_original_bbox = QRectF(source_bbox)
            self._edit_drag_current_bbox = QRectF(source_bbox)
            return True
        # Outside the bbox but within the rotation halo → rotate.
        halo = QRectF(
            widget_bbox.x() - self._EXTRACT_ROTATION_BAND_PX,
            widget_bbox.y() - self._EXTRACT_ROTATION_BAND_PX,
            widget_bbox.width() + 2 * self._EXTRACT_ROTATION_BAND_PX,
            widget_bbox.height() + 2 * self._EXTRACT_ROTATION_BAND_PX,
        )
        if not halo.contains(position):
            return False
        start_angle = math.atan2(
            source_point.y() - centre_source.y(),
            source_point.x() - centre_source.x(),
        )
        self._edit_drag_mode = "extract_rotate"
        self._edit_drag_handle = None
        self._extract_drag_center = centre_source
        self._extract_drag_start_angle = start_angle
        self._extract_drag_angle = 0.0
        self._edit_drag_source_anchor = source_point
        self._edit_drag_original_bbox = QRectF(source_bbox)
        self._edit_drag_current_bbox = QRectF(source_bbox)
        return True

    def _update_extract_scale(self, source_point: QPointF) -> None:
        """Track the live scale factor + preview bbox for an Extract Box scale drag."""

        if (
            self._extract_drag_center is None
            or self._extract_drag_start_radius <= 0.0
            or self._edit_drag_original_bbox is None
        ):
            return
        radius = math.hypot(
            source_point.x() - self._extract_drag_center.x(),
            source_point.y() - self._extract_drag_center.y(),
        )
        scale = max(0.05, radius / self._extract_drag_start_radius)
        self._extract_drag_scale = scale
        original = self._edit_drag_original_bbox
        cx = self._extract_drag_center.x()
        cy = self._extract_drag_center.y()
        new_w = original.width() * scale
        new_h = original.height() * scale
        self._edit_drag_current_bbox = QRectF(cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)

    def _update_extract_rotate(self, source_point: QPointF) -> None:
        """Track the live rotation delta for an Extract Box rotation drag.

        The preview bbox isn't rotated visually (the displayed rect stays
        axis-aligned) — the rotation only takes effect on commit because
        ``rotate_face`` is the source of truth.
        """

        if self._extract_drag_center is None:
            return
        angle = math.atan2(
            source_point.y() - self._extract_drag_center.y(),
            source_point.x() - self._extract_drag_center.x(),
        )
        self._extract_drag_angle = angle - self._extract_drag_start_angle


class ExtractBoxWindowEditorMixin:
    """Root-window adapters for Extract Box model updates."""

    def _is_extract_mode_active(self) -> bool:
        """Return whether the Extract Box editor (F3) is active."""
        return self._editor_state.editor_mode == "ExtractBox"

    def _on_face_scale_requested(self, face_index: int, scale: float) -> None:
        """Apply an Extract Box corner-drag scale to the active face (#102)."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        if not self._editable.scale_face(frame_index, int(face_index), float(scale)):
            self.statusBar().showMessage("Scale failed (degenerate result)", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()

    def _on_face_rotate_requested(self, face_index: int, angle: float) -> None:
        """Apply an Extract Box rotation-zone drag to the active face (#102)."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return
        if not self._editable.rotate_face(frame_index, int(face_index), float(angle)):
            self.statusBar().showMessage("Rotate failed (no active face)", 3000)
            return
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()
