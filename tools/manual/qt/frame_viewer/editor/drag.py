#!/usr/bin/env python3
"""Shared edit-drag dispatcher for Qt Manual Tool editor controllers."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from tools.manual.qt.types import FrameViewport


class FrameEditDragMixin:
    """Dispatch active edit-drag previews and commits across editor controllers."""

    def _update_edit_drag(self, position: QPointF) -> None:
        """Apply pointer motion to the active drag preview.

        Handles bbox move/resize/add plus the new landmark, landmark-group,
        and marquee modes (#103).  All previews update incrementally and
        request a repaint so the view + overlays stay in sync.
        """
        source_point = self._widget_to_source(position)
        if source_point is None or self._edit_drag_source_anchor is None:
            return
        anchor = self._edit_drag_source_anchor
        dx = source_point.x() - anchor.x()
        dy = source_point.y() - anchor.y()
        mode = self._edit_drag_mode
        if mode == "landmark":
            # Single-point drag: the overlay reads the live coord from
            # ``landmark_drag_preview`` so we just store the new position.
            self._edit_drag_current_bbox = QRectF(source_point.x(), source_point.y(), 0.0, 0.0)
            self.update()
            return
        if mode == "landmark_group":
            # Group move: overlay reads ``landmark_drag_preview`` for each
            # selected index — we just store the delta in the current bbox.
            self._edit_drag_current_bbox = QRectF(dx, dy, 0.0, 0.0)
            self.update()
            return
        if mode == "landmark_marquee":
            rect = QRectF(anchor, source_point).normalized()
            self._edit_drag_current_bbox = rect
            self.update()
            return
        if mode == "extract_translate":
            origin = self._edit_drag_original_bbox
            if origin is not None:
                self._edit_drag_current_bbox = QRectF(
                    origin.x() + dx,
                    origin.y() + dy,
                    origin.width(),
                    origin.height(),
                )
            self.update()
            return
        if mode == "extract_scale":
            self._update_extract_scale(source_point)
            self.update()
            return
        if mode == "extract_rotate":
            self._update_extract_rotate(source_point)
            self.update()
            return
        if self._edit_drag_original_bbox is None:
            return
        if mode == "move":
            origin = self._edit_drag_original_bbox
            self._edit_drag_current_bbox = QRectF(
                origin.x() + dx,
                origin.y() + dy,
                origin.width(),
                origin.height(),
            )
        elif mode == "add":
            self._edit_drag_current_bbox = QRectF(anchor, source_point).normalized()
        else:
            self._edit_drag_current_bbox = self._resize_bbox(
                self._edit_drag_original_bbox, self._edit_drag_handle or "", dx, dy
            )
        self.update()

    def _commit_edit_drag(self) -> None:
        """Emit the appropriate signal then clear the in-progress drag state."""
        original = self._edit_drag_original_bbox
        current = self._edit_drag_current_bbox
        mode = self._edit_drag_mode
        anchor = self._edit_drag_source_anchor
        landmark_index = self._landmark_drag_index
        landmark_indices = self._landmark_drag_indices
        extract_scale = self._extract_drag_scale
        extract_angle = self._extract_drag_angle
        face_index = self._active_face_provider() if self._active_face_provider else None
        self._reset_edit_drag()
        if mode == "add":
            self._emit_add_request(anchor, current)
            self.update()
            return
        if mode == "landmark":
            self._emit_landmark_move(face_index, landmark_index, anchor, current)
            self.update()
            return
        if mode == "landmark_group":
            self._emit_landmark_group_move(face_index, landmark_indices, current)
            self.update()
            return
        if mode == "landmark_marquee":
            self._emit_landmark_marquee(face_index, current)
            self.update()
            return
        if mode == "extract_translate":
            if face_index is not None and original is not None and current is not None:
                dx = current.x() - original.x()
                dy = current.y() - original.y()
                if dx != 0.0 or dy != 0.0:
                    self.face_move_requested.emit(int(face_index), float(dx), float(dy))
            self.update()
            return
        if mode == "extract_scale":
            if face_index is not None and extract_scale != 1.0:
                self.face_scale_requested.emit(int(face_index), float(extract_scale))
            self.update()
            return
        if mode == "extract_rotate":
            if face_index is not None and extract_angle != 0.0:
                self.face_rotate_requested.emit(int(face_index), float(extract_angle))
            self.update()
            return
        if face_index is None or original is None or current is None:
            self.update()
            return
        if mode == "move":
            dx = current.x() - original.x()
            dy = current.y() - original.y()
            if dx == 0.0 and dy == 0.0:
                self.update()
                return
            self.face_move_requested.emit(face_index, float(dx), float(dy))
        elif mode == "resize":
            if (
                current.x() == original.x()
                and current.y() == original.y()
                and current.width() == original.width()
                and current.height() == original.height()
            ):
                self.update()
                return
            self.face_resize_requested.emit(face_index, QRectF(current))
        self.update()

    def _reset_edit_drag(self) -> None:
        """Clear all in-progress edit-drag state without emitting signals."""
        self._edit_drag_mode = None
        self._edit_drag_handle = None
        self._edit_drag_source_anchor = None
        self._edit_drag_original_bbox = None
        self._edit_drag_current_bbox = None
        self._landmark_drag_index = None
        self._landmark_drag_indices = ()
        self._landmark_drag_origins = ()
        self._extract_drag_center = None
        self._extract_drag_start_radius = 0.0
        self._extract_drag_start_angle = 0.0
        self._extract_drag_scale = 1.0
        self._extract_drag_angle = 0.0

    def _paint_add_preview(self, painter: QPainter, viewport: FrameViewport) -> None:
        """Draw a dashed preview rect for in-progress add or marquee gestures."""
        mode = self._edit_drag_mode
        if mode not in ("add", "landmark_marquee"):
            return
        source = self._edit_drag_current_bbox
        if source is None:
            return
        if source.width() <= 0.0 and source.height() <= 0.0:
            return
        widget_rect = viewport.source_rect_to_widget(
            source.x(), source.y(), source.width(), source.height()
        )
        painter.save()
        pen = QPen(QColor("#88d4ff"))
        pen.setStyle(Qt.DashLine)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(widget_rect)
        painter.restore()
