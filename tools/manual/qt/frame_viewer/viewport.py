#!/usr/bin/env python3
"""Frame-view geometry types for the Qt Manual Tool."""

from __future__ import annotations

import typing as T

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainter

OverlayPainter = T.Callable[[QPainter, "FrameViewport"], None]


class FrameViewport(T.NamedTuple):
    """Snapshot of a frame view geometry passed to overlay painters."""

    source_size: tuple[int, int]
    """(width, height) of the source frame in pixels."""
    target_rect: QRectF
    """Destination rectangle in widget coordinates where the frame is drawn."""
    zoom: float
    """Effective zoom factor relative to the fit-to-widget baseline."""

    def source_to_widget(self, sx: float, sy: float) -> QPointF:
        """Translate a source-image point into widget-coordinate space."""
        src_w, src_h = self.source_size
        if src_w <= 0 or src_h <= 0:
            return QPointF(self.target_rect.x(), self.target_rect.y())
        rx = sx / src_w
        ry = sy / src_h
        return QPointF(
            self.target_rect.x() + rx * self.target_rect.width(),
            self.target_rect.y() + ry * self.target_rect.height(),
        )

    def source_rect_to_widget(self, x: float, y: float, w: float, h: float) -> QRectF:
        """Translate a source-image rect into widget-coordinate space."""
        top_left = self.source_to_widget(x, y)
        bottom_right = self.source_to_widget(x + w, y + h)
        return QRectF(top_left, bottom_right)
