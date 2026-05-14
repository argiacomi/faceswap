#!/usr/bin/env python3
"""Native Qt line chart widget for training graph data."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPaintEvent, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget

from lib.gui.services.training_graph_service import (
    TrainingGraphSeries,
    TrainingGraphSnapshot,
)


class TrainingGraphWidget(QWidget):
    """Dependency-free Qt line chart for training loss series."""

    MAX_POINTS_PER_PIXEL = 2
    MAX_ZOOM = 50.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-shell-training-graph-widget")
        self.setMinimumSize(240, 180)
        self._series: tuple[TrainingGraphSeries, ...] = ()
        self._selected_keys: tuple[str, ...] = ()
        self._status_text = "No graph data loaded"
        self._zoom = 1.0
        self._y_zoom = 1.0
        self._pan = 0.0
        self._y_pan = 0.0
        self._drag_start: tuple[float, float] | None = None
        self._last_decimated_count = 0

    @property
    def series(self) -> tuple[TrainingGraphSeries, ...]:
        """Return currently rendered series."""
        return self._series

    @property
    def selected_keys(self) -> tuple[str, ...]:
        """Return selected series keys."""
        return self._selected_keys

    @property
    def status_text(self) -> str:
        """Return graph widget status text."""
        return self._status_text

    @property
    def zoom(self) -> float:
        """Return the current horizontal zoom factor."""
        return self._zoom

    @property
    def y_zoom(self) -> float:
        """Return the current vertical zoom factor."""
        return self._y_zoom

    @property
    def pan(self) -> float:
        """Return the current horizontal pan offset from 0.0 to 1.0."""
        return self._pan

    @property
    def y_pan(self) -> float:
        """Return the current vertical pan offset from 0.0 to 1.0."""
        return self._y_pan

    @property
    def last_decimated_count(self) -> int:
        """Return point count used for the most recent generated path."""
        return self._last_decimated_count

    @property
    def viewport(self) -> tuple[float, float, float, float]:
        """Return zoom/pan state as ``x_zoom, y_zoom, x_pan, y_pan``."""
        return self._zoom, self._y_zoom, self._pan, self._y_pan

    def set_snapshot(
        self,
        snapshot: TrainingGraphSnapshot,
        *,
        selected_keys: tuple[str, ...] = (),
    ) -> None:
        """Set graph data and repaint."""
        if snapshot.is_empty:
            self._series = ()
            self._selected_keys = ()
            self._status_text = "No graph data loaded"
            self.reset_view(update=False)
            self.update()
            return
        selected = selected_keys or tuple(series.name for series in snapshot.series)
        self._selected_keys = tuple(selected)
        self._series = tuple(series for series in snapshot.series if series.name in selected)
        point_count = max((series.count for series in self._series), default=0)
        if not self._series or point_count == 0:
            self._status_text = "No selected graph data loaded"
        else:
            names = ", ".join(series.name for series in self._series)
            self._status_text = (
                f"Rendered {len(self._series)} series, {point_count} points: {names}"
            )
        self.update()

    def clear(self) -> None:
        """Clear rendered graph data."""
        self._series = ()
        self._selected_keys = ()
        self._status_text = "No graph data loaded"
        self._last_decimated_count = 0
        self.reset_view(update=False)
        self.update()

    def zoom_in(self) -> None:
        """Zoom into the graph history on the x axis."""
        self._set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        """Zoom out of the graph history on the x axis."""
        self._set_zoom(self._zoom / 1.25)

    def zoom_y_in(self) -> None:
        """Zoom into the graph value range on the y axis."""
        self._set_y_zoom(self._y_zoom * 1.25)

    def zoom_y_out(self) -> None:
        """Zoom out of the graph value range on the y axis."""
        self._set_y_zoom(self._y_zoom / 1.25)

    def reset_view(self, *, update: bool = True) -> None:
        """Reset graph zoom and pan."""
        self._zoom = 1.0
        self._y_zoom = 1.0
        self._pan = 0.0
        self._y_pan = 0.0
        if update:
            self.update()

    def save_image(self, filename: str | Path) -> bool:
        """Render the chart to an image file inferred from filename suffix."""
        if not self._series:
            return False
        path = Path(filename)
        image_format = path.suffix.lstrip(".").upper() or "PNG"
        if image_format == "JPG":
            image_format = "JPEG"
        pixmap = QPixmap(self.size())
        pixmap.fill(self.palette().base().color())
        painter = QPainter(pixmap)
        try:
            self._draw_chart(painter)
        finally:
            painter.end()
        return pixmap.save(str(path), image_format)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa:N802
        """Paint the graph widget."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setClipRect(event.rect())
        try:
            self._draw_chart(painter)
        finally:
            painter.end()

    def wheelEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Zoom graph history with the mouse wheel.

        Holding Shift mirrors Tk-style value-axis zoom while the default wheel zooms x/history.
        """
        if not self._series:
            return
        vertical = event.angleDelta().y() > 0
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.zoom_y_in() if vertical else self.zoom_y_out()
        else:
            self.zoom_in() if vertical else self.zoom_out()
        event.accept()

    def mousePressEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Start graph panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (float(event.position().x()), float(event.position().y()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Pan graph history while dragging."""
        if self._drag_start is None or (self._zoom <= 1.0 and self._y_zoom <= 1.0):
            super().mouseMoveEvent(event)
            return
        current_x = float(event.position().x())
        current_y = float(event.position().y())
        start_x, start_y = self._drag_start
        self._drag_start = (current_x, current_y)
        if self._zoom > 1.0:
            self._set_pan(self._pan + ((start_x - current_x) / max(1, self.width())))
        if self._y_zoom > 1.0:
            self._set_y_pan(self._y_pan + ((current_y - start_y) / max(1, self.height())))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Stop graph panning."""
        self._drag_start = None
        super().mouseReleaseEvent(event)

    def _draw_chart(self, painter: QPainter) -> None:
        """Draw axes, message, and any loaded line series."""
        rect = self.rect().adjusted(44, 16, -16, -28)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(self.palette().mid().color())
        painter.drawRect(rect)
        if not self._series:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._status_text)
            return
        minimum, maximum = self._value_range()
        if minimum == maximum:
            minimum -= 1.0
            maximum += 1.0
        painter.drawText(4, 18, f"{maximum:.4g}")
        painter.drawText(4, rect.bottom(), f"{minimum:.4g}")
        painter.save()
        painter.setClipRect(rect)
        for index, series in enumerate(self._series):
            pen = QPen(self.palette().highlight().color())
            pen.setWidth(2 + (index % 2))
            painter.setPen(pen)
            painter.drawPath(self._path_for_series(series, minimum, maximum, rect))
        painter.restore()
        painter.setPen(self.palette().text().color())
        painter.drawText(rect.left(), self.height() - 8, self._legend())

    def _path_for_series(self, series: TrainingGraphSeries, minimum: float, maximum: float, rect) -> QPainterPath:
        """Return a batched painter path for a series in widget coordinates."""
        points = self._points_for_series(series, minimum, maximum, rect)
        if not points:
            return QPainterPath()
        polygon = QPolygonF(points)
        path = QPainterPath()
        path.addPolygon(polygon)
        return path

    def _points_for_series(
        self,
        series: TrainingGraphSeries,
        minimum: float,
        maximum: float,
        rect,
    ) -> list[QPointF]:
        """Return painted points for a series in widget coordinates."""
        values = self._decimated_values(self._visible_values(series.values), max(2, int(rect.width()) * self.MAX_POINTS_PER_PIXEL))
        self._last_decimated_count = max(self._last_decimated_count, len(values))
        if not values:
            return []
        x_span = max(1, len(values) - 1)
        y_span = maximum - minimum
        points = []
        for index, value in enumerate(values):
            x_pos = rect.left() + (index / x_span) * rect.width()
            y_pos = rect.bottom() - ((value - minimum) / y_span) * rect.height()
            points.append(QPointF(x_pos, y_pos))
        return points

    def _value_range(self) -> tuple[float, float]:
        """Return min/max values across rendered series after x/y viewport transforms."""
        values = [
            value for series in self._series for value in self._visible_values(series.values)
        ]
        minimum = min(values)
        maximum = max(values)
        if self._y_zoom <= 1.0 or minimum == maximum:
            return minimum, maximum
        span = maximum - minimum
        window = span / self._y_zoom
        max_start = span - window
        start = minimum + (max_start * self._y_pan)
        return start, start + window

    def _legend(self) -> str:
        """Return a compact legend string."""
        return ", ".join(series.name for series in self._series)

    def _visible_values(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Return values in the current x zoom/pan window."""
        if self._zoom <= 1.0 or len(values) <= 2:
            return values
        window = max(2, int(round(len(values) / self._zoom)))
        max_start = max(0, len(values) - window)
        start = int(round(max_start * self._pan))
        return values[start : start + window]

    @staticmethod
    def _decimated_values(values: tuple[float, ...], max_points: int) -> tuple[float, ...]:
        """Return evenly sampled values capped for long-history paint performance."""
        if len(values) <= max_points:
            return values
        step = len(values) / max_points
        return tuple(values[int(index * step)] for index in range(max_points))

    def _set_zoom(self, zoom: float) -> None:
        """Clamp and apply horizontal zoom."""
        self._zoom = max(1.0, min(self.MAX_ZOOM, zoom))
        if self._zoom == 1.0:
            self._pan = 0.0
        self.update()

    def _set_y_zoom(self, zoom: float) -> None:
        """Clamp and apply vertical zoom."""
        self._y_zoom = max(1.0, min(self.MAX_ZOOM, zoom))
        if self._y_zoom == 1.0:
            self._y_pan = 0.0
        self.update()

    def _set_pan(self, pan: float) -> None:
        """Clamp and apply horizontal pan."""
        self._pan = max(0.0, min(1.0, pan))
        self.update()

    def _set_y_pan(self, pan: float) -> None:
        """Clamp and apply vertical pan."""
        self._y_pan = max(0.0, min(1.0, pan))
        self.update()
