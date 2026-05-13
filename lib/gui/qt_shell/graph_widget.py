#!/usr/bin/env python3
"""Native Qt line chart widget for training graph data."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from lib.gui.services.training_graph_service import TrainingGraphSeries, TrainingGraphSnapshot


class TrainingGraphWidget(QWidget):
    """Simple dependency-free Qt line chart for training loss series."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-shell-training-graph-widget")
        self.setMinimumSize(240, 180)
        self._series: tuple[TrainingGraphSeries, ...] = ()
        self._selected_keys: tuple[str, ...] = ()
        self._status_text = "No graph data loaded"
        self._zoom = 1.0
        self._pan = 0.0
        self._drag_start_x: float | None = None

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
    def pan(self) -> float:
        """Return the current horizontal pan offset from 0.0 to 1.0."""
        return self._pan

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
        self.reset_view(update=False)
        self.update()

    def zoom_in(self) -> None:
        """Zoom into the graph history."""
        self._set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        """Zoom out of the graph history."""
        self._set_zoom(self._zoom / 1.25)

    def reset_view(self, *, update: bool = True) -> None:
        """Reset graph zoom and pan."""
        self._zoom = 1.0
        self._pan = 0.0
        if update:
            self.update()

    def save_image(self, filename: str | Path) -> bool:
        """Render the chart to a PNG image."""
        if not self._series:
            return False
        pixmap = QPixmap(self.size())
        pixmap.fill(self.palette().base().color())
        painter = QPainter(pixmap)
        try:
            self._draw_chart(painter)
        finally:
            painter.end()
        return pixmap.save(str(filename), "PNG")

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa:N802
        """Paint the graph widget."""
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            self._draw_chart(painter)
        finally:
            painter.end()

    def wheelEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Zoom graph history with the mouse wheel."""
        if not self._series:
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def mousePressEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Start horizontal panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_x = float(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Pan graph history while dragging."""
        if self._drag_start_x is None or self._zoom <= 1.0:
            super().mouseMoveEvent(event)
            return
        current_x = float(event.position().x())
        delta = (self._drag_start_x - current_x) / max(1, self.width())
        self._drag_start_x = current_x
        self._set_pan(self._pan + delta)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Stop graph panning."""
        self._drag_start_x = None
        super().mouseReleaseEvent(event)

    def _draw_chart(self, painter: QPainter) -> None:
        """Draw axes, message, and any loaded line series."""
        rect = self.rect().adjusted(36, 16, -16, -28)
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
        for index, series in enumerate(self._series):
            pen = QPen(self.palette().highlight().color())
            pen.setWidth(2 + (index % 2))
            painter.setPen(pen)
            points = self._points_for_series(series, minimum, maximum, rect)
            for start, end in zip(points, points[1:], strict=False):
                painter.drawLine(start, end)
        painter.setPen(self.palette().text().color())
        painter.drawText(rect.left(), self.height() - 8, self._legend())

    def _points_for_series(
        self,
        series: TrainingGraphSeries,
        minimum: float,
        maximum: float,
        rect,
    ) -> list[QPointF]:
        """Return painted points for a series in widget coordinates."""
        values = self._visible_values(series.values)
        if not values:
            return []
        max_points = max(2, int(rect.width()) * 2)
        if len(values) > max_points:
            step = len(values) / max_points
            values = tuple(values[int(index * step)] for index in range(max_points))
        x_span = max(1, len(values) - 1)
        y_span = maximum - minimum
        points = []
        for index, value in enumerate(values):
            x_pos = rect.left() + (index / x_span) * rect.width()
            y_pos = rect.bottom() - ((value - minimum) / y_span) * rect.height()
            points.append(QPointF(x_pos, y_pos))
        return points

    def _value_range(self) -> tuple[float, float]:
        """Return min/max values across rendered series."""
        values = [
            value for series in self._series for value in self._visible_values(series.values)
        ]
        return min(values), max(values)

    def _legend(self) -> str:
        """Return a compact legend string."""
        return ", ".join(series.name for series in self._series)

    def _visible_values(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Return values in the current zoom/pan window."""
        if self._zoom <= 1.0 or len(values) <= 2:
            return values
        window = max(2, int(round(len(values) / self._zoom)))
        max_start = max(0, len(values) - window)
        start = int(round(max_start * self._pan))
        return values[start : start + window]

    def _set_zoom(self, zoom: float) -> None:
        """Clamp and apply horizontal zoom."""
        self._zoom = max(1.0, min(50.0, zoom))
        if self._zoom == 1.0:
            self._pan = 0.0
        self.update()

    def _set_pan(self, pan: float) -> None:
        """Clamp and apply horizontal pan."""
        self._pan = max(0.0, min(1.0, pan))
        self.update()
