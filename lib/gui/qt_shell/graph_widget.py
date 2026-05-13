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
        if not series.values:
            return []
        x_span = max(1, series.count - 1)
        y_span = maximum - minimum
        points = []
        for index, value in enumerate(series.values):
            x_pos = rect.left() + (index / x_span) * rect.width()
            y_pos = rect.bottom() - ((value - minimum) / y_span) * rect.height()
            points.append(QPointF(x_pos, y_pos))
        return points

    def _value_range(self) -> tuple[float, float]:
        """Return min/max values across rendered series."""
        values = [value for series in self._series for value in series.values]
        return min(values), max(values)

    def _legend(self) -> str:
        """Return a compact legend string."""
        return ", ".join(series.name for series in self._series)
