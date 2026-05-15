#!/usr/bin/env python3
"""Qt Graph panel renderer control tests."""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QLabel, QPushButton

from lib.gui.qt_shell.graph_panel import GraphPanel
from lib.gui.qt_shell.graph_widget import TrainingGraphWidget
from lib.gui.services.training_graph_service import (
    TrainingGraphSeries,
    TrainingGraphSnapshot,
)


class _GraphServiceDouble:
    """Small TrainingGraphService stand-in."""

    def __init__(self) -> None:
        self.source = None
        self.session_id = None
        self.session_ids = (1, 2)
        self.is_loaded = True
        self.snapshot = TrainingGraphSnapshot(
            source=None,
            session_id=None,
            series=(
                TrainingGraphSeries("metric_a", (3.0, 2.0, 1.0)),
                TrainingGraphSeries("metric_b", (1.0, 1.5, 2.0)),
            ),
        )
        self.clear_count = 0

    def refresh(self) -> TrainingGraphSnapshot:
        """Return current snapshot."""
        return self.snapshot

    def set_session_id(self, session_id) -> TrainingGraphSnapshot:  # type:ignore[no-untyped-def]
        """Capture session id selection."""
        self.session_id = session_id
        return self.snapshot

    def save_csv(self, filename, *, selected_keys=()):  # type:ignore[no-untyped-def]
        """Save selected series as CSV."""
        series = self.snapshot.series_for_keys(selected_keys)
        if not series:
            return 0
        max_count = max(item.count for item in series)
        with Path(filename).open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(
                outfile, fieldnames=("iteration", *(item.name for item in series))
            )
            writer.writeheader()
            for index in range(max_count):
                writer.writerow(
                    {
                        "iteration": index + 1,
                        **{item.name: item.values[index] for item in series if index < item.count},
                    }
                )
        return max_count

    def clear(self) -> None:
        """Clear state."""
        self.clear_count += 1
        self.snapshot = TrainingGraphSnapshot(None, None, ())


def _combo(panel: GraphPanel, name: str) -> QComboBox:
    """Return a named combo box."""
    combo = panel.findChild(QComboBox, f"qt-shell-graph-{name}")
    assert combo is not None
    return combo


def _label(panel: GraphPanel, name: str) -> QLabel:
    """Return a named label."""
    label = panel.findChild(QLabel, f"qt-shell-graph-{name}")
    assert label is not None
    return label


def _button(panel: GraphPanel, name: str) -> QPushButton:
    """Return a named button."""
    button = panel.findChild(QPushButton, f"qt-shell-graph-{name}")
    assert button is not None
    return button


def _widget(panel: GraphPanel) -> TrainingGraphWidget:
    """Return the graph widget."""
    widget = panel.findChild(TrainingGraphWidget, "qt-shell-training-graph-widget")
    assert widget is not None
    return widget


def test_graph_panel_renders_chart_and_populates_loss_keys(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Refreshing should render chart data and populate the loss-key selector."""
    panel = GraphPanel(service=_GraphServiceDouble())  # type:ignore[arg-type]
    qtbot.addWidget(panel)

    assert panel.refresh_graph() is True

    key_combo = _combo(panel, "key")
    assert [key_combo.itemText(index) for index in range(key_combo.count())] == [
        "All losses",
        "metric_a",
        "metric_b",
    ]
    assert _widget(panel).status_text == "Rendered 2 series, 3 points: metric_a, metric_b"
    assert _label(panel, "status").text() == "Rendered 2 series, 3 points: metric_a, metric_b"
    assert _button(panel, "export-image").isEnabled() is True
    assert _button(panel, "export-csv").isEnabled() is True
    assert _button(panel, "y-zoom-in").isEnabled() is True
    assert _button(panel, "y-zoom-out").isEnabled() is True


def test_graph_panel_loss_key_selection_filters_chart(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Changing the loss-key selector should filter the rendered chart."""
    panel = GraphPanel(service=_GraphServiceDouble())  # type:ignore[arg-type]
    qtbot.addWidget(panel)
    assert panel.refresh_graph() is True
    key_combo = _combo(panel, "key")

    key_combo.setCurrentIndex(2)

    assert [series.name for series in _widget(panel).series] == ["metric_b"]
    assert _label(panel, "status").text() == "Rendered 1 series, 3 points: metric_b"


def test_graph_panel_save_image_uses_chart_widget(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Saving should delegate to the chart widget and update status."""
    panel = GraphPanel(service=_GraphServiceDouble())  # type:ignore[arg-type]
    qtbot.addWidget(panel)
    assert panel.refresh_graph() is True
    filename = tmp_path / "chart.png"

    assert panel.save_graph_image(filename) is True

    assert filename.exists()
    assert _label(panel, "status").text() == "Graph image saved"


def test_graph_panel_save_csv_uses_selected_loss_keys(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """CSV export should write the currently selected graph series only."""
    panel = GraphPanel(service=_GraphServiceDouble())  # type:ignore[arg-type]
    qtbot.addWidget(panel)
    assert panel.refresh_graph() is True
    _combo(panel, "key").setCurrentText("metric_b")
    filename = tmp_path / "chart.csv"

    written = panel.save_graph_csv(filename)

    assert written == 3
    assert filename.read_text(encoding="utf-8").splitlines()[0] == "iteration,metric_b"
    assert _label(panel, "status").text() == "Graph CSV saved: 3 rows"


def test_graph_panel_clear_resets_renderer_controls(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Clear should reset chart, key selector and export buttons."""
    service = _GraphServiceDouble()
    panel = GraphPanel(service=service)  # type:ignore[arg-type]
    qtbot.addWidget(panel)
    assert panel.refresh_graph() is True

    panel.cleanup_graph("Graph cleared after reload")

    assert service.clear_count == 1
    assert _combo(panel, "key").count() == 0
    assert _widget(panel).series == ()
    assert _button(panel, "export-image").isEnabled() is False
    assert _button(panel, "export-csv").isEnabled() is False
    assert _label(panel, "status").text() == "Graph cleared after reload"
