#!/usr/bin/env python3
"""Tests for the native Qt training graph widget."""

from __future__ import annotations

from pathlib import Path

from lib.gui.qt_shell.graph_widget import TrainingGraphWidget
from lib.gui.services.training_graph_service import TrainingGraphSeries, TrainingGraphSnapshot


def _snapshot() -> TrainingGraphSnapshot:
    """Return a graph snapshot with two series."""
    return TrainingGraphSnapshot(
        source=None,
        session_id=None,
        series=(
            TrainingGraphSeries("metric_a", (3.0, 2.0, 1.0)),
            TrainingGraphSeries("metric_b", (1.0, 1.5, 2.0)),
        ),
    )


def test_graph_widget_renders_all_series_by_default(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Graph widget should render every series when no selection is provided."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)

    widget.set_snapshot(_snapshot())

    assert [series.name for series in widget.series] == ["metric_a", "metric_b"]
    assert widget.status_text == "Rendered 2 series, 3 points: metric_a, metric_b"


def test_graph_widget_filters_selected_series(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Graph widget should render only requested series."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)

    widget.set_snapshot(_snapshot(), selected_keys=("metric_b",))

    assert [series.name for series in widget.series] == ["metric_b"]
    assert widget.status_text == "Rendered 1 series, 3 points: metric_b"


def test_graph_widget_handles_empty_data(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Empty snapshots should show empty state."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)

    widget.set_snapshot(TrainingGraphSnapshot(None, None, ()))

    assert widget.series == ()
    assert widget.status_text == "No graph data loaded"


def test_graph_widget_clear_resets_state(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Clear should reset series, selection and status."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)
    widget.set_snapshot(_snapshot())

    widget.clear()

    assert widget.series == ()
    assert widget.selected_keys == ()
    assert widget.status_text == "No graph data loaded"


def test_graph_widget_saves_image_when_data_loaded(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Saving should create an image when graph data exists."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 240)
    widget.set_snapshot(_snapshot())
    filename = tmp_path / "chart.png"

    assert widget.save_image(filename) is True
    assert filename.exists()


def test_graph_widget_does_not_save_empty_image(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Saving should fail cleanly when no graph data is rendered."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)

    assert widget.save_image(tmp_path / "chart.png") is False
