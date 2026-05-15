#!/usr/bin/env python3
"""Tests for the native Qt training graph widget."""

from __future__ import annotations

from pathlib import Path

from lib.gui.qt_shell.graph_widget import TrainingGraphWidget
from lib.gui.services.training_graph_service import (
    TrainingGraphSeries,
    TrainingGraphSnapshot,
)


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


def _long_snapshot(count: int = 100_000) -> TrainingGraphSnapshot:
    """Return a large graph snapshot for long-history renderer checks."""
    return TrainingGraphSnapshot(
        source=None,
        session_id=None,
        series=(TrainingGraphSeries("loss", tuple(float(index % 100) for index in range(count))),),
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
    widget.zoom_in()
    widget.zoom_y_in()

    widget.clear()

    assert widget.series == ()
    assert widget.selected_keys == ()
    assert widget.status_text == "No graph data loaded"
    assert widget.viewport == (1.0, 1.0, 0.0, 0.0)


def test_graph_widget_zoom_and_reset_view(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Graph widget should expose bounded x/y zoom controls."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)
    widget.set_snapshot(_snapshot())

    widget.zoom_in()
    widget.zoom_y_in()

    assert widget.zoom > 1.0
    assert widget.y_zoom > 1.0

    widget.zoom_out()
    widget.zoom_y_out()

    assert widget.zoom == 1.0
    assert widget.y_zoom == 1.0

    widget.zoom_in()
    widget.zoom_y_in()
    widget.reset_view()

    assert widget.viewport == (1.0, 1.0, 0.0, 0.0)


def test_graph_widget_decimates_long_history_for_fast_paint(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Long histories should be capped to a viewport-sized point budget."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 240)
    widget.set_snapshot(_long_snapshot())
    rect = widget.rect().adjusted(44, 16, -16, -28)

    points = widget._points_for_series(  # pylint:disable=protected-access
        widget.series[0],
        0.0,
        99.0,
        rect,
    )

    assert len(points) <= rect.width() * widget.MAX_POINTS_PER_PIXEL
    assert widget.last_decimated_count == len(points)


def test_graph_widget_visible_values_respect_x_zoom_and_pan(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Zoom/pan should expose a bounded subsection of the graph history."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)
    widget.set_snapshot(_long_snapshot(100))

    widget.zoom_in()
    widget._set_pan(1.0)  # pylint:disable=protected-access

    visible = widget._visible_values(widget.series[0].values)  # pylint:disable=protected-access

    assert 2 <= len(visible) < 100
    assert visible[-1] == 99.0


def test_graph_widget_saves_image_when_data_loaded(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Saving should create an image when graph data exists."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 240)
    widget.set_snapshot(_snapshot())
    filename = tmp_path / "chart.png"

    assert widget.save_image(filename) is True
    assert filename.exists()


def test_graph_widget_saves_jpeg_when_requested(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Image export should infer common formats from filename suffixes."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)
    widget.resize(320, 240)
    widget.set_snapshot(_snapshot())
    filename = tmp_path / "chart.jpg"

    assert widget.save_image(filename) is True
    assert filename.exists()


def test_graph_widget_does_not_save_empty_image(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Saving should fail cleanly when no graph data is rendered."""
    widget = TrainingGraphWidget()
    qtbot.addWidget(widget)

    assert widget.save_image(tmp_path / "chart.png") is False
