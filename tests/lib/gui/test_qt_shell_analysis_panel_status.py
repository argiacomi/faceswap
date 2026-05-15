#!/usr/bin/env python3
"""Qt Analysis panel summary status tests."""

from __future__ import annotations

from pathlib import Path

from lib.gui.qt_shell.analysis_panel import AnalysisPanel
from lib.gui.services.analysis_session_service import (
    AnalysisSessionSource,
    AnalysisTableRow,
)


class _AnalysisServiceDouble:
    """Small AnalysisSessionService stand-in for panel tests."""

    TABLE_HEADERS = ("Graphs", "#", "Start", "End", "Elapsed", "Batch", "Iterations", "EGs/sec")

    def __init__(self) -> None:
        self.source = AnalysisSessionSource(
            state_file=Path("/models/model_state.json"),
            model_dir=Path("/models"),
            model_name="model",
        )
        self.is_loaded = False
        self.is_training = False
        self.table_rows: tuple[AnalysisTableRow, ...] = ()
        self.clear_count = 0

    def load_session(self, _source) -> tuple[AnalysisTableRow, ...]:  # type:ignore[no-untyped-def]
        """Load rows for the panel."""
        self.is_loaded = True
        self.table_rows = (
            AnalysisTableRow(True, 1, "start", "end", "1h", 4, "1,000", 12.5),
            AnalysisTableRow(False, 2, "start", "end", "1h", 0, 0, 0),
        )
        return self.table_rows

    def refresh_summaries(self) -> tuple[AnalysisTableRow, ...]:
        """Refresh rows for the panel."""
        return self.table_rows

    def save_csv(self, _filename) -> int:  # type:ignore[no-untyped-def]
        """Return the number of current rows."""
        return len(self.table_rows)

    def clear_session(self) -> None:
        """Clear rows."""
        self.is_loaded = False
        self.table_rows = ()
        self.clear_count += 1


def _label(panel: AnalysisPanel, name: str):  # type:ignore[no-untyped-def]
    """Return a named label."""
    return panel.findChild(type(panel._status_label), f"qt-shell-analysis-{name}")  # pylint:disable=protected-access


def test_analysis_panel_uses_summary_metrics_status(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Loading rows should show graph, iteration and detail summary metrics."""
    service = _AnalysisServiceDouble()
    panel = AnalysisPanel(service=service)  # type:ignore[arg-type]
    qtbot.addWidget(panel)

    assert panel.load_session("ignored") is True

    assert _label(panel, "status").text() == "Loaded session: 2 rows, 1 graph, 1000 iterations"
    assert (
        _label(panel, "detail").text()
        == "Rows: 2 | Graphs: 1 | Iterations: 1000 | Avg EGs/sec: 6.25"
    )
    assert _label(panel, "selection").text() == "No session selected"


def test_analysis_panel_training_status(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Training sessions should be identified in the status text."""
    service = _AnalysisServiceDouble()
    service.is_training = True
    panel = AnalysisPanel(service=service)  # type:ignore[arg-type]
    qtbot.addWidget(panel)

    assert panel.load_session("ignored") is True

    assert _label(panel, "status").text() == "Training session: 2 rows, 1 graph, 1000 iterations"
    assert _label(panel, "source").text() == "Training: model  |  /models"


def test_analysis_panel_loaded_empty_status(qtbot) -> None:  # type:ignore[no-untyped-def]
    """A loaded session with no rows should have a distinct empty status."""
    service = _AnalysisServiceDouble()
    panel = AnalysisPanel(service=service)  # type:ignore[arg-type]
    qtbot.addWidget(panel)
    service.is_loaded = True
    service.table_rows = ()

    assert panel.refresh_session() is True

    assert _label(panel, "status").text() == "Session loaded with no summary rows"
    assert (
        _label(panel, "detail").text() == "Rows: 0 | Graphs: 0 | Iterations: 0 | Avg EGs/sec: 0.00"
    )


def test_analysis_panel_clear_resets_summary_status(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Clear should reset the richer status text to the unloaded state."""
    service = _AnalysisServiceDouble()
    panel = AnalysisPanel(service=service)  # type:ignore[arg-type]
    qtbot.addWidget(panel)
    assert panel.load_session("ignored") is True

    panel.clear_session()

    assert _label(panel, "status").text() == "No session data loaded"
    assert (
        _label(panel, "detail").text() == "Rows: 0 | Graphs: 0 | Iterations: 0 | Avg EGs/sec: 0.00"
    )
    assert _label(panel, "selection").text() == "No session selected"
    assert service.clear_count == 1
