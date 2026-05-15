#!/usr/bin/env python3
"""Qt Analysis runtime panel tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QTableWidget

from lib.gui.services.command_context import CommandExecutionContext


class _SessionDouble:
    """Small legacy Analysis Session test double."""

    def __init__(self, summary: list[dict[str, object]] | None = None) -> None:
        self.is_loaded = False
        self.is_training = False
        self.full_summary = [] if summary is None else summary
        self.initialized: list[tuple[str, str, bool]] = []
        self.clear_count = 0

    def initialize_session(
        self,
        model_folder: str,
        model_name: str,
        is_training: bool = False,
    ) -> None:
        """Capture session initialization calls."""
        self.is_loaded = True
        self.is_training = is_training
        self.initialized.append((model_folder, model_name, is_training))

    def clear(self) -> None:
        """Capture clear calls."""
        self.is_loaded = False
        self.clear_count += 1


def _summary_row(
    session: object = 1,
    *,
    batch: object = 16,
    iterations: object = 100,
    rate: object = "53.3",
) -> dict[str, object]:
    """Return a complete legacy Analysis summary row."""
    return {
        "session": session,
        "start": "01/01/26 00:00:00",
        "end": "01/01/26 00:10:00",
        "elapsed": "00:10:00",
        "batch": batch,
        "iterations": iterations,
        "rate": rate,
    }


def _panel(session: _SessionDouble | None = None):  # type:ignore[no-untyped-def]
    """Return an AnalysisPanel with an injected service."""
    from lib.gui.qt_shell.analysis_panel import AnalysisPanel
    from lib.gui.services.analysis_session_service import AnalysisSessionService

    session = _SessionDouble([_summary_row()]) if session is None else session
    return AnalysisPanel(AnalysisSessionService(session, require_logs=False))


def _state_file(tmp_path: Path, name: str = "model") -> Path:
    """Create a model state file."""
    state_file = tmp_path / f"{name}_state.json"
    state_file.write_text("{}", encoding="utf-8")
    return state_file


def _button(panel, name: str) -> QPushButton:  # type:ignore[no-untyped-def]
    """Return an AnalysisPanel button by object name suffix."""
    button = panel.findChild(QPushButton, f"qt-shell-analysis-{name}")
    assert button is not None
    return button


def _label(panel, name: str) -> QLabel:  # type:ignore[no-untyped-def]
    """Return an AnalysisPanel label by object name suffix."""
    label = panel.findChild(QLabel, f"qt-shell-analysis-{name}")
    assert label is not None
    return label


def _combo(panel, name: str) -> QComboBox:  # type:ignore[no-untyped-def]
    """Return an AnalysisPanel combo by object name suffix."""
    combo = panel.findChild(QComboBox, f"qt-shell-analysis-{name}")
    assert combo is not None
    return combo


def _table(panel) -> QTableWidget:  # type:ignore[no-untyped-def]
    """Return the AnalysisPanel table."""
    table = panel.findChild(QTableWidget, "qt-shell-session-stats")
    assert table is not None
    return table


def test_analysis_panel_initial_state(qtbot) -> None:  # type:ignore[no-untyped-def]
    """AnalysisPanel should start empty with only Open enabled."""
    panel = _panel(_SessionDouble([]))
    qtbot.addWidget(panel)
    table = _table(panel)

    assert table.rowCount() == 0
    assert table.columnCount() == 8
    assert [table.horizontalHeaderItem(idx).text() for idx in range(8)] == [
        "Graphs",
        "#",
        "Start",
        "End",
        "Elapsed",
        "Batch",
        "Iterations",
        "EGs/sec",
    ]
    assert _label(panel, "source").text() == "No session source loaded"
    assert _label(panel, "status").text() == "No session data loaded"
    assert (
        _label(panel, "detail").text() == "Rows: 0 | Graphs: 0 | Iterations: 0 | Avg EGs/sec: 0.00"
    )
    assert _label(panel, "selection").text() == "No session selected"
    assert _button(panel, "open").isEnabled() is True
    assert _button(panel, "refresh").isEnabled() is False
    assert _button(panel, "save").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False
    assert _combo(panel, "filter").isEnabled() is False
    assert _combo(panel, "group").isEnabled() is False


def test_analysis_panel_loads_session_rows(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Loading a source should initialize the service and render table values."""
    session = _SessionDouble(
        [
            _summary_row(1),
            _summary_row("Total", batch="0", iterations=100),
        ]
    )
    panel = _panel(session)
    qtbot.addWidget(panel)
    state_file = _state_file(tmp_path, "my_model")

    loaded = panel.load_session(state_file)
    table = _table(panel)

    assert loaded is True
    assert session.initialized == [(str(tmp_path), "my_model", False)]
    assert table.rowCount() == 2
    assert table.item(0, 0).text() == "✓"
    assert table.item(0, 1).text() == "1"
    assert table.item(0, 5).text() == "16"
    assert table.item(0, 7).text() == "53.3"
    assert table.item(1, 0).text() == ""
    assert table.item(1, 1).data(0x0100) == "total"
    assert _label(panel, "source").text() == f"my_model  |  {tmp_path}"
    assert _label(panel, "status").text() == "Loaded session: 2 rows, 1 graph, 200 iterations"
    assert (
        _label(panel, "detail").text()
        == "Rows: 2 | Graphs: 1 | Iterations: 200 | Avg EGs/sec: 53.30"
    )
    assert _button(panel, "refresh").isEnabled() is True
    assert _button(panel, "save").isEnabled() is True
    assert _button(panel, "clear").isEnabled() is True
    assert _combo(panel, "filter").isEnabled() is True
    assert _combo(panel, "group").isEnabled() is True


def test_analysis_panel_refresh_replaces_rows(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Refresh should pull latest rows from the backing session service."""
    session = _SessionDouble([_summary_row(1)])
    panel = _panel(session)
    qtbot.addWidget(panel)
    assert panel.load_session(_state_file(tmp_path)) is True

    session.full_summary = [_summary_row(2, batch=32, iterations=200)]
    refreshed = panel.refresh_session()
    table = _table(panel)

    assert refreshed is True
    assert table.rowCount() == 1
    assert table.item(0, 1).text() == "2"
    assert table.item(0, 5).text() == "32"
    assert table.item(0, 6).text() == "200"
    assert _label(panel, "status").text() == "Loaded session: 1 row, 1 graph, 200 iterations"


def test_analysis_panel_filters_groups_and_selects_session_rows(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Filter/group controls and row selection should mirror Tk stats tree behavior."""
    session = _SessionDouble(
        [
            _summary_row(2, batch=0, iterations=0, rate="0"),
            _summary_row(1, batch=16, iterations=100, rate="50.0"),
            _summary_row("Total", batch=16, iterations=100, rate="50.0"),
        ]
    )
    panel = _panel(session)
    qtbot.addWidget(panel)
    assert panel.load_session(_state_file(tmp_path)) is True

    _combo(panel, "filter").setCurrentText("Graphable only")
    table = _table(panel)

    assert table.rowCount() == 2
    assert [table.item(row, 1).text() for row in range(table.rowCount())] == ["1", "Total"]
    assert (
        _label(panel, "detail").text()
        == "Rows: 2 | Graphs: 2 | Iterations: 200 | Avg EGs/sec: 50.00"
    )

    _combo(panel, "filter").setCurrentText("All sessions")
    _combo(panel, "group").setCurrentText("Total vs sessions")

    assert [table.item(row, 1).text() for row in range(table.rowCount())] == ["Total", "1", "2"]

    table.setCurrentCell(1, 1)

    assert (
        _label(panel, "selection").text()
        == "Selected session 1: 100 iterations, batch 16, graph available"
    )


def test_analysis_panel_loads_from_model_context(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Training context should auto-attach to the current model folder/name."""
    session = _SessionDouble([_summary_row()])
    panel = _panel(session)
    qtbot.addWidget(panel)
    _state_file(tmp_path, "original")
    context = CommandExecutionContext(model_name="original", model_folder=str(tmp_path))

    with qtbot.waitSignal(panel.session_loaded):
        attached = panel.apply_context(context)

    assert attached is True
    assert session.initialized == [(str(tmp_path), "original", True)]
    assert _label(panel, "source").text() == f"Training: original  |  {tmp_path}"
    assert _label(panel, "status").text() == "Training session: 1 row, 1 graph, 100 iterations"
    assert _button(panel, "save").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False


def test_analysis_panel_saves_csv(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Saving should delegate to AnalysisSessionService and update status."""
    panel = _panel(_SessionDouble([_summary_row()]))
    qtbot.addWidget(panel)
    assert panel.load_session(_state_file(tmp_path)) is True
    output = tmp_path / "summary.csv"

    written = panel.save_csv(output)

    assert written == 1
    assert output.read_text(encoding="utf-8").startswith(
        "batch,elapsed,end,iterations,rate,session,start"
    )
    assert _label(panel, "status").text() == "Saved 1 rows"


def test_analysis_panel_clear_resets_table_and_service(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Clear should reset labels, buttons and service state."""
    session = _SessionDouble([_summary_row()])
    panel = _panel(session)
    qtbot.addWidget(panel)
    assert panel.load_session(_state_file(tmp_path)) is True

    with qtbot.waitSignal(panel.session_cleared):
        panel.clear_session()
    table = _table(panel)

    assert table.rowCount() == 0
    assert session.clear_count == 1
    assert panel.displayed_rows == ()
    assert _label(panel, "source").text() == "No session source loaded"
    assert _label(panel, "status").text() == "No session data loaded"
    assert (
        _label(panel, "detail").text() == "Rows: 0 | Graphs: 0 | Iterations: 0 | Avg EGs/sec: 0.00"
    )
    assert _label(panel, "selection").text() == "No session selected"
    assert _button(panel, "refresh").isEnabled() is False
    assert _button(panel, "save").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False


def test_analysis_panel_cleanup_resets_with_terminal_message(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Terminal cleanup paths should clear state and preserve the terminal reason."""
    session = _SessionDouble([_summary_row()])
    panel = _panel(session)
    qtbot.addWidget(panel)
    assert panel.load_session(_state_file(tmp_path)) is True

    panel.cleanup_session("Analysis cleared after reload")

    assert _table(panel).rowCount() == 0
    assert panel.service.source is None
    assert _label(panel, "status").text() == "Analysis cleared after reload"


def test_analysis_panel_load_failure_displays_error(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Load errors should stay in-panel instead of raising dialogs."""
    panel = _panel(_SessionDouble([]))
    qtbot.addWidget(panel)

    loaded = panel.load_session(tmp_path / "missing_state.json")

    assert loaded is False
    assert "does not exist" in _label(panel, "status").text()
    assert _button(panel, "refresh").isEnabled() is False
    assert _button(panel, "save").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False
