#!/usr/bin/env python3
"""Qt Analysis runtime panel tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QPushButton, QTableWidget


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
) -> dict[str, object]:
    """Return a complete legacy Analysis summary row."""
    return {
        "session": session,
        "start": "01/01/26 00:00:00",
        "end": "01/01/26 00:10:00",
        "elapsed": "00:10:00",
        "batch": batch,
        "iterations": iterations,
        "rate": "53.3",
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


def test_analysis_panel_initial_state(qtbot) -> None:  # type:ignore[no-untyped-def]
    """AnalysisPanel should start empty with only Open enabled."""
    panel = _panel(_SessionDouble([]))
    qtbot.addWidget(panel)
    table = panel.findChild(QTableWidget, "qt-shell-session-stats")
    assert table is not None

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
    assert _button(panel, "open").isEnabled() is True
    assert _button(panel, "refresh").isEnabled() is False
    assert _button(panel, "save").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False


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
    table = panel.findChild(QTableWidget, "qt-shell-session-stats")
    assert table is not None

    assert loaded is True
    assert session.initialized == [(str(tmp_path), "my_model", False)]
    assert table.rowCount() == 2
    assert table.item(0, 0).text() == "✓"
    assert table.item(0, 1).text() == "1"
    assert table.item(0, 5).text() == "16"
    assert table.item(0, 7).text() == "53.3"
    assert table.item(1, 0).text() == ""
    assert _label(panel, "source").text() == f"my_model  |  {tmp_path}"
    assert _label(panel, "status").text() == "Loaded 2 session rows"
    assert _button(panel, "refresh").isEnabled() is True
    assert _button(panel, "save").isEnabled() is True
    assert _button(panel, "clear").isEnabled() is True


def test_analysis_panel_refresh_replaces_rows(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Refresh should pull latest rows from the backing session service."""
    session = _SessionDouble([_summary_row(1)])
    panel = _panel(session)
    qtbot.addWidget(panel)
    assert panel.load_session(_state_file(tmp_path)) is True

    session.full_summary = [_summary_row(2, batch=32, iterations=200)]
    refreshed = panel.refresh_session()
    table = panel.findChild(QTableWidget, "qt-shell-session-stats")
    assert table is not None

    assert refreshed is True
    assert table.rowCount() == 1
    assert table.item(0, 1).text() == "2"
    assert table.item(0, 5).text() == "32"
    assert table.item(0, 6).text() == "200"
    assert _label(panel, "status").text() == "Loaded 1 session row"


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

    panel.clear_session()
    table = panel.findChild(QTableWidget, "qt-shell-session-stats")
    assert table is not None

    assert table.rowCount() == 0
    assert session.clear_count == 1
    assert _label(panel, "source").text() == "No session source loaded"
    assert _label(panel, "status").text() == "No session data loaded"
    assert _button(panel, "refresh").isEnabled() is False
    assert _button(panel, "save").isEnabled() is False
    assert _button(panel, "clear").isEnabled() is False


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
