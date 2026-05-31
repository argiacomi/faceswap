#!/usr/bin/env python3
"""Qt right display panel tests."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QSplitter, QTableWidget, QTabWidget


@dataclass(frozen=True)
class _RuntimeEvent:
    """RuntimeEvent-shaped test double."""

    kind: str
    message: str = ""
    progress: float | None = None
    payload: dict[str, object] | None = None


def _main_window(monkeypatch, tmp_path):
    """Return a MainWindow with a small deterministic schema."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (OptionSpec("Input", "-i"),),
            ),
        )
    )
    return MainWindow(schema)


def _display_tabs(window):
    """Return the Qt shell display tab widget."""
    tabs = window.findChild(QTabWidget, "qt-shell-display-tabs")
    assert tabs is not None
    return tabs


def _view_action(window, label: str):
    """Return a persisted View menu action by label."""
    return window._view_actions[label]  # pylint:disable=protected-access


def _view_action_enabled(window, label: str) -> bool:
    """Return whether a View menu action is enabled."""
    return _view_action(window, label).isEnabled()  # type: ignore[no-any-return]


def _trigger_view_action(window, label: str) -> None:
    """Trigger a View menu action by label."""
    _view_action(window, label).trigger()


def test_analysis_panel_renders_session_stats_placeholder(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """MainWindow should render the Analysis tab session table skeleton."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)
    table = window.findChild(QTableWidget, "qt-shell-session-stats")
    labels = {label.text() for label in window.findChildren(QLabel)}

    assert table is not None
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
    assert "Session Stats" in labels
    assert "No session data loaded" in labels


def test_analysis_panel_disables_horizontal_scrolling(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """Analysis table should fit the available width instead of scrolling sideways."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)
    table = window.findChild(QTableWidget, "qt-shell-session-stats")

    assert table is not None
    assert table.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff  # type: ignore[attr-defined]
    assert table.minimumWidth() == 0


def test_right_display_panel_is_display_only(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """The large right panel should expose display tabs, not command output."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)
    tabs = _display_tabs(window)

    assert tabs.minimumWidth() == 0
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Analysis",
        "Preview",
        "Graph",
    ]


def test_view_menu_actions_follow_runtime_display_visibility(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """View menu actions should switch only currently visible display tabs."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)
    tabs = _display_tabs(window)

    assert _view_action_enabled(window, "Analysis") is True
    assert _view_action_enabled(window, "Preview") is False
    assert _view_action_enabled(window, "Graph") is False

    window._consume_runtime_event(  # pylint:disable=protected-access
        _RuntimeEvent("runtime", payload={"command": "train", "state": "started"})
    )

    assert _view_action_enabled(window, "Preview") is False
    assert _view_action_enabled(window, "Graph") is True

    _trigger_view_action(window, "Graph")
    assert tabs.tabText(tabs.currentIndex()) == "Graph"

    _trigger_view_action(window, "Analysis")
    assert tabs.tabText(tabs.currentIndex()) == "Analysis"

    window._consume_runtime_event(  # pylint:disable=protected-access
        _RuntimeEvent("process", payload={"command": "train", "state": "finished"})
    )

    assert _view_action_enabled(window, "Preview") is False
    assert _view_action_enabled(window, "Graph") is False
    assert tabs.tabText(tabs.currentIndex()) == "Analysis"


def test_progress_runtime_event_updates_status_and_progress_bar(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """Determinate progress events should update status text and progress value."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)
    progress = window.findChild(QProgressBar, "qt-shell-progress")
    assert progress is not None

    window._set_running(True)  # pylint:disable=protected-access
    window._consume_runtime_event(  # pylint:disable=protected-access
        _RuntimeEvent("progress", "Halfway there", 42.9, {"command": "extract"})
    )

    assert window.statusBar().currentMessage() == "Halfway there"
    assert progress.minimum() == 0
    assert progress.maximum() == 100
    assert progress.value() == 42


def test_status_runtime_event_can_set_indeterminate_progress(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """Runtime status mode events should control busy progress mode."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)
    progress = window.findChild(QProgressBar, "qt-shell-progress")
    assert progress is not None

    window._set_running(True)  # pylint:disable=protected-access
    window._consume_runtime_event(  # pylint:disable=protected-access
        _RuntimeEvent(
            "status",
            "Training progress is indeterminate",
            payload={"command": "train", "mode": "indeterminate"},
        )
    )

    assert window.statusBar().currentMessage() == "Training progress is indeterminate"
    assert progress.minimum() == 0
    assert progress.maximum() == 0


def test_job_finished_resets_progress_and_view_actions(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """Finishing a job should reset progress and hide runtime display actions."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)
    progress = window.findChild(QProgressBar, "qt-shell-progress")
    assert progress is not None

    window._set_running(True)  # pylint:disable=protected-access
    window._consume_runtime_event(  # pylint:disable=protected-access
        _RuntimeEvent("runtime", payload={"command": "train", "state": "started"})
    )
    window._consume_runtime_event(  # pylint:disable=protected-access
        _RuntimeEvent("progress", "Nearly done", 99.0, {"command": "train"})
    )

    window._job_finished(0)  # pylint:disable=protected-access

    assert progress.isVisible() is False
    assert progress.minimum() == 0
    assert progress.maximum() == 100
    assert progress.value() == 0
    assert _view_action_enabled(window, "Preview") is False
    assert _view_action_enabled(window, "Graph") is False


def test_main_panels_are_adjustable(
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """Main command/display and top/console panels should be QSplitters."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)

    assert isinstance(window.findChild(QSplitter, "qt-shell-main-splitter"), QSplitter)
    assert isinstance(window.findChild(QSplitter, "qt-shell-vertical-splitter"), QSplitter)
