#!/usr/bin/env python3
"""Qt right display panel tests."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter, QTableWidget, QTabWidget


def _main_window(monkeypatch, tmp_path):  # type:ignore[no-untyped-def]
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


def test_analysis_panel_renders_session_stats_placeholder(  # type:ignore[no-untyped-def]
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


def test_analysis_panel_disables_horizontal_scrolling(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """Analysis table should fit the available width instead of scrolling sideways."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)
    table = window.findChild(QTableWidget, "qt-shell-session-stats")

    assert table is not None
    assert table.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert table.minimumWidth() == 0


def test_right_display_panel_is_display_only(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """The large right panel should expose display tabs, not command output."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)
    tabs = window.findChild(QTabWidget, "qt-shell-display-tabs")

    assert tabs is not None
    assert tabs.minimumWidth() == 0
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        "Analysis",
        "Preview",
        "Graph",
    ]


def test_main_panels_are_adjustable(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """Main command/display and top/console panels should be QSplitters."""
    window = _main_window(monkeypatch, tmp_path)
    qtbot.addWidget(window)

    assert isinstance(window.findChild(QSplitter, "qt-shell-main-splitter"), QSplitter)
    assert isinstance(window.findChild(QSplitter, "qt-shell-vertical-splitter"), QSplitter)
