#!/usr/bin/env python3
"""Qt Analysis panel tests."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTableWidget


def test_analysis_panel_renders_session_stats_placeholder(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path,
) -> None:
    """MainWindow should render the Analysis tab session table skeleton."""
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
    window = MainWindow(schema)
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
