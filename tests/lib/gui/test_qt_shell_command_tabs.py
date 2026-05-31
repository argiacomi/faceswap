#!/usr/bin/env python3
"""Qt command tab selection tests."""

from __future__ import annotations

from PySide6.QtWidgets import QTabBar


def _panel():
    """Return a CommandPanel with Faceswap and tool commands."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec("faceswap", "extract", (OptionSpec("Input", "-i"),)),
            CommandSpec("faceswap", "train", (OptionSpec("Model", "-m"),)),
            CommandSpec("faceswap", "convert", (OptionSpec("Output", "-o"),)),
            CommandSpec("tools", "alignments", (OptionSpec("Input", "-i"),)),
            CommandSpec("tools", "sort", (OptionSpec("Output", "-o"),)),
        )
    )
    return CommandPanel(schema)


def _find_tab(tabs: QTabBar, label: str) -> int:
    """Return the index for a visible tab label."""
    for index in range(tabs.count()):
        if tabs.tabText(index) == label:
            return index
    raise AssertionError(f"Tab not found: {label}")


def test_faceswap_tab_selection_updates_command(qtbot) -> None:
    """Selecting primary Faceswap tabs should update category and command."""
    panel = _panel()
    qtbot.addWidget(panel)
    primary_tabs = panel.findChild(QTabBar, "qt-shell-command-tabs")
    tool_tabs = panel.findChild(QTabBar, "qt-shell-tool-tabs")
    assert primary_tabs is not None
    assert tool_tabs is not None

    primary_tabs.setCurrentIndex(_find_tab(primary_tabs, "Train"))

    assert panel.command_spec()[0:2] == ("faceswap", "train")
    assert tool_tabs.isHidden() is True

    primary_tabs.setCurrentIndex(_find_tab(primary_tabs, "Convert"))

    assert panel.command_spec()[0:2] == ("faceswap", "convert")
    assert tool_tabs.isHidden() is True


def test_tool_tab_selection_updates_command(qtbot) -> None:
    """Selecting Tools and then a tool tab should update category and command."""
    panel = _panel()
    qtbot.addWidget(panel)
    primary_tabs = panel.findChild(QTabBar, "qt-shell-command-tabs")
    tool_tabs = panel.findChild(QTabBar, "qt-shell-tool-tabs")
    assert primary_tabs is not None
    assert tool_tabs is not None

    primary_tabs.setCurrentIndex(_find_tab(primary_tabs, "Tools"))

    assert tool_tabs.isHidden() is False
    assert panel.command_spec()[0:2] == ("tools", "alignments")

    tool_tabs.setCurrentIndex(_find_tab(tool_tabs, "Sort"))

    assert panel.command_spec()[0:2] == ("tools", "sort")


def test_set_command_selects_matching_faceswap_or_tool_tab(qtbot) -> None:
    """Programmatic command selection should keep the visible tabs in sync."""
    panel = _panel()
    qtbot.addWidget(panel)
    primary_tabs = panel.findChild(QTabBar, "qt-shell-command-tabs")
    tool_tabs = panel.findChild(QTabBar, "qt-shell-tool-tabs")
    assert primary_tabs is not None
    assert tool_tabs is not None

    panel.set_command("sort", {"-o": "/sorted"})

    assert primary_tabs.tabText(primary_tabs.currentIndex()) == "Tools"
    assert tool_tabs.tabText(tool_tabs.currentIndex()) == "Sort"
    assert tool_tabs.isHidden() is False
    assert panel.command_spec() == ("tools", "sort", {"-o": "/sorted"})

    panel.set_command("extract", {"-i": "/input"})

    assert primary_tabs.tabText(primary_tabs.currentIndex()) == "Extract"
    assert tool_tabs.isHidden() is True
    assert panel.command_spec() == ("faceswap", "extract", {"-i": "/input"})
