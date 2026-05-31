#!/usr/bin/env python3
"""Tests for Qt command option state preservation."""

from __future__ import annotations

from PySide6.QtWidgets import QLineEdit


def _panel():
    """Return a CommandPanel with multiple simple commands."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec("faceswap", "extract", (OptionSpec("Input", "-i"),)),
            CommandSpec("faceswap", "train", (OptionSpec("Model", "-m"),)),
            CommandSpec("faceswap", "convert", (OptionSpec("Output", "-o"),)),
        )
    )
    return CommandPanel(schema)


def _line_edit(panel, switch: str) -> QLineEdit:
    """Return the line edit for a rendered switch."""
    widget = panel.renderer.widget_for_switch(switch)
    assert isinstance(widget, QLineEdit)
    return widget


def test_command_option_state_preserves_values_across_switches(qtbot) -> None:
    """Switching commands should cache previous command values and restore them later."""
    panel = _panel()
    qtbot.addWidget(panel)

    panel._set_command_options("extract")  # pylint:disable=protected-access
    _line_edit(panel, "-i").setText("/input")
    panel._set_command_options("train")  # pylint:disable=protected-access
    _line_edit(panel, "-m").setText("/models")
    panel._set_command_options("extract")  # pylint:disable=protected-access

    assert panel.command_spec()[2] == {"-i": "/input"}

    panel._set_command_options("train")  # pylint:disable=protected-access

    assert panel.command_spec()[2] == {"-m": "/models"}


def test_command_option_state_caches_programmatic_set_command_values(qtbot) -> None:
    """Project/session command restore should seed the per-command cache."""
    panel = _panel()
    qtbot.addWidget(panel)

    panel.set_command("convert", {"-o": "/output"})
    panel._set_command_options("extract")  # pylint:disable=protected-access
    panel._set_command_options("convert")  # pylint:disable=protected-access

    assert panel.command_spec()[2] == {"-o": "/output"}


def test_command_option_state_clear_values_drops_active_command_cache(qtbot) -> None:
    """Clear should reset the active command and not restore stale cached values."""
    panel = _panel()
    qtbot.addWidget(panel)
    panel.set_command("extract", {"-i": "/input"})

    panel.clear_values()
    panel._set_command_options("train")  # pylint:disable=protected-access
    panel._set_command_options("extract")  # pylint:disable=protected-access

    assert panel.command_spec()[2] == {"-i": ""}
