#!/usr/bin/env python3
"""Shared Qt GUI test helpers.

These helpers replace the small ``_command_panel``/``_main_window``/``_option_spec``
factories that were copy-pasted across many of the Qt test files.  Centralising
them keeps the tests focused on the behaviour they assert and gives one obvious
place to extend the setup if the underlying schema/widget APIs change.

Imports of ``lib.gui.qt_shell`` and ``PySide6`` are kept lazy so test modules
that only need ``option_spec`` do not pull in the whole Qt shell at collection
time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtWidgets import (
        QLabel,
        QListWidget,
        QMainWindow,
        QPushButton,
        QToolBar,
        QWidget,
    )

    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow


# ---------------------------------------------------------------------------
# Schema factories
# ---------------------------------------------------------------------------


def option_spec(*args: Any, **kwargs: Any) -> OptionSpec:
    """Build an ``OptionSpec`` without import-time Qt dependencies."""
    from lib.gui.qt_shell.command_schema import OptionSpec

    return OptionSpec(*args, **kwargs)


def command_spec(
    command: str = "extract",
    *options: OptionSpec,
    program: str = "faceswap",
) -> CommandSpec:
    """Build a single-command ``CommandSpec``."""
    from lib.gui.qt_shell.command_schema import CommandSpec

    return CommandSpec(program, command, tuple(options))


def command_schema(*commands: CommandSpec) -> CommandSchema:
    """Wrap ``CommandSpec`` entries in a ``CommandSchema``."""
    from lib.gui.qt_shell.command_schema import CommandSchema

    return CommandSchema(tuple(commands))


def default_schema() -> CommandSchema:
    """Return the deterministic 3-command schema used by toolbar/window tests."""
    return command_schema(
        command_spec("extract", option_spec("Input", "-i")),
        command_spec("train", option_spec("Model", "-m")),
        command_spec("convert", option_spec("Output", "-o")),
    )


# ---------------------------------------------------------------------------
# Widget factories
# ---------------------------------------------------------------------------


def command_panel(qtbot, *options: OptionSpec, command: str = "extract") -> CommandPanel:
    """Build and register a ``CommandPanel`` backed by a single-command schema."""
    from lib.gui.qt_shell.command_panel import CommandPanel

    panel = CommandPanel(command_schema(command_spec(command, *options)))
    qtbot.addWidget(panel)
    return panel


def main_window(
    qtbot,
    monkeypatch,
    tmp_path: Path,
    schema: CommandSchema | None = None,
) -> MainWindow:
    """Build a registered ``MainWindow`` with a stable HOME and schema."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.main_window import MainWindow

    window = MainWindow(schema or default_schema())
    qtbot.addWidget(window)
    return window


# ---------------------------------------------------------------------------
# Widget lookups
# ---------------------------------------------------------------------------


def toolbar(window: QMainWindow, name: str = "qt-shell-toolbar") -> QToolBar:
    """Return the named toolbar from ``window``; assert it exists."""
    from PySide6.QtWidgets import QToolBar

    found = window.findChild(QToolBar, name)
    assert found is not None, f"Toolbar {name!r} not found"
    return found


def button(parent: QWidget, name: str) -> QPushButton:
    """Return the named ``QPushButton`` child; assert it exists."""
    from PySide6.QtWidgets import QPushButton

    found = parent.findChild(QPushButton, name)
    assert found is not None, f"Button {name!r} not found"
    return found


def label(parent: QWidget, name: str) -> QLabel:
    """Return the named ``QLabel`` child; assert it exists."""
    from PySide6.QtWidgets import QLabel

    found = parent.findChild(QLabel, name)
    assert found is not None, f"Label {name!r} not found"
    return found


def list_widget(parent: QWidget, name: str) -> QListWidget:
    """Return the named ``QListWidget`` child; assert it exists."""
    from PySide6.QtWidgets import QListWidget

    found = parent.findChild(QListWidget, name)
    assert found is not None, f"List widget {name!r} not found"
    return found
