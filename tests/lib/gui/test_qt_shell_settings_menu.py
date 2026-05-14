#!/usr/bin/env python3
"""Tests for Qt Settings menu parity in MainWindow."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QMainWindow, QMenu  # noqa:E402

SETTINGS_MENU_LABEL = "Settings"
CONFIGURE_SETTINGS_LABEL = "Configure Settings..."


def _main_window(qtbot, monkeypatch, tmp_path: Path):  # type:ignore[no-untyped-def]
    """Return a MainWindow with a deterministic schema."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow

    schema = CommandSchema(
        (
            CommandSpec("faceswap", "extract", (OptionSpec("Input", "-i"),)),
            CommandSpec("faceswap", "train", (OptionSpec("Model", "-m"),)),
        )
    )
    window = MainWindow(schema)
    qtbot.addWidget(window)
    return window


def _menu_labels(window: QMainWindow) -> list[str]:
    """Return top-level menu labels without accelerator markers."""
    return [action.text().replace("&", "") for action in window.menuBar().actions()]


def _settings_menu(window: QMainWindow) -> QMenu:
    """Return the Settings menu."""
    menu = window.findChild(QMenu, "qt-shell-settings-menu")
    assert menu is not None, "Settings menu not found"
    return menu


def test_settings_menu_is_inserted_between_project_and_command(
    qtbot, monkeypatch, tmp_path: Path
) -> None:  # type:ignore[no-untyped-def]
    """Top-level menu order should place Settings between Project and Command."""
    window = _main_window(qtbot, monkeypatch, tmp_path)

    labels = _menu_labels(window)

    assert labels[:3] == ["Project", "Settings", "Command"]


def test_settings_menu_contains_configure_settings_action(
    qtbot, monkeypatch, tmp_path: Path
) -> None:  # type:ignore[no-untyped-def]
    """Settings menu should expose the Configure Settings entry point."""
    window = _main_window(qtbot, monkeypatch, tmp_path)

    settings_menu = _settings_menu(window)
    actions = [action for action in settings_menu.actions() if not action.isSeparator()]

    assert [action.text() for action in actions] == [CONFIGURE_SETTINGS_LABEL]
    assert actions[0].objectName() == "qt-shell-settings-configure"
    assert actions[0].toolTip() == CONFIGURE_SETTINGS_LABEL


def test_settings_menu_action_routes_to_dialog(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Configure Settings should open and reuse the Qt settings dialog."""
    import lib.gui.qt_shell.main_window as main_window_module

    created: list[object] = []

    class _DialogDouble:
        def __init__(self, section=None, parent=None) -> None:  # type:ignore[no-untyped-def]
            self.section = section
            self.parent = parent
            self.show_count = 0
            self.raise_count = 0
            self.activate_count = 0
            self.selected: list[str] = []
            created.append(self)

        def show(self) -> None:
            self.show_count += 1

        def raise_(self) -> None:
            self.raise_count += 1

        def activateWindow(self) -> None:  # noqa:N802
            self.activate_count += 1

        def _select_initial_section(self, section: str) -> None:
            self.selected.append(section)

    monkeypatch.setattr(main_window_module, "SettingsDialog", _DialogDouble)
    window = _main_window(qtbot, monkeypatch, tmp_path)
    action = _settings_menu(window).actions()[0]

    action.trigger()
    dialog = window._open_settings_dialog("train")  # pylint:disable=protected-access

    assert len(created) == 1
    assert created[0].parent is window
    assert created[0].show_count == 2
    assert created[0].raise_count == 2
    assert created[0].activate_count == 2
    assert created[0].selected == ["train"]
    assert dialog is created[0]
