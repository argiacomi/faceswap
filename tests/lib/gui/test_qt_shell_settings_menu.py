#!/usr/bin/env python3
"""Tests for Qt Settings menu parity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QMainWindow  # noqa:E402

from lib.gui.qt_shell.settings_menu import (  # noqa:E402
    CONFIGURE_SETTINGS_LABEL,
    SETTINGS_MENU_LABEL,
    install_settings_menu,
)


class _WindowDouble(QMainWindow):
    """Small MainWindow-like class for menu adapter tests."""

    def __init__(self) -> None:
        super().__init__()
        self._menus = []
        self.build_count = 0

    def _build_menus(self) -> None:
        """Build the pre-existing Qt shell menus."""
        self.build_count += 1
        menu_bar = self.menuBar()
        for label in ("Project", "Command", "View"):
            menu = menu_bar.addMenu(label)
            self._menus.append(menu)


def _menu_labels(window: QMainWindow) -> list[str]:
    """Return top-level menu labels without accelerator markers."""
    return [action.text().replace("&", "") for action in window.menuBar().actions()]


def _settings_menu(window: QMainWindow):  # type:ignore[no-untyped-def]
    """Return the Settings menu."""
    for action in window.menuBar().actions():
        if action.text().replace("&", "") == SETTINGS_MENU_LABEL:
            return action.menu()
    raise AssertionError("Settings menu not found")


def test_settings_menu_is_inserted_between_project_and_command(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Qt top-level menu order should match Tk File/Settings/Help placement intent."""
    Window = type("Window", (_WindowDouble,), {})
    install_settings_menu(Window)
    window = Window()
    qtbot.addWidget(window)

    window._build_menus()

    assert _menu_labels(window) == ["Project", "Settings", "Command", "View"]
    assert window.build_count == 1


def test_settings_menu_contains_configure_settings_action(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Settings menu should expose the Tk Configure Settings entry point."""
    Window = type("Window", (_WindowDouble,), {})
    install_settings_menu(Window)
    window = Window()
    qtbot.addWidget(window)
    window._build_menus()

    settings_menu = _settings_menu(window)
    actions = [action for action in settings_menu.actions() if not action.isSeparator()]

    assert [action.text() for action in actions] == [CONFIGURE_SETTINGS_LABEL]
    assert actions[0].objectName() == "qt-shell-settings-configure"
    assert actions[0].toolTip() == CONFIGURE_SETTINGS_LABEL


def test_settings_menu_action_routes_to_dialog(monkeypatch, qtbot) -> None:  # type:ignore[no-untyped-def]
    """Configure Settings should open and reuse the Qt settings dialog."""
    import lib.gui.qt_shell.settings_menu as settings_menu_module

    created = []

    class _DialogDouble:
        def __init__(self, section=None, parent=None) -> None:  # type:ignore[no-untyped-def]
            self.section = section
            self.parent = parent
            self.show_count = 0
            self.raise_count = 0
            self.activate_count = 0
            self.selected = []
            created.append(self)

        def show(self) -> None:
            self.show_count += 1

        def raise_(self) -> None:
            self.raise_count += 1

        def activateWindow(self) -> None:  # noqa:N802
            self.activate_count += 1

        def _select_initial_section(self, section: str) -> None:
            self.selected.append(section)

    monkeypatch.setattr(settings_menu_module, "SettingsDialog", _DialogDouble)
    Window = type("Window", (_WindowDouble,), {})
    install_settings_menu(Window)
    window = Window()
    qtbot.addWidget(window)
    window._build_menus()
    action = _settings_menu(window).actions()[0]

    action.trigger()
    dialog = window._open_settings_dialog("train")

    assert len(created) == 1
    assert created[0].parent is window
    assert created[0].show_count == 2
    assert created[0].raise_count == 2
    assert created[0].activate_count == 2
    assert created[0].selected == ["train"]
    assert dialog is created[0]


def test_settings_menu_installation_is_idempotent(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Installing the adapter twice should not duplicate Settings menu wrapping."""
    Window = type("Window", (_WindowDouble,), {})
    install_settings_menu(Window)
    wrapped = Window._build_menus
    install_settings_menu(Window)
    window = Window()
    qtbot.addWidget(window)

    window._build_menus()

    assert Window._build_menus is wrapped
    assert _menu_labels(window).count(SETTINGS_MENU_LABEL) == 1


def test_qt_package_import_installs_settings_menu() -> None:
    """Importing MainWindow from the Qt package should install the Settings adapter."""
    from lib.gui.qt_shell import MainWindow

    assert getattr(MainWindow, "_settings_menu_installed", False) is True
