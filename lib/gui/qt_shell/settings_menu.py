#!/usr/bin/env python3
"""Settings menu parity adapter for the Qt shell."""

from __future__ import annotations

import functools
import typing as T

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from lib.gui.qt_shell.settings_dialog import SettingsDialog
from lib.utils import get_module_objects


SETTINGS_MENU_LABEL = "Settings"
CONFIGURE_SETTINGS_LABEL = "Configure Settings..."


def install_settings_menu(main_window_class: type) -> None:
    """Install Tk-parity Settings menu behavior on MainWindow."""
    if getattr(main_window_class, "_settings_menu_installed", False):
        return

    original_build_menus = main_window_class._build_menus

    @functools.wraps(original_build_menus)
    def build_menus(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        original_build_menus(self, *args, **kwargs)
        _add_settings_menu(self)
        return None

    main_window_class._build_menus = build_menus
    main_window_class._open_settings_dialog = _open_settings_dialog
    main_window_class._settings_menu_installed = True


def _add_settings_menu(window: object) -> None:
    """Add the Settings menu between Project and Command, matching Tk top-level order."""
    menu_bar = window.menuBar()
    if _menu_action(menu_bar, SETTINGS_MENU_LABEL) is not None:
        return
    settings_menu = QMenu(SETTINGS_MENU_LABEL, menu_bar)
    settings_menu.setObjectName("qt-shell-settings-menu")
    action = QAction(CONFIGURE_SETTINGS_LABEL, settings_menu)
    action.setObjectName("qt-shell-settings-configure")
    action.setToolTip(CONFIGURE_SETTINGS_LABEL)
    action.triggered.connect(lambda _checked=False: window._open_settings_dialog())
    settings_menu.addAction(action)
    before = _menu_action(menu_bar, "Command")
    if before is None:
        menu_bar.addMenu(settings_menu)
    else:
        menu_bar.insertMenu(before, settings_menu)
    menus = getattr(window, "_menus", None)
    if isinstance(menus, list):
        menus.append(settings_menu)
    window._settings_menu = settings_menu
    window._settings_actions = {"configure": action}


def _menu_action(menu_bar: object, label: str) -> QAction | None:
    """Return a top-level menu action by visible text."""
    for action in menu_bar.actions():
        if action.text().replace("&", "") == label:
            return action
    return None


def _open_settings_dialog(self, section: str | None = None) -> SettingsDialog:
    """Open the Qt settings dialog, reusing an existing window when possible."""
    dialog = getattr(self, "_settings_dialog", None)
    if dialog is None:
        dialog = SettingsDialog(section=section, parent=self)
        self._settings_dialog = dialog
    else:
        if section is not None:
            dialog._select_initial_section(section)  # pylint:disable=protected-access
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return T.cast(SettingsDialog, dialog)


__all__ = get_module_objects(__name__)
