#!/usr/bin/env python3
"""Prototype Qt shell for validating the Faceswap GUI service layer."""

from lib.gui.qt_shell.main import main
from lib.gui.qt_shell.main_window import MainWindow
from lib.gui.qt_shell.settings_menu import install_settings_menu as _install_settings_menu

_install_settings_menu(MainWindow)

__all__ = ["MainWindow", "main"]
