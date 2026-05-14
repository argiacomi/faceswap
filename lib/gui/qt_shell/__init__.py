#!/usr/bin/env python3
"""Prototype Qt shell for validating the Faceswap GUI service layer."""

from lib.gui.qt_shell.main import main
from lib.gui.qt_shell.main_window import MainWindow
from lib.gui.qt_shell.preview_job_lifecycle import install_preview_job_lifecycle as _install_preview_job_lifecycle
from lib.gui.qt_shell.recent_menu import install_recent_menu_display as _install_recent_menu_display

_install_preview_job_lifecycle(MainWindow)
_install_recent_menu_display(MainWindow)

__all__ = ["MainWindow", "main"]
