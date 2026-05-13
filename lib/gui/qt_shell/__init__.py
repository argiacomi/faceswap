#!/usr/bin/env python3
"""Prototype Qt shell for validating the Faceswap GUI service layer."""

from lib.gui.qt_shell.main import main
from lib.gui.qt_shell.main_window import MainWindow
from lib.gui.qt_shell.preview_job_lifecycle import install_preview_job_lifecycle as _install_hooks

_install_hooks(MainWindow)

__all__ = ["MainWindow", "main"]
