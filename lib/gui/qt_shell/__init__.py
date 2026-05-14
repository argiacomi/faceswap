#!/usr/bin/env python3
"""Prototype Qt shell for validating the Faceswap GUI service layer."""

from lib.gui.qt_shell.command_option_state import install_command_option_state as _install_command_option_state
from lib.gui.qt_shell.command_panel import CommandPanel
from lib.gui.qt_shell.main import main
from lib.gui.qt_shell.main_window import MainWindow

_install_command_option_state(CommandPanel)

__all__ = ["MainWindow", "main"]
