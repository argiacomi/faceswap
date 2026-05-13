#!/usr/bin/env python3
"""Optional Qt GUI shell entry point for faceswap."""

from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from lib.gui.qt_shell.main_window import MainWindow
from lib.utils import get_module_objects

QT_NO_EXEC_ENV = "FACESWAP_QT_NO_EXEC"
_TRUE_VALUES = {"1", "true", "yes", "on"}


class Gui:
    """Qt GUI process wrapper used by the standard Faceswap launcher."""

    def __init__(self, arguments) -> None:  # type:ignore[no-untyped-def]
        self._arguments = arguments
        self._owns_app = QApplication.instance() is None
        self._no_exec = self._resolve_no_exec(arguments)
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.root = MainWindow()

    def process(self) -> None:
        """Show and execute the Qt event loop unless running in smoke-test mode."""
        if self._no_exec:
            return
        self.root.show()
        if self._owns_app:
            self.app.exec()

    @staticmethod
    def _resolve_no_exec(arguments) -> bool:  # type:ignore[no-untyped-def]
        """Return whether Qt should skip the event loop for launch smoke tests."""
        if bool(getattr(arguments, "no_gui_exec", False)):
            return True
        return os.environ.get(QT_NO_EXEC_ENV, "").lower() in _TRUE_VALUES


__all__ = get_module_objects(__name__)
