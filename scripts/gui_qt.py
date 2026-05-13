#!/usr/bin/env python3
"""Optional Qt GUI shell entry point for faceswap."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from lib.gui.qt_shell.main_window import MainWindow
from lib.utils import get_module_objects


class Gui:
    """Qt GUI process wrapper used by the standard Faceswap launcher."""

    def __init__(self, arguments) -> None:  # type:ignore[no-untyped-def]
        self._arguments = arguments
        self._owns_app = QApplication.instance() is None
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.root = MainWindow()

    def process(self) -> None:
        """Show and execute the Qt event loop."""
        self.root.show()
        if self._owns_app:
            self.app.exec()


__all__ = get_module_objects(__name__)
