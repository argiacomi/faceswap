#!/usr/bin/env python3
"""Entrypoint for the Qt shell prototype."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from lib.gui.qt_shell.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    """Run the Qt shell prototype."""
    args = sys.argv if argv is None else argv
    app = QApplication(args)
    window = MainWindow()
    window.resize(1200, 640)
    window.show()
    return app.exec()
