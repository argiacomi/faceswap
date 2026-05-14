#!/usr/bin/env python3
"""Entrypoint for the Qt shell prototype."""

from __future__ import annotations

import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from lib.gui.qt_shell.main_window import MainWindow
from lib.gui.qt_shell.settings_menu import install_settings_menu
from lib.gui.qt_shell.theme import apply_theme


def main(argv: list[str] | None = None) -> int:
    """Run the Qt shell prototype."""
    args = sys.argv if argv is None else argv
    install_settings_menu(MainWindow)
    app = QApplication(args)
    theme = apply_theme(app)
    _install_signal_handlers(app)
    window = MainWindow(theme=theme)
    window.resize(1280, 760)
    window.show()
    return app.exec()


def _install_signal_handlers(app: QApplication) -> None:
    """Let Ctrl-C terminate the Qt event loop immediately."""

    def exit_from_signal(_signum: int, _frame: object) -> None:
        app.exit(130)

    signal.signal(signal.SIGINT, exit_from_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, exit_from_signal)

    timer = QTimer(app)
    timer.timeout.connect(lambda: None)
    timer.start(100)
