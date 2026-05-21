#!/usr/bin/env python3
"""Entrypoint for the Qt shell prototype."""

from __future__ import annotations

import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from lib.gui import gui_config as cfg
from lib.gui.qt_shell.main_window import MainWindow
from lib.gui.qt_shell.theme import apply_theme

INTERRUPT_EXIT_CODE = 130


def main(argv: list[str] | None = None) -> int:
    """Run the Qt shell prototype."""
    args = sys.argv if argv is None else argv
    app = QApplication(args)
    cfg.load_config(None)
    theme = apply_theme(app)
    window = MainWindow(theme=theme)
    install_signal_handlers(app, window)
    window.resize(1280, 760)
    window.show()
    try:
        return app.exec()
    except KeyboardInterrupt:
        interrupt_window(window)
        return INTERRUPT_EXIT_CODE


def install_signal_handlers(app: QApplication, window: MainWindow) -> None:
    """Install SIGINT/SIGTERM handlers that cleanly stop the Qt event loop."""

    def exit_from_signal(_signum: int, _frame: object) -> None:
        interrupt_window(window)
        app.exit(INTERRUPT_EXIT_CODE)

    signal.signal(signal.SIGINT, exit_from_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, exit_from_signal)

    # Qt's C++ event loop blocks Python signal delivery; a no-op timer pumps the
    # interpreter often enough for SIGINT to run our handler.
    timer = QTimer(app)
    timer.setObjectName("qt-shell-signal-pump")
    timer.timeout.connect(lambda: None)
    timer.start(50)
    app.setProperty("qt_shell_signal_timer", timer)


def interrupt_window(window: MainWindow) -> None:
    """Clean up active Qt shell runtime state before an interrupt exit."""
    try:
        window.statusBar().showMessage("Interrupted. Shutting down Qt shell.")
        window._stop_preview_live_refresh()  # pylint:disable=protected-access
        window._runner.stop()  # pylint:disable=protected-access
    except RuntimeError:
        return
