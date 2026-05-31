#!/usr/bin/env python3
"""Optional Qt GUI shell entry point for faceswap."""

from __future__ import annotations

import logging
import os
import sys
import typing as T

from PySide6.QtWidgets import QApplication

from lib.gui import gui_config as cfg
from lib.gui.qt_shell.console_router import (
    QtConsoleLogHandler,
    install_console_routers,
    restore_console_routers,
)
from lib.gui.qt_shell.main import (
    INTERRUPT_EXIT_CODE,
    install_signal_handlers,
    interrupt_window,
)
from lib.gui.qt_shell.main_window import MainWindow
from lib.gui.qt_shell.theme import QtTheme, apply_theme, theme_from_gui_config
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

QT_NO_EXEC_ENV = "FACESWAP_QT_NO_EXEC"
_TRUE_VALUES = {"1", "true", "yes", "on"}


class Gui:
    """Qt GUI process wrapper used by the standard Faceswap launcher."""

    def __init__(self, arguments) -> None:
        self._arguments = arguments
        self._owns_app = QApplication.instance() is None
        self._no_exec = self._resolve_no_exec(arguments)
        self.app = QApplication.instance() or QApplication(sys.argv)
        cfg.load_config(getattr(arguments, "config_file", None))
        self.theme = apply_theme(self.app, theme_from_gui_config(QtTheme.default()))
        self.root = MainWindow(theme=self.theme)
        resize = getattr(self.root, "resize", None)
        if callable(resize):
            resize(1280, 760)
        self._console_routers: tuple[T.Any, T.Any] | None = None
        self._log_handler: QtConsoleLogHandler | None = None
        # Console routing (stdout/stderr swap, root-logger handler) is deferred
        # to process() so that early-return paths (``--no-gui-exec`` smoke runs,
        # callers reusing an existing QApplication) don't leak the redirection.

    def _install_console_logging(self) -> None:
        """Redirect stdout/stderr and attach a console-aware logging handler.

        Mirrors the Tk ``ConsoleOut`` setup so any ``print`` or ``logging`` call
        made by the GUI process surfaces in the Qt console pane. Skipped when
        ``--debug`` is passed (matches Tk behaviour).
        """
        console = getattr(self.root, "console", None)
        if console is None:  # pragma: no cover - defensive
            return
        self._console_routers = install_console_routers(console)
        handler = QtConsoleLogHandler(console)
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

    def _restore_console_logging(self) -> None:
        """Undo :meth:`_install_console_logging`."""
        if self._console_routers is not None:
            restore_console_routers(*self._console_routers)
            self._console_routers = None
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    def process(self) -> None:
        """Show and execute the Qt event loop unless running in smoke-test mode."""
        if self._no_exec:
            return
        self.root.show()
        self.root.apply_gui_settings()
        if not self._owns_app:
            return
        try:
            if not bool(getattr(self._arguments, "debug", False)):
                self._install_console_logging()
            else:
                logger.info("Console debug activated. Outputting to main terminal")
            install_signal_handlers(self.app, self.root)
            self.app.exec()
        except KeyboardInterrupt:
            interrupt_window(self.root)
            sys.exit(INTERRUPT_EXIT_CODE)
        finally:
            self._restore_console_logging()

    @staticmethod
    def _resolve_no_exec(arguments) -> bool:
        """Return whether Qt should skip the event loop for launch smoke tests."""
        if bool(getattr(arguments, "no_gui_exec", False)):
            return True
        return os.environ.get(QT_NO_EXEC_ENV, "").lower() in _TRUE_VALUES


__all__ = get_module_objects(__name__)
