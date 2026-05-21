#!/usr/bin/env python3
"""Stdout/stderr and logging routers for the Qt shell console.

Mirrors :class:`lib.gui.custom_widgets._SysOutRouter` from the Tk shell so any
``print`` call or Python ``logging`` record made by the GUI process surfaces in
the Qt console pane with the same level-aware colorization the Tk console uses.

The router is a :class:`QObject` so it can ``emit`` to the GUI thread via a
``Qt.QueuedConnection`` signal — Qt widgets must not be touched from worker
threads, and subprocess readers in :mod:`lib.gui.qt_shell.job_runner` already
publish on background threads.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sys
import typing as T

from PySide6.QtCore import QObject, Qt, Signal

from lib.utils import get_module_objects

if T.TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from lib.gui.qt_shell.main_window import ConsolePane

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_LOG_LEVEL_RE = re.compile(r".+?(\s\d+:\d+:\d+\s)(?P<lvl>[A-Z]+)\s")
_KNOWN_TAGS = {"info", "verbose", "warning", "error", "critical", "stdout", "stderr"}


def _tag_for(stream: str, line: str) -> str:
    """Return the Tk-parity colour tag for one console line."""
    if stream == "stderr":
        return "stderr"
    match = _LOG_LEVEL_RE.match(line)
    if not match:
        return "stdout"
    tag = match.groupdict()["lvl"].strip().lower()
    return tag if tag in _KNOWN_TAGS else "stdout"


class QtConsoleRouter(QObject):
    """Thread-safe text router that pushes to a :class:`ConsolePane`.

    Used both as a drop-in replacement for ``sys.stdout``/``sys.stderr`` and as
    a transport for subprocess reader threads. ``write`` is a file-like API so
    callers don't need to know about the underlying signal.

    Parameters
    ----------
    console : ConsolePane
        Destination console widget. Must live on the Qt main thread.
    stream : str
        ``"stdout"`` or ``"stderr"``; controls default colour tagging.
    """

    text_ready = Signal(str, str)  # (text, stream)

    def __init__(self, console: ConsolePane, stream: str) -> None:
        super().__init__(console)
        self._console = console
        self._stream = stream if stream in ("stdout", "stderr") else "stdout"
        self.text_ready.connect(self._deliver, Qt.QueuedConnection)

    def write(self, text: str) -> int:
        """File-like write: emit the chunk for delivery on the GUI thread."""
        if not text:
            return 0
        cleaned = _ANSI_ESCAPE_RE.sub("", text)
        self.text_ready.emit(cleaned, self._stream)
        return len(text)

    @staticmethod
    def flush() -> None:
        """File-like flush — pass through to the real terminal stdout."""
        with contextlib.suppress(AttributeError, OSError):
            sys.__stdout__.flush()

    @staticmethod
    def isatty() -> bool:
        """The Qt console is never an interactive terminal."""
        return False

    def _deliver(self, text: str, stream: str) -> None:
        """Forward queued text to the console on the GUI thread."""
        self._console.write(text, tag=_tag_for(stream, text))


class QtConsoleLogHandler(logging.Handler):
    """Logging handler that pipes records into a :class:`ConsolePane`.

    Mirrors the Tk stream handler (``%(asctime)s %(levelname)-8s %(message)s``)
    so :func:`_tag_for` can pick the colour tag back out from the formatted
    line.
    """

    def __init__(self, console: ConsolePane, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._console = console
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # noqa:BLE001 - logging must never raise
            self.handleError(record)
            return
        tag = _tag_for("stdout", message)
        # ConsolePane.write must be called from the GUI thread; logging records
        # may originate anywhere, so route through the router signal whenever a
        # router is installed on the console.
        router: QtConsoleRouter | None = getattr(self._console, "_log_router", None)
        if router is not None:
            router.text_ready.emit(message + "\n", "stdout")
            return
        self._console.write(message + "\n", tag=tag)


def install_console_routers(console: ConsolePane) -> tuple[QtConsoleRouter, QtConsoleRouter]:
    """Replace ``sys.stdout`` / ``sys.stderr`` with Qt console routers.

    Stores the originals on the router objects so :func:`restore_console_routers`
    can undo the redirection cleanly (e.g. when the Qt shell shuts down).
    """
    stdout_router = QtConsoleRouter(console, "stdout")
    stderr_router = QtConsoleRouter(console, "stderr")
    stdout_router._original_stream = sys.stdout  # type:ignore[attr-defined]
    stderr_router._original_stream = sys.stderr  # type:ignore[attr-defined]
    sys.stdout = stdout_router  # type:ignore[assignment]
    sys.stderr = stderr_router  # type:ignore[assignment]
    console._log_router = stdout_router  # type:ignore[attr-defined]
    return stdout_router, stderr_router


def restore_console_routers(
    stdout_router: QtConsoleRouter, stderr_router: QtConsoleRouter
) -> None:
    """Undo :func:`install_console_routers`, restoring the original streams."""
    original_stdout = getattr(stdout_router, "_original_stream", sys.__stdout__)
    original_stderr = getattr(stderr_router, "_original_stream", sys.__stderr__)
    if sys.stdout is stdout_router:
        sys.stdout = original_stdout
    if sys.stderr is stderr_router:
        sys.stderr = original_stderr


__all__ = get_module_objects(__name__)
