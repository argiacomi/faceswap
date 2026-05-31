#!/usr/bin/env python3
"""Stdout/stderr and logging routers for the Qt shell console."""

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
_LOG_LEVEL_RE = re.compile(r"(?:^|.*?\s)(?P<time>\d{1,2}:\d{2}:\d{2})\s+(?P<lvl>[A-Z]+)\b")
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


def _is_log_handler_output(text: str) -> bool:
    """Return whether ``text`` appears to be formatted logging output."""
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and all(_LOG_LEVEL_RE.match(line) for line in lines)


class QtConsoleRouter(QObject):
    """Thread-safe text router that pushes to a :class:`ConsolePane`."""

    text_ready = Signal(str, str)  # (text, stream)

    def __init__(self, console: ConsolePane, stream: str) -> None:
        super().__init__(console)
        self._console = console
        self._stream = stream if stream in ("stdout", "stderr") else "stdout"
        self.text_ready.connect(self._deliver, Qt.QueuedConnection)  # type: ignore[attr-defined]

    def write(self, text: str) -> int:
        """File-like write: emit the chunk for delivery on the GUI thread."""
        if not text:
            return 0
        cleaned = _ANSI_ESCAPE_RE.sub("", text)
        if bool(
            getattr(self._console, "_qt_console_log_handler_active", False)
        ) and _is_log_handler_output(cleaned):
            return len(text)
        self.text_ready.emit(cleaned, self._stream)
        return len(text)

    @staticmethod
    def flush() -> None:
        """File-like flush: pass through to the real terminal stdout."""
        with contextlib.suppress(AttributeError, OSError):
            sys.__stdout__.flush()  # type: ignore[union-attr]

    @staticmethod
    def isatty() -> bool:
        """The Qt console is never an interactive terminal."""
        return False

    def _deliver(self, text: str, stream: str) -> None:
        """Forward queued text to the console on the GUI thread."""
        self._console.write(text, tag=_tag_for(stream, text))


class QtConsoleLogHandler(logging.Handler):
    """Logging handler that pipes records into a :class:`ConsolePane`."""

    def __init__(self, console: ConsolePane, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._console = console
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%H:%M:%S"))
        self._set_active(True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # noqa:BLE001 - logging must never raise
            self.handleError(record)
            return
        tag = _tag_for("stdout", message)
        router: QtConsoleRouter | None = getattr(self._console, "_log_router", None)
        if router is not None:
            router.text_ready.emit(message + "\n", "stdout")
            return
        self._console.write(message + "\n", tag=tag)

    def close(self) -> None:
        """Mark the direct log route inactive before closing the handler."""
        self._set_active(False)
        super().close()

    def _set_active(self, active: bool) -> None:
        """Set the console flag used by stream routers to avoid duplicated log records."""
        with contextlib.suppress(Exception):
            self._console._qt_console_log_handler_active = active


def install_console_routers(console: ConsolePane) -> tuple[QtConsoleRouter, QtConsoleRouter]:
    """Replace ``sys.stdout`` / ``sys.stderr`` with Qt console routers."""
    stdout_router = QtConsoleRouter(console, "stdout")
    stderr_router = QtConsoleRouter(console, "stderr")
    stdout_router._original_stream = sys.stdout  # type:ignore[attr-defined]
    stderr_router._original_stream = sys.stderr  # type:ignore[attr-defined]
    sys.stdout = stdout_router
    sys.stderr = stderr_router
    console._log_router = stdout_router
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
