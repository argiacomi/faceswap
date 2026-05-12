#!/usr/bin/env python3
"""Runtime process event facade for GUI process output."""

from __future__ import annotations

import typing as T

from lib.utils import get_module_objects

from .progress_parser import ProgressParser
from .runtime_events import ParsedRuntimeOutput, RuntimeEvent

RuntimeEventCallback = T.Callable[[RuntimeEvent], None]
JobEventCallback = RuntimeEventCallback


class ProcessRuntimeService:
    """Emit structured events parsed from a running job's stdout/stderr.

    This is intentionally narrow for the first runtime-events pass. Existing process launching and
    console rendering can keep their behavior while callers route process output through this class
    to receive progress/status events.
    """

    def __init__(self, command: str | None = None) -> None:
        self._command = command
        self._parser = ProgressParser()
        self._event_callbacks: list[RuntimeEventCallback] = []

    @property
    def command(self) -> str | None:
        """The command currently associated with this runner."""
        return self._command

    @command.setter
    def command(self, value: str | None) -> None:
        self._command = value
        self._parser.reset()

    def add_event_callback(self, callback: RuntimeEventCallback) -> None:
        """Register a callback for emitted runtime events."""
        self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: RuntimeEventCallback) -> None:
        """Remove a previously registered callback."""
        self._event_callbacks.remove(callback)

    def process_stdout(self, output: str) -> ParsedRuntimeOutput:
        """Parse one stdout line and emit any structured events."""
        parsed = self._parser.parse_stdout(self._command, output)
        self._emit_all(parsed.events)
        return parsed

    def process_stderr(self, output: str) -> ParsedRuntimeOutput:
        """Parse one stderr line and emit any structured events."""
        parsed = self._parser.parse_stderr(self._command, output)
        self._emit_all(parsed.events)
        return parsed

    def finish(self, returncode: int) -> RuntimeEvent:
        """Emit the final status event for the job."""
        event = self._parser.final_status(self._command, returncode)
        self._emit(event)
        return event

    def _emit_all(self, events: T.Iterable[RuntimeEvent]) -> None:
        """Emit multiple events."""
        for event in events:
            self._emit(event)

    def _emit(self, event: RuntimeEvent) -> None:
        """Emit one event to each registered callback."""
        for callback in tuple(self._event_callbacks):
            callback(event)


JobRunner = ProcessRuntimeService

__all__ = get_module_objects(__name__)
