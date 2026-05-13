#!/usr/bin/env python3
"""Runtime process event facade for GUI process output."""

from __future__ import annotations

import typing as T

from lib.utils import get_module_objects

from .command_context import CommandExecutionContext
from .progress_parser import ProgressParser
from .runtime_events import ParsedRuntimeOutput, RuntimeEvent
from .runtime_session_services import (
    PreviewOutputRuntimeService,
    RuntimeTerminationPolicy,
    TrainingSessionRuntimeService,
)

RuntimeEventCallback = T.Callable[[RuntimeEvent], None]
JobEventCallback = RuntimeEventCallback


class ProcessRuntimeService:
    """Emit structured events parsed from a running job's stdout/stderr."""

    def __init__(
        self,
        command: str | None = None,
        *,
        training_session: TrainingSessionRuntimeService | None = None,
        preview_output: PreviewOutputRuntimeService | None = None,
    ) -> None:
        self._command = command
        self._parser = ProgressParser()
        self._training_session = training_session or TrainingSessionRuntimeService()
        self._preview_output = preview_output or PreviewOutputRuntimeService()
        self._event_callbacks: list[RuntimeEventCallback] = []

    @property
    def command(self) -> str | None:
        """The command currently associated with this runner."""
        return self._command

    @command.setter
    def command(self, value: str | None) -> None:
        self._command = value
        self._parser.reset()

    @property
    def training_session(self) -> TrainingSessionRuntimeService:
        """Return the training-session service used for saved-model runtime events."""
        return self._training_session

    @property
    def preview_output(self) -> PreviewOutputRuntimeService:
        """Return the preview-output side-effect service for this runtime."""
        return self._preview_output

    def configure_context(self, context: CommandExecutionContext) -> tuple[RuntimeEvent, ...]:
        """Configure shared runtime services with command-derived context."""
        self._training_session.configure(context)
        try:
            preview_configured = self._preview_output.apply_context(context)
        except (AssertionError, ImportError, ModuleNotFoundError):
            preview_configured = context.preview_output_path is not None
        events: tuple[RuntimeEvent, ...] = ()
        if preview_configured:
            events = (
                RuntimeEvent(
                    kind="preview_output",
                    message="Preview output configured",
                    payload={
                        "preview_output": True,
                        "preview_refresh": True,
                        "preview_output_path": context.preview_output_path or "",
                        "batch_mode": context.batch_mode,
                    },
                ),
            )
            self._emit_all(events)
        return events

    def add_event_callback(self, callback: RuntimeEventCallback) -> None:
        """Register a callback for emitted runtime events."""
        self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: RuntimeEventCallback) -> None:
        """Remove a previously registered callback."""
        self._event_callbacks.remove(callback)

    def process_stdout(self, output: str) -> ParsedRuntimeOutput:
        """Parse one stdout line and emit any structured events."""
        parsed = self._parser.parse_stdout(self._command, output)
        session_events = self._training_session.events_for_stdout(self._command, output)
        self._emit_all((*parsed.events, *session_events))
        if session_events:
            return ParsedRuntimeOutput(
                events=(*parsed.events, *session_events), consumed=parsed.consumed
            )
        return parsed

    def process_stderr(self, output: str) -> ParsedRuntimeOutput:
        """Parse one stderr line and emit any structured events."""
        parsed = self._parser.parse_stderr(self._command, output)
        self._emit_all(parsed.events)
        return parsed

    def finish(self, returncode: int) -> RuntimeEvent:
        """Emit the final status event for the job and clean runtime side effects."""
        event = self._parser.final_status(self._command, returncode)
        if RuntimeTerminationPolicy.normalize_command(self._command) == "train":
            self._training_session.stop()
        self._preview_output.cleanup()
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
