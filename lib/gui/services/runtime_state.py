#!/usr/bin/env python3
"""GUI runtime display state helpers."""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

from lib.utils import get_module_objects

from .runtime_events import RuntimeEvent

ProgressMode = T.Literal["determinate", "indeterminate"]


@dataclass(frozen=True)
class RuntimeDisplayState:
    """Tk/Qt-neutral display state for a process lifecycle transition."""

    running_task: bool
    is_training: bool
    status_message: str
    display: str
    progress_mode: ProgressMode | None = None
    clear_console: bool = False


class DisplayStateService:
    """Build display state values without depending on a GUI toolkit."""

    _INDETERMINATE_START_COMMANDS = frozenset(("effmpeg", "train"))

    def start(self, command: str) -> RuntimeDisplayState:
        """Return the display state for a newly started command."""
        mode: ProgressMode = (
            "indeterminate" if command in self._INDETERMINATE_START_COMMANDS else "determinate"
        )
        return RuntimeDisplayState(
            running_task=True,
            is_training=command == "train",
            clear_console=True,
            status_message=f"Executing - {command}.py",
            progress_mode=mode,
            display=command,
        )

    def finish(self, command: str | None, event: RuntimeEvent) -> RuntimeDisplayState:
        """Return the display state for a completed command."""
        return RuntimeDisplayState(
            running_task=False,
            is_training=False,
            status_message=event.message,
            display="",
            progress_mode=None,
            clear_console=False,
        )


__all__ = get_module_objects(__name__)
