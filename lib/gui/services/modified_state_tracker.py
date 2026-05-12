#!/usr/bin/env python3
"""Modified-state adapter for Faceswap GUI project and task tabs."""

from __future__ import annotations

from typing import Protocol

from lib.utils import get_module_objects


class BoolVarLike(Protocol):
    """Small protocol for the tkinter BooleanVar methods we need."""

    def get(self) -> bool:
        """Return the current boolean value."""

    def set(self, value: bool) -> None:
        """Set the current boolean value."""


class ModifiedStateTracker:
    """Small adapter around per-command modified variables."""

    def __init__(self, vars_by_command: dict[str, BoolVarLike]) -> None:
        self._vars = vars_by_command

    def any_modified(self) -> bool:
        """Return ``True`` if any command has been modified."""
        return any(var.get() for var in self._vars.values())

    def is_modified(self, command: str) -> bool:
        """Return ``True`` if the given command has been modified."""
        var = self._vars.get(command)
        return False if var is None else var.get()

    def reset(self, command: str | None = None) -> None:
        """Reset modified state for all commands or one command."""
        for key, var in self._vars.items():
            if command is None or command == key:
                var.set(False)

    def set(self, command: str, modified: bool = True) -> None:
        """Set modified state for a command when it exists."""
        if command in self._vars:
            self._vars[command].set(modified)


__all__ = get_module_objects(__name__)
