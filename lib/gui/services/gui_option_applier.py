#!/usr/bin/env python3
"""Apply loaded project and task option values to GUI variables."""

from __future__ import annotations

import typing as T

from lib.utils import get_module_objects


class VarLike(T.Protocol):
    """Small protocol for variables that can receive option values."""

    def set(self, value: T.Any) -> None:
        """Set the variable value."""


class CliOptionsLike(T.Protocol):
    """Small protocol for the CLI option lookup used by the applier."""

    def get_one_option_variable(self, command: str, title: str) -> VarLike | None:
        """Return the variable for a command option, if it exists."""


class GuiOptionApplier:
    """Apply loaded project/task values back into GUI option variables."""

    def __init__(
        self,
        cli_options: CliOptionsLike,
        set_active_tab_by_name: T.Callable[[str], None],
    ) -> None:
        self._cli_options = cli_options
        self._set_active_tab_by_name = set_active_tab_by_name

    def apply_project(
        self,
        options: dict[str, str | dict[str, T.Any]],
        *,
        command: str | None = None,
    ) -> bool:
        """Apply a loaded project/task payload to GUI variables."""
        if command is None:
            command_options = {
                key: value for key, value in options.items() if isinstance(value, dict)
            }
            active_tab = self._tab_name(options)
        else:
            value = options.get(command)
            if not isinstance(value, dict):
                return False
            command_options = {command: value}
            active_tab = command

        for cmd, values in command_options.items():
            self.apply_command(cmd, values)

        self._set_active_tab_by_name(active_tab)
        return True

    def apply_command(self, command: str, values: dict[str, T.Any]) -> None:
        """Apply loaded values for a single command."""
        for option_name, option_value in values.items():
            variable = self._cli_options.get_one_option_variable(command, option_name)
            if variable is not None:
                variable.set(option_value)

    @staticmethod
    def _tab_name(options: dict[str, str | dict[str, T.Any]]) -> str:
        """Return the stored tab name or the default extract tab."""
        tab_name = options.get("tab_name")
        return tab_name if isinstance(tab_name, str) and tab_name else "extract"


__all__ = get_module_objects(__name__)
