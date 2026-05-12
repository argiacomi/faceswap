#!/usr/bin/env python3
"""Adapter from GUI CLI option metadata into Qt command schema."""

from __future__ import annotations

import typing as T

from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec


class CommandSchemaService:
    """Build a Qt command schema from existing GUI CLI option metadata."""

    def from_cli_options(self, cli_options: T.Any) -> CommandSchema:
        """Build a CommandSchema from a CliOptions-like object.

        This deliberately uses duck typing so the Qt shell can keep this seam without importing
        Tk-heavy GUI option discovery at module import time.
        """
        command_specs: list[CommandSpec] = []
        commands_by_category = getattr(cli_options, "commands", {})
        options_by_command = getattr(cli_options, "opts", {})

        for category in getattr(cli_options, "categories", ()):
            for command in commands_by_category.get(category, ()):
                command_specs.append(
                    CommandSpec(
                        str(category),
                        str(command),
                        self._options_for_command(options_by_command.get(command, {})),
                    )
                )

        return CommandSchema(command_specs)

    def _options_for_command(
        self, options: T.Mapping[str, object]
    ) -> tuple[OptionSpec, ...]:
        """Return Qt option specs from a command's GUI option metadata."""
        option_specs = [
            spec
            for title, option in options.items()
            if (spec := self._option_from_cli(str(title), option)) is not None
        ]
        return tuple(option_specs)

    @staticmethod
    def _option_from_cli(title: str, option: object) -> OptionSpec | None:
        """Translate one CliOption-like object to OptionSpec."""
        panel_option = getattr(option, "panel_option", None)
        switches = getattr(option, "opts", ())
        if panel_option is None or not switches:
            return None

        choices = getattr(panel_option, "choices", None)
        if isinstance(choices, (list, tuple)):
            choices = tuple(str(choice) for choice in choices)
        else:
            choices = ()

        value_type = getattr(panel_option, "dtype", str)
        if not isinstance(value_type, type):
            value_type = str

        default = getattr(panel_option, "default", "")
        return OptionSpec(
            title=getattr(panel_option, "title", title),
            switch=str(switches[0]),
            value_type=value_type,
            default="" if default is None else default,
            choices=choices,
            nargs=getattr(option, "nargs", None) is not None,
        )
