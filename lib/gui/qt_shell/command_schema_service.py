#!/usr/bin/env python3
"""Adapter from CLI metadata into Qt command schema."""

from __future__ import annotations

import typing as T

from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
from lib.gui.services.command_schema_discovery import (
    CommandSchemaDiscovery,
    DiscoveredCommand,
    DiscoveredCliOption,
)


class CommandSchemaService:
    """Build a Qt command schema from Faceswap CLI option metadata."""

    def from_real_cli_metadata(
        self, categories: T.Iterable[str] | None = None
    ) -> CommandSchema:
        """Build a CommandSchema from real Faceswap CLI metadata."""
        return self.from_discovered_commands(
            CommandSchemaDiscovery().discover(categories=categories)
        )

    def from_discovered_commands(
        self, commands: T.Iterable[DiscoveredCommand]
    ) -> CommandSchema:
        """Build a CommandSchema from GUI-neutral CLI discovery results."""
        return CommandSchema(
            CommandSpec(
                command.category,
                command.command,
                self._options_from_discovered(command.options),
                command.info,
            )
            for command in commands
        )

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

    def _options_from_discovered(
        self, options: T.Iterable[DiscoveredCliOption]
    ) -> tuple[OptionSpec, ...]:
        """Return Qt option specs from discovered CLI metadata."""
        return tuple(
            OptionSpec(
                title=option.title,
                switch=option.opts[0],
                value_type=option.value_type,
                default="" if option.default is None else option.default,
                choices=option.choices,
                nargs=option.nargs is not None,
                action=option.action,
                group=option.group,
                helptext=option.helptext,
                browser_modes=option.browser_modes,
                is_radio=option.is_radio,
                is_multi_option=option.is_multi_option,
                slider_min=option.slider_min,
                slider_max=option.slider_max,
                slider_rounding=option.slider_rounding,
            )
            for option in options
            if option.opts
        )

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
        browser_modes = CommandSchemaService._browser_modes_from_panel(panel_option)
        slider_min, slider_max = CommandSchemaService._slider_min_max(panel_option)
        return OptionSpec(
            title=getattr(panel_option, "title", title),
            switch=str(switches[0]),
            value_type=value_type,
            default="" if default is None else default,
            choices=choices,
            nargs=getattr(option, "nargs", None) is not None,
            group=getattr(panel_option, "group", None),
            helptext=getattr(panel_option, "helptext", "") or "",
            browser_modes=browser_modes,
            is_radio=bool(getattr(panel_option, "is_radio", False)),
            is_multi_option=bool(getattr(panel_option, "is_multi_option", False)),
            slider_min=slider_min,
            slider_max=slider_max,
            slider_rounding=CommandSchemaService._slider_rounding(panel_option),
        )

    @staticmethod
    def _browser_modes_from_panel(panel_option: object) -> tuple[str, ...]:
        """Return browser modes from legacy CliOptions panel metadata."""
        sysbrowser = getattr(panel_option, "sysbrowser", None)
        if not isinstance(sysbrowser, dict):
            return ()
        browser = sysbrowser.get("browser", ())
        if not isinstance(browser, (list, tuple)):
            return ()
        return tuple(str(mode) for mode in browser if str(mode) != "context")

    @staticmethod
    def _slider_min_max(panel_option: object) -> tuple[float | None, float | None]:
        """Return slider bounds from legacy panel metadata."""
        min_max = getattr(panel_option, "min_max", None)
        if not isinstance(min_max, (list, tuple)) or len(min_max) != 2:
            return None, None
        return float(min_max[0]), float(min_max[1])

    @staticmethod
    def _slider_rounding(panel_option: object) -> float | None:
        """Return slider rounding from legacy panel metadata."""
        rounding = getattr(panel_option, "rounding", None)
        if isinstance(rounding, (float, int)):
            return float(rounding)
        return None
