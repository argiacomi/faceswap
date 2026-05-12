#!/usr/bin/env python3
"""Tk-free discovery of Faceswap CLI metadata for schema-backed GUIs."""

from __future__ import annotations

import inspect
import os
import typing as T
from argparse import SUPPRESS
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType


@dataclass(frozen=True)
class DiscoveredCliOption:
    """A GUI-neutral option extracted from Faceswap CLI metadata."""

    title: str
    opts: tuple[str, ...]
    value_type: type = str
    default: object = ""
    choices: tuple[str, ...] = ()
    nargs: str | None = None
    group: str | None = None
    helptext: str = ""
    action: str | None = None
    browser_modes: tuple[str, ...] = ()
    is_radio: bool = False
    is_multi_option: bool = False
    slider_min: float | None = None
    slider_max: float | None = None
    slider_rounding: float | None = None


@dataclass(frozen=True)
class DiscoveredCommand:
    """A GUI-neutral command extracted from Faceswap CLI metadata."""

    category: str
    command: str
    info: str
    options: tuple[DiscoveredCliOption, ...]


class CommandSchemaDiscovery:
    """Discover real Faceswap CLI metadata without importing Tk GUI controls."""

    def __init__(self, base_path: str | os.PathLike[str] | None = None) -> None:
        self._base_path = (
            Path(__file__).resolve().parents[3]
            if base_path is None
            else Path(base_path)
        )

    def discover(
        self, categories: T.Iterable[str] | None = None
    ) -> tuple[DiscoveredCommand, ...]:
        """Return discovered commands for the requested categories."""
        wanted_categories = (
            ("faceswap", "tools") if categories is None else tuple(categories)
        )
        commands: list[DiscoveredCommand] = []
        for category in wanted_categories:
            modules = self._get_modules(str(category))
            classes = self._get_all_classes(modules)
            commands.extend(self._commands_for_category(str(category), classes))
        return tuple(commands)

    def _get_modules(self, category: str) -> tuple[ModuleType, ...]:
        """Return CLI metadata modules for a Faceswap category."""
        if category == "tools":
            return self._get_modules_tools()
        if category == "faceswap":
            return self._get_modules_faceswap()
        return ()

    def _get_modules_tools(self) -> tuple[ModuleType, ...]:
        """Return tool CLI modules."""
        tools_dir = self._base_path / "tools"
        if not tools_dir.exists():
            return ()

        modules: list[ModuleType] = []
        for tool_name in sorted(os.listdir(tools_dir)):
            cli_file = tools_dir / tool_name / "cli.py"
            if not cli_file.exists():
                continue
            modules.append(import_module(".".join(("tools", tool_name, "cli"))))
        return tuple(modules)

    def _get_modules_faceswap(self) -> tuple[ModuleType, ...]:
        """Return core Faceswap CLI modules."""
        cli_dir = self._base_path / "lib" / "cli"
        if not cli_dir.exists():
            return ()

        modules: list[ModuleType] = []
        for filename in sorted(os.listdir(cli_dir)):
            if not filename.startswith("args"):
                continue
            module_name = ".".join(("lib", "cli", Path(filename).stem))
            modules.append(import_module(module_name))
        return tuple(modules)

    @classmethod
    def _get_all_classes(
        cls, modules: T.Iterable[ModuleType]
    ) -> tuple[type[T.Any], ...]:
        """Return valid FaceSwapArgs classes from CLI metadata modules."""
        classes: list[type[T.Any]] = []
        for module in modules:
            classes.extend(cls._get_classes(module))
        return tuple(classes)

    @staticmethod
    def _get_classes(module: ModuleType) -> tuple[type[T.Any], ...]:
        """Return command argument classes from one CLI metadata module."""
        classes: list[type[T.Any]] = []
        skipped = {"faceswapargs", "extractconvertargs", "guiargs"}
        for name, obj in inspect.getmembers(module):
            if not inspect.isclass(obj) or not name.lower().endswith("args"):
                continue
            if name.lower() in skipped:
                continue
            classes.append(obj)
        return tuple(classes)

    def _commands_for_category(
        self, category: str, classes: T.Iterable[type[T.Any]]
    ) -> tuple[DiscoveredCommand, ...]:
        """Return ordered command metadata for a category."""
        by_command = {
            self._class_name_to_command(arg_class.__name__): arg_class
            for arg_class in classes
        }
        command_names = sorted(by_command)
        if category == "faceswap":
            workflow_order = ("extract", "train", "convert")
            command_names = [
                *[name for name in workflow_order if name in by_command],
                *[name for name in command_names if name not in workflow_order],
            ]

        discovered: list[DiscoveredCommand] = []
        for command in command_names:
            info, options = self._get_cli_arguments(by_command[command], command)
            discovered.append(
                DiscoveredCommand(
                    category=category,
                    command=command,
                    info=info,
                    options=self._process_options(options),
                )
            )
        return tuple(discovered)

    @staticmethod
    def _class_name_to_command(class_name: str) -> str:
        """Convert a FaceSwapArgs class name to a command name."""
        return class_name.lower()[:-4]

    @staticmethod
    def _get_cli_arguments(
        arg_class: type[T.Any], command: str
    ) -> tuple[str, list[dict[str, T.Any]]]:
        """Extract raw CLI option dictionaries from one argument class."""
        args = T.cast(T.Any, arg_class(None, command))
        arguments = T.cast(
            list[dict[str, T.Any]],
            args.argument_list + args.optional_arguments + args.global_arguments,
        )
        return str(args.info), arguments

    @classmethod
    def _process_options(
        cls, command_options: T.Iterable[dict[str, T.Any]]
    ) -> tuple[DiscoveredCliOption, ...]:
        """Convert raw CLI option dictionaries into GUI-neutral option metadata."""
        options = [
            option
            for cli_option in command_options
            if (option := cls._process_option(cli_option)) is not None
        ]
        return tuple(options)

    @classmethod
    def _process_option(cls, option: dict[str, T.Any]) -> DiscoveredCliOption | None:
        """Convert one raw CLI option dictionary into GUI-neutral metadata."""
        if option.get("help", "") == SUPPRESS:
            return None

        raw_opts = option.get("opts", ())
        if not isinstance(raw_opts, (list, tuple)) or not raw_opts:
            return None
        opts = tuple(str(opt) for opt in raw_opts)
        nargs = option.get("nargs")
        group = option.get("group")
        action = cls._action_name(option.get("action"))
        slider_min, slider_max = cls._slider_min_max(option, action)

        return DiscoveredCliOption(
            title=cls._set_control_title(opts),
            opts=opts,
            value_type=cls._get_data_type(option),
            default=option.get("default", ""),
            choices=cls._get_choices(option.get("choices")),
            nargs=None if nargs is None else str(nargs),
            group=None if group is None else str(group),
            helptext=str(option.get("help", "")),
            action=action,
            browser_modes=cls._browser_modes(action),
            is_radio=action == "Radio",
            is_multi_option=action == "MultiOption",
            slider_min=slider_min,
            slider_max=slider_max,
            slider_rounding=cls._slider_rounding(option, action),
        )

    @staticmethod
    def _set_control_title(opts: tuple[str, ...]) -> str:
        """Return the CLI option title used by the existing GUI."""
        control_title = opts[1] if len(opts) == 2 else opts[0]
        return control_title.replace("-", " ").replace("_", " ").strip().title()

    @staticmethod
    def _get_data_type(option: dict[str, T.Any]) -> type:
        """Return the value type implied by raw CLI option metadata."""
        opt_type = option.get("type")
        if isinstance(opt_type, type):
            return opt_type
        if option.get("action", "") in ("store_true", "store_false"):
            return bool
        return str

    @staticmethod
    def _get_choices(choices: object) -> tuple[str, ...]:
        """Normalize CLI choices for Qt option rendering."""
        if isinstance(choices, (list, tuple)):
            return tuple(str(choice) for choice in choices)
        return ()

    @staticmethod
    def _action_name(action: object) -> str | None:
        """Return a stable CLI action name for schema consumers."""
        if action is None:
            return None
        if isinstance(action, str):
            return action
        name = getattr(action, "__name__", None)
        return str(name) if name is not None else str(action)

    @staticmethod
    def _browser_modes(action: str | None) -> tuple[str, ...]:
        """Return simple path browser modes for known path actions."""
        modes = {
            "DirFullPaths": ("folder",),
            "FileFullPaths": ("file",),
            "FilesFullPaths": ("files",),
            "DirOrFileFullPaths": ("folder", "file"),
            "DirOrFilesFullPaths": ("folder", "files"),
            "SaveFileFullPaths": ("save",),
        }
        return modes.get(action or "", ())

    @staticmethod
    def _slider_min_max(
        option: dict[str, T.Any], action: str | None
    ) -> tuple[float | None, float | None]:
        """Return slider bounds from CLI metadata."""
        if action != "Slider":
            return None, None
        min_max = option.get("min_max")
        if not isinstance(min_max, (list, tuple)) or len(min_max) != 2:
            return None, None
        return float(min_max[0]), float(min_max[1])

    @staticmethod
    def _slider_rounding(option: dict[str, T.Any], action: str | None) -> float | None:
        """Return slider rounding metadata."""
        if action != "Slider":
            return None
        rounding = option.get("rounding")
        if isinstance(rounding, (float, int)):
            return float(rounding)
        return None


__all__ = ["CommandSchemaDiscovery", "DiscoveredCliOption", "DiscoveredCommand"]
