#!/usr/bin python3
"""The Faceswap GUI package."""

from __future__ import annotations

import importlib
import typing as T

_LAZY_EXPORTS = {
    "CliOptions": ("lib.gui.options", "CliOptions"),
    "CommandNotebook": ("lib.gui.command", "CommandNotebook"),
    "ConsoleOut": ("lib.gui.custom_widgets", "ConsoleOut"),
    "DisplayNotebook": ("lib.gui.display", "DisplayNotebook"),
    "LastSession": ("lib.gui.project", "LastSession"),
    "MainMenuBar": ("lib.gui.menu", "MainMenuBar"),
    "ProcessWrapper": ("lib.gui.wrapper", "ProcessWrapper"),
    "StatusBar": ("lib.gui.custom_widgets", "StatusBar"),
    "TaskBar": ("lib.gui.menu", "TaskBar"),
    "get_config": ("lib.gui.utils", "get_config"),
    "get_images": ("lib.gui.utils", "get_images"),
    "initialize_config": ("lib.gui.utils", "initialize_config"),
    "initialize_images": ("lib.gui.utils", "initialize_images"),
    "preview_trigger": ("lib.gui.utils", "preview_trigger"),
}

__all__ = [*_LAZY_EXPORTS, "gui_config"]


def __getattr__(name: str) -> object:
    """Load Tk GUI exports only when callers request them."""
    if name == "gui_config":
        module = importlib.import_module("lib.gui.gui_config")
        globals()[name] = module
        return module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attributes including lazy GUI exports."""
    return sorted([*globals(), *__all__])


if T.TYPE_CHECKING:
    from lib.gui.command import CommandNotebook
    from lib.gui.custom_widgets import ConsoleOut, StatusBar
    from lib.gui.display import DisplayNotebook
    from lib.gui.menu import MainMenuBar, TaskBar
    from lib.gui.options import CliOptions
    from lib.gui.project import LastSession
    from lib.gui.utils import (
        get_config,
        get_images,
        initialize_config,
        initialize_images,
        preview_trigger,
    )
    from lib.gui.wrapper import ProcessWrapper
