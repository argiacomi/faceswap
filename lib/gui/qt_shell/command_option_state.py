#!/usr/bin/env python3
"""Command option state preservation for the Qt shell."""

from __future__ import annotations

import functools

from lib.utils import get_module_objects


def install_command_option_state(command_panel_class: type) -> None:
    """Install per-command option state preservation on CommandPanel."""
    if getattr(command_panel_class, "_command_option_state_installed", False):
        return

    original_init = command_panel_class.__init__
    original_set_options = command_panel_class._set_command_options
    original_set_command = command_panel_class.set_command
    original_clear_values = command_panel_class.clear_values

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        self._command_value_cache = {}
        self._command_value_cache_enabled = True
        return original_init(self, *args, **kwargs)

    @functools.wraps(original_set_options)
    def set_command_options(self, command: str) -> None:
        previous = getattr(self, "_active_cached_command", None)
        cache = getattr(self, "_command_value_cache", None)
        if isinstance(previous, str) and cache is not None and previous:
            cache[previous] = self._renderer.values()
        original_set_options(self, command)
        if cache is not None and command in cache:
            self._renderer.apply_values(cache[command])
            self._update_validation()
        self._active_cached_command = command

    @functools.wraps(original_set_command)
    def set_command(self, command, values):  # type:ignore[no-untyped-def]
        cache = getattr(self, "_command_value_cache", None)
        if cache is not None and command:
            cache[command] = dict(values)
        return original_set_command(self, command, values)

    @functools.wraps(original_clear_values)
    def clear_values(self) -> None:
        cache = getattr(self, "_command_value_cache", None)
        command = self._command.currentText()
        if cache is not None and command in cache:
            cache.pop(command, None)
        return original_clear_values(self)

    command_panel_class.__init__ = init
    command_panel_class._set_command_options = set_command_options
    command_panel_class.set_command = set_command
    command_panel_class.clear_values = clear_values
    command_panel_class._command_option_state_installed = True


__all__ = get_module_objects(__name__)
