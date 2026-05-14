#!/usr/bin/env python3
"""Tests for Qt command option state preservation."""

from __future__ import annotations

from lib.gui.qt_shell.command_option_state import install_command_option_state


class _RendererDouble:
    """Small renderer stand-in."""

    def __init__(self) -> None:
        self.current_values = {}
        self.applied = []

    def values(self) -> dict[str, object]:
        """Return current renderer values."""
        return dict(self.current_values)

    def apply_values(self, values) -> None:  # type:ignore[no-untyped-def]
        """Capture applied values."""
        self.applied.append(dict(values))
        self.current_values = dict(values)


class _CommandDouble:
    """Small command combo stand-in."""

    def __init__(self) -> None:
        self.text = "extract"

    def currentText(self) -> str:  # noqa:N802
        """Return current command text."""
        return self.text


class _PanelBase:
    """Small CommandPanel-like stand-in."""

    def __init__(self) -> None:
        self._renderer = _RendererDouble()
        self._command = _CommandDouble()
        self.validation_count = 0
        self.set_options_calls = []
        self.clear_count = 0

    def _set_command_options(self, command: str) -> None:
        """Capture command option rendering."""
        self.set_options_calls.append(command)
        self._command.text = command

    def set_command(self, command: str, values) -> None:  # type:ignore[no-untyped-def]
        """Capture programmatic command setting."""
        self._command.text = command
        self._renderer.apply_values(values)

    def clear_values(self) -> None:
        """Capture clear behavior."""
        self.clear_count += 1

    def _update_validation(self) -> None:
        """Capture validation refresh."""
        self.validation_count += 1


def _panel_class():
    """Return a fresh panel class for each test."""
    Panel = type("Panel", (_PanelBase,), {})
    install_command_option_state(Panel)
    return Panel


def test_command_option_state_preserves_values_across_switches() -> None:
    """Switching commands should cache previous command values and restore them later."""
    panel = _panel_class()()
    panel._set_command_options("extract")
    panel._renderer.current_values = {"-i": "/input"}

    panel._set_command_options("train")
    panel._renderer.current_values = {"-m": "/models"}
    panel._set_command_options("extract")

    assert panel._command_value_cache["extract"] == {"-i": "/input"}
    assert panel._command_value_cache["train"] == {"-m": "/models"}
    assert panel._renderer.current_values == {"-i": "/input"}
    assert panel.validation_count == 1


def test_command_option_state_caches_programmatic_set_command_values() -> None:
    """Project/session command restore should seed the per-command cache."""
    panel = _panel_class()()

    panel.set_command("convert", {"-o": "/output"})
    panel._set_command_options("convert")

    assert panel._command_value_cache["convert"] == {"-o": "/output"}
    assert panel._renderer.current_values == {"-o": "/output"}


def test_command_option_state_clear_values_drops_active_command_cache() -> None:
    """Clearing current command values should also clear the cached values for that command."""
    panel = _panel_class()()
    panel._command.text = "extract"
    panel._command_value_cache["extract"] = {"-i": "/input"}

    panel.clear_values()

    assert "extract" not in panel._command_value_cache
    assert panel.clear_count == 1


def test_command_option_state_install_is_idempotent() -> None:
    """Repeated installation should not replace already wrapped methods."""
    Panel = _panel_class()
    wrapped = Panel._set_command_options

    install_command_option_state(Panel)

    assert Panel._set_command_options is wrapped
