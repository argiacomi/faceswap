#!/usr/bin/env python3
"""GUI launch hardening helpers for shell-specific preflight behavior."""

from __future__ import annotations

import functools
import os

from lib.utils import get_module_objects

QT_NO_EXEC_ENV = "FACESWAP_QT_NO_EXEC"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def install_gui_launch_hardening(script_executor_class: type) -> None:
    """Install GUI launch hardening on ScriptExecutor."""
    if getattr(script_executor_class, "_gui_launch_hardening_installed", False):
        return

    original_execute = script_executor_class.execute_script

    @functools.wraps(original_execute)
    def execute_script(self, arguments):  # type:ignore[no-untyped-def]
        if getattr(self, "_command", None) == "gui":
            self._gui_no_exec = _resolve_gui_no_exec(arguments)
        return original_execute(self, arguments)

    def test_for_gui(self) -> None:  # type:ignore[no-untyped-def]
        if getattr(self, "_command", None) != "gui":
            return
        if getattr(self, "_gui_shell", "tk") == "qt":
            self._test_pyside6()
            if getattr(self, "_gui_no_exec", False):
                return
        else:
            self._test_tkinter()
        self._check_display()

    script_executor_class.execute_script = execute_script
    script_executor_class._test_for_gui = test_for_gui
    script_executor_class._resolve_gui_no_exec = staticmethod(_resolve_gui_no_exec)
    script_executor_class._gui_launch_hardening_installed = True


def _resolve_gui_no_exec(arguments) -> bool:  # type:ignore[no-untyped-def]
    """Return whether Qt GUI launch should skip display preflight for smoke tests."""
    if bool(getattr(arguments, "no_gui_exec", False)):
        return True
    return os.environ.get(QT_NO_EXEC_ENV, "").lower() in _TRUE_VALUES


__all__ = get_module_objects(__name__)
