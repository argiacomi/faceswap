#!/usr/bin/env python3
"""Tests for GUI launch hardening hooks."""

from __future__ import annotations

from argparse import Namespace

from lib.cli.gui_launch_hardening import install_gui_launch_hardening


class _ExecutorDouble:
    """Small ScriptExecutor stand-in."""

    def __init__(self) -> None:
        self._command = "gui"
        self._gui_shell = "qt"
        self._gui_no_exec = False
        self.calls: list[str] = []

    def execute_script(self, arguments: Namespace) -> None:
        """Capture execution."""
        self.calls.append(f"execute:{bool(getattr(arguments, 'no_gui_exec', False))}")

    def _test_for_gui(self) -> None:
        """Original preflight placeholder."""
        self.calls.append("original")

    def _test_pyside6(self) -> None:
        """Capture PySide6 preflight."""
        self.calls.append("qt")

    def _test_tkinter(self) -> None:
        """Capture Tk preflight."""
        self.calls.append("tk")

    def _check_display(self) -> None:
        """Capture display preflight."""
        self.calls.append("display")


def _executor_class():
    """Return a fresh executor class."""
    Executor = type("Executor", (_ExecutorDouble,), {})
    install_gui_launch_hardening(Executor)
    return Executor


def test_qt_no_exec_skips_display_preflight() -> None:
    """Qt no-exec smoke launch should check PySide6 but not require a display."""
    executor = _executor_class()()
    executor.execute_script(Namespace(no_gui_exec=True))

    executor._test_for_gui()

    assert executor._gui_no_exec is True
    assert executor.calls == ["execute:True", "qt"]


def test_qt_normal_launch_keeps_display_preflight() -> None:
    """Normal Qt launch should still require display preflight."""
    executor = _executor_class()()
    executor.execute_script(Namespace(no_gui_exec=False))

    executor._test_for_gui()

    assert executor._gui_no_exec is False
    assert executor.calls == ["execute:False", "qt", "display"]


def test_tk_launch_keeps_tk_and_display_preflight() -> None:
    """Tk launch should keep existing Tk and display checks."""
    executor = _executor_class()()
    executor._gui_shell = "tk"
    executor.execute_script(Namespace(no_gui_exec=True))

    executor._test_for_gui()

    assert executor.calls == ["execute:True", "tk", "display"]


def test_no_exec_can_be_resolved_from_environment(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Qt no-exec smoke mode should also be available through the environment."""
    Executor = _executor_class()
    monkeypatch.setenv("FACESWAP_QT_NO_EXEC", "1")

    assert Executor._resolve_gui_no_exec(Namespace(no_gui_exec=False)) is True


def test_gui_launch_hardening_install_is_idempotent() -> None:
    """Repeated installation should not replace wrapped methods."""
    Executor = _executor_class()
    wrapped = Executor.execute_script

    install_gui_launch_hardening(Executor)

    assert Executor.execute_script is wrapped
