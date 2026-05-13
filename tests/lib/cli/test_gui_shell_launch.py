#!/usr/bin/env python3
"""Tests for GUI shell launch selection."""

from __future__ import annotations

import argparse
from argparse import Namespace

import pytest

from lib.cli.args import GuiArgs
from lib.cli.launcher import GUI_SHELL_ENV, FaceswapError, ScriptExecutor


def test_gui_args_include_hidden_no_exec_flag() -> None:
    """GUI args should expose a hidden no-exec flag for launch smoke tests."""
    hidden_arg = next(
        option
        for option in GuiArgs.get_argument_list()
        if "--no-gui-exec" in option["opts"]
    )

    assert hidden_arg["action"] == "store_true"
    assert hidden_arg["dest"] == "no_gui_exec"
    assert hidden_arg["help"] == argparse.SUPPRESS


def test_resolve_gui_shell_defaults_to_tk(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """GUI shell should default to Tk when no CLI/env selector is provided."""
    monkeypatch.delenv(GUI_SHELL_ENV, raising=False)

    shell = ScriptExecutor._resolve_gui_shell(Namespace(gui_shell=None))  # pylint:disable=protected-access

    assert shell == "tk"


def test_resolve_gui_shell_prefers_cli_over_env(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """CLI shell selection should override the environment selector."""
    monkeypatch.setenv(GUI_SHELL_ENV, "tk")

    shell = ScriptExecutor._resolve_gui_shell(Namespace(gui_shell="qt"))  # pylint:disable=protected-access

    assert shell == "qt"


def test_resolve_gui_shell_reads_environment(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Environment shell selection should be accepted when CLI selector is omitted."""
    monkeypatch.setenv(GUI_SHELL_ENV, "qt")

    shell = ScriptExecutor._resolve_gui_shell(Namespace(gui_shell=None))  # pylint:disable=protected-access

    assert shell == "qt"


def test_resolve_gui_shell_rejects_invalid_environment(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Invalid GUI shell names should fail clearly."""
    monkeypatch.setenv(GUI_SHELL_ENV, "bad")

    with pytest.raises(FaceswapError, match="Invalid GUI shell"):
        ScriptExecutor._resolve_gui_shell(Namespace(gui_shell=None))  # pylint:disable=protected-access


def test_import_script_uses_qt_gui_module(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Qt shell selection should import scripts.gui_qt instead of scripts.gui."""
    imported = []

    class _Module:
        Gui = object

    def fake_import_module(module_name: str):  # type:ignore[no-untyped-def]
        imported.append(module_name)
        return _Module

    executor = ScriptExecutor("gui")
    executor._gui_shell = "qt"  # pylint:disable=protected-access
    monkeypatch.setattr(executor, "_set_environment_variables", lambda: None)
    monkeypatch.setattr(executor, "_test_for_torch_version", lambda: None)
    monkeypatch.setattr(executor, "_test_for_gui", lambda: None)
    monkeypatch.setattr("lib.cli.launcher.import_module", fake_import_module)

    script = executor._import_script()  # pylint:disable=protected-access

    assert script is object
    assert imported == ["scripts.gui_qt"]


def test_import_script_uses_tk_gui_module(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Tk shell selection should keep the existing scripts.gui launch path."""
    imported = []

    class _Module:
        Gui = object

    def fake_import_module(module_name: str):  # type:ignore[no-untyped-def]
        imported.append(module_name)
        return _Module

    executor = ScriptExecutor("gui")
    executor._gui_shell = "tk"  # pylint:disable=protected-access
    monkeypatch.setattr(executor, "_set_environment_variables", lambda: None)
    monkeypatch.setattr(executor, "_test_for_torch_version", lambda: None)
    monkeypatch.setattr(executor, "_test_for_gui", lambda: None)
    monkeypatch.setattr("lib.cli.launcher.import_module", fake_import_module)

    script = executor._import_script()  # pylint:disable=protected-access

    assert script is object
    assert imported == ["scripts.gui"]


def test_qt_shell_checks_pyside6_not_tk(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Qt GUI preflight should check PySide6 and skip Tkinter checks."""
    calls = []
    executor = ScriptExecutor("gui")
    executor._gui_shell = "qt"  # pylint:disable=protected-access
    monkeypatch.setattr(executor, "_test_pyside6", lambda: calls.append("qt"))
    monkeypatch.setattr(executor, "_test_tkinter", lambda: calls.append("tk"))
    monkeypatch.setattr(executor, "_check_display", lambda: calls.append("display"))

    executor._test_for_gui()  # pylint:disable=protected-access

    assert calls == ["qt", "display"]


def test_tk_shell_checks_tk_not_pyside6(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Tk GUI preflight should keep existing Tkinter checks."""
    calls = []
    executor = ScriptExecutor("gui")
    executor._gui_shell = "tk"  # pylint:disable=protected-access
    monkeypatch.setattr(executor, "_test_pyside6", lambda: calls.append("qt"))
    monkeypatch.setattr(executor, "_test_tkinter", lambda: calls.append("tk"))
    monkeypatch.setattr(executor, "_check_display", lambda: calls.append("display"))

    executor._test_for_gui()  # pylint:disable=protected-access

    assert calls == ["tk", "display"]
