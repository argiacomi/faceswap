#!/usr/bin/env python3
"""Tests for applying saved GUI option values."""

from __future__ import annotations

import typing as T

from lib.gui.services.gui_option_applier import GuiOptionApplier


class _Var:
    """Variable test double."""

    def __init__(self) -> None:
        self.value: T.Any = None

    def set(self, value: T.Any) -> None:
        """Capture set value."""
        self.value = value


class _CliOptions:
    """CLI options test double."""

    def __init__(self) -> None:
        self.variables: dict[tuple[str, str], _Var] = {}

    def add(self, command: str, title: str) -> _Var:
        """Add an option variable."""
        variable = _Var()
        self.variables[(command, title)] = variable
        return variable

    def get_one_option_variable(self, command: str, title: str) -> _Var | None:
        """Return an option variable if known."""
        return self.variables.get((command, title))


class _ActiveTab:
    """Active tab callback test double."""

    def __init__(self) -> None:
        self.name: str | None = None

    def set(self, name: str) -> None:
        """Capture active tab name."""
        self.name = name


def test_apply_project_applies_known_variables_and_skips_unknowns() -> None:
    """Saved values should be applied only when a GUI variable exists."""
    cli_options = _CliOptions()
    input_dir = cli_options.add("extract", "Input Dir")
    active_tab = _ActiveTab()
    applier = GuiOptionApplier(cli_options, active_tab.set)

    applied = applier.apply_project(
        {
            "tab_name": "extract",
            "extract": {
                "Input Dir": "/input",
                "Unknown": "ignored",
            },
        }
    )

    assert applied is True
    assert input_dir.value == "/input"
    assert active_tab.name == "extract"


def test_apply_project_applies_all_command_sections() -> None:
    """Project application should apply each command section."""
    cli_options = _CliOptions()
    input_dir = cli_options.add("extract", "Input Dir")
    model_dir = cli_options.add("train", "Model Dir")
    active_tab = _ActiveTab()
    applier = GuiOptionApplier(cli_options, active_tab.set)

    applied = applier.apply_project(
        {
            "tab_name": "train",
            "extract": {"Input Dir": "/input"},
            "train": {"Model Dir": "/model"},
        }
    )

    assert applied is True
    assert input_dir.value == "/input"
    assert model_dir.value == "/model"
    assert active_tab.name == "train"


def test_apply_project_defaults_missing_tab_to_extract() -> None:
    """Project application should default to the extract tab."""
    cli_options = _CliOptions()
    active_tab = _ActiveTab()
    applier = GuiOptionApplier(cli_options, active_tab.set)

    applied = applier.apply_project({"extract": {}})

    assert applied is True
    assert active_tab.name == "extract"


def test_apply_project_for_command_applies_only_requested_command() -> None:
    """Command application should apply only the requested command section."""
    cli_options = _CliOptions()
    input_dir = cli_options.add("extract", "Input Dir")
    model_dir = cli_options.add("train", "Model Dir")
    active_tab = _ActiveTab()
    applier = GuiOptionApplier(cli_options, active_tab.set)

    applied = applier.apply_project(
        {
            "tab_name": "train",
            "extract": {"Input Dir": "/input"},
            "train": {"Model Dir": "/model"},
        },
        command="extract",
    )

    assert applied is True
    assert input_dir.value == "/input"
    assert model_dir.value is None
    assert active_tab.name == "extract"


def test_apply_project_returns_false_when_requested_command_missing() -> None:
    """Missing requested command sections should return False."""
    cli_options = _CliOptions()
    active_tab = _ActiveTab()
    applier = GuiOptionApplier(cli_options, active_tab.set)

    applied = applier.apply_project({"extract": {}}, command="train")

    assert applied is False
    assert active_tab.name is None
