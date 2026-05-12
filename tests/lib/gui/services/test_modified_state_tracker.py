#!/usr/bin/env python3
"""Tests for GUI modified-state tracking."""

from __future__ import annotations

from dataclasses import dataclass

from lib.gui.services.modified_state_tracker import ModifiedStateTracker


@dataclass
class _BoolVar:
    """Small bool var test double."""

    value: bool

    def get(self) -> bool:
        """Return stored value."""
        return self.value

    def set(self, value: bool) -> None:
        """Set stored value."""
        self.value = value


def test_any_modified() -> None:
    """Tracker should report whether any command is modified."""
    tracker = ModifiedStateTracker({"extract": _BoolVar(False), "train": _BoolVar(True)})

    assert tracker.any_modified() is True


def test_is_modified_handles_missing_command() -> None:
    """Missing commands should not be considered modified."""
    tracker = ModifiedStateTracker({"extract": _BoolVar(True)})

    assert tracker.is_modified("convert") is False


def test_reset_all() -> None:
    """Reset without a command should reset all modified vars."""
    vars_by_command = {"extract": _BoolVar(True), "train": _BoolVar(True)}
    tracker = ModifiedStateTracker(vars_by_command)

    tracker.reset()

    assert vars_by_command["extract"].get() is False
    assert vars_by_command["train"].get() is False


def test_reset_one_command() -> None:
    """Reset with a command should reset only that command."""
    vars_by_command = {"extract": _BoolVar(True), "train": _BoolVar(True)}
    tracker = ModifiedStateTracker(vars_by_command)

    tracker.reset("extract")

    assert vars_by_command["extract"].get() is False
    assert vars_by_command["train"].get() is True


def test_set_one_command() -> None:
    """Set should update only existing commands."""
    vars_by_command = {"extract": _BoolVar(False)}
    tracker = ModifiedStateTracker(vars_by_command)

    tracker.set("extract")
    tracker.set("missing")

    assert vars_by_command["extract"].get() is True
