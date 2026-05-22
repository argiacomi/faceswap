#!/usr/bin/env python3
"""Shared pytest fixtures for Qt GUI tests.

Auto-accepts modal QMessageBox prompts so test teardown does not hang waiting
for user input.  Without this, closing a ManualToolWindow that has unsaved
edits (or any other widget that prompts for confirmation in ``closeEvent``)
blocks pytest forever in headless mode.

The native Manual Tool also reports startup progress through the normal
application logger.  That is useful at runtime, but in widget tests the
background startup worker can finish between pytest progress dots and print
non-pytest log lines.  Keep those routine INFO messages quiet for this test
package while still allowing warnings/errors through.
"""

from __future__ import annotations

import logging

import pytest
from PySide6.QtWidgets import QMessageBox


@pytest.fixture(scope="session", autouse=True)
def _suppress_manual_tool_startup_logs() -> None:
    """Silence routine Manual Tool startup INFO logs for Qt GUI tests."""
    logger = logging.getLogger("lib.gui.qt_shell.manual_tool")
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    yield
    logger.setLevel(original_level)


@pytest.fixture(autouse=True)
def _autoaccept_modal_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force QMessageBox prompts to return Yes / Ok so headless tests cannot stall."""
    monkeypatch.setattr(QMessageBox, "question", staticmethod(_auto_yes), raising=True)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_auto_ok), raising=True)
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_auto_ok), raising=True)
    monkeypatch.setattr(QMessageBox, "information", staticmethod(_auto_ok), raising=True)


def _auto_yes(*_args: object, **_kwargs: object) -> QMessageBox.StandardButton:
    """Stand-in for QMessageBox.question that auto-accepts."""
    return QMessageBox.StandardButton.Yes


def _auto_ok(*_args: object, **_kwargs: object) -> QMessageBox.StandardButton:
    """Stand-in for warning/critical/information dialogs that auto-acknowledges."""
    return QMessageBox.StandardButton.Ok
