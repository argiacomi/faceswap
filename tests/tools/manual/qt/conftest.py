#!/usr/bin/env python3
"""Shared pytest fixtures for Qt Manual Tool tests."""

from __future__ import annotations

import logging

import pytest
from PySide6.QtWidgets import QMessageBox


@pytest.fixture(scope="session", autouse=True)
def _suppress_manual_qt_info_logs() -> None:
    """Silence routine Manual Tool INFO logs while preserving warnings/errors."""
    logger_names = (
        "tools.manual.qt",
        "lib.align.alignments",
    )
    loggers = tuple(logging.getLogger(name) for name in logger_names)
    original_levels = tuple(logger.level for logger in loggers)
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    yield
    for logger, level in zip(loggers, original_levels, strict=True):
        logger.setLevel(level)


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
