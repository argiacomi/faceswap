#!/usr/bin/env python3
"""Tests for Qt Preview lifecycle hook installation."""

from __future__ import annotations


def test_qt_shell_package_installs_preview_lifecycle_hook() -> None:
    """Importing the Qt shell package should install Preview lifecycle hooks."""
    from lib.gui.qt_shell import MainWindow

    assert getattr(MainWindow, "_preview_job_lifecycle_installed", False) is True


def test_qt_shell_package_does_not_double_wrap_preview_lifecycle_hook() -> None:
    """Repeated installation should not replace already wrapped methods."""
    from lib.gui.qt_shell import MainWindow
    from lib.gui.qt_shell.preview_job_lifecycle import install_preview_job_lifecycle

    run_command = MainWindow._run_command
    stop_job = MainWindow._stop_job
    job_finished = MainWindow._job_finished

    install_preview_job_lifecycle(MainWindow)

    assert MainWindow._run_command is run_command
    assert MainWindow._stop_job is stop_job
    assert MainWindow._job_finished is job_finished
