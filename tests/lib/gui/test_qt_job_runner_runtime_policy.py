#!/usr/bin/env python3
"""Tests for Qt JobRunner runtime-service policy hooks."""

from __future__ import annotations

from PySide6.QtCore import QProcess

from lib.gui.services.command_context import CommandExecutionContext


class _ProcessDouble:
    """Small running QProcess double."""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def state(self) -> QProcess.ProcessState:
        """Return a running process state."""
        return QProcess.Running

    def terminate(self) -> None:
        """Capture terminate calls."""
        self.terminated = True

    def kill(self) -> None:
        """Capture kill calls."""
        self.killed = True


class _TimerDouble:
    """Capture QTimer.singleShot calls without running callbacks."""

    calls: list[tuple[int, object]] = []

    @classmethod
    def singleShot(cls, interval: int, callback: object) -> None:
        """Capture scheduled callbacks."""
        cls.calls.append((interval, callback))


def test_job_runner_stop_uses_train_termination_policy(monkeypatch) -> None:
    """Train stop should emit exit-signal text and use the train grace period."""
    from lib.gui.qt_shell import job_runner

    _TimerDouble.calls = []
    monkeypatch.setattr(job_runner, "QTimer", _TimerDouble)
    runner = job_runner.JobRunner()
    process = _ProcessDouble()
    runner.process = process  # type:ignore[assignment]
    runner._runtime.command = "train"  # pylint:disable=protected-access
    stdout = []
    runner.stdout.connect(stdout.append)

    runner.stop()

    assert stdout == ["Sending Exit Signal\n"]
    assert process.terminated is True
    assert _TimerDouble.calls == [(5000, runner._kill_if_running)]  # pylint:disable=protected-access


def test_job_runner_stop_uses_default_termination_policy(monkeypatch) -> None:
    """Non-train stop should emit standard termination text."""
    from lib.gui.qt_shell import job_runner

    _TimerDouble.calls = []
    monkeypatch.setattr(job_runner, "QTimer", _TimerDouble)
    runner = job_runner.JobRunner()
    process = _ProcessDouble()
    runner.process = process  # type:ignore[assignment]
    runner._runtime.command = "extract"  # pylint:disable=protected-access
    stdout = []
    runner.stdout.connect(stdout.append)

    runner.stop()

    assert stdout == ["Terminating Process...\n"]
    assert process.terminated is True
    assert _TimerDouble.calls == [(5000, runner._kill_if_running)]  # pylint:disable=protected-access


def test_job_runner_configures_training_session_from_command_context() -> None:
    """Command context should flow into the shared training-session service."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    context = CommandExecutionContext.from_values(
        "train",
        {"-m": "/models", "-t": "My-Model"},
    )

    runner.configure_runtime_context(context)

    location = runner._runtime.training_session.location  # pylint:disable=protected-access
    assert location.model_folder == "/models"
    assert location.model_name == "my_model"
