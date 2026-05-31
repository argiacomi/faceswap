#!/usr/bin/env python3
"""Tests for Qt JobRunner command-aware termination behavior."""

from __future__ import annotations

import os
import signal

from PySide6.QtCore import QProcess

from lib.gui.qt_shell.job_runner import JobRunner


class _ProcessDouble:
    """Small QProcess stand-in for stop-policy tests."""

    def __init__(self, *, pid: int = 1234, state: QProcess.ProcessState | None = None) -> None:
        self._pid = pid
        self._state = state or QProcess.ProcessState.Running
        self.terminate_count = 0
        self.kill_count = 0

    def state(self) -> QProcess.ProcessState:
        """Return current process state."""
        return self._state

    def processId(self) -> int:  # noqa:N802
        """Return fake process ID."""
        return self._pid

    def terminate(self) -> None:
        """Capture terminate calls."""
        self.terminate_count += 1

    def kill(self) -> None:
        """Capture kill calls."""
        self.kill_count += 1


def test_stop_sends_interrupt_for_train_before_terminate(qtbot, monkeypatch) -> None:
    """Train stop should request SIGINT and skip immediate terminate when it succeeds."""
    runner = JobRunner()
    process = _ProcessDouble(pid=4321)
    runner.process = process  # type:ignore[assignment]
    runner._runtime.command = "train"  # pylint:disable=protected-access
    killed: list[tuple[int, signal.Signals]] = []
    messages: list[str] = []
    runner.stdout.connect(messages.append)
    monkeypatch.setattr("lib.gui.qt_shell.job_runner.IS_WINDOWS", False)
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr("lib.gui.qt_shell.job_runner.QTimer.singleShot", lambda *_args: None)

    runner.stop()

    assert killed == [(4321, signal.SIGINT)]
    assert process.terminate_count == 0
    assert messages == ["Sending Exit Signal\n"]


def test_stop_falls_back_to_terminate_when_train_interrupt_fails(qtbot, monkeypatch) -> None:
    """Train stop should use terminate when SIGINT cannot be sent."""
    runner = JobRunner()
    process = _ProcessDouble(pid=4321)
    runner.process = process  # type:ignore[assignment]
    runner._runtime.command = "train"  # pylint:disable=protected-access
    monkeypatch.setattr("lib.gui.qt_shell.job_runner.IS_WINDOWS", False)
    monkeypatch.setattr(os, "kill", lambda _pid, _sig: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr("lib.gui.qt_shell.job_runner.QTimer.singleShot", lambda *_args: None)

    runner.stop()

    assert process.terminate_count == 1


def test_stop_uses_terminate_for_train_on_windows(qtbot, monkeypatch) -> None:
    """Windows train stop should use QProcess terminate because SIGINT is unavailable."""
    runner = JobRunner()
    process = _ProcessDouble(pid=4321)
    runner.process = process  # type:ignore[assignment]
    runner._runtime.command = "train"  # pylint:disable=protected-access
    monkeypatch.setattr("lib.gui.qt_shell.job_runner.IS_WINDOWS", True)
    monkeypatch.setattr("lib.gui.qt_shell.job_runner.QTimer.singleShot", lambda *_args: None)

    runner.stop()

    assert process.terminate_count == 1


def test_stop_uses_terminate_for_non_train(qtbot, monkeypatch) -> None:
    """Non-train commands should use standard QProcess terminate first."""
    runner = JobRunner()
    process = _ProcessDouble(pid=4321)
    runner.process = process  # type:ignore[assignment]
    runner._runtime.command = "extract"  # pylint:disable=protected-access
    killed: list[tuple[int, signal.Signals]] = []
    messages: list[str] = []
    runner.stdout.connect(messages.append)
    monkeypatch.setattr("lib.gui.qt_shell.job_runner.IS_WINDOWS", False)
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr("lib.gui.qt_shell.job_runner.QTimer.singleShot", lambda *_args: None)

    runner.stop()

    assert killed == []
    assert process.terminate_count == 1
    assert messages == ["Terminating Process...\n"]


def test_stop_schedules_kill_fallback_for_train(qtbot, monkeypatch) -> None:
    """Train stop should still schedule kill fallback after graceful interrupt."""
    runner = JobRunner()
    process = _ProcessDouble(pid=4321)
    runner.process = process  # type:ignore[assignment]
    runner._runtime.command = "train"  # pylint:disable=protected-access
    scheduled: list[int] = []
    monkeypatch.setattr("lib.gui.qt_shell.job_runner.IS_WINDOWS", False)
    monkeypatch.setattr(os, "kill", lambda _pid, _sig: None)
    monkeypatch.setattr(
        "lib.gui.qt_shell.job_runner.QTimer.singleShot",
        lambda delay, callback: scheduled.append(delay),
    )

    runner.stop()

    assert scheduled == [5000]


def test_kill_fallback_kills_still_running_process() -> None:
    """Kill fallback should escalate when the process is still running."""
    runner = JobRunner()
    process = _ProcessDouble()
    runner.process = process  # type:ignore[assignment]

    runner._kill_if_running()  # pylint:disable=protected-access

    assert process.kill_count == 1


def test_kill_fallback_invokes_tree_terminator() -> None:
    """Kill fallback should attempt child-process cleanup before root kill."""
    runner = JobRunner()
    process = _ProcessDouble(pid=2468)
    runner.process = process  # type:ignore[assignment]
    runner._pending_termination_plan = runner._termination_policy.plan("extract")  # pylint:disable=protected-access
    calls: list[tuple[int, float]] = []
    runner._tree_terminator.terminate = (  # type:ignore[method-assign] # pylint:disable=protected-access
        lambda pid, *, timeout_seconds: calls.append((pid, timeout_seconds)) or True  # type: ignore[func-returns-value]
    )

    runner._kill_if_running()  # pylint:disable=protected-access

    assert calls == [(2468, 5.0)]
    assert process.kill_count == 1


def test_kill_fallback_ignores_finished_process() -> None:
    """Kill fallback should not kill a process that already stopped."""
    runner = JobRunner()
    process = _ProcessDouble(state=QProcess.ProcessState.NotRunning)
    runner.process = process  # type:ignore[assignment]

    runner._kill_if_running()  # pylint:disable=protected-access

    assert process.kill_count == 0
