#!/usr/bin/env python3
"""Validation tests for shared runtime termination behavior."""

from __future__ import annotations

import sys
import types

from lib.gui.services.runtime_session_services import (
    ProcessTreeTerminator,
    RuntimeTerminationPolicy,
)


class _PsutilError(Exception):
    """Fake psutil base error."""


class _ChildProcess:
    """Fake child process for process-tree cleanup tests."""

    def __init__(self, *, fail_terminate: bool = False, fail_kill: bool = False) -> None:
        self.fail_terminate = fail_terminate
        self.fail_kill = fail_kill
        self.terminate_count = 0
        self.kill_count = 0

    def terminate(self) -> None:
        """Capture terminate calls."""
        self.terminate_count += 1
        if self.fail_terminate:
            raise _PsutilError()

    def kill(self) -> None:
        """Capture kill calls."""
        self.kill_count += 1
        if self.fail_kill:
            raise _PsutilError()


class _RootProcess:
    """Fake root process."""

    def __init__(self, children: list[_ChildProcess]) -> None:
        self._children = children

    def children(self, *, recursive: bool = False) -> list[_ChildProcess]:
        """Return child processes."""
        assert recursive is True
        return self._children


def _install_fake_psutil(  # type:ignore[no-untyped-def]
    monkeypatch,
    children: list[_ChildProcess],
    alive: list[_ChildProcess],
):
    """Install a fake psutil module."""
    wait_calls = []

    def process_factory(pid: int) -> _RootProcess:
        assert pid == 1234
        return _RootProcess(children)

    def wait_procs(procs: list[_ChildProcess], *, timeout: float):  # type:ignore[no-untyped-def]
        wait_calls.append((procs, timeout))
        gone = [child for child in procs if child not in alive]
        return gone, alive

    fake_psutil = types.SimpleNamespace(
        Error=_PsutilError,
        Process=process_factory,
        wait_procs=wait_procs,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    return wait_calls


def test_runtime_termination_policy_train_plan_is_interrupt_first() -> None:
    """Train termination plans should request graceful interrupt before escalation."""
    plan = RuntimeTerminationPolicy(train_timeout_seconds=2.5).plan("train.py")

    assert plan.command == "train"
    assert plan.interrupt_first is True
    assert plan.grace_ms == 2500
    assert plan.tree_timeout_seconds == 2.5
    assert plan.message == "Sending Exit Signal"


def test_runtime_termination_policy_default_plan_uses_terminate() -> None:
    """Non-train termination plans should use standard process termination."""
    plan = RuntimeTerminationPolicy(default_timeout_seconds=7).plan("convert")

    assert plan.command == "convert"
    assert plan.interrupt_first is False
    assert plan.grace_ms == 7000
    assert plan.tree_timeout_seconds == 7
    assert plan.message == "Terminating Process..."


def test_process_tree_terminator_returns_false_without_psutil(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Process-tree cleanup should be a no-op when psutil is unavailable."""
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert ProcessTreeTerminator().terminate(1234, timeout_seconds=1.0) is False


def test_process_tree_terminator_terminates_children_and_kills_survivors(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Process-tree cleanup should terminate descendants and kill survivors after timeout."""
    first = _ChildProcess()
    survivor = _ChildProcess()
    wait_calls = _install_fake_psutil(monkeypatch, [first, survivor], [survivor])

    handled = ProcessTreeTerminator().terminate(1234, timeout_seconds=0.25)

    assert handled is True
    assert first.terminate_count == 1
    assert first.kill_count == 0
    assert survivor.terminate_count == 1
    assert survivor.kill_count == 1
    assert wait_calls == [([first, survivor], 0.25)]


def test_process_tree_terminator_ignores_child_errors(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Process-tree cleanup should keep going when individual child calls fail."""
    failed_terminate = _ChildProcess(fail_terminate=True)
    failed_kill = _ChildProcess(fail_kill=True)
    _install_fake_psutil(monkeypatch, [failed_terminate, failed_kill], [failed_kill])

    handled = ProcessTreeTerminator().terminate(1234, timeout_seconds=0.25)

    assert handled is True
    assert failed_terminate.terminate_count == 1
    assert failed_kill.terminate_count == 1
    assert failed_kill.kill_count == 1
