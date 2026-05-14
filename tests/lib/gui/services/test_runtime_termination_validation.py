#!/usr/bin/env python3
"""Validation tests for shared runtime termination behavior."""

from __future__ import annotations

import sys
import types

from lib.gui.services.runtime_session_services import ProcessTreeTerminator, RuntimeTerminationPolicy


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


def _install_fake_psutil(monkeypatch, children: list[_ChildProcess], alive: list[_ChildProcess]):  # type:ignore[no-untyped-def]
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


def test_process_tree_terminator_terminates_children_then_kills_alive(  # type:ignore[no-untyped-def]
    monkeypatch,
) -> None:
    """Process-tree cleanup should terminate children and kill survivors after timeout."""
    child_done = _ChildProcess()
    child_alive = _ChildProcess()
    wait_calls = _install_fake_psutil(monkeypatch, [child_done, child_alive], [child_alive])

    handled = ProcessTreeTerminator().terminate(1234, timeout_seconds=3.0)

    assert handled is True
    assert child_done.terminate_count == 1
    assert child_done.kill_count == 0
    assert child_alive.terminate_count == 1
    assert child_alive.kill_count == 1
    assert wait_calls == [([child_done, child_alive], 3.0)]


def test_process_tree_terminator_ignores_psutil_errors(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Process-tree cleanup should continue when individual child operations fail."""
    child = _ChildProcess(fail_terminate=True, fail_kill=True)
    _install_fake_psutil(monkeypatch, [child], [child])

    handled = ProcessTreeTerminator().terminate(1234, timeout_seconds=0.0)

    assert handled is True
    assert child.terminate_count == 1
    assert child.kill_count == 1
