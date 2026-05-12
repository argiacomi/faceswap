#!/usr/bin/env python3
"""Tests for Qt JobRunner runtime-event integration."""

from __future__ import annotations

from PySide6.QtCore import QProcess


class _BufferedProcessDouble:
    """QProcess test double with buffered stdout/stderr."""

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._state = QProcess.Running

    def state(self) -> QProcess.ProcessState:
        """Return the configured process state."""
        return self._state

    def readAllStandardOutput(self) -> bytes:
        """Return buffered stdout once."""
        stdout = self._stdout
        self._stdout = b""
        return stdout

    def readAllStandardError(self) -> bytes:
        """Return buffered stderr once."""
        stderr = self._stderr
        self._stderr = b""
        return stderr


def test_job_runner_emits_progress_event_from_tqdm_stdout() -> None:
    """Qt JobRunner should emit RuntimeEvent values parsed from stdout."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    runner.process = _BufferedProcessDouble(
        b"Extracting:  25%|##5       | 5/20 [00:02<00:06,  2.50it/s]\n"
    )  # type:ignore[assignment]
    runner._runtime.command = "extract"  # pylint:disable=protected-access
    progress = []
    stdout = []
    runner.progress.connect(progress.append)
    runner.stdout.connect(stdout.append)

    runner._stdout()  # pylint:disable=protected-access

    assert stdout == []
    assert len(progress) == 1
    assert progress[0].kind == "progress"
    assert progress[0].progress == 25.0
    assert progress[0].payload["parser"] == "tqdm"


def test_job_runner_emits_final_status_event_on_finish(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Qt JobRunner should emit final RuntimeEvent status before finished."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    runner.process = _BufferedProcessDouble()  # type:ignore[assignment]
    runner._runtime.command = "train"  # pylint:disable=protected-access
    progress = []
    runner.progress.connect(progress.append)

    with qtbot.waitSignal(runner.finished, timeout=1000) as signal:
        runner._finished(-9, QProcess.NormalExit)  # pylint:disable=protected-access

    assert signal.args == [-9]
    assert progress[-1].kind == "status"
    assert progress[-1].message == "Killed - train.py"
    assert progress[-1].payload == {"command": "train", "returncode": -9, "state": "killed"}
    assert runner.process is None


def test_job_runner_suppresses_consumed_stderr_output() -> None:
    """Consumed parser output should not be duplicated to stderr."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    runner.process = _BufferedProcessDouble(
        stderr=b"Training:  50%|#####     | 10/20 [00:04<00:04,  2.50it/s]\n"
    )  # type:ignore[assignment]
    runner._runtime.command = "train"  # pylint:disable=protected-access
    progress = []
    stderr = []
    runner.progress.connect(progress.append)
    runner.stderr.connect(stderr.append)

    runner._stderr()  # pylint:disable=protected-access

    assert stderr == []
    assert [event.kind for event in progress] == ["progress", "status"]


def test_job_runner_preserves_plain_stdout_output() -> None:
    """Plain process output should still pass through to stdout."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    runner.process = _BufferedProcessDouble(b"ordinary console line\n")  # type:ignore[assignment]
    runner._runtime.command = "extract"  # pylint:disable=protected-access
    progress = []
    stdout = []
    runner.progress.connect(progress.append)
    runner.stdout.connect(stdout.append)

    runner._stdout()  # pylint:disable=protected-access

    assert stdout == ["ordinary console line\n"]
    assert progress == []


def test_job_runner_accepts_command_when_starting(monkeypatch) -> None:
    """Starting a process should set the runtime command before output is parsed."""
    from lib.gui.qt_shell import job_runner

    class _SignalDouble:
        def connect(self, _callback):  # type:ignore[no-untyped-def]
            return None

    class _ProcessStartDouble:
        def __init__(self, _parent=None) -> None:  # type:ignore[no-untyped-def]
            self.readyReadStandardOutput = _SignalDouble()
            self.readyReadStandardError = _SignalDouble()
            self.errorOccurred = _SignalDouble()
            self.finished = _SignalDouble()
            self.program = None
            self.arguments = None
            self.started = False

        def state(self):  # type:ignore[no-untyped-def]
            return QProcess.NotRunning

        def setProgram(self, program):  # type:ignore[no-untyped-def]
            self.program = program

        def setArguments(self, arguments):  # type:ignore[no-untyped-def]
            self.arguments = arguments

        def start(self):  # type:ignore[no-untyped-def]
            self.started = True

    monkeypatch.setattr(job_runner, "QProcess", _ProcessStartDouble)
    runner = job_runner.JobRunner()

    runner.start(["python", "-u", "/repo/faceswap.py", "extract"], command="convert")

    assert runner._runtime.command == "convert"  # pylint:disable=protected-access
    assert runner.process.started is True


def test_job_runner_infers_command_from_argv(monkeypatch) -> None:
    """Existing MainWindow start(args) calls should still provide a runtime command."""
    from lib.gui.qt_shell import job_runner

    class _SignalDouble:
        def connect(self, _callback):  # type:ignore[no-untyped-def]
            return None

    class _ProcessStartDouble:
        def __init__(self, _parent=None) -> None:  # type:ignore[no-untyped-def]
            self.readyReadStandardOutput = _SignalDouble()
            self.readyReadStandardError = _SignalDouble()
            self.errorOccurred = _SignalDouble()
            self.finished = _SignalDouble()

        def state(self):  # type:ignore[no-untyped-def]
            return QProcess.NotRunning

        def setProgram(self, _program):  # type:ignore[no-untyped-def]
            return None

        def setArguments(self, _arguments):  # type:ignore[no-untyped-def]
            return None

        def start(self):  # type:ignore[no-untyped-def]
            return None

    monkeypatch.setattr(job_runner, "QProcess", _ProcessStartDouble)
    runner = job_runner.JobRunner()

    runner.start(["python", "-u", "/repo/tools.py", "effmpeg"])

    assert runner._runtime.command == "effmpeg"  # pylint:disable=protected-access
