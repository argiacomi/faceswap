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

    def write_stdout(self, stdout: bytes) -> None:
        """Replace stdout buffer for another fake ready-read cycle."""
        self._stdout = stdout

    def write_stderr(self, stderr: bytes) -> None:
        """Replace stderr buffer for another fake ready-read cycle."""
        self._stderr = stderr


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


def test_job_runner_handles_mixed_plain_and_tqdm_stdout_chunk() -> None:
    """One stdout read can contain both console output and parsed progress."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    runner.process = _BufferedProcessDouble(
        b"plain line\n"
        b"Extracting:  25%|##5       | 5/20 [00:02<00:06,  2.50it/s]\n"
        b"another plain line\n"
    )  # type:ignore[assignment]
    runner._runtime.command = "extract"  # pylint:disable=protected-access
    progress = []
    stdout = []
    runner.progress.connect(progress.append)
    runner.stdout.connect(stdout.append)

    runner._stdout()  # pylint:disable=protected-access

    assert stdout == ["plain line\n", "another plain line\n"]
    assert len(progress) == 1
    assert progress[0].kind == "progress"
    assert progress[0].progress == 25.0


def test_job_runner_handles_multiple_progress_lines_in_one_chunk() -> None:
    """One stdout read can contain multiple parsed progress lines."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    runner.process = _BufferedProcessDouble(
        b"Extracting:  25%|##5       | 5/20 [00:02<00:06,  2.50it/s]\n"
        b"Extracting:  50%|#####     | 10/20 [00:04<00:04,  2.50it/s]\n"
    )  # type:ignore[assignment]
    runner._runtime.command = "extract"  # pylint:disable=protected-access
    progress = []
    stdout = []
    runner.progress.connect(progress.append)
    runner.stdout.connect(stdout.append)

    runner._stdout()  # pylint:disable=protected-access

    assert stdout == []
    assert [event.progress for event in progress] == [25.0, 50.0]


def test_job_runner_preserves_partial_tqdm_stdout_until_complete() -> None:
    """Progress split across QProcess reads should parse only after terminator arrives."""
    from lib.gui.qt_shell.job_runner import JobRunner

    process = _BufferedProcessDouble(b"Extracting:  25%|##5       | 5/20 [00:02<")
    runner = JobRunner()
    runner.process = process  # type:ignore[assignment]
    runner._runtime.command = "extract"  # pylint:disable=protected-access
    progress = []
    stdout = []
    runner.progress.connect(progress.append)
    runner.stdout.connect(stdout.append)

    runner._stdout()  # pylint:disable=protected-access
    process.write_stdout(b"00:06,  2.50it/s]\n")
    runner._stdout()  # pylint:disable=protected-access

    assert stdout == []
    assert len(progress) == 1
    assert progress[0].progress == 25.0


def test_job_runner_preserves_partial_plain_stdout_until_finished(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Trailing partial plain output should be flushed when the process finishes."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    runner.process = _BufferedProcessDouble(b"partial plain output")  # type:ignore[assignment]
    runner._runtime.command = "extract"  # pylint:disable=protected-access
    stdout = []
    runner.stdout.connect(stdout.append)

    with qtbot.waitSignal(runner.finished, timeout=1000):
        runner._finished(0, QProcess.NormalExit)  # pylint:disable=protected-access

    assert stdout == ["partial plain output"]


def test_job_runner_accepts_command_when_starting(monkeypatch) -> None:
    """Starting a process should set the runtime command before output is parsed."""
    from lib.gui.qt_shell import job_runner

    class _SignalDouble:
        def connect(self, _callback):  # type:ignore[no-untyped-def]
            return None

    class _ProcessStartDouble:
        NotRunning = QProcess.NotRunning

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
        NotRunning = QProcess.NotRunning

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


def test_main_window_display_controller_reacts_to_runner_progress(qtbot) -> None:  # type:ignore[no-untyped-def]
    """MainWindow should route runner RuntimeEvent values to DisplayController."""
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec
    from lib.gui.qt_shell.main_window import MainWindow
    from lib.gui.services.runtime_events import RuntimeEvent

    schema = CommandSchema((CommandSpec("faceswap", "train", ()),))
    window = MainWindow(schema)
    qtbot.addWidget(window)

    assert window._display_controller.visible_tab_names() == ("Analysis",)  # pylint:disable=protected-access

    window._runner.progress.emit(  # pylint:disable=protected-access
        RuntimeEvent(kind="progress", progress=50.0, payload={"command": "train"})
    )

    assert window._display_controller.visible_tab_names() == (  # pylint:disable=protected-access
        "Analysis",
        "Graph",
    )


def test_split_output_units_handles_carriage_return_updates() -> None:
    """Carriage-return progress updates should be split into parseable units."""
    from lib.gui.qt_shell.job_runner import JobRunner

    units, remainder = JobRunner._split_output_units("25%\r50%\r100%\n")

    assert units == ["25%\r", "50%\r", "100%\n"]
    assert remainder == ""


def test_split_output_units_preserves_partial_trailing_output() -> None:
    """Trailing output without a terminator should remain buffered."""
    from lib.gui.qt_shell.job_runner import JobRunner

    units, remainder = JobRunner._split_output_units("plain\npartial")

    assert units == ["plain\n"]
    assert remainder == "partial"


def test_split_output_units_flushes_partial_trailing_output() -> None:
    """Flush should emit the final unterminated output unit."""
    from lib.gui.qt_shell.job_runner import JobRunner

    units, remainder = JobRunner._split_output_units("plain\npartial", flush=True)

    assert units == ["plain\n", "partial"]
    assert remainder == ""
