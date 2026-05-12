#!/usr/bin/env python3
"""Smoke tests for the Qt shell prototype."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit


class _ProcessDouble:
    """Minimal QProcess test double."""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self._state = QProcess.Running

    def state(self) -> QProcess.ProcessState:
        """Return the configured process state."""
        return self._state

    def terminate(self) -> None:
        """Capture terminate calls."""
        self.terminated = True

    def kill(self) -> None:
        """Capture kill calls."""
        self.killed = True

    def readAllStandardOutput(self) -> bytes:
        """Return empty stdout."""
        return b""

    def readAllStandardError(self) -> bytes:
        """Return empty stderr."""
        return b""


class _BufferedProcessDouble(_ProcessDouble):
    """QProcess test double with buffered output."""

    def __init__(self, stdout: bytes, stderr: bytes) -> None:
        super().__init__()
        self._stdout = stdout
        self._stderr = stderr

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


@dataclass(frozen=True)
class _PanelOption:
    """CliOption panel option test double."""

    title: str
    dtype: type
    default: object = ""
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class _CliOption:
    """CliOption test double."""

    panel_option: _PanelOption
    opts: tuple[str, ...]
    nargs: str | None = None


class _CliOptions:
    """CliOptions test double."""

    categories = ("faceswap",)
    commands = {"faceswap": ("extract",)}
    opts = {
        "extract": {
            "Input Dir": _CliOption(
                _PanelOption("Input Dir", str), ("-i", "--input-dir")
            ),
            "Batch Mode": _CliOption(_PanelOption("Batch Mode", bool, False), ("-b",)),
            "Mode": _CliOption(
                _PanelOption("Mode", str, "one", ("one", "two")), ("--mode",)
            ),
            "Files": _CliOption(_PanelOption("Files", str), ("--files",), "+"),
            "helptext": "ignored",
        }
    }


def test_qt_shell_imports_without_tk_config() -> None:
    """The Qt shell should import without initializing Tk GUI configuration."""
    importlib.import_module("lib.gui.qt_shell.main_window")


def test_main_window_constructs(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The Qt shell main window should construct without starting the app loop."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "Faceswap Qt Shell Prototype"


def test_command_schema_service_adapts_cli_like_options() -> None:
    """CommandSchemaService should adapt CliOptions-like metadata without importing Tk."""
    from lib.gui.qt_shell.command_schema_service import CommandSchemaService

    schema = CommandSchemaService().from_cli_options(_CliOptions())
    options = schema.options("extract")

    assert schema.categories == ("faceswap",)
    assert schema.commands("faceswap") == ("extract",)
    assert [option.switch for option in options] == ["-i", "-b", "--mode", "--files"]
    assert options[1].value_type is bool
    assert options[2].choices == ("one", "two")
    assert options[3].nargs is True


def test_command_schema_renders_widgets(qtbot) -> None:  # type:ignore[no-untyped-def]
    """CommandPanel should render widgets from CommandSchema option specs."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec("Input Dir", "-i"),
                    OptionSpec("Batch Mode", "-b", bool, False),
                    OptionSpec("Mode", "--mode", str, "two", ("one", "two")),
                    OptionSpec("Files", "--files", str, "", (), True),
                ),
            ),
        )
    )

    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    renderer = panel.renderer

    assert panel.command_spec()[0:2] == ("faceswap", "extract")
    assert renderer.rendered_switches == ("-i", "-b", "--mode", "--files")
    assert isinstance(renderer.widget_for_switch("-i"), QLineEdit)
    assert isinstance(renderer.widget_for_switch("-b"), QCheckBox)
    assert isinstance(renderer.widget_for_switch("--mode"), QComboBox)


def test_command_schema_widget_values(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Rendered schema widgets should return switch-keyed command values."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec("Input Dir", "-i"),
                    OptionSpec("Batch Mode", "-b", bool, False),
                    OptionSpec("Mode", "--mode", str, "one", ("one", "two")),
                    OptionSpec("Files", "--files", str, "", (), True),
                ),
            ),
        )
    )

    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    renderer = panel.renderer
    input_widget = renderer.widget_for_switch("-i")
    batch_widget = renderer.widget_for_switch("-b")
    mode_widget = renderer.widget_for_switch("--mode")
    files_widget = renderer.widget_for_switch("--files")
    assert isinstance(input_widget, QLineEdit)
    assert isinstance(batch_widget, QCheckBox)
    assert isinstance(mode_widget, QComboBox)
    assert isinstance(files_widget, QLineEdit)

    input_widget.setText("/input")
    batch_widget.setChecked(True)
    mode_widget.setCurrentText("two")
    files_widget.setText('"file one.png" file_two.png')

    assert panel.command_spec() == (
        "faceswap",
        "extract",
        {
            "-i": "/input",
            "-b": True,
            "--mode": "two",
            "--files": ["file one.png", "file_two.png"],
        },
    )


def test_command_schema_nargs_splitting_preserves_windows_paths(monkeypatch) -> None:
    """nargs splitting should not mangle Windows paths with backslashes."""
    from lib.gui.qt_shell import command_panel

    monkeypatch.setattr(command_panel.os, "name", "nt")

    assert command_panel.OptionsFormRenderer._split_nargs(
        r'"C:\Input Folder\one.png" C:\out\two.png'
    ) == [r"C:\Input Folder\one.png", r"C:\out\two.png"]


def test_job_runner_stop_terminates_then_kills_if_still_running(monkeypatch) -> None:
    """Stopping should terminate first, then escalate if the process is still running."""
    from lib.gui.qt_shell import job_runner

    runner = job_runner.JobRunner()
    process = _ProcessDouble()
    runner.process = process  # type:ignore[assignment]
    callbacks = []
    monkeypatch.setattr(
        job_runner.QTimer,
        "singleShot",
        lambda _timeout, callback: callbacks.append(callback),
    )

    runner.stop()
    callbacks[0]()

    assert process.terminated is True
    assert process.killed is True


def test_job_runner_finished_drains_output_before_clearing_process() -> None:
    """Finished jobs should emit final output before clearing the process reference."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    runner.process = _BufferedProcessDouble(b"final stdout", b"final stderr")  # type:ignore[assignment]
    stdout = []
    stderr = []
    runner.stdout.connect(stdout.append)
    runner.stderr.connect(stderr.append)

    runner._finished(0, QProcess.NormalExit)  # pylint:disable=protected-access

    assert stdout == ["final stdout"]
    assert stderr == ["final stderr"]
    assert runner.process is None


def test_job_runner_finished_clears_process(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Finished jobs should clear the old QProcess reference."""
    from lib.gui.qt_shell.job_runner import JobRunner

    runner = JobRunner()
    runner.process = _ProcessDouble()  # type:ignore[assignment]

    with qtbot.waitSignal(runner.finished, timeout=1000) as signal:
        runner._finished(0, QProcess.NormalExit)  # pylint:disable=protected-access

    assert signal.args == [0]
    assert runner.process is None
