#!/usr/bin/env python3
"""Smoke tests for the Qt shell prototype."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QLineEdit, QPushButton


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
            "Input Dir": _CliOption(_PanelOption("Input Dir", str), ("-i", "--input-dir")),
            "Batch Mode": _CliOption(_PanelOption("Batch Mode", bool, False), ("-b",)),
            "Mode": _CliOption(_PanelOption("Mode", str, "one", ("one", "two")), ("--mode",)),
            "Files": _CliOption(_PanelOption("Files", str), ("--files",), "+"),
            "helptext": "ignored",
        }
    }


def _force_cpu_backend(monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """Force stable CPU CLI metadata for real CLI discovery tests."""
    monkeypatch.setenv("FACESWAP_BACKEND", "cpu")
    from lib.utils import set_backend

    set_backend("cpu")


def test_qt_shell_imports_without_tk_config() -> None:
    """The Qt shell should import without initializing Tk GUI configuration."""
    sys.modules.pop("lib.gui.control_helper", None)

    import lib.gui.qt_shell.main_window  # noqa: F401

    assert "lib.gui.control_helper" not in sys.modules


def test_qt_shell_does_not_initialize_tk_config(monkeypatch, qtbot) -> None:  # type:ignore[no-untyped-def]
    """The Qt shell should construct without initializing Tk GUI config."""
    _force_cpu_backend(monkeypatch)

    import lib.gui.utils as gui_utils

    def fail(*args, **kwargs):  # type:ignore[no-untyped-def]
        raise AssertionError("Tk GUI config initialized")

    monkeypatch.setattr(gui_utils, "initialize_config", fail)
    monkeypatch.setattr(gui_utils, "initialize_images", fail)

    from lib.gui.qt_shell.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)


def test_main_window_constructs(qtbot, monkeypatch, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The Qt shell main window should construct without starting the app loop."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _force_cpu_backend(monkeypatch)

    from lib.gui.qt_shell.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    assert "Faceswap Qt Shell Prototype" in window.windowTitle()


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


def test_command_schema_service_discovers_real_faceswap_cli_metadata(
    monkeypatch,
) -> None:
    """CommandSchemaService should build extract/train/convert from real CLI metadata."""
    _force_cpu_backend(monkeypatch)
    sys.modules.pop("lib.gui.control_helper", None)

    from lib.gui.qt_shell.command_schema_service import CommandSchemaService

    schema = CommandSchemaService().from_real_cli_metadata(categories=("faceswap",))
    extract_options = {option.switch: option for option in schema.options("extract")}

    assert schema.categories == ("faceswap",)
    assert schema.commands("faceswap")[:3] == ("extract", "train", "convert")
    assert {"-i", "-o", "-D", "-A"}.issubset(extract_options)
    assert {"-A", "-B", "-m", "-t"}.issubset({option.switch for option in schema.options("train")})
    assert {"-i", "-o", "-m", "-w"}.issubset(
        {option.switch for option in schema.options("convert")}
    )
    assert extract_options["-i"].group == "Data"
    assert extract_options["-i"].browser_modes == ("folder", "file")
    assert "Input directory" in extract_options["-i"].helptext
    assert "lib.gui.control_helper" not in sys.modules


def test_real_cli_metadata_renders_extract_train_convert(qtbot, monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """CommandPanel should render real extract/train/convert CLI metadata."""
    _force_cpu_backend(monkeypatch)

    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema_service import CommandSchemaService

    schema = CommandSchemaService().from_real_cli_metadata(categories=("faceswap",))
    panel = CommandPanel(schema)
    qtbot.addWidget(panel)

    for command, switches in {
        "extract": {"-i", "-o", "-D", "-A"},
        "train": {"-A", "-B", "-m", "-t"},
        "convert": {"-i", "-o", "-m", "-w"},
    }.items():
        panel.set_command(command, {})
        assert panel.command_spec()[0:2] == ("faceswap", command)
        assert switches.issubset(set(panel.renderer.rendered_switches))


def test_command_schema_renders_groups_tooltips_and_browse_buttons(qtbot) -> None:  # type:ignore[no-untyped-def]
    """CommandPanel should expose group/help/path metadata without full Tk parity."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec(
                        "Input Dir",
                        "-i",
                        group="Data",
                        helptext="Pick an input directory or file.",
                        browser_modes=("folder", "file"),
                    ),
                ),
            ),
        )
    )

    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    renderer = panel.renderer
    input_widget = renderer.widget_for_switch("-i")
    labels = {
        label.text().replace("<b>", "").replace("</b>", "")
        for label in renderer.findChildren(QLabel)
    }

    assert isinstance(input_widget, QLineEdit)
    assert input_widget.toolTip() == "Pick an input directory or file."
    assert "Data" in labels
    assert [button.text() for button in renderer.findChildren(QPushButton)] == [
        "Folder",
        "File",
    ]


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


def test_store_false_option_emits_when_unchecked(qtbot) -> None:  # type:ignore[no-untyped-def]
    """store_false checkboxes should emit their switch when unchecked."""
    from lib.gui.qt_shell.command_panel import CommandPanel
    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.services.command_builder import CommandBuilder

    schema = CommandSchema(
        (
            CommandSpec(
                "faceswap",
                "extract",
                (
                    OptionSpec(
                        "Enabled Feature",
                        "--disable-feature",
                        bool,
                        True,
                        action="store_false",
                    ),
                ),
            ),
        )
    )

    panel = CommandPanel(schema)
    qtbot.addWidget(panel)
    widget = panel.renderer.widget_for_switch("--disable-feature")
    assert isinstance(widget, QCheckBox)

    assert widget.isChecked() is True
    assert CommandBuilder.build_options(panel.command_spec()[2]) == []

    widget.setChecked(False)

    assert panel.command_spec()[2] == {"--disable-feature": True}
    assert CommandBuilder.build_options(panel.command_spec()[2]) == ["--disable-feature"]

    panel.set_command("extract", {"--disable-feature": True})

    assert widget.isChecked() is False
    assert CommandBuilder.build_options(panel.command_spec()[2]) == ["--disable-feature"]


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
