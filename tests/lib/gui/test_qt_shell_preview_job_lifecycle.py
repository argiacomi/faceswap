#!/usr/bin/env python3
"""Tests for MainWindow Preview live-refresh lifecycle behavior."""

from __future__ import annotations

from pathlib import Path

import pytest


def _main_window(qtbot, monkeypatch, tmp_path: Path):  # type:ignore[no-untyped-def]
    """Return a MainWindow with deterministic command metadata."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from lib.gui.qt_shell.command_schema import CommandSchema, CommandSpec, OptionSpec
    from lib.gui.qt_shell.main_window import MainWindow

    schema = CommandSchema(
        (
            CommandSpec("faceswap", "extract", (OptionSpec("Output", "-o"),)),
            CommandSpec("faceswap", "convert", (OptionSpec("Output", "-o"),)),
            CommandSpec(
                "faceswap",
                "train",
                (OptionSpec("Model Dir", "-m"), OptionSpec("Trainer", "-t")),
            ),
        )
    )
    window = MainWindow(schema)
    qtbot.addWidget(window)
    return window


def _capture_preview_refresh(window, monkeypatch):  # type:ignore[no-untyped-def]
    """Patch Preview live refresh methods and return captured calls."""
    calls: list[str] = []
    preview = window._preview_panel_widget  # pylint:disable=protected-access
    assert preview is not None
    monkeypatch.setattr(preview, "start_live_refresh", lambda: calls.append("start"))
    monkeypatch.setattr(preview, "stop_live_refresh", lambda: calls.append("stop"))
    return calls


def _patch_runner_start(
    window, monkeypatch, *, fail: bool = False
) -> list[tuple[str, tuple[str, ...]]]:  # type:ignore[no-untyped-def]
    """Patch the job runner start method and return captured invocations."""
    starts: list[tuple[str, tuple[str, ...]]] = []

    def start(args, *, command: str) -> None:  # type:ignore[no-untyped-def]
        if fail:
            raise RuntimeError("run failed")
        starts.append((command, tuple(args)))

    monkeypatch.setattr(window._runner, "start", start)  # pylint:disable=protected-access
    return starts


def test_extract_job_with_preview_output_starts_live_refresh(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Extract jobs with output paths should start Preview polling after launch."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    calls = _capture_preview_refresh(window, monkeypatch)
    starts = _patch_runner_start(window, monkeypatch)
    window._command_panel.set_command("extract", {"-o": "/preview"})  # pylint:disable=protected-access

    window._run_command()  # pylint:disable=protected-access

    assert calls == ["start"]
    assert starts[0][0] == "extract"


def test_convert_job_with_preview_output_starts_live_refresh(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Convert jobs with output paths should start Preview polling after launch."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    calls = _capture_preview_refresh(window, monkeypatch)
    _patch_runner_start(window, monkeypatch)
    window._command_panel.set_command("convert", {"-o": "/preview"})  # pylint:disable=protected-access

    window._run_command()  # pylint:disable=protected-access

    assert calls == ["start"]


def test_train_job_does_not_start_preview_live_refresh(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Train jobs should not start Preview polling because train preview is separate."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    calls = _capture_preview_refresh(window, monkeypatch)
    _patch_runner_start(window, monkeypatch)
    window._command_panel.set_command("train", {"-m": "/models", "-t": "model"})  # pylint:disable=protected-access

    window._run_command()  # pylint:disable=protected-access

    assert calls == ["stop"]


def test_preview_capable_job_without_output_stops_live_refresh(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Extract/convert jobs without output paths should not leave Preview polling active."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    calls = _capture_preview_refresh(window, monkeypatch)
    _patch_runner_start(window, monkeypatch)
    window._command_panel.set_command("extract", {})  # pylint:disable=protected-access

    window._run_command()  # pylint:disable=protected-access

    assert calls == ["stop"]


def test_failed_run_stops_live_refresh_and_shows_error(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Run failures should stop Preview polling immediately."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    calls = _capture_preview_refresh(window, monkeypatch)
    errors: list[str] = []
    _patch_runner_start(window, monkeypatch, fail=True)
    monkeypatch.setattr(window, "_show_error", lambda message: errors.append(message))
    window._command_panel.set_command("extract", {"-o": "/preview"})  # pylint:disable=protected-access

    window._run_command()  # pylint:disable=protected-access

    assert calls == ["stop"]
    assert errors == ["run failed"]


def test_stop_finish_reset_open_and_reload_stop_live_refresh(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Lifecycle reset paths should stop Preview polling."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    calls = _capture_preview_refresh(window, monkeypatch)
    monkeypatch.setattr(window._runner, "stop", lambda: None)  # pylint:disable=protected-access
    monkeypatch.setattr(window, "_confirm_discard_changes", lambda _action: True)
    monkeypatch.setattr(window, "_show_error", lambda _message: None)
    project_file = tmp_path / "project.fsw"
    project_file.write_text("{}", encoding="utf-8")
    window._project_filename = str(project_file)  # pylint:disable=protected-access

    window._stop_job()  # pylint:disable=protected-access
    window._job_finished(0)  # pylint:disable=protected-access
    window._reset_project_state()  # pylint:disable=protected-access
    window._open_session_file(str(project_file))  # pylint:disable=protected-access
    window._project_filename = str(project_file)  # pylint:disable=protected-access
    window._reload_current_file()  # pylint:disable=protected-access

    assert calls == ["stop"] * 8


def test_run_command_still_raises_unexpected_build_errors(  # type:ignore[no-untyped-def]
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Unexpected command-build errors should not be swallowed by preview lifecycle."""
    window = _main_window(qtbot, monkeypatch, tmp_path)
    _capture_preview_refresh(window, monkeypatch)
    monkeypatch.setattr(
        window,
        "_build_command",
        lambda *, generate: (_ for _ in ()).throw(TypeError("bad build")),
    )

    with pytest.raises(TypeError, match="bad build"):
        window._run_command()  # pylint:disable=protected-access
