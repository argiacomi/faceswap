#!/usr/bin/env python3
"""Tests for Preview live-refresh lifecycle hooks."""

from __future__ import annotations

import pytest

from lib.gui.qt_shell.preview_job_lifecycle import install_preview_job_lifecycle


class _PreviewPanelDouble:
    """Small PreviewPanel stand-in."""

    def __init__(self) -> None:
        self.start_count = 0
        self.stop_count = 0

    def start_live_refresh(self) -> None:
        """Capture start requests."""
        self.start_count += 1

    def stop_live_refresh(self) -> None:
        """Capture stop requests."""
        self.stop_count += 1


class _CommandPanelDouble:
    """Small CommandPanel stand-in."""

    def __init__(self, command: str, values: dict[str, object]) -> None:
        self._command = command
        self._values = values

    def command_spec(self) -> tuple[str, str, dict[str, object]]:
        """Return command state."""
        return "faceswap", self._command, self._values


def _window_class(command: str, values: dict[str, object], *, fail_run: bool = False):
    """Return a fresh MainWindow-like class for lifecycle hook tests."""

    class _WindowDouble:
        def __init__(self) -> None:
            self._command_panel = _CommandPanelDouble(command, values)
            self._preview_panel_widget = _PreviewPanelDouble()
            self._running = False
            self.calls: list[str] = []

        def _run_command(self) -> None:
            self.calls.append("run")
            if fail_run:
                raise RuntimeError("run failed")
            self._running = True

        def _stop_job(self) -> None:
            self.calls.append("stop")

        def _job_finished(self, exit_code: int) -> None:
            self.calls.append(f"finish:{exit_code}")
            self._running = False

        def _reset_project_state(self) -> None:
            self.calls.append("reset")

        def _apply_project(self, *args, **kwargs) -> bool:  # type:ignore[no-untyped-def]
            self.calls.append("apply")
            return True

        def _open_session_file(self, *args, **kwargs) -> bool:  # type:ignore[no-untyped-def]
            self.calls.append("open")
            return True

        def _reload_current_file(self) -> bool:
            self.calls.append("reload")
            return True

    install_preview_job_lifecycle(_WindowDouble)
    return _WindowDouble


def test_extract_job_with_preview_output_starts_live_refresh() -> None:
    """Extract jobs with output paths should start Preview polling after launch."""
    window = _window_class("extract", {"-o": "/preview"})()

    window._run_command()

    assert window._preview_panel_widget.start_count == 1
    assert window._preview_panel_widget.stop_count == 0
    assert window.calls == ["run"]


def test_convert_job_with_preview_output_starts_live_refresh() -> None:
    """Convert jobs with output paths should start Preview polling after launch."""
    window = _window_class("convert", {"-o": "/preview"})()

    window._run_command()

    assert window._preview_panel_widget.start_count == 1
    assert window._preview_panel_widget.stop_count == 0


def test_train_job_does_not_start_preview_live_refresh() -> None:
    """Train jobs should not start Preview polling because train preview is separate."""
    window = _window_class("train", {"-m": "/models", "-t": "model"})()

    window._run_command()

    assert window._preview_panel_widget.start_count == 0
    assert window._preview_panel_widget.stop_count == 1


def test_preview_capable_job_without_output_stops_live_refresh() -> None:
    """Extract/convert jobs without output paths should not leave Preview polling active."""
    window = _window_class("extract", {})()

    window._run_command()

    assert window._preview_panel_widget.start_count == 0
    assert window._preview_panel_widget.stop_count == 1


def test_failed_run_stops_live_refresh_and_reraises() -> None:
    """Run failures should stop Preview polling immediately."""
    window = _window_class("extract", {"-o": "/preview"}, fail_run=True)()

    with pytest.raises(RuntimeError, match="run failed"):
        window._run_command()

    assert window._preview_panel_widget.start_count == 0
    assert window._preview_panel_widget.stop_count == 1


def test_run_that_does_not_enter_running_state_stops_live_refresh() -> None:
    """A handled launch failure should stop Preview polling when running remains false."""
    Window = _window_class("extract", {"-o": "/preview"})

    def _run_without_start(self) -> None:  # type:ignore[no-untyped-def]
        self.calls.append("run")
        self._running = False

    Window._run_command = _run_without_start
    install_preview_job_lifecycle(Window)
    window = Window()

    window._run_command()

    assert window._preview_panel_widget.start_count == 0
    assert window._preview_panel_widget.stop_count == 1


def test_stop_job_stops_live_refresh_before_stopping_process() -> None:
    """Manual stop should stop Preview polling before delegating to the job runner."""
    window = _window_class("extract", {"-o": "/preview"})()

    window._stop_job()

    assert window._preview_panel_widget.stop_count == 1
    assert window.calls == ["stop"]


def test_job_finished_stops_live_refresh_before_finish_handling() -> None:
    """Finished jobs should stop Preview polling before normal finish handling."""
    window = _window_class("extract", {"-o": "/preview"})()

    window._job_finished(0)

    assert window._preview_panel_widget.stop_count == 1
    assert window.calls == ["finish:0"]


def test_project_reset_stops_live_refresh() -> None:
    """Project close/new flows should stop Preview polling through reset."""
    window = _window_class("extract", {"-o": "/preview"})()

    window._reset_project_state()

    assert window._preview_panel_widget.stop_count == 1
    assert window.calls == ["reset"]


def test_project_apply_stops_live_refresh() -> None:
    """Opening or restoring a project should stop active Preview polling."""
    window = _window_class("extract", {"-o": "/preview"})()

    assert window._apply_project("project.fsw", object()) is True

    assert window._preview_panel_widget.stop_count == 1
    assert window.calls == ["apply"]


def test_open_session_file_stops_live_refresh_even_before_load_result() -> None:
    """Opening another project/task should stop Preview polling before loading."""
    window = _window_class("extract", {"-o": "/preview"})()

    assert window._open_session_file("project.fsw") is True

    assert window._preview_panel_widget.stop_count == 1
    assert window.calls == ["open"]


def test_reload_current_file_stops_live_refresh() -> None:
    """Reloading the current file should stop Preview polling."""
    window = _window_class("extract", {"-o": "/preview"})()

    assert window._reload_current_file() is True

    assert window._preview_panel_widget.stop_count == 1
    assert window.calls == ["reload"]
