#!/usr/bin/env python3
"""Preview live-refresh lifecycle hooks for the Qt shell."""

from __future__ import annotations

import functools
import typing as T

from lib.gui.services.command_context import CommandExecutionContext
from lib.utils import get_module_objects

_PREVIEW_COMMANDS = {"extract", "convert"}


def install_preview_job_lifecycle(main_window_class: type) -> None:
    """Install Preview live-refresh hooks on the Qt MainWindow class."""
    if getattr(main_window_class, "_preview_job_lifecycle_installed", False):
        return

    original_run = main_window_class._run_command
    original_stop = main_window_class._stop_job
    original_finish = main_window_class._job_finished
    original_reset = main_window_class._reset_project_state
    original_apply_project = main_window_class._apply_project
    original_open_file = main_window_class._open_session_file
    original_reload = main_window_class._reload_current_file

    @functools.wraps(original_run)
    def run_command(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        command_context = _command_context(self)
        try:
            result = original_run(self, *args, **kwargs)
        except Exception:
            _stop_preview_live_refresh(self)
            raise
        if getattr(self, "_running", False):
            _start_preview_live_refresh(self, command_context)
        else:
            _stop_preview_live_refresh(self)
        return result

    @functools.wraps(original_stop)
    def stop_job(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        _stop_preview_live_refresh(self)
        return original_stop(self, *args, **kwargs)

    @functools.wraps(original_finish)
    def job_finished(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        _stop_preview_live_refresh(self)
        return original_finish(self, *args, **kwargs)

    @functools.wraps(original_reset)
    def reset_project_state(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        _stop_preview_live_refresh(self)
        return original_reset(self, *args, **kwargs)

    @functools.wraps(original_apply_project)
    def apply_project(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        _stop_preview_live_refresh(self)
        return original_apply_project(self, *args, **kwargs)

    @functools.wraps(original_open_file)
    def open_session_file(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        _stop_preview_live_refresh(self)
        return original_open_file(self, *args, **kwargs)

    @functools.wraps(original_reload)
    def reload_current_file(self, *args, **kwargs):  # type:ignore[no-untyped-def]
        _stop_preview_live_refresh(self)
        return original_reload(self, *args, **kwargs)

    main_window_class._run_command = run_command
    main_window_class._stop_job = stop_job
    main_window_class._job_finished = job_finished
    main_window_class._reset_project_state = reset_project_state
    main_window_class._apply_project = apply_project
    main_window_class._open_session_file = open_session_file
    main_window_class._reload_current_file = reload_current_file
    main_window_class._preview_job_lifecycle_installed = True


def _command_context(window: object) -> tuple[str, CommandExecutionContext] | None:
    """Return current command context for preview lifecycle decisions."""
    panel = getattr(window, "_command_panel", None)
    if panel is None:
        return None
    try:
        _category, command, values = panel.command_spec()
    except (AttributeError, TypeError, ValueError):
        return None
    if not isinstance(command, str):
        return None
    if not isinstance(values, T.Mapping):
        return None
    return command, CommandExecutionContext.from_values(command, values)


def _start_preview_live_refresh(
    window: object,
    command_context: tuple[str, CommandExecutionContext] | None,
) -> None:
    """Start live refresh for extract/convert jobs with preview output."""
    if command_context is None:
        return
    command, context = command_context
    if command not in _PREVIEW_COMMANDS or context.preview_output_path is None:
        _stop_preview_live_refresh(window)
        return
    preview_panel = getattr(window, "_preview_panel_widget", None)
    if preview_panel is None:
        return
    start_live_refresh = getattr(preview_panel, "start_live_refresh", None)
    if callable(start_live_refresh):
        start_live_refresh()


def _stop_preview_live_refresh(window: object) -> None:
    """Stop live refresh if the Preview panel has been created."""
    preview_panel = getattr(window, "_preview_panel_widget", None)
    if preview_panel is None:
        return
    stop_live_refresh = getattr(preview_panel, "stop_live_refresh", None)
    if callable(stop_live_refresh):
        stop_live_refresh()


__all__ = get_module_objects(__name__)
