#!/usr/bin/env python3
"""Focused GUI regression tests."""

from __future__ import annotations

import typing as T
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from lib.gui.display import DisplayNotebook
from lib.gui.display_page import DisplayOptionalPage
from lib.gui.options import CliOption, CliOptions
from lib.gui.services.runtime_events import RuntimeEvent
from lib.gui.services.runtime_service import ProcessRuntimeService
from lib.gui.utils import image as gui_image
from lib.gui.wrapper import FaceswapControl, ProcessWrapper


class _FakePhotoImage:
    """Minimal stand-in for ImageTk.PhotoImage."""

    def __init__(self, image: Image.Image) -> None:
        self.image = image

    def width(self) -> int:
        """Return image width."""
        return self.image.width  # type: ignore[no-any-return]

    def height(self) -> int:
        """Return image height."""
        return self.image.height  # type: ignore[no-any-return]


class _FakePanelOption:
    """Minimal stand-in for ControlPanelOption."""

    def __init__(self, value: T.Any, default: T.Any = None) -> None:
        self._value = value
        self.default = default
        self.set_values: list[T.Any] = []

    def get(self) -> T.Any:
        """Return the current value."""
        return self._value

    def set(self, value: T.Any) -> None:
        """Set the current value."""
        self._value = value
        self.set_values.append(value)


class _FakeVar:
    """Minimal stand-in for a Tk variable."""

    def __init__(self, value: T.Any = None) -> None:
        self.value = value
        self.set_values: list[T.Any] = []

    def get(self) -> T.Any:
        """Return the current value."""
        return self.value

    def set(self, value: T.Any) -> None:
        """Set the current value."""
        self.value = value
        self.set_values.append(value)


class _FakeStatusbar:
    """Minimal stand-in for the GUI status bar."""

    def __init__(self) -> None:
        self.message = _FakeVar()
        self.stop_called = False
        self.modes: list[str] = []
        self.progress_updates: list[tuple[str, int, bool]] = []

    def stop(self) -> None:
        """Record that the status bar was stopped."""
        self.stop_called = True

    def set_mode(self, mode: str) -> None:
        """Record status bar mode updates."""
        self.modes.append(mode)

    def progress_update(self, message: str, position: int, determinate: bool) -> None:
        """Record progress bar updates."""
        self.progress_updates.append((message, position, determinate))


class _FakePsutilProcess:
    """Minimal stand-in for a psutil.Process."""

    def __init__(self, children: list[_FakePsutilProcess] | None = None) -> None:
        self._children = children or []
        self.terminated = False
        self.killed = False

    def children(self, recursive: bool = False) -> list[_FakePsutilProcess]:
        """Return child processes."""
        assert recursive is True
        return self._children

    def terminate(self) -> None:
        """Record termination."""
        self.terminated = True

    def kill(self) -> None:
        """Record kill."""
        self.killed = True


def _preview_extract_with_cache(num_images: int = 4, size: int = 10) -> gui_image.PreviewExtract:
    """Return a PreviewExtract instance with synthetic cached thumbnails."""
    preview = gui_image.PreviewExtract.__new__(gui_image.PreviewExtract)
    preview._images = np.stack(
        [np.full((size, size, 3), fill_value=idx, dtype=np.uint8) for idx in range(num_images)]
    )
    preview._placeholder = None
    preview._preview_image = None
    preview._preview_image_tk = None
    return preview


@pytest.fixture(name="cli_opts")
def cli_opts_fixture() -> CliOptions:
    """Return a CliOptions instance with synthetic options."""
    cli_opts = CliOptions.__new__(CliOptions)
    extract_input = CliOption(_FakePanelOption("current", default="default"), ("-i",), None)  # type: ignore[arg-type]
    extract_flag = CliOption(_FakePanelOption(True, default=True), ("-b",), None)  # type: ignore[arg-type]
    train_input = CliOption(_FakePanelOption("train", default="train_default"), ("-t",), None)  # type: ignore[arg-type]
    cli_opts._opts = {
        "extract": {
            "Input": extract_input,
            "Flag": extract_flag,
            "helptext": "extract help",
        },
        "train": {"Input": train_input, "helptext": "train help"},
    }
    return cli_opts


def test_options_to_process_all_commands(cli_opts: CliOptions) -> None:
    """All commands return only CliOption values, not helptext or dict keys."""
    options = cli_opts._options_to_process()

    assert len(options) == 3
    assert all(isinstance(option, CliOption) for option in options)


def test_options_to_process_single_command(cli_opts: CliOptions) -> None:
    """A single command returns only that command's CliOption values."""
    options = cli_opts._options_to_process("extract")

    assert len(options) == 2
    assert {option.opts[0] for option in options} == {"-i", "-b"}


def test_reset_single_command(cli_opts: CliOptions) -> None:
    """Reset updates only the requested command back to defaults."""
    cli_opts.reset("extract")

    assert cli_opts._opts["extract"]["Input"].panel_option.get() == "default"  # type: ignore[union-attr]
    assert cli_opts._opts["extract"]["Flag"].panel_option.get() is True  # type: ignore[union-attr]
    assert cli_opts._opts["train"]["Input"].panel_option.get() == "train"  # type: ignore[union-attr]


def test_clear_single_command(cli_opts: CliOptions) -> None:
    """Clear updates only the requested command to empty values."""
    cli_opts.clear("extract")

    assert cli_opts._opts["extract"]["Input"].panel_option.get() == ""  # type: ignore[union-attr]
    assert cli_opts._opts["extract"]["Flag"].panel_option.get() is False  # type: ignore[union-attr]
    assert cli_opts._opts["train"]["Input"].panel_option.get() == "train"  # type: ignore[union-attr]


def test_gen_cli_arguments_with_quoted_nargs() -> None:
    """Quoted nargs values preserve mixed quoted and unquoted values."""
    cli_opts = CliOptions.__new__(CliOptions)
    input_option = CliOption(_FakePanelOption('"foo bar" baz'), ("-i",), "+")  # type: ignore[arg-type]
    cli_opts._opts = {"extract": {"Input": input_option, "helptext": "extract help"}}

    with pytest.deprecated_call(match="CliOptions.gen_cli_arguments"):
        args = list(cli_opts.gen_cli_arguments("extract"))

    assert args == [("-i", "foo bar", "baz")]


def test_invalid_quote_rolls_back_gui_state(capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid generated arguments do not leave the GUI marked as running."""
    wrapper = ProcessWrapper.__new__(ProcessWrapper)
    wrapper._tk_vars = SimpleNamespace(  # type: ignore[assignment]
        action_command=_FakeVar("faceswap,extract"),
        running_task=_FakeVar(False),
        is_training=_FakeVar(True),
        display=_FakeVar("extract"),
    )
    wrapper._statusbar = _FakeStatusbar()  # type: ignore[assignment]
    wrapper._task = SimpleNamespace(terminate=lambda: None, execute_script=lambda *_args: None)  # type: ignore[assignment]
    wrapper._command = None
    wrapper._build_args = lambda _category: (_ for _ in ()).throw(ValueError("bad quote"))  # type: ignore[assignment, method-assign, misc]
    wrapper._prepare_after_args_built = lambda: pytest.fail("prepare should not run")  # type: ignore[method-assign]

    wrapper._action_command()

    assert wrapper._tk_vars.running_task.get() is False
    assert wrapper._tk_vars.is_training.get() is False
    assert wrapper._tk_vars.display.get() == ""
    assert wrapper._tk_vars.action_command.get() == ""
    assert wrapper._statusbar.stop_called is True  # type: ignore[attr-defined]
    assert wrapper._statusbar.message.get() == "Invalid command options"
    assert "bad quote" in capsys.readouterr().err


def test_invalid_generate_does_not_clear_running_task(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid Generate options do not reset an already running task."""
    wrapper = ProcessWrapper.__new__(ProcessWrapper)
    wrapper._tk_vars = SimpleNamespace(  # type: ignore[assignment]
        generate_command=_FakeVar("faceswap,extract"),
        running_task=_FakeVar(True),
        is_training=_FakeVar(True),
        display=_FakeVar("train"),
        console_clear=_FakeVar(False),
    )
    wrapper._statusbar = _FakeStatusbar()  # type: ignore[assignment]
    wrapper._command = "train"
    wrapper._build_args = lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad quote"))  # type: ignore[method-assign]

    wrapper._generate_command()

    assert wrapper._tk_vars.running_task.get() is True
    assert wrapper._tk_vars.is_training.get() is True
    assert wrapper._tk_vars.display.get() == "train"
    assert wrapper._tk_vars.console_clear.get() is True
    assert wrapper._tk_vars.generate_command.get() == ""
    assert wrapper._command == "train"
    assert wrapper._statusbar.stop_called is False  # type: ignore[attr-defined]
    assert "bad quote" in capsys.readouterr().err


def test_windows_path_parsing_for_nargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows nargs parsing preserves backslashes in quoted paths."""
    monkeypatch.setattr("lib.gui.options.IS_WINDOWS", True)
    value = r'"C:\Users\Drew\input one.jpg" "C:\Users\Drew\input two.jpg"'

    assert CliOptions._split_nargs(value, "-i") == [
        r"C:\Users\Drew\input one.jpg",
        r"C:\Users\Drew\input two.jpg",
    ]


def test_invalid_nargs_quote_raises_value_error() -> None:
    """Invalid quoted nargs values raise a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid quoted argument for -i"):
        CliOptions._split_nargs('unterminated "quote', "-i")


def test_terminate_targets_launched_process_pid_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Termination targets the launched process tree, not all GUI child processes."""
    from lib.gui import wrapper as wrapper_mod  # pylint:disable=import-outside-toplevel

    child = _FakePsutilProcess()
    root = _FakePsutilProcess(children=[child])
    process_ids: list[int | None] = []

    def fake_process(pid: int | None = None) -> _FakePsutilProcess:
        process_ids.append(pid)
        if pid is None:
            pytest.fail("psutil.Process() must be called with the launched process pid")
        return root

    monkeypatch.setattr(wrapper_mod.psutil, "Process", fake_process)
    monkeypatch.setattr(wrapper_mod.psutil, "wait_procs", lambda procs, timeout: (procs, []))

    control = FaceswapControl.__new__(FaceswapControl)
    control._queue_ui_update = lambda *_args: None  # type: ignore[method-assign]

    control._terminate_process_tree(SimpleNamespace(pid=12345))  # type: ignore[arg-type]

    assert process_ids == [12345]
    assert root.terminated is True
    assert child.terminated is True


def test_faceswap_control_process_stdout_queues_runtime_events() -> None:
    """FaceswapControl should pass parsed stdout through the shared runtime service."""
    control = FaceswapControl.__new__(FaceswapControl)
    control._runtime = ProcessRuntimeService(command="extract")  # pylint:disable=protected-access
    queued: list[tuple[str, object]] = []
    control._queue_ui_update = lambda *args: queued.append(args)  # type:ignore[method-assign]
    control._runtime.add_event_callback(  # pylint:disable=protected-access
        control._queue_runtime_event  # pylint:disable=protected-access
    )

    consumed = control._process_progress_stdout(  # pylint:disable=protected-access
        "Extracting:  25%|##5       | 5/20 [00:02<00:06,  2.50it/s]\n"
    )

    assert consumed is True
    assert len(queued) == 1
    assert queued[0][0] == "runtime_event"
    event = queued[0][1]
    assert isinstance(event, RuntimeEvent)
    assert event.kind == "progress"
    assert event.progress == 25.0


def test_faceswap_control_runtime_progress_event_updates_statusbar() -> None:
    """Runtime progress events should update the legacy Tk status bar."""
    control = FaceswapControl.__new__(FaceswapControl)
    statusbar = _FakeStatusbar()
    control._statusbar = statusbar  # type: ignore[assignment]  # pylint:disable=protected-access

    control._handle_runtime_event(  # pylint:disable=protected-access
        RuntimeEvent("progress", "Halfway", 42.9)
    )

    assert statusbar.progress_updates == [("Halfway", 42, True)]


def test_faceswap_control_runtime_status_event_updates_mode_and_loss() -> None:
    """Runtime status events should preserve training loss status behavior."""
    control = FaceswapControl.__new__(FaceswapControl)
    statusbar = _FakeStatusbar()
    control._statusbar = statusbar  # type: ignore[assignment]  # pylint:disable=protected-access

    control._handle_runtime_event(  # pylint:disable=protected-access
        RuntimeEvent(
            "status",
            "Elapsed: 00:00:01 | loss: 1.0",
            payload={"mode": "indeterminate", "parser": "loss"},
        )
    )

    assert statusbar.modes == ["indeterminate"]
    assert statusbar.progress_updates == [("Elapsed: 00:00:01 | loss: 1.0", 0, False)]


def test_faceswap_control_training_session_event_refreshes_graph() -> None:
    """Training session runtime events should refresh the legacy graph display."""
    control = FaceswapControl.__new__(FaceswapControl)
    refresh_graph = _FakeVar(False)
    control._config = SimpleNamespace(  # type: ignore[assignment]  # pylint:disable=protected-access
        tk_vars=SimpleNamespace(refresh_graph=refresh_graph)
    )
    control._statusbar = _FakeStatusbar()  # type: ignore[assignment]  # pylint:disable=protected-access

    control._handle_runtime_event(  # pylint:disable=protected-access
        RuntimeEvent("training_session", payload={"graph_refresh": True})
    )

    assert refresh_graph.get() is True


def test_display_optional_page_close_cancels_scheduled_callbacks() -> None:
    """Closing an optional page cancels tracked after callbacks before destroying children."""
    page = DisplayOptionalPage.__new__(DisplayOptionalPage)
    page._after_ids = ["retry-1", "retry-2"]
    page._update_after_id = "update-loop"
    cancelled: list[str] = []
    destroyed_children: list[str] = []
    child = SimpleNamespace(destroy=lambda: destroyed_children.append("child"))
    page.after_cancel = lambda after_id: cancelled.append(after_id)  # type: ignore[assignment, method-assign]
    page.winfo_children = lambda: [child]  # type: ignore[list-item, method-assign]
    page.destroy = lambda: pytest.fail("parent notebook owns page destruction")  # type: ignore[method-assign]

    page.close()

    assert cancelled == ["update-loop", "retry-1", "retry-2"]
    assert page._after_ids == []
    assert page._update_after_id is None
    assert destroyed_children == ["child"]


def test_display_notebook_tab_removal_does_not_error_after_close() -> None:
    """Notebook removal can forget and destroy optional tabs after page cleanup."""
    notebook = DisplayNotebook.__new__(DisplayNotebook)
    tabs = ["notebook.analysis", "notebook.preview"]
    notebook._static_tabs = ["notebook.analysis"]
    actions: list[str] = []
    optional_page = SimpleNamespace(
        close=lambda: actions.append("close"),
        destroy=lambda: actions.append("destroy"),
    )
    notebook.children = {"preview": optional_page}  # type: ignore[dict-item]
    notebook.tabs = lambda: list(tabs)  # type: ignore[method-assign]

    def forget(child: str) -> None:
        actions.append(f"forget:{child}")
        tabs.remove(child)

    notebook.forget = forget  # type: ignore[assignment, method-assign]

    notebook._remove_tabs()

    assert actions == ["close", "forget:notebook.preview", "destroy"]
    assert tabs == ["notebook.analysis"]


def test_display_notebook_tab_change_ignores_missing_child() -> None:
    """Tab-change events after removal do not assume the selected child still exists."""
    notebook = DisplayNotebook.__new__(DisplayNotebook)
    notebook.select = lambda: "notebook.missing"  # type: ignore[assignment, method-assign, misc]
    notebook.children = {}

    notebook._on_tab_change(None)


def test_preview_extract_reflow_truncates_cache_when_panel_shrinks() -> None:
    """Preview reflow should truncate cached thumbnails to the resized panel capacity."""
    preview = _preview_extract_with_cache(num_images=4, size=10)

    image = preview._place_previews((10, 10))  # pylint:disable=protected-access

    assert image is not None
    assert image.size == (10, 10)


def test_preview_extract_resize_preview_reflows_cached_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resize_preview should rebuild the PhotoImage from cached thumbnails."""
    monkeypatch.setattr(gui_image.ImageTk, "PhotoImage", _FakePhotoImage)
    preview = _preview_extract_with_cache(num_images=2, size=10)

    updated = preview.resize_preview(thumbnail_size=10, frame_dims=(20, 10))

    assert updated is True
    assert preview._preview_image is not None  # pylint:disable=protected-access
    assert preview._preview_image.size == (20, 10)  # pylint:disable=protected-access
    assert isinstance(preview._preview_image_tk, _FakePhotoImage)  # pylint:disable=protected-access
