#!/usr/bin/env python3
"""MainWindow-level tests for Qt project/session lifecycle parity."""

from __future__ import annotations

from pathlib import Path

from lib.gui.models.project import ProjectFile
from lib.gui.qt_shell.display_controller import DisplayController
from lib.gui.qt_shell.main_window import MainWindow
from lib.gui.services.project_session_service import PROJECT_KIND, TASK_KIND, ProjectSessionService


class _Status:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str, _timeout: int | None = None) -> None:  # noqa:N802
        self.messages.append(message)


class _Console:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, line: str = "") -> None:
        self.lines.append(line)


class _RecentFiles:
    def __init__(self) -> None:
        self.added: list[tuple[str, str]] = []
        self.removed: list[str] = []

    def add(self, filename: str, kind: str) -> None:
        self.added.append((filename, kind))

    def remove(self, filename: str) -> None:
        self.removed.append(filename)


class _LastSession:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, dict[str, object]]] = []

    def save(self, filename: str, kind: str, ui_state: dict[str, object]) -> None:
        self.saved.append((filename, kind, dict(ui_state)))


class _ProjectStore:
    def __init__(self, *, fail_save: bool = False) -> None:
        self.fail_save = fail_save
        self.saved: list[tuple[str, ProjectFile]] = []

    def save(self, filename: str, project: ProjectFile) -> None:
        if self.fail_save:
            raise OSError("save failed")
        self.saved.append((filename, project))


class _CommandPanel:
    def __init__(self) -> None:
        self.command = "extract"
        self.values: dict[str, object] = {"-i": "in"}
        self.set_commands: list[tuple[str, dict[str, object]]] = []
        self.clear_count = 0

    def command_spec(self) -> tuple[str, str, dict[str, object]]:
        return "faceswap", self.command, dict(self.values)

    def set_command(self, command: str, values: dict[str, object]) -> None:
        self.command = command
        self.values = dict(values)
        self.set_commands.append((command, dict(values)))

    def clear_values(self) -> None:
        self.clear_count += 1
        self.values = {}


class _Panel:
    def __init__(self, source: str | None = None) -> None:
        self.source_path = source
        self.cleared = False
        self.restored: list[str | None] = []

    def clear_session(self) -> None:
        self.cleared = True

    def clear_preview(self) -> None:
        self.cleared = True

    def clear_graph(self) -> None:
        self.cleared = True

    def restore_source(self, source: str | None) -> bool:
        self.restored.append(source)
        self.source_path = source
        return bool(source)


class _Tabs:
    def __init__(self) -> None:
        self.current = 1
        self.tabs = [
            DisplayController.ANALYSIS,
            DisplayController.PREVIEW,
            DisplayController.GRAPH,
        ]
        self.visible: list[tuple[int, bool]] = []

    def currentIndex(self) -> int:  # noqa:N802
        return self.current

    def tabText(self, index: int) -> str:  # noqa:N802
        return self.tabs[index]

    def count(self) -> int:
        return len(self.tabs)

    def setTabVisible(self, index: int, visible: bool) -> None:  # noqa:N802
        self.visible.append((index, visible))

    def setCurrentIndex(self, index: int) -> None:  # noqa:N802
        self.current = index


class _Splitter:
    def __init__(self, sizes: list[int]) -> None:
        self._sizes = sizes
        self.restored: list[list[int]] = []

    def sizes(self) -> list[int]:
        return list(self._sizes)

    def setSizes(self, sizes: list[int]) -> None:  # noqa:N802
        self.restored.append(list(sizes))
        self._sizes = list(sizes)


class _Window:
    def __init__(self) -> None:
        self._session_service = ProjectSessionService()
        self._project_store = _ProjectStore()
        self._recent_files = _RecentFiles()
        self._last_session = _LastSession()
        self._project = ProjectFile()
        self._project_filename = None
        self._file_kind = PROJECT_KIND
        self._modified = False
        self._suppress_dirty = False
        self._command_panel = _CommandPanel()
        self._console = _Console()
        self._status = _Status()
        self._main_splitter = _Splitter([400, 800])
        self._vertical_splitter = _Splitter([500, 200])
        self._display_tabs_widget = _Tabs()
        self._analysis_panel_widget = _Panel("/models/model_state.json")
        self._preview_panel_widget = _Panel("/tmp/preview")
        self._graph_panel_widget = _Panel("/models/model_state.json")
        self._display_controller = None
        self.preview_stopped = 0
        self.synced = 0
        self.errors: list[str] = []
        self.titles: list[str] = []
        self.refreshed_recent = 0
        self.size = [1280, 760]

    def width(self) -> int:
        return self.size[0]

    def height(self) -> int:
        return self.size[1]

    def resize(self, width: int, height: int) -> None:
        self.size = [width, height]

    def statusBar(self) -> _Status:  # noqa:N802
        return self._status

    def setWindowTitle(self, title: str) -> None:  # noqa:N802
        self.titles.append(title)

    def _stop_preview_live_refresh(self) -> None:
        self.preview_stopped += 1

    def _sync_view_actions(self) -> None:
        self.synced += 1

    def _show_error(self, message: str) -> None:
        self.errors.append(message)

    def _refresh_recent_menu(self) -> None:
        self.refreshed_recent += 1

    def _apply_preview_context(self, _context) -> None:  # type:ignore[no-untyped-def]
        return None

    def _apply_graph_context(self, _context) -> None:  # type:ignore[no-untyped-def]
        return None

    _capture_ui_state = MainWindow._capture_ui_state
    _restore_splitter = staticmethod(MainWindow._restore_splitter)
    _string_or_none = staticmethod(MainWindow._string_or_none)
    _restore_display_tab = MainWindow._restore_display_tab
    _set_modified = MainWindow._set_modified
    _update_window_title = MainWindow._update_window_title
    _mark_modified_from_user_event = MainWindow._mark_modified_from_user_event


def test_capture_ui_state_includes_display_and_runtime_sources() -> None:
    window = _Window()

    assert MainWindow._capture_ui_state(window) == {  # type:ignore[arg-type]
        "window_size": [1280, 760],
        "display_tab": DisplayController.PREVIEW,
        "main_splitter": [400, 800],
        "vertical_splitter": [500, 200],
        "analysis_source": "/models/model_state.json",
        "preview_source": "/tmp/preview",
        "graph_source": "/models/model_state.json",
    }


def test_restore_ui_state_restores_display_and_runtime_sources() -> None:
    window = _Window()

    MainWindow._restore_ui_state(  # type:ignore[arg-type]
        window,
        {
            "window_size": [1440, 900],
            "display_tab": DisplayController.GRAPH,
            "main_splitter": [300, 900],
            "vertical_splitter": [600, 250],
            "analysis_source": "/restore/analysis_state.json",
            "preview_source": "/restore/preview",
            "graph_source": "/restore/graph_state.json",
        },
    )

    assert window.size == [1440, 900]
    assert window._main_splitter.restored == [[300, 900]]
    assert window._vertical_splitter.restored == [[600, 250]]
    assert window._display_tabs_widget.current == 2
    assert window._analysis_panel_widget.restored == ["/restore/analysis_state.json"]
    assert window._preview_panel_widget.restored == ["/restore/preview"]
    assert window._graph_panel_widget.restored == ["/restore/graph_state.json"]


def test_save_file_updates_recent_last_session_and_menu(tmp_path: Path) -> None:
    window = _Window()
    filename = str(tmp_path / "project.fsw")
    project = ProjectFile(tab_name="extract", tasks={"extract": {"-i": "in"}})

    assert MainWindow._save_file(window, filename, project, PROJECT_KIND) is True  # type:ignore[arg-type]

    assert window._project_store.saved == [(filename, project)]
    assert window._recent_files.added == [(filename, PROJECT_KIND)]
    assert window._last_session.saved[-1][0:2] == (filename, PROJECT_KIND)
    assert window._last_session.saved[-1][2]["preview_source"] == "/tmp/preview"
    assert window.refreshed_recent == 1


def test_save_file_failure_does_not_update_session_state(tmp_path: Path) -> None:
    window = _Window()
    window._project_store = _ProjectStore(fail_save=True)

    assert (
        MainWindow._save_file(  # type:ignore[arg-type]
            window, str(tmp_path / "project.fsw"), ProjectFile(), PROJECT_KIND
        )
        is False
    )

    assert window.errors == ["save failed"]
    assert window._recent_files.added == []
    assert window._last_session.saved == []
    assert window.refreshed_recent == 0


def test_open_missing_file_removes_recent_and_keeps_current_state(tmp_path: Path) -> None:
    window = _Window()
    filename = str(tmp_path / "missing.fsw")

    assert MainWindow._open_session_file(window, filename, PROJECT_KIND) is False  # type:ignore[arg-type]

    assert window._recent_files.removed == [filename]
    assert window.refreshed_recent == 1
    assert "Recent file no longer exists" in window._status.messages[-1]
    assert window._project_filename is None


def test_apply_project_restores_command_context_and_last_session(tmp_path: Path) -> None:
    window = _Window()
    filename = str(tmp_path / "task.fst")
    project = ProjectFile(tab_name="train", tasks={"train": {"-m": "/models", "-t": "model"}})

    assert MainWindow._apply_project(window, filename, project, TASK_KIND) is True  # type:ignore[arg-type]

    assert window.preview_stopped == 1
    assert window._project is project
    assert window._project_filename == filename
    assert window._file_kind == TASK_KIND
    assert window._command_panel.set_commands == [("train", {"-m": "/models", "-t": "model"})]
    assert window._recent_files.added == [(filename, TASK_KIND)]
    assert window._last_session.saved[-1][0:2] == (filename, TASK_KIND)
    assert window._modified is False


def test_reset_project_state_clears_command_displays_and_dirty_state() -> None:
    window = _Window()
    window._modified = True
    window._project_filename = "/tmp/project.fsw"
    window._file_kind = TASK_KIND

    MainWindow._reset_project_state(window)  # type:ignore[arg-type]

    assert window.preview_stopped == 1
    assert window._project_filename is None
    assert window._file_kind == PROJECT_KIND
    assert window._command_panel.clear_count == 1
    assert window._analysis_panel_widget.cleared is True
    assert window._preview_panel_widget.cleared is True
    assert window._graph_panel_widget.cleared is True
    assert window._modified is False
    assert window.synced == 1


def test_window_title_dirty_state_matches_session_service_format(tmp_path: Path) -> None:
    window = _Window()
    window._project_filename = str(tmp_path / "sample.fsw")

    MainWindow._set_modified(window, True)  # type:ignore[arg-type]
    MainWindow._set_modified(window, False)  # type:ignore[arg-type]

    assert window.titles[-2:] == [
        "Faceswap Qt Shell Prototype - sample.fsw*",
        "Faceswap Qt Shell Prototype - sample.fsw",
    ]


def test_mark_modified_respects_restore_suppression() -> None:
    window = _Window()
    window._suppress_dirty = True

    MainWindow._mark_modified_from_user_event(window)  # type:ignore[arg-type]
    assert window._modified is False

    window._suppress_dirty = False
    MainWindow._mark_modified_from_user_event(window)  # type:ignore[arg-type]
    assert window._modified is True
