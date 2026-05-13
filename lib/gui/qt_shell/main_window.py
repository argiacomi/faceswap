#!/usr/bin/env python3
"""Minimal Qt shell for proving Faceswap GUI services outside Tk."""

from __future__ import annotations

import typing as T
from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QAction, QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from lib.gui.models.project import ProjectFile
from lib.gui.qt_shell.analysis_panel import AnalysisPanel
from lib.gui.qt_shell.command_panel import CommandPanel
from lib.gui.qt_shell.command_schema import CommandSchema
from lib.gui.qt_shell.command_schema_service import CommandSchemaService
from lib.gui.qt_shell.display_controller import DisplayController
from lib.gui.qt_shell.graph_panel import GraphPanel
from lib.gui.qt_shell.job_runner import JobRunner
from lib.gui.qt_shell.preview_panel import PreviewPanel
from lib.gui.services.command_builder import CommandBuilder
from lib.gui.services.command_context import CommandExecutionContext
from lib.gui.services.project_session_service import (
    PROJECT_EXTENSION,
    PROJECT_KIND,
    TASK_EXTENSION,
    TASK_KIND,
    LastSessionStore,
    ProjectSessionService,
)
from lib.gui.services.project_store import ProjectStore
from lib.gui.services.recent_files_store import RecentFilesStore
from lib.serializer import get_serializer
from lib.utils import PROJECT_ROOT


class ConsolePane(QPlainTextEdit):
    """Read-only console output pane."""

    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Generated commands and process output appear here.")

    def write(self, text: str) -> None:
        """Append raw console text."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def write_line(self, text: str = "") -> None:
        """Append one console line."""
        self.write(f"{text}\n")


class MainWindow(QMainWindow):
    """Qt QMainWindow shell that exercises service-layer command generation."""

    def __init__(self, schema: CommandSchema | None = None) -> None:
        super().__init__()
        root = Path(PROJECT_ROOT)
        serializer = get_serializer("json")
        self._schema = self._load_schema() if schema is None else schema
        self._builder = CommandBuilder(base_path=str(root))
        self._project_store = ProjectStore(serializer)
        self._session_service = ProjectSessionService()
        self._recent_files = RecentFilesStore(serializer, str(self._recent_cache()))
        self._last_session = LastSessionStore(serializer, str(self._last_session_cache()))
        self._project = ProjectFile()
        self._project_filename: str | None = None
        self._file_kind = PROJECT_KIND
        self._modified = False
        self._suppress_dirty = False
        self._runner = JobRunner(self)
        self._running = False
        self._command_panel = CommandPanel(self._schema)
        self._console = ConsolePane()
        self._progress = QProgressBar()
        self._menus: list[object] = []
        self._recent_menu: QMenu | None = None
        self._display_tabs_widget: QTabWidget | None = None
        self._display_controller: DisplayController | None = None
        self._preview_panel_widget: PreviewPanel | None = None
        self._graph_panel_widget: GraphPanel | None = None
        self._view_actions: dict[str, QAction] = {}
        self._run_action: QAction | None = None
        self._stop_action: QAction | None = None
        self._run_menu_action: QAction | None = None
        self._stop_menu_action: QAction | None = None
        self._build_ui()
        self._connect_signals()
        self._install_dirty_event_filters()
        self._set_running(False)
        self._set_modified(False)
        self._console.write_line("Faceswap Qt shell prototype")
        self._console.write_line(
            "Scope: real CLI metadata rendering, command generation and QProcess jobs"
        )

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa:N802
        """Track command-panel user edits for dirty-state updates."""
        event_type = event.type()
        if event_type == QEvent.Type.ChildAdded:
            child = event.child()  # type:ignore[attr-defined]
            if isinstance(child, QWidget):
                self._install_dirty_event_filters(child)
        elif event_type in {
            QEvent.Type.KeyRelease,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.Wheel,
        }:
            self._mark_modified_from_user_event()
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Prompt before closing with unsaved project changes."""
        if self._confirm_discard_changes("close this window"):
            event.accept()
        else:
            event.ignore()

    def _build_ui(self) -> None:
        """Build the Qt shell layout."""
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        top = QSplitter(Qt.Horizontal)
        top.setObjectName("qt-shell-main-splitter")
        top.setChildrenCollapsible(False)
        top.addWidget(self._command_panel)
        top.addWidget(self._display_tabs())
        top.setStretchFactor(0, 0)
        top.setStretchFactor(1, 1)
        top.setSizes([420, 840])
        main = QSplitter(Qt.Vertical)
        main.setObjectName("qt-shell-vertical-splitter")
        main.setChildrenCollapsible(False)
        main.addWidget(top)
        main.addWidget(self._console)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 1)
        main.setSizes([520, 180])
        self.setCentralWidget(main)

    def _display_tabs(self) -> QTabWidget:
        """Create right display tabs used only for Analysis, Preview and Graph."""
        tabs = QTabWidget()
        tabs.setObjectName("qt-shell-display-tabs")
        tabs.setMinimumWidth(0)
        tabs.addTab(self._analysis_panel(), "Analysis")
        tabs.addTab(self._preview_panel(), "Preview")
        tabs.addTab(self._graph_panel(), "Graph")
        self._display_tabs_widget = tabs
        self._display_controller = DisplayController(
            tabs,
            analysis_factory=self._analysis_panel,
            preview_factory=self._preview_panel,
            graph_factory=self._graph_panel,
            preserve_existing_tabs=True,
            parent=self,
        )
        self._sync_view_actions()
        return tabs

    def _analysis_panel(self) -> QWidget:
        """Create the right-side Analysis runtime panel."""
        return AnalysisPanel(parent=self)

    def _preview_panel(self) -> QWidget:
        """Create or return the right-side Preview runtime panel."""
        if self._preview_panel_widget is None:
            self._preview_panel_widget = PreviewPanel(parent=self)
        return self._preview_panel_widget

    def _graph_panel(self) -> QWidget:
        """Create or return the right-side Graph runtime panel."""
        if self._graph_panel_widget is None:
            self._graph_panel_widget = GraphPanel(parent=self)
        return self._graph_panel_widget

    @staticmethod
    def _display_placeholder(name: str) -> QWidget:
        """Create a right-panel placeholder for display-only tabs."""
        panel = QWidget()
        panel.setMinimumWidth(0)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 8)
        label = QLabel(f"{name} display placeholder")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setObjectName(f"qt-shell-{name.lower()}-placeholder")
        layout.addWidget(label, 1)
        return panel

    def _build_menus(self) -> None:
        """Build prototype menu bar."""
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        project_menu = menu_bar.addMenu("Project")
        self._menus.append(project_menu)
        project_menu.addAction("New Prototype Project", self._new_project)
        project_menu.addAction("Open Project...", self._open_project)
        project_menu.addAction("Open Task...", self._open_task)
        project_menu.addSeparator()
        project_menu.addAction("Save Project", self._save_project_current)
        project_menu.addAction("Save Project As...", self._save_project_as)
        project_menu.addAction("Save Task As...", self._save_task_as)
        project_menu.addAction("Reload Current File", self._reload_current_file)
        project_menu.addSeparator()
        project_menu.addAction("Restore Last Session", self._restore_last_session)
        self._recent_menu = project_menu.addMenu("Recent Files")
        self._refresh_recent_menu()

        command_menu = menu_bar.addMenu("Command")
        self._menus.append(command_menu)
        command_menu.addAction("Generate", self._generate_command)
        self._run_menu_action = command_menu.addAction("Run", self._run_command)
        self._stop_menu_action = command_menu.addAction("Stop", self._stop_job)

        view_menu = menu_bar.addMenu("View")
        self._menus.append(view_menu)
        for label in (
            DisplayController.ANALYSIS,
            DisplayController.PREVIEW,
            DisplayController.GRAPH,
        ):
            action = view_menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, tab_name=label: self._show_display_tab(tab_name)
            )
            self._view_actions[label] = action

    def _build_toolbar(self) -> None:
        """Build the top toolbar."""
        toolbar = QToolBar("Toolbar")
        toolbar.setObjectName("qt-shell-toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction("New", self._new_project)
        toolbar.addAction("Open", self._open_project)
        toolbar.addAction("Save", self._save_project_current)
        toolbar.addSeparator()
        toolbar.addAction("Generate", self._generate_command)
        self._run_action = toolbar.addAction("Run", self._run_command)
        self._stop_action = toolbar.addAction("Stop", self._stop_job)

    def _build_statusbar(self) -> None:
        """Build QStatusBar and QProgressBar status panel."""
        status = QStatusBar()
        self._progress.setObjectName("qt-shell-progress")
        self._reset_progress()
        self._progress.setVisible(False)
        status.addPermanentWidget(self._progress)
        self.setStatusBar(status)
        status.showMessage("Status Ready")

    def _connect_signals(self) -> None:
        """Connect command panel and QProcess runner signals."""
        self._command_panel.generate_requested.connect(self._generate_command)
        self._command_panel.run_requested.connect(self._run_command)
        self._runner.stdout.connect(self._console.write)
        self._runner.stderr.connect(self._console.write)
        self._runner.progress.connect(self._consume_runtime_event)
        self._runner.finished.connect(self._job_finished)

    def _install_dirty_event_filters(self, root: QWidget | None = None) -> None:
        """Install dirty-state event filters on the command panel and dynamic children."""
        widget = root or self._command_panel
        widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            child.installEventFilter(self)

    def _mark_modified_from_user_event(self) -> None:
        """Mark the active project/task dirty after a user command-panel edit."""
        if not self._suppress_dirty:
            self._set_modified(True)

    def _show_display_tab(self, name: str) -> None:
        """Switch to a visible display tab from the View menu."""
        if self._display_tabs_widget is None:
            return
        for index in range(self._display_tabs_widget.count()):
            if self._display_tabs_widget.tabText(index) != name:
                continue
            if self._display_tabs_widget.isTabVisible(index):
                self._display_tabs_widget.setCurrentIndex(index)
            return

    def _sync_view_actions(self) -> None:
        """Keep View menu actions in sync with display tab visibility."""
        if self._display_controller is None:
            visible_tabs: set[str] = {DisplayController.ANALYSIS}
        else:
            visible_tabs = set(self._display_controller.visible_tab_names())

        for name, action in self._view_actions.items():
            action.setEnabled(name in visible_tabs)

        tabs = self._display_tabs_widget
        if tabs is None or tabs.currentIndex() == -1:
            return
        if tabs.isTabVisible(tabs.currentIndex()):
            return
        for index in range(tabs.count()):
            if tabs.isTabVisible(index):
                tabs.setCurrentIndex(index)
                return

    def _generate_command(self) -> None:
        """Generate command text through CommandBuilder."""
        try:
            _, command, values, args = self._build_command(generate=True)
        except ValueError as err:
            self._show_error(str(err))
            return
        self._project = self._session_service.snapshot_project(self._project, command, values)
        command_text = " ".join(args)
        self._console.write_line(f"$ {command_text}")
        self._write_context(command, values)
        context = CommandExecutionContext.from_values(command, values)
        self._apply_preview_context(context)
        self._apply_graph_context(context)
        self._set_modified(True)
        self.statusBar().showMessage("Generated command through CommandBuilder", 5000)

    def _run_command(self) -> None:
        """Run the selected command using JobRunner/QProcess."""
        if self._running:
            self._show_error("A job is already running")
            return
        try:
            category, command, values, args = self._build_command(generate=False)
        except ValueError as err:
            self._show_error(str(err))
            return
        self._project = self._session_service.snapshot_project(self._project, command, values)
        self._console.write_line(f"$ {' '.join(CommandBuilder.quote_args(args))}")
        self._write_context(command, values)
        context = CommandExecutionContext.from_values(command, values)
        self._apply_preview_context(context)
        self._apply_graph_context(context)
        self._runner.configure_runtime_context(context)
        try:
            self._runner.start(args, command=command)
        except (RuntimeError, ValueError) as err:
            self._show_error(str(err))
            return
        if self._display_controller is not None:
            self._display_controller.set_runtime_state(command, running=True)
        self._sync_view_actions()
        self._set_running(True)
        self.statusBar().showMessage(f"Running {category} {command}")

    def _stop_job(self) -> None:
        """Stop the current QProcess job."""
        self._runner.stop()

    def _job_finished(self, exit_code: int) -> None:
        """Update UI state when the process exits."""
        self._set_running(False)
        if self._display_controller is not None:
            self._display_controller.set_runtime_state(None, running=False)
        if self._preview_panel_widget is not None:
            self._preview_panel_widget.refresh_preview()
        if self._graph_panel_widget is not None:
            self._graph_panel_widget.refresh_graph()
        self._sync_view_actions()
        self._console.write_line(f"\nProcess finished with exit code {exit_code}")
        self.statusBar().showMessage(f"Process finished with exit code {exit_code}", 5000)

    def _consume_runtime_event(self, event: object) -> None:
        """Route structured runtime events to runtime-aware UI widgets."""
        if self._display_controller is not None:
            self._display_controller.consume_event(event)
        self._update_runtime_status(event)
        self._refresh_graph_from_event(event)
        self._sync_view_actions()

    def _update_runtime_status(self, event: object) -> None:
        """Update status and progress widgets from a runtime event."""
        message = getattr(event, "message", "")
        if isinstance(message, str) and message:
            self.statusBar().showMessage(message)

        payload = getattr(event, "payload", None)
        payload = payload if isinstance(payload, dict) else {}
        progress = getattr(event, "progress", None)
        if isinstance(progress, int | float) and not isinstance(progress, bool):
            self._progress.setRange(0, 100)
            self._progress.setValue(max(0, min(100, int(progress))))
            return

        mode = payload.get("mode")
        kind = getattr(event, "kind", "")
        if mode == "indeterminate" or kind == "progress":
            self._progress.setRange(0, 0)
        elif mode == "determinate":
            self._progress.setRange(0, 100)

    def _reset_progress(self) -> None:
        """Reset the progress bar to an idle determinate state."""
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

    def _apply_preview_context(self, context: CommandExecutionContext) -> None:
        """Apply preview output context to the Preview panel when available."""
        if self._preview_panel_widget is not None:
            self._preview_panel_widget.apply_context(context)

    def _apply_graph_context(self, context: CommandExecutionContext) -> None:
        """Apply model context to the Graph panel when available."""
        if self._graph_panel_widget is not None:
            self._graph_panel_widget.apply_context(context)

    def _refresh_graph_from_event(self, event: object) -> None:
        """Refresh Graph panel when runtime events announce new graph data."""
        payload = getattr(event, "payload", None)
        payload = payload if isinstance(payload, dict) else {}
        if self._graph_panel_widget is not None and payload.get("graph_refresh"):
            self._graph_panel_widget.refresh_graph()

    def _build_command(self, *, generate: bool) -> tuple[str, str, dict[str, object], list[str]]:
        """Build command args from the selected panel state."""
        category, command, values = self._command_panel.command_spec()
        if not category or not command:
            raise ValueError("Select a command before generating arguments")
        args = self._builder.build(category, command, values, generate=generate)
        return category, command, values, args

    def _write_context(self, command: str, values: T.Mapping[str, object]) -> None:
        """Show derived CommandExecutionContext values without invoking previews."""
        context = CommandExecutionContext.from_values(command, values)
        if context == CommandExecutionContext():
            return
        self._console.write_line(
            "Context: "
            f"model={context.model_name or '-'}, "
            f"model_folder={context.model_folder or '-'}, "
            f"preview_output={context.preview_output_path or '-'}, "
            f"batch_mode={context.batch_mode}"
        )

    def _new_project(self) -> bool:
        """Clear the prototype project state."""
        if not self._confirm_discard_changes("create a new project"):
            return False
        self._project = ProjectFile()
        self._project_filename = None
        self._file_kind = PROJECT_KIND
        self._suppress_dirty = True
        try:
            self._command_panel.clear_values()
        finally:
            self._suppress_dirty = False
        if self._preview_panel_widget is not None:
            self._preview_panel_widget.clear_preview()
        if self._graph_panel_widget is not None:
            self._graph_panel_widget.clear_graph()
        if self._display_controller is not None:
            self._display_controller.set_runtime_state(None, running=False)
        self._sync_view_actions()
        self._set_modified(False)
        self._console.write_line("New prototype project")
        self.statusBar().showMessage("New prototype project", 5000)
        return True

    def _open_project(self) -> None:
        """Open a project file from disk."""
        self._open_file_dialog(PROJECT_KIND)

    def _open_task(self) -> None:
        """Open a task file from disk."""
        self._open_file_dialog(TASK_KIND)

    def _open_file_dialog(self, kind: str) -> None:
        """Load a project or task file through ProjectStore."""
        if not self._confirm_discard_changes("open another file"):
            return
        title = "Open Faceswap Task" if kind == TASK_KIND else "Open Faceswap Project"
        filter_text = "Faceswap task (*.fst);;JSON files (*.json);;All files (*)"
        if kind == PROJECT_KIND:
            filter_text = "Faceswap project (*.fsw);;JSON files (*.json);;All files (*)"
        filename, _ = QFileDialog.getOpenFileName(self, title, "", filter_text)
        if filename:
            self._open_session_file(filename, kind, confirm_discard=False)

    def _save_project_current(self) -> None:
        """Save the current project file, or prompt when no project file is active."""
        if self._project_filename is None or self._file_kind != PROJECT_KIND:
            self._save_project_as()
            return
        self._save_project(self._project_filename)

    def _save_project_as(self) -> None:
        """Save current command state as a project file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Faceswap Project As",
            self._project_filename if self._file_kind == PROJECT_KIND else "",
            "Faceswap project (*.fsw);;JSON files (*.json);;All files (*)",
        )
        if filename:
            self._save_project(self._with_suffix(filename, PROJECT_EXTENSION))

    def _save_task_as(self) -> None:
        """Save only the current command state as a task file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Faceswap Task As",
            "",
            "Faceswap task (*.fst);;JSON files (*.json);;All files (*)",
        )
        if filename:
            self._save_task(self._with_suffix(filename, TASK_EXTENSION))

    def _save_project(self, filename: str) -> bool:
        """Persist the full project state."""
        _, command, values = self._command_panel.command_spec()
        self._project = self._session_service.snapshot_project(self._project, command, values)
        if not self._save_file(filename, self._project, PROJECT_KIND):
            return False
        self._project_filename = filename
        self._file_kind = PROJECT_KIND
        self._set_modified(False)
        self._console.write_line(f"Saved prototype project: {filename}")
        self.statusBar().showMessage("Project saved", 5000)
        return True

    def _save_task(self, filename: str) -> bool:
        """Persist only the current command state as a task file."""
        _, command, values = self._command_panel.command_spec()
        task = self._session_service.snapshot_task(command, values)
        if not self._save_file(filename, task, TASK_KIND):
            return False
        self._console.write_line(f"Saved prototype task: {filename}")
        self.statusBar().showMessage("Task saved", 5000)
        return True

    def _save_file(self, filename: str, project: ProjectFile, kind: str) -> bool:
        """Save a project/task model and update recent/last-session stores."""
        try:
            self._project_store.save(filename, project)
            self._recent_files.add(filename, kind)
            self._last_session.save(filename, kind)
        except OSError as err:
            self._show_error(str(err))
            return False
        self._refresh_recent_menu()
        return True

    def _open_session_file(
        self,
        filename: str,
        kind: str | None = None,
        *,
        confirm_discard: bool = True,
    ) -> bool:
        """Open a project or task file by filename."""
        if confirm_discard and not self._confirm_discard_changes("open another file"):
            return False
        resolved_kind = kind or self._session_service.kind_from_filename(filename)
        try:
            project = self._project_store.load(filename)
        except (OSError, ValueError) as err:
            self._show_error(str(err))
            return False
        return self._apply_project(filename, project, resolved_kind)

    def _apply_project(
        self, filename: str, project: ProjectFile, kind: str = PROJECT_KIND
    ) -> bool:
        """Apply a loaded ProjectFile to the command panel and runtime displays."""
        try:
            command, values = self._session_service.selected_task(project)
        except ValueError as err:
            self._show_error(str(err))
            return False
        self._project = project
        self._project_filename = filename
        self._file_kind = kind
        self._suppress_dirty = True
        try:
            self._command_panel.set_command(command, values)
        finally:
            self._suppress_dirty = False
        context = CommandExecutionContext.from_values(command, values)
        self._apply_preview_context(context)
        self._apply_graph_context(context)
        self._recent_files.add(filename, kind)
        self._last_session.save(filename, kind)
        self._refresh_recent_menu()
        self._set_modified(False)
        self._console.write_line(f"Loaded prototype {kind}: {filename}")
        self.statusBar().showMessage(f"{kind.title()} loaded", 5000)
        return True

    def _reload_current_file(self) -> bool:
        """Reload the active project/task file from disk."""
        if self._project_filename is None:
            self.statusBar().showMessage("No current file to reload", 5000)
            self._console.write_line("No current file to reload")
            return False
        if not self._confirm_discard_changes("reload the current file"):
            return False
        return self._open_session_file(
            self._project_filename,
            self._file_kind,
            confirm_discard=False,
        )

    def _restore_last_session(self) -> bool:
        """Restore the last opened project or task file."""
        session = self._last_session.load()
        if session is None:
            self.statusBar().showMessage("No last session to restore", 5000)
            self._console.write_line("No last session to restore")
            return False
        return self._open_session_file(session.filename, session.kind)

    def _refresh_recent_menu(self) -> None:
        """Refresh recent-file menu actions."""
        if self._recent_menu is None:
            return
        self._recent_menu.clear()
        recent_files = self._recent_files.load()
        if not recent_files:
            action = self._recent_menu.addAction("No recent files")
            action.setEnabled(False)
            return
        for item in recent_files:
            label = f"{item.kind.title()}: {Path(item.filename).name}"
            action = self._recent_menu.addAction(label)
            action.setToolTip(item.filename)
            action.triggered.connect(
                lambda _checked=False, filename=item.filename, kind=item.kind: (
                    self._open_session_file(
                        filename,
                        kind,
                    )
                )
            )

    def _list_recent_files(self) -> None:
        """Print recent files from RecentFilesStore to the console."""
        recent_files = self._recent_files.load()
        if not recent_files:
            self._console.write_line("No recent files")
            return
        self._console.write_line("Recent files:")
        for item in recent_files:
            self._console.write_line(f"  {item.kind}: {item.filename}")

    def _set_running(self, running: bool) -> None:
        """Update running controls."""
        self._running = running
        for action in (self._run_action, self._run_menu_action):
            if action is not None:
                action.setEnabled(not running)
        for action in (self._stop_action, self._stop_menu_action):
            if action is not None:
                action.setEnabled(running)
        self._progress.setVisible(running)
        if not running:
            self._reset_progress()

    def _set_modified(self, modified: bool) -> None:
        """Set dirty-state flag and refresh window title."""
        self._modified = modified
        self._update_window_title()

    def _update_window_title(self) -> None:
        """Update the main window title from active file and dirty state."""
        self.setWindowTitle(
            self._session_service.title(self._project_filename, modified=self._modified)
        )

    def _confirm_discard_changes(self, action: str) -> bool:
        """Return whether the user confirmed discarding unsaved changes."""
        if not self._modified:
            return True
        response = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"Discard unsaved changes and {action}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return response == QMessageBox.StandardButton.Yes

    def _show_error(self, message: str) -> None:
        """Display an error in both console and dialog form."""
        self._console.write_line(f"Error: {message}")
        QMessageBox.critical(self, "Qt Shell Prototype", message)

    @staticmethod
    def _with_suffix(filename: str, suffix: str) -> str:
        """Return filename with suffix when the user omitted one."""
        path = Path(filename)
        if path.suffix:
            return str(path)
        return str(path.with_suffix(suffix))

    @staticmethod
    def _load_schema() -> CommandSchema:
        """Load real Faceswap and tools CLI metadata for the Qt shell."""
        return CommandSchemaService().from_real_cli_metadata(categories=("faceswap", "tools"))

    @staticmethod
    def _recent_cache() -> Path:
        """Return the prototype recent-files cache path outside the repository."""
        cache_dir = Path.home() / ".cache" / "faceswap"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / ".qt_recent.json"

    @staticmethod
    def _last_session_cache() -> Path:
        """Return the prototype last-session cache path outside the repository."""
        cache_dir = Path.home() / ".cache" / "faceswap"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / ".qt_last_session.json"
