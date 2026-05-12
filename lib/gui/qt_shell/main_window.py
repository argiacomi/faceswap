#!/usr/bin/env python3
"""Minimal Qt shell for proving Faceswap GUI services outside Tk."""

from __future__ import annotations

import typing as T
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from lib.gui.models.project import ProjectFile
from lib.gui.qt_shell.command_panel import CommandPanel
from lib.gui.qt_shell.command_schema import CommandSchema
from lib.gui.qt_shell.command_schema_service import CommandSchemaService
from lib.gui.qt_shell.display_controller import DisplayController
from lib.gui.qt_shell.job_runner import JobRunner
from lib.gui.services.command_builder import CommandBuilder
from lib.gui.services.command_context import CommandExecutionContext
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
        self._schema = self._load_schema() if schema is None else schema
        self._builder = CommandBuilder(base_path=str(root))
        self._project_store = ProjectStore(get_serializer("json"))
        self._recent_files = RecentFilesStore(get_serializer("json"), str(self._recent_cache()))
        self._project = ProjectFile()
        self._project_filename: str | None = None
        self._runner = JobRunner(self)
        self._running = False
        self._command_panel = CommandPanel(self._schema)
        self._console = ConsolePane()
        self._progress = QProgressBar()
        self._display_controller: DisplayController | None = None
        self._run_action: QAction | None = None
        self._stop_action: QAction | None = None
        self._run_menu_action: QAction | None = None
        self._stop_menu_action: QAction | None = None
        self._build_ui()
        self._connect_signals()
        self._set_running(False)
        self._console.write_line("Faceswap Qt shell prototype")
        self._console.write_line(
            "Scope: real CLI metadata rendering, command generation and QProcess jobs"
        )

    def _build_ui(self) -> None:
        """Build the Qt shell layout."""
        self.setWindowTitle("Faceswap Qt Shell Prototype")
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
        tabs.addTab(self._display_placeholder("Preview"), "Preview")
        tabs.addTab(self._display_placeholder("Graph"), "Graph")
        self._display_controller = DisplayController(
            tabs,
            analysis_factory=self._analysis_panel,
            preview_factory=lambda: self._display_placeholder("Preview"),
            graph_factory=lambda: self._display_placeholder("Graph"),
            preserve_existing_tabs=True,
            parent=self,
        )
        return tabs

    def _analysis_panel(self) -> QWidget:
        """Create the right-side Analysis panel skeleton."""
        panel = QWidget()
        panel.setMinimumWidth(0)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        title = QLabel("Session Stats")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("qt-shell-analysis-title")
        layout.addWidget(title)

        table = QTableWidget(0, 8)
        table.setObjectName("qt-shell-session-stats")
        table.setMinimumWidth(0)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setHorizontalHeaderLabels(
            (
                "Graphs",
                "#",
                "Start",
                "End",
                "Elapsed",
                "Batch",
                "Iterations",
                "EGs/sec",
            )
        )
        layout.addWidget(table, 1)

        footer = QHBoxLayout()
        footer.addWidget(QLabel("No session data loaded"))
        footer.addStretch(1)
        for text in ("Open", "Save", "Clear"):
            button = QPushButton(text)
            button.setObjectName(f"qt-shell-analysis-{text.lower()}")
            footer.addWidget(button)
        layout.addLayout(footer)
        return panel

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
        project_menu.addAction("New Prototype Project", self._new_project)
        project_menu.addAction("Open Project...", self._open_project)
        project_menu.addAction("Save Project As...", self._save_project_as)
        project_menu.addSeparator()
        project_menu.addAction("List Recent Files", self._list_recent_files)

        command_menu = menu_bar.addMenu("Command")
        command_menu.addAction("Generate", self._generate_command)
        self._run_menu_action = command_menu.addAction("Run", self._run_command)
        self._stop_menu_action = command_menu.addAction("Stop", self._stop_job)

        view_menu = menu_bar.addMenu("View")
        view_menu.addAction("Analysis")
        view_menu.addAction("Preview")
        view_menu.addAction("Graph")

    def _build_toolbar(self) -> None:
        """Build the top toolbar."""
        toolbar = QToolBar("Toolbar")
        toolbar.setObjectName("qt-shell-toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction("New", self._new_project)
        toolbar.addAction("Open", self._open_project)
        toolbar.addAction("Save", self._save_project_as)
        toolbar.addSeparator()
        toolbar.addAction("Generate", self._generate_command)
        self._run_action = toolbar.addAction("Run", self._run_command)
        self._stop_action = toolbar.addAction("Stop", self._stop_job)

    def _build_statusbar(self) -> None:
        """Build QStatusBar and QProgressBar status panel."""
        status = QStatusBar()
        self._progress.setRange(0, 0)
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

    def _generate_command(self) -> None:
        """Generate command text through CommandBuilder."""
        try:
            _, command, values, args = self._build_command(generate=True)
        except ValueError as err:
            self._show_error(str(err))
            return
        self._project = ProjectFile(tab_name=command, tasks={command: values})
        command_text = " ".join(args)
        self._console.write_line(f"$ {command_text}")
        self._write_context(command, values)
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
        self._project = ProjectFile(tab_name=command, tasks={command: values})
        self._console.write_line(f"$ {' '.join(CommandBuilder.quote_args(args))}")
        self._write_context(command, values)
        try:
            self._runner.start(args)
        except (RuntimeError, ValueError) as err:
            self._show_error(str(err))
            return
        if self._display_controller is not None:
            self._display_controller.set_runtime_state(command, running=True)
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
        self._console.write_line(f"\nProcess finished with exit code {exit_code}")
        self.statusBar().showMessage(f"Process finished with exit code {exit_code}", 5000)

    def _consume_runtime_event(self, event: object) -> None:
        """Route structured runtime events to the display controller."""
        if self._display_controller is not None:
            self._display_controller.consume_event(event)

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

    def _new_project(self) -> None:
        """Clear the prototype project state."""
        self._project = ProjectFile()
        self._project_filename = None
        self._command_panel.clear_values()
        self._console.write_line("New prototype project")
        self.statusBar().showMessage("New prototype project", 5000)

    def _open_project(self) -> None:
        """Load a project file through ProjectStore."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Faceswap Project",
            "",
            "Faceswap project (*.fsw *.fst);;JSON files (*.json);;All files (*)",
        )
        if not filename:
            return
        try:
            project = self._project_store.load(filename)
        except (OSError, ValueError) as err:
            self._show_error(str(err))
            return
        self._apply_project(filename, project)

    def _save_project_as(self) -> None:
        """Save current command state through ProjectStore."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Faceswap Project As",
            self._project_filename or "",
            "Faceswap project (*.fsw);;JSON files (*.json);;All files (*)",
        )
        if filename:
            self._save_project(filename)

    def _save_project(self, filename: str) -> None:
        """Persist current command state through project and recent stores."""
        _, command, values = self._command_panel.command_spec()
        self._project = ProjectFile(tab_name=command, tasks={command: values})
        try:
            self._project_store.save(filename, self._project)
            self._recent_files.add(filename, "project")
        except OSError as err:
            self._show_error(str(err))
            return
        self._project_filename = filename
        self._console.write_line(f"Saved prototype project: {filename}")
        self.statusBar().showMessage("Project saved", 5000)

    def _apply_project(self, filename: str, project: ProjectFile) -> None:
        """Apply a loaded ProjectFile to the placeholder command panel."""
        command = project.tab_name if project.tab_name in project.tasks else None
        if command is None and project.tasks:
            command = next(iter(project.tasks))
        if command is None:
            self._show_error("Project does not contain any tasks")
            return
        values = project.tasks.get(command, {})
        self._project = project
        self._project_filename = filename
        self._command_panel.set_command(command, values)
        self._recent_files.add(filename, "project")
        self._console.write_line(f"Loaded prototype project: {filename}")
        self.statusBar().showMessage("Project loaded", 5000)

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

    def _show_error(self, message: str) -> None:
        """Display an error in both console and dialog form."""
        self._console.write_line(f"Error: {message}")
        QMessageBox.critical(self, "Qt Shell Prototype", message)

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
