#!/usr/bin/env python3
"""Minimal Qt shell for proving Faceswap GUI services outside Tk."""

from __future__ import annotations

import typing as T
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QAction, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
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
from lib.gui.qt_shell.console_router import QtConsoleRouter
from lib.gui.qt_shell.display_controller import DisplayController
from lib.gui.qt_shell.graph_panel import GraphPanel
from lib.gui.qt_shell.job_runner import JobRunner
from lib.gui.qt_shell.manual_tool import ManualToolWindow
from lib.gui.qt_shell.preview_panel import PreviewPanel
from lib.gui.qt_shell.settings_dialog import SettingsDialog
from lib.gui.qt_shell.theme import QtTheme
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


class FreeSplitter(QSplitter):
    """QSplitter that lets users drag panes to any size in [0, full].

    QSplitter normally respects each child's ``minimumSizeHint`` which prevents
    continuous dragging past the layout's preferred minimum. Faceswap users
    expect Tk-parity behavior where a panel can be expanded to fill the entire
    window or collapsed to nothing without snapping back.
    """

    def __init__(self, orientation: Qt.Orientation, parent: QWidget | None = None) -> None:
        super().__init__(orientation, parent)
        self.setChildrenCollapsible(True)
        self.setHandleWidth(6)
        self.setOpaqueResize(True)

    def addWidget(self, widget: QWidget) -> None:  # type:ignore[override]  # noqa:N802
        """Force each child's minimum to zero so the handle can move anywhere.

        ``QSizePolicy.Ignored`` makes QSplitter ignore the child's minimumSizeHint
        so the user can drag continuously between zero and the full splitter range.
        """
        widget.setMinimumSize(0, 0)
        widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        super().addWidget(widget)
        self.setCollapsible(self.count() - 1, True)


_CONSOLE_TAG_COLORS: dict[str, str] = {
    "stdout": "console_text",
    "stderr": "error",
    "info": "info",
    "verbose": "verbose",
    "warning": "warning",
    "error": "error",
    "critical": "critical",
}


class ConsolePane(QPlainTextEdit):
    """Read-only console output pane with Tk-parity level-aware colorization."""

    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-shell-console")
        self.setProperty("console", "stdout")
        self.setReadOnly(True)
        self.setPlaceholderText("Generated commands and process output appear here.")
        self.setMinimumSize(0, 0)
        self._formats: dict[str, T.Any] = {}
        self._theme: T.Any = None

    def set_theme(self, theme: T.Any) -> None:
        """Cache theme colour lookup for tagged inserts."""
        self._theme = theme
        self._formats.clear()

    def write(self, text: str, tag: str = "stdout") -> None:
        """Append console text styled for the supplied tag (Tk parity)."""
        if not text:
            return
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        char_format = self._format_for(tag)
        if char_format is None:
            cursor.insertText(text)
        else:
            cursor.insertText(text, char_format)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def write_line(self, text: str = "", tag: str = "stdout") -> None:
        """Append one console line with the requested tag."""
        self.write(f"{text}\n", tag=tag)

    def _format_for(self, tag: str):  # type:ignore[no-untyped-def]
        """Return a cached QTextCharFormat for one console tag."""
        if tag not in _CONSOLE_TAG_COLORS or self._theme is None:
            return None
        cached = self._formats.get(tag)
        if cached is not None:
            return cached
        from PySide6.QtGui import QColor, QTextCharFormat

        char_format = QTextCharFormat()
        char_format.setForeground(QColor(self._theme.color(_CONSOLE_TAG_COLORS[tag])))
        self._formats[tag] = char_format
        return char_format


class MainWindow(QMainWindow):
    """Qt QMainWindow shell that exercises service-layer command generation."""

    def __init__(self, schema: CommandSchema | None = None, theme: QtTheme | None = None) -> None:
        super().__init__()
        root = Path(PROJECT_ROOT)
        serializer = get_serializer("json")
        self._theme = theme or QtTheme.default()
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
        self._native_tool_windows: list[QWidget] = []
        self._command_panel = CommandPanel(self._schema)
        self._console = ConsolePane()
        self._console.set_theme(self._theme)
        self._console_stdout_router = QtConsoleRouter(self._console, "stdout")
        self._console_stderr_router = QtConsoleRouter(self._console, "stderr")
        self._progress = QProgressBar()
        self._menus: list[object] = []
        self._recent_menu: QMenu | None = None
        self._main_splitter: QSplitter | None = None
        self._vertical_splitter: QSplitter | None = None
        self._display_tabs_widget: QTabWidget | None = None
        self._display_controller: DisplayController | None = None
        self._analysis_panel_widget: AnalysisPanel | None = None
        self._preview_panel_widget: PreviewPanel | None = None
        self._graph_panel_widget: GraphPanel | None = None
        self._view_actions: dict[str, QAction] = {}
        self._run_action: QAction | None = None
        self._stop_action: QAction | None = None
        self._run_menu_action: QAction | None = None
        self._stop_menu_action: QAction | None = None
        self._settings_dialog: SettingsDialog | None = None
        self._build_ui()
        self._connect_signals()
        self._install_dirty_event_filters()
        self._set_running(False)
        self._set_modified(False)
        self._console.write_line("Faceswap Qt shell prototype")
        self._console.write_line(
            "Scope: real CLI metadata rendering, command generation and QProcess jobs"
        )

    @property
    def console(self) -> ConsolePane:
        """Return the bottom console pane (Tk-parity logger destination)."""
        return self._console

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
        if not self.isVisible():
            self._stop_preview_live_refresh()
            event.accept()
            return
        if self._confirm_discard_changes("close this window"):
            self._stop_preview_live_refresh()
            self._save_last_session_state()
            event.accept()
        else:
            event.ignore()

    def apply_gui_settings(self) -> None:
        """Apply ``config/gui.ini`` runtime preferences.

        Reads ``lib.gui.gui_config`` values for the initial command tab, splitter
        ratios and fullscreen flag and applies them to the shell. Safe to call
        multiple times; missing keys fall back to the existing layout.
        """
        try:
            from lib.gui import gui_config as cfg
        except ImportError:  # pragma: no cover - gui_config always ships
            return
        tab_name = self._safe_call(cfg, "tab")
        if isinstance(tab_name, str) and tab_name:
            self._command_panel.select_command(tab_name)
        width_pct = self._safe_call(cfg, "options_panel_width")
        height_pct = self._safe_call(cfg, "console_panel_height")
        self._apply_splitter_ratios(width_pct, height_pct)
        fullscreen = self._safe_call(cfg, "fullscreen")
        if fullscreen:
            self.showMaximized()

    def _apply_splitter_ratios(
        self, options_panel_width_pct: object, console_panel_height_pct: object
    ) -> None:
        """Reflow main splitters from gui.ini percentage settings."""
        if self._main_splitter is not None and isinstance(options_panel_width_pct, int | float):
            total = max(1, self._main_splitter.width() or self.width())
            left = max(0, int(total * float(options_panel_width_pct) / 100.0))
            self._main_splitter.setSizes([left, max(0, total - left)])
        if self._vertical_splitter is not None and isinstance(
            console_panel_height_pct, int | float
        ):
            total = max(1, self._vertical_splitter.height() or self.height())
            bottom = max(0, int(total * float(console_panel_height_pct) / 100.0))
            self._vertical_splitter.setSizes([max(0, total - bottom), bottom])

    @staticmethod
    def _safe_call(module: T.Any, attribute: str) -> object:
        """Return the result of calling a gui_config ConfigItem, or ``None``."""
        func = getattr(module, attribute, None)
        try:
            return func() if callable(func) else None
        except (AttributeError, KeyError, ValueError):
            return None

    def _build_ui(self) -> None:
        """Build the Qt shell layout."""
        self._apply_window_chrome()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        top = FreeSplitter(Qt.Horizontal)
        top.setObjectName("qt-shell-main-splitter")
        top.addWidget(self._command_panel)
        top.addWidget(self._display_tabs())
        top.setStretchFactor(0, 0)
        top.setStretchFactor(1, 1)
        top.setSizes([420, 840])
        self._main_splitter = top
        main = FreeSplitter(Qt.Vertical)
        main.setObjectName("qt-shell-vertical-splitter")
        main.addWidget(top)
        main.addWidget(self._console)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 1)
        main.setSizes([520, 180])
        self._vertical_splitter = main
        self.setCentralWidget(main)

    def _apply_window_chrome(self) -> None:
        """Apply startup window icon and sizing polish."""
        icon = self._icon("favicon")
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.setMinimumSize(640, 400)

    def _display_tabs(self) -> QTabWidget:
        """Create right display tabs used only for Analysis, Preview and Graph."""
        tabs = QTabWidget()
        tabs.setObjectName("qt-shell-display-tabs")
        tabs.setMinimumSize(0, 0)
        tabs.tabBar().setExpanding(False)
        tabs.tabBar().setUsesScrollButtons(True)
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
        """Create or return the right-side Analysis runtime panel."""
        if self._analysis_panel_widget is None:
            self._analysis_panel_widget = AnalysisPanel(parent=self)
        return self._analysis_panel_widget

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
        project_menu.addAction(self._action("New Prototype Project", self._new_project, "new"))
        project_menu.addAction(self._action("Close Project", self._close_project, "clear"))
        project_menu.addAction(self._action("Open Project...", self._open_project, "open"))
        project_menu.addAction(self._action("Open Task...", self._open_task, "open"))
        project_menu.addSeparator()
        project_menu.addAction(self._action("Save Project", self._save_project_current, "save"))
        project_menu.addAction(
            self._action("Save Project As...", self._save_project_as, "save_as")
        )
        project_menu.addAction(self._action("Save Task As...", self._save_task_as, "save_as"))
        project_menu.addAction(
            self._action("Reload Current File", self._reload_current_file, "reload")
        )
        project_menu.addSeparator()
        project_menu.addAction(
            self._action("Restore Last Session", self._restore_last_session, "reload")
        )
        self._recent_menu = project_menu.addMenu("Recent Files")
        self._refresh_recent_menu()

        settings_menu = menu_bar.addMenu("Settings")
        settings_menu.setObjectName("qt-shell-settings-menu")
        self._menus.append(settings_menu)
        configure = QAction("Configure Settings...", settings_menu)
        configure.setObjectName("qt-shell-settings-configure")
        configure.setToolTip("Configure Settings...")
        configure.triggered.connect(lambda _checked=False: self._open_settings_dialog())
        settings_menu.addAction(configure)
        settings_menu.addSeparator()
        for name in ("extract", "train", "convert"):
            entry = QAction(
                self._icon(f"settings_{name}"),
                f"Configure {name.title()} Settings...",
                settings_menu,
            )
            entry.setObjectName(f"qt-shell-settings-configure-{name}")
            entry.setToolTip(f"Configure {name.title()} settings...")
            entry.triggered.connect(
                lambda _checked=False, section=name: self._open_settings_dialog(section)
            )
            settings_menu.addAction(entry)
        gui_entry = QAction(
            self._icon("settings"),
            "Faceswap GUI Options...",
            settings_menu,
        )
        gui_entry.setObjectName("qt-shell-settings-configure-gui")
        gui_entry.setToolTip("Configure the appearance and behavior of the GUI")
        gui_entry.triggered.connect(lambda _checked=False: self._open_settings_dialog("gui"))
        settings_menu.addAction(gui_entry)

        command_menu = menu_bar.addMenu("Command")
        self._menus.append(command_menu)
        command_menu.addAction(self._action("Generate", self._generate_command, "generate"))
        self._run_menu_action = self._action("Run", self._run_command, "run")
        self._stop_menu_action = self._action("Stop", self._stop_job, "stop")
        command_menu.addAction(self._run_menu_action)
        command_menu.addAction(self._stop_menu_action)

        view_menu = menu_bar.addMenu("View")
        self._menus.append(view_menu)
        for label in (
            DisplayController.ANALYSIS,
            DisplayController.PREVIEW,
            DisplayController.GRAPH,
        ):
            action = view_menu.addAction(label)
            action.setIcon(self._icon(label.lower()))
            action.triggered.connect(
                lambda _checked=False, tab_name=label: self._show_display_tab(tab_name)
            )
            self._view_actions[label] = action

    def _open_settings_dialog(self, section: str | None = None) -> SettingsDialog:
        """Open the Qt settings dialog, reusing an existing window when possible."""
        dialog = self._settings_dialog
        if dialog is None:
            dialog = SettingsDialog(section=section, parent=self)
            self._settings_dialog = dialog
        elif section is not None:
            dialog._select_initial_section(section)  # pylint:disable=protected-access
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _build_toolbar(self) -> None:
        """Build the top toolbar."""
        toolbar = QToolBar("Toolbar")
        toolbar.setObjectName("qt-shell-toolbar")
        toolbar.setIconSize(QSize(self._theme.icon_size, self._theme.icon_size))
        self.addToolBar(toolbar)
        new_action = self._action("New Project", self._new_project, "new")
        new_action.setObjectName("qt-shell-toolbar-new")
        toolbar.addAction(new_action)
        open_action = self._action("Load Project...", self._open_project, "open")
        open_action.setObjectName("qt-shell-toolbar-open")
        toolbar.addAction(open_action)
        save_action = self._action("Save Project", self._save_project_current, "save")
        save_action.setObjectName("qt-shell-toolbar-save")
        toolbar.addAction(save_action)
        save_as_action = self._action("Save Project As...", self._save_project_as, "save_as")
        save_as_action.setObjectName("qt-shell-toolbar-save-as")
        toolbar.addAction(save_as_action)
        reload_action = self._action("Reload Project", self._reload_current_file, "reload")
        reload_action.setObjectName("qt-shell-toolbar-reload")
        toolbar.addAction(reload_action)
        toolbar.addSeparator()
        task_load = self._action("Load Task...", self._open_task, "task_open")
        task_load.setObjectName("qt-shell-toolbar-task-open")
        toolbar.addAction(task_load)
        task_save = self._action("Save Task", self._save_task_current, "task_save")
        task_save.setObjectName("qt-shell-toolbar-task-save")
        toolbar.addAction(task_save)
        task_save_as = self._action("Save Task As...", self._save_task_as, "task_save_as")
        task_save_as.setObjectName("qt-shell-toolbar-task-save-as")
        toolbar.addAction(task_save_as)
        task_reset = self._action("Reset Task", self._reset_task, "task_reset")
        task_reset.setObjectName("qt-shell-toolbar-task-reset")
        toolbar.addAction(task_reset)
        task_reload = self._action("Reload Task", self._reload_task, "task_reload")
        task_reload.setObjectName("qt-shell-toolbar-task-reload")
        toolbar.addAction(task_reload)
        toolbar.addSeparator()
        for name in ("extract", "train", "convert"):
            action = self._settings_action(name)
            toolbar.addAction(action)

    def _settings_action(self, name: str) -> QAction:
        """Build a taskbar settings action that opens the dialog for one command."""
        tooltip = f"Configure {name.title()} settings..."
        action = QAction(self._icon(f"settings_{name}"), tooltip, self)
        action.setObjectName(f"qt-shell-toolbar-settings-{name}")
        action.setToolTip(tooltip)
        action.triggered.connect(
            lambda _checked=False, section=name: self._open_settings_dialog(section)
        )
        return action

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
        self._command_panel.value_changed.connect(self._mark_modified_from_user_event)
        self._runner.stdout.connect(self._console_stdout_router.write)
        self._runner.stderr.connect(self._console_stderr_router.write)
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
        command_text = " ".join(CommandBuilder.quote_args(args))
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
        if self._try_launch_native_tool(category, command, values):
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
            self._stop_preview_live_refresh()
            self._show_error(str(err))
            return
        if self._display_controller is not None:
            self._display_controller.set_runtime_state(command, running=True)
        self._sync_view_actions()
        self._set_running(True)
        self._start_preview_live_refresh(command, context)
        self.statusBar().showMessage(f"Running {category} {command}")

    def _try_launch_native_tool(
        self,
        category: str,
        command: str,
        values: T.Mapping[str, object],
    ) -> bool:
        """Open a native Qt window for tools that have a Qt implementation.

        Returns ``True`` when a native window has been launched (Run is fully
        handled here); ``False`` when the caller should fall back to the
        subprocess JobRunner path.
        """
        if category != "tools" or command != "manual":
            return False
        try:
            window = ManualToolWindow.from_command_values(
                values,
                builder=self._builder,
                parent=self,
            )
        except ValueError as err:
            message = str(err)
            self._command_panel.set_external_errors((message,))
            self._console.write_line(f"Manual Tool: {message}")
            self.statusBar().showMessage(message, 5000)
            return True
        self._command_panel.set_external_errors(())
        self._project = self._session_service.snapshot_project(self._project, command, values)
        window.setAttribute(Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(lambda _obj=None, w=window: self._discard_native_tool(w))
        self._native_tool_windows.append(window)
        window.show()
        self._console.write_line("Launched native Qt Manual Tool")
        self.statusBar().showMessage("Launched native Qt Manual Tool", 5000)
        return True

    def _discard_native_tool(self, window: QWidget) -> None:
        """Drop a closed native tool window from the tracked list."""
        if window in self._native_tool_windows:
            self._native_tool_windows.remove(window)

    def _stop_job(self) -> None:
        """Stop the current QProcess job."""
        self._stop_preview_live_refresh()
        self._runner.stop()

    def _job_finished(self, exit_code: int) -> None:
        """Update UI state when the process exits."""
        self._stop_preview_live_refresh()
        self._set_running(False)
        if self._display_controller is not None:
            self._display_controller.set_runtime_state(None, running=False)
        self._refresh_runtime_panels(preview=True, graph=True, analysis=True)
        self._sync_view_actions()
        self._save_last_session_state()
        self._console.write_line(f"\nProcess finished with exit code {exit_code}")
        self.statusBar().showMessage(f"Process finished with exit code {exit_code}", 5000)

    def _consume_runtime_event(self, event: object) -> None:
        """Route structured runtime events to runtime-aware UI widgets."""
        if self._display_controller is not None:
            self._display_controller.consume_event(event)
        self._update_runtime_status(event)
        self._refresh_displays_from_event(event)
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

    def _refresh_displays_from_event(self, event: object) -> None:
        """Refresh runtime display panels when structured events request it."""
        payload = getattr(event, "payload", None)
        payload = payload if isinstance(payload, dict) else {}
        kind = getattr(event, "kind", "")
        refresh_all = bool(payload.get("saved_model") or payload.get("session_refresh"))
        self._refresh_runtime_panels(
            preview=refresh_all or bool(payload.get("preview_refresh")),
            graph=refresh_all or bool(payload.get("graph_refresh")),
            analysis=refresh_all or bool(payload.get("analysis_refresh") or kind == "saved_model"),
        )

    def _refresh_runtime_panels(
        self,
        *,
        preview: bool = False,
        graph: bool = False,
        analysis: bool = False,
    ) -> None:
        """Refresh loaded runtime panels without changing their configured sources."""
        if analysis and self._analysis_panel_widget is not None:
            self._analysis_panel_widget.refresh_session()
        if preview and self._preview_panel_widget is not None:
            self._preview_panel_widget.refresh_preview()
        if graph and self._graph_panel_widget is not None:
            self._graph_panel_widget.refresh_graph()

    def _build_command(self, *, generate: bool) -> tuple[str, str, dict[str, object], list[str]]:
        """Build command args from the selected panel state."""
        category, command, values = self._command_panel.command_spec()
        if not category or not command:
            raise ValueError("Select a command before generating arguments")
        if not generate:
            errors = self._command_panel.validation_errors()
            if errors:
                raise ValueError("\n".join(errors))
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
        self._reset_project_state()
        self._console.write_line("New prototype project")
        self.statusBar().showMessage("New prototype project", 5000)
        return True

    def _close_project(self) -> bool:
        """Close the current project/task state."""
        if not self._confirm_discard_changes("close the current project"):
            return False
        self._reset_project_state()
        self._last_session.clear()
        self._console.write_line("Closed current project")
        self.statusBar().showMessage("Project closed", 5000)
        return True

    def _reset_project_state(self) -> None:
        """Reset project/task, command and display state to the initial shell state."""
        self._stop_preview_live_refresh()
        self._project = ProjectFile()
        self._project_filename = None
        self._file_kind = PROJECT_KIND
        self._suppress_dirty = True
        try:
            self._command_panel.clear_values()
        finally:
            self._suppress_dirty = False
        if self._analysis_panel_widget is not None:
            self._analysis_panel_widget.clear_session()
        if self._preview_panel_widget is not None:
            self._preview_panel_widget.clear_preview()
        if self._graph_panel_widget is not None:
            self._graph_panel_widget.clear_graph()
        if self._display_controller is not None:
            self._display_controller.set_runtime_state(None, running=False)
        self._sync_view_actions()
        self._set_modified(False)

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
            self._project_filename if self._file_kind == TASK_KIND else "",
            "Faceswap task (*.fst);;JSON files (*.json);;All files (*)",
        )
        if filename:
            self._save_task(self._with_suffix(filename, TASK_EXTENSION))

    def _save_task_current(self) -> None:
        """Overwrite the active task file, or prompt when none is active."""
        if self._project_filename is None or self._file_kind != TASK_KIND:
            self._save_task_as()
            return
        self._save_task(self._project_filename)

    def _reset_task(self) -> None:
        """Reset the current command's options to their defaults (Tk parity)."""
        if not self._confirm_discard_changes("reset the current task"):
            return
        self._suppress_dirty = True
        try:
            self._command_panel.clear_values()
        finally:
            self._suppress_dirty = False
        self._set_modified(False)
        self._console.write_line("Reset current task to defaults")
        self.statusBar().showMessage("Task reset", 5000)

    def _reload_task(self) -> bool:
        """Reload the active task file from disk."""
        if self._project_filename is None or self._file_kind != TASK_KIND:
            self.statusBar().showMessage("No task file to reload", 5000)
            self._console.write_line("No task file to reload")
            return False
        return self._reload_current_file()

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
            normalized_kind = self._session_service.normalize_kind(kind, filename)
            self._recent_files.add(filename, normalized_kind)
            self._last_session.save(filename, normalized_kind, self._capture_ui_state())
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
        if not Path(filename).exists():
            self._recent_files.remove(filename)
            self._refresh_recent_menu()
            self.statusBar().showMessage("Recent file no longer exists", 5000)
            self._console.write_line(f"Recent file no longer exists: {filename}")
            return False
        if confirm_discard and not self._confirm_discard_changes("open another file"):
            return False
        self._stop_preview_live_refresh()
        resolved_kind = (
            self._session_service.kind_from_filename(filename)
            if kind is None
            else self._session_service.normalize_kind(kind, filename)
        )
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
        self._stop_preview_live_refresh()
        try:
            command, values = self._session_service.selected_task(project)
        except ValueError as err:
            self._show_error(str(err))
            return False
        self._project = project
        self._project_filename = filename
        self._file_kind = self._session_service.normalize_kind(kind, filename)
        self._suppress_dirty = True
        try:
            self._command_panel.set_command(command, values)
        finally:
            self._suppress_dirty = False
        context = CommandExecutionContext.from_values(command, values)
        self._apply_preview_context(context)
        self._apply_graph_context(context)
        self._recent_files.add(filename, self._file_kind)
        self._last_session.save(filename, self._file_kind, self._capture_ui_state())
        self._refresh_recent_menu()
        self._set_modified(False)
        self._console.write_line(f"Loaded prototype {self._file_kind}: {filename}")
        self.statusBar().showMessage(f"{self._file_kind.title()} loaded", 5000)
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
        restored = self._open_session_file(session.filename, session.kind)
        if restored:
            self._restore_ui_state(session.ui_state)
            self._save_last_session_state()
        return restored

    def _capture_ui_state(self) -> dict[str, object]:
        """Capture restorable Qt shell UI/session state."""
        state: dict[str, object] = {"window_size": [self.width(), self.height()]}
        if self._display_tabs_widget is not None and self._display_tabs_widget.currentIndex() >= 0:
            state["display_tab"] = self._display_tabs_widget.tabText(
                self._display_tabs_widget.currentIndex()
            )
        if self._main_splitter is not None:
            state["main_splitter"] = self._main_splitter.sizes()
        if self._vertical_splitter is not None:
            state["vertical_splitter"] = self._vertical_splitter.sizes()
        if self._analysis_panel_widget is not None:
            state["analysis_source"] = self._analysis_panel_widget.source_path
        if self._preview_panel_widget is not None:
            state["preview_source"] = self._preview_panel_widget.source_path
        if self._graph_panel_widget is not None:
            state["graph_source"] = self._graph_panel_widget.source_path
        return state

    def _restore_ui_state(self, state: T.Mapping[str, object]) -> None:
        """Restore Qt shell UI/session state from the last-session cache."""
        window_size = state.get("window_size")
        if isinstance(window_size, list | tuple) and len(window_size) == 2:
            width, height = window_size
            if isinstance(width, int) and isinstance(height, int):
                self.resize(width, height)
        self._restore_splitter(self._main_splitter, state.get("main_splitter"))
        self._restore_splitter(self._vertical_splitter, state.get("vertical_splitter"))
        if self._analysis_panel_widget is not None:
            self._analysis_panel_widget.restore_source(
                self._string_or_none(state.get("analysis_source"))
            )
        if self._preview_panel_widget is not None:
            self._preview_panel_widget.restore_source(
                self._string_or_none(state.get("preview_source"))
            )
        if self._graph_panel_widget is not None:
            self._graph_panel_widget.restore_source(
                self._string_or_none(state.get("graph_source"))
            )
        display_tab = self._string_or_none(state.get("display_tab"))
        if display_tab:
            self._restore_display_tab(display_tab)

    def _restore_display_tab(self, name: str) -> None:
        """Restore a saved display tab, making it visible when it still exists."""
        if self._display_tabs_widget is None:
            return
        for index in range(self._display_tabs_widget.count()):
            if self._display_tabs_widget.tabText(index) != name:
                continue
            self._display_tabs_widget.setTabVisible(index, True)
            self._display_tabs_widget.setCurrentIndex(index)
            self._sync_view_actions()
            return

    @staticmethod
    def _restore_splitter(splitter: QSplitter | None, sizes: object) -> None:
        """Restore splitter sizes from saved state."""
        if splitter is None or not isinstance(sizes, list | tuple):
            return
        if all(isinstance(size, int) for size in sizes):
            splitter.setSizes(list(sizes))

    @staticmethod
    def _string_or_none(value: object) -> str | None:
        """Return a non-empty string or None."""
        return value if isinstance(value, str) and value else None

    def _save_last_session_state(self) -> None:
        """Persist current UI state for the active project/task file."""
        if self._project_filename is not None:
            self._last_session.save(
                self._project_filename,
                self._file_kind,
                self._capture_ui_state(),
            )

    def _refresh_recent_menu(self) -> None:
        """Refresh recent-file menu actions."""
        if self._recent_menu is None:
            return
        self._recent_menu.clear()
        recent_files = self._recent_files.prune_missing()
        display_items = self._recent_files.display_items(recent_files)
        if not display_items:
            action = self._recent_menu.addAction("No recent files")
            action.setEnabled(False)
        else:
            for display_item in display_items:
                item = display_item.file
                normalized_kind = self._session_service.normalize_kind(item.kind, item.filename)
                action = self._recent_menu.addAction(display_item.label)
                action.setToolTip(display_item.tooltip)
                action.triggered.connect(
                    lambda _checked=False, filename=item.filename, kind=normalized_kind: (
                        self._open_session_file(
                            filename,
                            kind,
                        )
                    )
                )
        self._recent_menu.addSeparator()
        clear_action = self._recent_menu.addAction("Clear Recent Files", self._clear_recent_files)
        clear_action.setEnabled(bool(display_items))

    def _clear_recent_files(self) -> None:
        """Clear all recent-file entries."""
        self._recent_files.clear()
        self._refresh_recent_menu()
        self._console.write_line("Cleared recent files")
        self.statusBar().showMessage("Recent files cleared", 5000)

    def _list_recent_files(self) -> None:
        """Print recent files from RecentFilesStore to the console."""
        recent_files = self._recent_files.prune_missing()
        self._refresh_recent_menu()
        if not recent_files:
            self._console.write_line("No recent files")
            return
        self._console.write_line("Recent files:")
        for item in recent_files:
            self._console.write_line(f"  {item.kind}: {item.filename}")

    def _set_running(self, running: bool) -> None:
        """Update running controls."""
        self._running = running
        self._command_panel.set_running(running)
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

    def _start_preview_live_refresh(
        self,
        command: str,
        context: CommandExecutionContext,
    ) -> None:
        """Start Preview polling for extract/convert jobs that emit preview output."""
        if command not in {"extract", "convert"} or context.preview_output_path is None:
            self._stop_preview_live_refresh()
            return
        if self._preview_panel_widget is not None:
            self._preview_panel_widget.start_live_refresh()

    def _stop_preview_live_refresh(self) -> None:
        """Stop Preview polling if the panel exists."""
        if self._preview_panel_widget is not None:
            self._preview_panel_widget.stop_live_refresh()

    def _action(self, text: str, callback: T.Callable[[], object], icon_key: str) -> QAction:
        """Create a menu/toolbar action with the mapped Faceswap icon."""
        action = QAction(self._icon(icon_key), text, self)
        action.setToolTip(text)
        action.triggered.connect(lambda _checked=False: callback())
        return action

    def _icon(self, action: str) -> QIcon:
        """Return a QIcon from the legacy icon cache for a theme action."""
        icon_name = self._theme.icon_name(action)
        path = Path(PROJECT_ROOT) / "lib" / "gui" / ".cache" / "icons" / f"{icon_name}.png"
        return QIcon(str(path)) if path.is_file() else QIcon()

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
