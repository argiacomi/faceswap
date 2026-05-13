#!/usr/bin/env python3
"""Qt Graph panel backed by TrainingGraphService."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from lib.gui.services.command_context import CommandExecutionContext
from lib.gui.services.training_graph_service import (
    TrainingGraphError,
    TrainingGraphService,
    TrainingGraphSnapshot,
)


class GraphPanel(QWidget):
    """Runtime Graph panel for loaded Analysis session series."""

    _BARS = "▁▂▃▄▅▆▇█"

    def __init__(
        self,
        service: TrainingGraphService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service or TrainingGraphService()
        self._source_label = QLabel("No graph source configured")
        self._status_label = QLabel("No graph data loaded")
        self._session_combo = QComboBox()
        self._graph_text = QPlainTextEdit()
        self._open_button = QPushButton("Open")
        self._refresh_button = QPushButton("Refresh")
        self._clear_button = QPushButton("Clear")
        self._build_ui()
        self._connect_signals()
        self._sync_actions()

    @property
    def service(self) -> TrainingGraphService:
        """Return the backing graph service."""
        return self._service

    @property
    def source_path(self) -> str | None:
        """Return the current graph model state file path, if configured."""
        source = self._service.source
        return None if source is None else str(source.state_file)

    def restore_source(self, source: str | None) -> bool:
        """Restore a Graph source from saved UI state."""
        if not source:
            return False
        return self.load_source(source)

    def apply_context(self, context: CommandExecutionContext) -> bool:
        """Apply command-derived model context to the graph panel."""
        source = self._service.configure(
            model_folder=context.model_folder,
            model_name=context.model_name,
        )
        self._update_source_label()
        self._sync_actions()
        return source is not None

    def load_source(self, source: str | Path) -> bool:
        """Load graph data from a model state file or model folder."""
        try:
            snapshot = self._service.load_source(source)
        except (TrainingGraphError, OSError, ValueError) as err:
            self._set_error(str(err))
            return False
        self._render_snapshot(snapshot)
        self._update_source_label()
        self._sync_session_combo()
        self._sync_actions()
        return True

    def refresh_graph(self) -> bool:
        """Refresh graph data from the loaded or configured source."""
        try:
            if not self._service.is_loaded and self._service.source is not None:
                snapshot = self._service.load_configured_source(is_training=True)
            else:
                snapshot = self._service.refresh()
        except (TrainingGraphError, OSError, ValueError, AssertionError) as err:
            self._set_error(str(err))
            return False
        self._render_snapshot(snapshot)
        self._update_source_label()
        self._sync_session_combo()
        self._sync_actions()
        return True

    def clear_graph(self) -> None:
        """Clear graph source and rendered graph state."""
        self._service.clear()
        self._session_combo.clear()
        self._graph_text.setPlainText("No graph data loaded")
        self._source_label.setText("No graph source configured")
        self._status_label.setText("No graph data loaded")
        self._sync_actions()

    def _build_ui(self) -> None:
        """Build graph panel layout."""
        self.setObjectName("qt-shell-graph-panel")
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        title = QLabel("Training Graph")
        title.setObjectName("qt-shell-graph-title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self._source_label.setObjectName("qt-shell-graph-source")
        self._source_label.setAlignment(Qt.AlignCenter)
        self._source_label.setWordWrap(True)
        layout.addWidget(self._source_label)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Session:"))
        self._session_combo.setObjectName("qt-shell-graph-session")
        controls.addWidget(self._session_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._graph_text.setObjectName("qt-shell-graph-text")
        self._graph_text.setReadOnly(True)
        self._graph_text.setPlainText("No graph data loaded")
        layout.addWidget(self._graph_text, 1)

        footer = QHBoxLayout()
        self._status_label.setObjectName("qt-shell-graph-status")
        footer.addWidget(self._status_label)
        footer.addStretch(1)
        for button in (self._open_button, self._refresh_button, self._clear_button):
            button.setObjectName(f"qt-shell-graph-{button.text().lower()}")
            footer.addWidget(button)
        layout.addLayout(footer)

    def _connect_signals(self) -> None:
        """Connect panel signals."""
        self._open_button.clicked.connect(lambda _checked=False: self._open_source_dialog())
        self._refresh_button.clicked.connect(lambda _checked=False: self.refresh_graph())
        self._clear_button.clicked.connect(lambda _checked=False: self.clear_graph())
        self._session_combo.currentIndexChanged.connect(self._session_changed)

    def _open_source_dialog(self) -> None:
        """Prompt for a model state file or folder and load it."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Training Graph State",
            "",
            "Faceswap state files (*_state.json);;JSON files (*.json);;All files (*)",
        )
        if filename:
            self.load_source(filename)
            return
        folder = QFileDialog.getExistingDirectory(self, "Open Model Folder", "")
        if folder:
            self.load_source(folder)

    def _session_changed(self, index: int) -> None:
        """Refresh graph data when the selected session changes."""
        if index < 0 or not self._service.is_loaded:
            return
        try:
            snapshot = self._service.set_session_id(self._session_combo.itemData(index))
        except (TrainingGraphError, OSError, ValueError, AssertionError) as err:
            self._set_error(str(err))
            return
        self._render_snapshot(snapshot)
        self._sync_actions()

    def _render_snapshot(self, snapshot: TrainingGraphSnapshot) -> None:
        """Render graph snapshot and status text."""
        if snapshot.is_empty:
            self._graph_text.setPlainText("No graph data loaded")
            self._status_label.setText("No graph data loaded")
            return
        lines = []
        for series in snapshot.series:
            lines.append(f"{series.name}: {self._sparkline(series.values)}")
            lines.append(
                f"  points={series.count} min={series.minimum:.6g} max={series.maximum:.6g}"
            )
        self._graph_text.setPlainText("\n".join(lines))
        names = ", ".join(series.name for series in snapshot.series)
        self._status_label.setText(
            f"Loaded {len(snapshot.series)} series, {snapshot.point_count} points: {names}"
        )

    def _sync_session_combo(self) -> None:
        """Populate session selector from the backing session."""
        current = self._service.session_id
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        self._session_combo.addItem("All sessions", None)
        if self._service.is_loaded:
            for session_id in self._service.session_ids:
                self._session_combo.addItem(str(session_id), session_id)
        for index in range(self._session_combo.count()):
            if self._session_combo.itemData(index) == current:
                self._session_combo.setCurrentIndex(index)
                break
        self._session_combo.blockSignals(False)

    def _update_source_label(self) -> None:
        """Update source label from current service source."""
        source = self._service.source
        if source is None:
            self._source_label.setText("No graph source configured")
        else:
            self._source_label.setText(f"Graph source: {source.model_name}  |  {source.model_dir}")

    def _sync_actions(self) -> None:
        """Enable buttons based on current graph state."""
        has_source = self._service.source is not None
        has_graph_data = not self._service.snapshot.is_empty
        active_graph = has_source or has_graph_data
        self._refresh_button.setEnabled(active_graph)
        self._clear_button.setEnabled(active_graph)
        self._session_combo.setEnabled(active_graph and self._service.is_loaded)

    def _set_error(self, message: str) -> None:
        """Render an in-panel error."""
        self._status_label.setText(message)
        self._graph_text.setPlainText("No graph data loaded")
        self._sync_actions()

    @classmethod
    def _sparkline(cls, values: tuple[float, ...]) -> str:
        """Return a compact unicode sparkline for values."""
        if not values:
            return ""
        minimum = min(values)
        maximum = max(values)
        if minimum == maximum:
            return cls._BARS[0] * len(values)
        scale = len(cls._BARS) - 1
        return "".join(
            cls._BARS[int(round(((value - minimum) / (maximum - minimum)) * scale))]
            for value in values
        )
