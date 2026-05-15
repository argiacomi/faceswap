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

from lib.gui.qt_shell.graph_widget import TrainingGraphWidget
from lib.gui.services.command_context import CommandExecutionContext
from lib.gui.services.training_graph_service import (
    TrainingGraphError,
    TrainingGraphService,
    TrainingGraphSnapshot,
)


class GraphPanel(QWidget):
    """Runtime Graph panel for loaded Analysis session series."""

    _BARS = "▁▂▃▄▅▆▇█"
    IMAGE_FILTER = (
        "PNG files (*.png);;JPEG files (*.jpg *.jpeg);;Bitmap files (*.bmp);;All files (*)"
    )
    CSV_FILTER = "CSV files (*.csv);;All files (*)"

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
        self._key_combo = QComboBox()
        self._graph_widget = TrainingGraphWidget()
        self._graph_text = QPlainTextEdit()
        self._open_button = QPushButton("Open")
        self._refresh_button = QPushButton("Refresh")
        self._export_image_button = QPushButton("Export Image")
        self._export_csv_button = QPushButton("Export CSV")
        self._zoom_in_button = QPushButton("Zoom In")
        self._zoom_out_button = QPushButton("Zoom Out")
        self._zoom_y_in_button = QPushButton("Y Zoom In")
        self._zoom_y_out_button = QPushButton("Y Zoom Out")
        self._reset_view_button = QPushButton("Reset")
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
            snapshot = (
                self._service.load_configured_source(is_training=True)
                if not self._service.is_loaded and self._service.source is not None
                else self._service.refresh()
            )
        except (TrainingGraphError, OSError, ValueError, AssertionError) as err:
            self._set_error(str(err))
            return False
        self._render_snapshot(snapshot)
        self._update_source_label()
        self._sync_session_combo()
        self._sync_actions()
        return True

    def save_graph_image(self, filename: str | Path) -> bool:
        """Save the currently rendered graph image to disk."""
        saved = self._graph_widget.save_image(filename)
        self._status_label.setText("Graph image saved" if saved else "No graph image to save")
        return saved

    def save_graph_csv(self, filename: str | Path) -> int:
        """Save the currently selected graph series to CSV."""
        written = self._service.save_csv(
            filename, selected_keys=self._selected_loss_keys(self._service.snapshot)
        )
        self._status_label.setText(
            "No graph data to save" if written == 0 else f"Graph CSV saved: {written} rows"
        )
        return written

    def clear_graph(self) -> None:
        """Clear graph source and rendered graph state."""
        self.cleanup_graph()

    def cleanup_graph(self, message: str = "No graph data loaded") -> None:
        """Clear graph state for stop/failure/reload/close/project-change terminal paths."""
        self._service.clear()
        self._session_combo.clear()
        self._key_combo.clear()
        self._graph_widget.clear()
        self._graph_text.setPlainText("No graph data loaded")
        self._source_label.setText("No graph source configured")
        self._status_label.setText(message)
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
        controls.addWidget(QLabel("Loss:"))
        self._key_combo.setObjectName("qt-shell-graph-key")
        controls.addWidget(self._key_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        layout.addWidget(self._graph_widget, 2)

        self._graph_text.setObjectName("qt-shell-graph-text")
        self._graph_text.setReadOnly(True)
        self._graph_text.setPlainText("No graph data loaded")
        self._graph_text.setMaximumHeight(120)
        layout.addWidget(self._graph_text, 1)

        footer = QHBoxLayout()
        self._status_label.setObjectName("qt-shell-graph-status")
        footer.addWidget(self._status_label)
        footer.addStretch(1)
        for button in (
            self._open_button,
            self._refresh_button,
            self._export_image_button,
            self._export_csv_button,
            self._zoom_in_button,
            self._zoom_out_button,
            self._zoom_y_in_button,
            self._zoom_y_out_button,
            self._reset_view_button,
            self._clear_button,
        ):
            button.setObjectName(f"qt-shell-graph-{button.text().lower().replace(' ', '-')}")
            footer.addWidget(button)
        layout.addLayout(footer)

    def _connect_signals(self) -> None:
        """Connect panel signals."""
        self._open_button.clicked.connect(lambda _checked=False: self._open_source_dialog())
        self._refresh_button.clicked.connect(lambda _checked=False: self.refresh_graph())
        self._export_image_button.clicked.connect(
            lambda _checked=False: self._export_graph_image_dialog()
        )
        self._export_csv_button.clicked.connect(
            lambda _checked=False: self._export_graph_csv_dialog()
        )
        self._zoom_in_button.clicked.connect(lambda _checked=False: self._graph_widget.zoom_in())
        self._zoom_out_button.clicked.connect(lambda _checked=False: self._graph_widget.zoom_out())
        self._zoom_y_in_button.clicked.connect(
            lambda _checked=False: self._graph_widget.zoom_y_in()
        )
        self._zoom_y_out_button.clicked.connect(
            lambda _checked=False: self._graph_widget.zoom_y_out()
        )
        self._reset_view_button.clicked.connect(
            lambda _checked=False: self._graph_widget.reset_view()
        )
        self._clear_button.clicked.connect(lambda _checked=False: self.clear_graph())
        self._session_combo.currentIndexChanged.connect(self._session_changed)
        self._key_combo.currentIndexChanged.connect(self._loss_key_changed)

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

    def _export_graph_image_dialog(self) -> None:
        """Prompt for an image output path and save the rendered graph."""
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Training Graph Image",
            self._default_export_name("png"),
            self.IMAGE_FILTER,
        )
        if filename:
            suffix = (
                ".jpg"
                if selected_filter.startswith("JPEG")
                else ".bmp"
                if selected_filter.startswith("Bitmap")
                else ".png"
            )
            self.save_graph_image(self._with_suffix(filename, suffix))

    def _export_graph_csv_dialog(self) -> None:
        """Prompt for a CSV output path and save graph series data."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Training Graph CSV",
            self._default_export_name("csv"),
            self.CSV_FILTER,
        )
        if filename:
            self.save_graph_csv(self._with_suffix(filename, ".csv"))

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

    def _loss_key_changed(self, index: int) -> None:
        """Rerender graph when the selected loss key changes."""
        if index < 0:
            return
        self._render_snapshot(self._service.snapshot, sync_keys=False)
        self._sync_actions()

    def _render_snapshot(self, snapshot: TrainingGraphSnapshot, *, sync_keys: bool = True) -> None:
        """Render graph snapshot and status text."""
        if snapshot.is_empty:
            self._graph_widget.clear()
            self._graph_text.setPlainText("No graph data loaded")
            self._status_label.setText("No graph data loaded")
            if sync_keys:
                self._sync_key_combo(snapshot)
            return
        if sync_keys:
            self._sync_key_combo(snapshot)
        selected_keys = self._selected_loss_keys(snapshot)
        self._graph_widget.set_snapshot(snapshot, selected_keys=selected_keys)
        lines = []
        for series in snapshot.series:
            if selected_keys and series.name not in selected_keys:
                continue
            lines.append(f"{series.name}: {self._sparkline(series.values)}")
            lines.append(
                f"  points={series.count} min={series.minimum:.6g} max={series.maximum:.6g}"
            )
        self._graph_text.setPlainText("\n".join(lines) or "No selected graph data loaded")
        self._status_label.setText(self._graph_widget.status_text)

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

    def _sync_key_combo(self, snapshot: TrainingGraphSnapshot) -> None:
        """Populate loss-key selector from the current graph snapshot."""
        current = self._key_combo.currentData()
        keys = tuple(series.name for series in snapshot.series)
        self._key_combo.blockSignals(True)
        self._key_combo.clear()
        if keys:
            self._key_combo.addItem("All losses", None)
            for key in keys:
                self._key_combo.addItem(key, key)
        for index in range(self._key_combo.count()):
            if self._key_combo.itemData(index) == current:
                self._key_combo.setCurrentIndex(index)
                break
        self._key_combo.blockSignals(False)

    def _selected_loss_keys(self, snapshot: TrainingGraphSnapshot) -> tuple[str, ...]:
        """Return selected loss keys, or all available keys."""
        selected = self._key_combo.currentData()
        if isinstance(selected, str):
            return (selected,)
        return tuple(series.name for series in snapshot.series)

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
        has_rendered_series = bool(self._graph_widget.series)
        self._refresh_button.setEnabled(active_graph)
        self._export_image_button.setEnabled(has_rendered_series)
        self._export_csv_button.setEnabled(has_graph_data)
        self._zoom_in_button.setEnabled(has_rendered_series)
        self._zoom_out_button.setEnabled(has_rendered_series)
        self._zoom_y_in_button.setEnabled(has_rendered_series)
        self._zoom_y_out_button.setEnabled(has_rendered_series)
        self._reset_view_button.setEnabled(has_rendered_series)
        self._clear_button.setEnabled(active_graph)
        self._session_combo.setEnabled(active_graph and self._service.is_loaded)
        self._key_combo.setEnabled(has_graph_data)

    def _set_error(self, message: str) -> None:
        """Render an in-panel error."""
        self._status_label.setText(message)
        self._graph_widget.clear()
        self._graph_text.setPlainText("No graph data loaded")
        self._sync_actions()

    def _default_export_name(self, suffix: str) -> str:
        """Return a default export filename matching the loaded source/session."""
        source = self._service.source
        stem = "training_graph" if source is None else f"{source.model_name}_training_graph"
        session = self._service.session_id
        if session is not None:
            stem = f"{stem}_session_{session}"
        return f"{stem}.{suffix}"

    @staticmethod
    def _with_suffix(filename: str, suffix: str) -> str:
        """Return filename with suffix when the user omitted one."""
        path = Path(filename)
        return str(path if path.suffix else path.with_suffix(suffix))

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
