#!/usr/bin/env python3
"""Qt Analysis panel backed by AnalysisSessionService."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lib.gui.services.analysis_session_service import (
    AnalysisSessionError,
    AnalysisSessionService,
    AnalysisTableRow,
)
from lib.gui.services.analysis_summary_service import AnalysisSummaryService
from lib.gui.services.command_context import CommandExecutionContext


class AnalysisPanel(QWidget):
    """Runtime Analysis panel for loading, refreshing and exporting session summaries."""

    session_loaded = Signal(str)
    session_cleared = Signal()

    def __init__(
        self,
        service: AnalysisSessionService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service or AnalysisSessionService()
        self._summary_service = AnalysisSummaryService()
        self._title = QLabel("Session Stats")
        self._source_label = QLabel("No session source loaded")
        self._status_label = QLabel("No session data loaded")
        self._detail_label = QLabel("Rows: 0 | Graphs: 0 | Iterations: 0 | Avg EGs/sec: 0.00")
        self._selection_label = QLabel("No session selected")
        self._filter_combo = QComboBox()
        self._group_combo = QComboBox()
        self._table = QTableWidget(0, len(AnalysisSessionService.TABLE_HEADERS))
        self._open_button = QPushButton("Open")
        self._refresh_button = QPushButton("Refresh")
        self._save_button = QPushButton("Save")
        self._clear_button = QPushButton("Clear")
        self._rows: tuple[AnalysisTableRow, ...] = ()
        self._display_rows: tuple[AnalysisTableRow, ...] = ()
        self._build_ui()
        self._connect_signals()
        self._sync_actions()

    @property
    def service(self) -> AnalysisSessionService:
        """Return the backing Analysis session service."""
        return self._service

    @property
    def source_path(self) -> str | None:
        """Return the current Analysis source state file path, if loaded."""
        source = self._service.source
        return None if source is None else str(source.state_file)

    @property
    def displayed_rows(self) -> tuple[AnalysisTableRow, ...]:
        """Return currently displayed rows after filter/group controls."""
        return self._display_rows

    def restore_source(self, source: str | None) -> bool:
        """Restore an Analysis source from saved UI state."""
        if not source:
            return False
        return self.load_session(source)

    def apply_context(self, context: CommandExecutionContext) -> bool:
        """Attach Analysis to the currently selected training model context."""
        if context.model_folder is None:
            return False
        return self.load_model_context(
            context.model_folder,
            context.model_name,
            is_training=True,
        )

    def load_model_context(
        self,
        model_folder: str | Path,
        model_name: str | None = None,
        *,
        is_training: bool = False,
    ) -> bool:
        """Load a session from a model folder/name pair."""
        try:
            rows = self._service.load_model(
                model_folder,
                model_name,
                is_training=is_training,
            )
        except (AnalysisSessionError, OSError, ValueError) as err:
            self._status_label.setText(str(err))
            self._sync_actions()
            return False
        self._after_rows_loaded(rows)
        return True

    def load_session(self, source: str | Path) -> bool:
        """Load a session source and render summary rows."""
        try:
            rows = self._service.load_session(source)
        except (AnalysisSessionError, OSError, ValueError) as err:
            self._status_label.setText(str(err))
            self._sync_actions()
            return False
        self._after_rows_loaded(rows)
        return True

    def refresh_session(self) -> bool:
        """Refresh the currently loaded session summary rows."""
        try:
            rows = self._service.refresh_summaries()
        except (AnalysisSessionError, OSError, ValueError) as err:
            self._status_label.setText(str(err))
            self._sync_actions()
            return False
        self._rows = tuple(rows)
        self._apply_row_controls()
        self._sync_actions()
        return True

    def save_csv(self, filename: str | Path) -> int:
        """Save the current summary rows to CSV."""
        try:
            written = self._service.save_csv(filename)
        except OSError as err:
            self._status_label.setText(str(err))
            return 0
        self._status_label.setText(
            "No session data to save" if written == 0 else f"Saved {written} rows"
        )
        return written

    def clear_session(self) -> None:
        """Clear the current Analysis session and reset the table."""
        self._service.clear_session()
        self._rows = ()
        self._display_rows = ()
        self._table.setRowCount(0)
        self._source_label.setText("No session source loaded")
        self._status_label.setText("No session data loaded")
        self._detail_label.setText("Rows: 0 | Graphs: 0 | Iterations: 0 | Avg EGs/sec: 0.00")
        self._selection_label.setText("No session selected")
        self._sync_actions()
        self.session_cleared.emit()

    def cleanup_session(self, message: str = "No session data loaded") -> None:
        """Terminal lifecycle cleanup for stop/failure/reload/close/project change."""
        self.clear_session()
        self._status_label.setText(message)

    def _after_rows_loaded(self, rows: tuple[AnalysisTableRow, ...]) -> None:
        """Render and signal a successfully loaded session."""
        self._rows = tuple(rows)
        self._update_source_label()
        self._apply_row_controls()
        self._sync_actions()
        if self.source_path is not None:
            self.session_loaded.emit(self.source_path)

    def _build_ui(self) -> None:
        """Build the panel layout."""
        self.setObjectName("qt-shell-analysis-panel")
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        self._title.setAlignment(Qt.AlignCenter)
        self._title.setObjectName("qt-shell-analysis-title")
        layout.addWidget(self._title)

        self._source_label.setObjectName("qt-shell-analysis-source")
        self._source_label.setAlignment(Qt.AlignCenter)
        self._source_label.setWordWrap(True)
        layout.addWidget(self._source_label)

        controls = QHBoxLayout()
        filter_label = QLabel("Filter:")
        filter_label.setObjectName("qt-shell-analysis-filter-label")
        self._filter_combo.setObjectName("qt-shell-analysis-filter")
        self._filter_combo.addItems(AnalysisSummaryService.FILTERS)
        group_label = QLabel("Group:")
        group_label.setObjectName("qt-shell-analysis-group-label")
        self._group_combo.setObjectName("qt-shell-analysis-group")
        self._group_combo.addItems(AnalysisSummaryService.GROUPS)
        controls.addWidget(filter_label)
        controls.addWidget(self._filter_combo)
        controls.addWidget(group_label)
        controls.addWidget(self._group_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._detail_label.setObjectName("qt-shell-analysis-detail")
        self._detail_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._detail_label)

        self._table.setObjectName("qt-shell-session-stats")
        self._table.setMinimumWidth(0)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setHorizontalHeaderLabels(AnalysisSessionService.TABLE_HEADERS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self._table, 1)

        self._selection_label.setObjectName("qt-shell-analysis-selection")
        self._selection_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._selection_label)

        footer = QHBoxLayout()
        self._status_label.setObjectName("qt-shell-analysis-status")
        footer.addWidget(self._status_label)
        footer.addStretch(1)
        for button in (
            self._open_button,
            self._refresh_button,
            self._save_button,
            self._clear_button,
        ):
            button.setObjectName(f"qt-shell-analysis-{button.text().lower()}")
            footer.addWidget(button)
        layout.addLayout(footer)

    def _connect_signals(self) -> None:
        """Connect panel button signals."""
        self._open_button.clicked.connect(self._open_session_dialog)
        self._refresh_button.clicked.connect(self.refresh_session)
        self._save_button.clicked.connect(self._save_csv_dialog)
        self._clear_button.clicked.connect(self.clear_session)
        self._filter_combo.currentTextChanged.connect(lambda _text: self._apply_row_controls())
        self._group_combo.currentTextChanged.connect(lambda _text: self._apply_row_controls())
        self._table.itemSelectionChanged.connect(self._selection_changed)

    def _open_session_dialog(self) -> None:
        """Prompt for a model state file and load it."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Analysis Session",
            "",
            "Faceswap state files (*_state.json);;JSON files (*.json);;All files (*)",
        )
        if filename:
            self.load_session(filename)

    def _save_csv_dialog(self) -> None:
        """Prompt for a CSV output path and save summaries."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis Summary",
            "analysis_summary.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if filename:
            self.save_csv(filename)

    def _apply_row_controls(self) -> None:
        """Render rows after applying filter and group controls."""
        self._display_rows = self._summary_service.display_rows(
            self._rows,
            filter_name=self._filter_combo.currentText(),
            group_name=self._group_combo.currentText(),
        )
        self._render_rows(self._display_rows)
        self._update_summary_status()
        self._selection_changed()
        self._sync_actions()

    def _render_rows(self, rows: tuple[AnalysisTableRow, ...]) -> None:
        """Render Analysis summary rows into the table."""
        self._table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row.values):
                item = QTableWidgetItem(self._display_value(value, column_index))
                if column_index in (0, 1, 5, 6, 7):
                    item.setTextAlignment(Qt.AlignCenter)
                if row.is_total:
                    item.setData(Qt.ItemDataRole.UserRole, "total")
                self._table.setItem(row_index, column_index, item)

    @staticmethod
    def _display_value(value: object, column_index: int) -> str:
        """Return a display string for one table value."""
        if column_index == 0:
            return "✓" if value else ""
        return "" if value is None else str(value)

    def _selection_changed(self) -> None:
        """Update selected-session detail from the current table selection."""
        row_index = self._table.currentRow()
        row = (
            None
            if row_index < 0 or row_index >= len(self._display_rows)
            else self._display_rows[row_index]
        )
        self._selection_label.setText(self._summary_service.row_detail(row))

    def _update_summary_status(self) -> None:
        """Update footer status and detail metrics from current summary rows."""
        metrics = self._summary_service.from_session(self._service, rows=self._display_rows)
        self._status_label.setText(metrics.status_text)
        self._detail_label.setText(metrics.detail_text)

    def _update_source_label(self) -> None:
        """Update source label from service source."""
        source = self._service.source
        if source is None:
            self._source_label.setText("No session source loaded")
            return
        label = f"{source.model_name}  |  {source.model_dir}"
        if self._service.is_training:
            label = f"Training: {label}"
        self._source_label.setText(label)

    def _sync_actions(self) -> None:
        """Enable buttons based on current session state."""
        loaded = self._service.is_loaded
        has_rows = self._table.rowCount() > 0
        training = self._service.is_training
        self._refresh_button.setEnabled(loaded)
        self._save_button.setEnabled(has_rows and not training)
        self._clear_button.setEnabled((loaded or has_rows) and not training)
        self._filter_combo.setEnabled(loaded or has_rows)
        self._group_combo.setEnabled(loaded or has_rows)
