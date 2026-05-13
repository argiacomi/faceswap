#!/usr/bin/env python3
"""Qt Analysis panel backed by AnalysisSessionService."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
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


class AnalysisPanel(QWidget):
    """Runtime Analysis panel for loading, refreshing and exporting session summaries."""

    def __init__(
        self,
        service: AnalysisSessionService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service or AnalysisSessionService()
        self._title = QLabel("Session Stats")
        self._source_label = QLabel("No session source loaded")
        self._status_label = QLabel("No session data loaded")
        self._table = QTableWidget(0, len(AnalysisSessionService.TABLE_HEADERS))
        self._open_button = QPushButton("Open")
        self._refresh_button = QPushButton("Refresh")
        self._save_button = QPushButton("Save")
        self._clear_button = QPushButton("Clear")
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

    def restore_source(self, source: str | None) -> bool:
        """Restore an Analysis source from saved UI state."""
        if not source:
            return False
        return self.load_session(source)

    def load_session(self, source: str | Path) -> bool:
        """Load a session source and render summary rows."""
        try:
            rows = self._service.load_session(source)
        except (AnalysisSessionError, OSError, ValueError) as err:
            self._status_label.setText(str(err))
            self._sync_actions()
            return False
        self._render_rows(rows)
        self._update_source_label()
        self._set_loaded_status(len(rows))
        self._sync_actions()
        return True

    def refresh_session(self) -> bool:
        """Refresh the currently loaded session summary rows."""
        try:
            rows = self._service.refresh_summaries()
        except (AnalysisSessionError, OSError, ValueError) as err:
            self._status_label.setText(str(err))
            self._sync_actions()
            return False
        self._render_rows(rows)
        self._set_loaded_status(len(rows))
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
        self._table.setRowCount(0)
        self._source_label.setText("No session source loaded")
        self._status_label.setText("No session data loaded")
        self._sync_actions()

    def _build_ui(self) -> None:
        """Build the panel layout."""
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

        self._table.setObjectName("qt-shell-session-stats")
        self._table.setMinimumWidth(0)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setHorizontalHeaderLabels(AnalysisSessionService.TABLE_HEADERS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self._table, 1)

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

    def _render_rows(self, rows: tuple[AnalysisTableRow, ...]) -> None:
        """Render Analysis summary rows into the table."""
        self._table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row.values):
                item = QTableWidgetItem(self._display_value(value, column_index))
                if column_index in (0, 1, 5, 6, 7):
                    item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(row_index, column_index, item)

    @staticmethod
    def _display_value(value: object, column_index: int) -> str:
        """Return a display string for one table value."""
        if column_index == 0:
            return "✓" if value else ""
        return "" if value is None else str(value)

    def _set_loaded_status(self, row_count: int) -> None:
        """Update status text for current row count."""
        if row_count == 0:
            self._status_label.setText("No session data loaded")
        elif row_count == 1:
            self._status_label.setText("Loaded 1 session row")
        else:
            self._status_label.setText(f"Loaded {row_count} session rows")

    def _update_source_label(self) -> None:
        """Update source label from service source."""
        source = self._service.source
        if source is None:
            self._source_label.setText("No session source loaded")
            return
        self._source_label.setText(f"{source.model_name}  |  {source.model_dir}")

    def _sync_actions(self) -> None:
        """Enable buttons based on current session state."""
        loaded = self._service.is_loaded
        has_rows = self._table.rowCount() > 0
        self._refresh_button.setEnabled(loaded)
        self._save_button.setEnabled(has_rows)
        self._clear_button.setEnabled(loaded or has_rows)
