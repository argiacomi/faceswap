#!/usr/bin/env python3
"""Native Qt Manual Tool shell.

This is the first Qt-native surface for the Manual Tool migration.  It provides
startup validation, frame display, thumbnail/selection placeholders, action
state, dirty-state lifecycle and a legacy fallback launcher without importing
``tkinter`` on the Qt path.
"""

from __future__ import annotations

import subprocess
import typing as T

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from lib.gui.services.command_builder import CommandBuilder
from tools.manual.session import (
    ManualEditorState,
    ManualFrame,
    ManualSession,
    ManualVideoMetadata,
)


class ManualFrameView(QLabel):
    """Manual Tool frame display with basic zoom/pan seams."""

    view_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("No frame selected", parent)
        self.setObjectName("qt-manual-frame-view")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._source_pixmap = QPixmap()
        self._zoom = 1.0
        self._drag_start: tuple[float, float] | None = None

    @property
    def zoom(self) -> float:
        """Return current zoom factor."""
        return self._zoom

    @property
    def has_frame(self) -> bool:
        """Return whether a frame pixmap is loaded."""
        return not self._source_pixmap.isNull()

    def load_frame(self, frame: ManualFrame) -> bool:
        """Load and display a source frame image."""
        pixmap = QPixmap(frame.path)
        if pixmap.isNull():
            self.clear_frame(f"Could not load frame: {frame.name}")
            return False
        self._source_pixmap = pixmap
        self.setText("")
        self._render_pixmap()
        return True

    def clear_frame(self, message: str = "No frame selected") -> None:
        """Clear the current frame."""
        self._source_pixmap = QPixmap()
        self._zoom = 1.0
        self.setPixmap(QPixmap())
        self.setText(message)
        self.view_changed.emit()

    def zoom_in(self) -> None:
        """Zoom into the frame display."""
        self._set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        """Zoom out of the frame display."""
        self._set_zoom(self._zoom / 1.25)

    def reset_view(self) -> None:
        """Reset zoom and redraw."""
        self._set_zoom(1.0)

    def resizeEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Re-render scaled pixmap on resize."""
        super().resizeEvent(event)
        self._render_pixmap()

    def wheelEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Zoom with mouse wheel."""
        if self._source_pixmap.isNull():
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def _set_zoom(self, zoom: float) -> None:
        """Clamp and apply zoom."""
        self._zoom = max(1.0, min(20.0, zoom))
        self._render_pixmap()
        self.view_changed.emit()

    def _render_pixmap(self) -> None:
        """Render the source pixmap scaled to the available panel."""
        if self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        target_width = max(1, int(self.width() * self._zoom))
        target_height = max(1, int(self.height() * self._zoom))
        self.setPixmap(
            self._source_pixmap.scaled(
                target_width,
                target_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class ManualThumbnailPanel(QListWidget):
    """Frame thumbnail/selection panel placeholder for Qt Manual Tool."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-thumbnail-panel")
        self.setMinimumSize(0, 96)

    def set_frames(self, frames: tuple[ManualFrame, ...]) -> None:
        """Display selectable entries for available frames."""
        self.clear()
        if not frames:
            item = QListWidgetItem("No image frames available yet")
            item.setFlags(Qt.NoItemFlags)
            self.addItem(item)
            return
        for frame in frames:
            item = QListWidgetItem(f"{frame.index + 1}: {frame.name}")
            item.setData(Qt.UserRole, frame.index)
            self.addItem(item)


class ManualToolWindow(QMainWindow):
    """Qt-native Manual Tool window with legacy fallback support."""

    dirty_changed = Signal(bool)

    def __init__(
        self,
        session: ManualSession,
        *,
        legacy_args: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-tool-window")
        self.setWindowTitle("Faceswap Manual Tool")
        self._session = session
        self._editor_state = session.create_editor_state()
        self._video_metadata: ManualVideoMetadata | None = None
        self._legacy_args = legacy_args or []
        self._current_frame: ManualFrame | None = None
        self._frame_view = ManualFrameView()
        self._thumbnail_panel = ManualThumbnailPanel()
        self._status_label = QLabel()
        self._metadata_label = QLabel()
        self._save_action: QAction | None = None
        self._legacy_action: QAction | None = None
        self._editor_state.subscribe("unsaved", self._on_unsaved_changed)
        self._build_ui()
        self._connect_signals()
        self._load_session()

    @property
    def session(self) -> ManualSession:
        """Return the backing GUI-neutral session."""
        return self._session

    @property
    def is_dirty(self) -> bool:
        """Return whether the Qt Manual Tool has unsaved edits."""
        return self._editor_state.unsaved

    @property
    def editor_state(self) -> ManualEditorState:
        """Return the shared GUI-neutral editor state."""
        return self._editor_state

    @property
    def video_metadata(self) -> ManualVideoMetadata | None:
        """Return cached neutral video metadata, if loaded."""
        return self._video_metadata

    @classmethod
    def from_command_values(
        cls,
        values: T.Mapping[str, object],
        *,
        builder: CommandBuilder,
        parent: QWidget | None = None,
    ) -> ManualToolWindow:
        """Create a Manual Tool window from command-panel values."""
        session = ManualSession.from_cli_values(values)
        legacy_args = builder.build("tools", "manual", values, generate=False)
        return cls(session, legacy_args=legacy_args, parent=parent)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa:N802
        """Prompt before closing when the editor has unsaved changes."""
        if not self._editor_state.unsaved:
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "Unsaved Manual Tool Changes",
            "Close the Manual Tool and discard unsaved changes?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def mark_dirty(self, dirty: bool = True) -> None:
        """Set dirty state and update action availability."""
        self._editor_state.set("unsaved", dirty)

    def save(self) -> bool:
        """Save seam for future alignment persistence integration."""
        self.mark_dirty(False)
        self.statusBar().showMessage("Manual Tool session saved", 5000)
        return True

    def launch_legacy(self) -> bool:
        """Launch the existing Tk Manual Tool as a fallback subprocess."""
        if not self._legacy_args:
            QMessageBox.warning(self, "Legacy Manual Tool", "Legacy command is unavailable.")
            return False
        try:
            subprocess.Popen(self._legacy_args)  # noqa:S603 - args are built internally
        except OSError as err:
            QMessageBox.critical(self, "Legacy Manual Tool", str(err))
            return False
        self.statusBar().showMessage("Launched legacy Manual Tool", 5000)
        return True

    def _build_ui(self) -> None:
        """Build Manual Tool widgets."""
        self.resize(980, 680)
        self._build_toolbar()
        status = QStatusBar()
        self.setStatusBar(status)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)
        self._metadata_label.setObjectName("qt-manual-session-metadata")
        self._metadata_label.setWordWrap(True)
        left_layout.addWidget(self._metadata_label)
        left_layout.addWidget(self._frame_view, 1)
        left_layout.addWidget(self._status_label)

        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("qt-manual-main-splitter")
        splitter.addWidget(left)
        splitter.addWidget(self._thumbnail_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

    def _build_toolbar(self) -> None:
        """Build Manual Tool action toolbar."""
        toolbar = QToolBar("Manual Tool")
        toolbar.setObjectName("qt-manual-toolbar")
        self.addToolBar(toolbar)
        self._save_action = toolbar.addAction("Save", self.save)
        toolbar.addSeparator()
        toolbar.addAction("Previous", self._previous_frame)
        toolbar.addAction("Next", self._next_frame)
        toolbar.addAction("Zoom In", self._frame_view.zoom_in)
        toolbar.addAction("Zoom Out", self._frame_view.zoom_out)
        toolbar.addAction("Reset View", self._frame_view.reset_view)
        toolbar.addSeparator()
        toolbar.addAction("Mark Dirty", lambda: self.mark_dirty(True))
        self._legacy_action = toolbar.addAction("Open Legacy Tool", self.launch_legacy)

    def _connect_signals(self) -> None:
        """Connect selection and dirty-state signals."""
        self._thumbnail_panel.currentItemChanged.connect(self._thumbnail_selected)
        self.dirty_changed.connect(
            lambda dirty: self.statusBar().showMessage(
                "Manual Tool has unsaved changes" if dirty else "Manual Tool ready",
                5000,
            )
        )

    def _load_session(self) -> None:
        """Populate the Qt Manual Tool from a neutral session."""
        alignments = self._session.alignments_handle()
        if self._session.is_video_input:
            self._video_metadata = alignments.video_metadata()
        frame_summary: str
        if self._session.has_images:
            frame_summary = str(self._session.frame_count)
        elif self._video_metadata is not None and self._video_metadata.is_valid:
            frame_summary = f"{self._video_metadata.frame_count} (video)"
        else:
            frame_summary = "video input"
        thumbs_state = "cached" if alignments.has_thumbnails() else "needs generation"
        if self._session.thumb_regenerate:
            thumbs_state = "regenerate forced"
        self._metadata_label.setText(
            "\n".join(
                (
                    f"Input: {self._session.frames}",
                    f"Alignments: {alignments.path}",
                    f"Frames: {frame_summary}",
                    f"Thumbnails: {thumbs_state}",
                )
            )
        )
        self._thumbnail_panel.set_frames(self._session.frame_list)
        if self._session.frame_list:
            self._thumbnail_panel.setCurrentRow(0)
        else:
            self._frame_view.clear_frame(
                "Video input detected. Frame extraction will be wired in a follow-up."
            )
        self._status_label.setText("Native Qt Manual Tool loaded")
        self._sync_actions()

    def _on_unsaved_changed(self, dirty: bool) -> None:
        """Forward editor-state unsaved changes to UI signals."""
        self.dirty_changed.emit(bool(dirty))
        self._sync_actions()

    def _thumbnail_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Display the selected source frame."""
        if current is None:
            return
        index = current.data(Qt.UserRole)
        if not isinstance(index, int) or index >= len(self._session.frame_list):
            return
        frame = self._session.frame_list[index]
        self._current_frame = frame
        self._editor_state.set("frame_index", frame.index)
        if self._frame_view.load_frame(frame):
            self._status_label.setText(
                f"Frame {frame.index + 1} of {self._session.frame_count}: {frame.name}"
            )

    def _previous_frame(self) -> None:
        """Select previous frame."""
        row = self._thumbnail_panel.currentRow()
        if row > 0:
            self._thumbnail_panel.setCurrentRow(row - 1)

    def _next_frame(self) -> None:
        """Select next frame."""
        row = self._thumbnail_panel.currentRow()
        if 0 <= row < self._thumbnail_panel.count() - 1:
            self._thumbnail_panel.setCurrentRow(row + 1)

    def _sync_actions(self) -> None:
        """Update action availability from session and dirty state."""
        if self._save_action is not None:
            self._save_action.setEnabled(self._editor_state.unsaved)
        if self._legacy_action is not None:
            self._legacy_action.setEnabled(bool(self._legacy_args))


__all__ = [
    "ManualFrameView",
    "ManualThumbnailPanel",
    "ManualToolWindow",
]
