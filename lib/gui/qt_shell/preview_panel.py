#!/usr/bin/env python3
"""Qt Preview panel backed by PreviewOutputService."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from lib.gui.services.command_context import CommandExecutionContext
from lib.gui.services.preview_output_service import (
    PreviewOutputError,
    PreviewOutputImage,
    PreviewOutputService,
)


class PreviewImageView(QLabel):
    """Image display with simple zoom and pan behavior matching the Qt graph idiom."""

    view_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("No preview selected", parent)
        self.setObjectName("qt-shell-preview-image")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(240, 180)
        self._source_pixmap = QPixmap()
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start: tuple[float, float] | None = None

    @property
    def zoom(self) -> float:
        """Return the current zoom factor."""
        return self._zoom

    @property
    def pan(self) -> tuple[float, float]:
        """Return normalized horizontal/vertical pan offsets."""
        return self._pan_x, self._pan_y

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        """Set the source pixmap and render it through the current view transform."""
        self._source_pixmap = pixmap
        if pixmap.isNull():
            self.reset_view(update=False)
            self.setPixmap(QPixmap())
            return
        self.setText("")
        self._render_pixmap()

    def clear_preview(self, message: str = "No preview selected") -> None:
        """Clear pixmap, zoom and pan state."""
        self._source_pixmap = QPixmap()
        self.reset_view(update=False)
        self.setPixmap(QPixmap())
        self.setText(message)

    def zoom_in(self) -> None:
        """Zoom into the preview image."""
        self._set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        """Zoom out of the preview image."""
        self._set_zoom(self._zoom / 1.25)

    def reset_view(self, *, update: bool = True) -> None:
        """Reset preview zoom and pan."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        if update:
            self._render_pixmap()
            self.view_changed.emit()

    def resizeEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Re-render scaled preview on resize."""
        super().resizeEvent(event)
        self._render_pixmap()

    def wheelEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Zoom the preview with the mouse wheel."""
        if self._source_pixmap.isNull():
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def mousePressEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Start image panning when zoomed in."""
        if event.button() == Qt.MouseButton.LeftButton and self._zoom > 1.0:
            self._drag_start = (float(event.position().x()), float(event.position().y()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Pan the preview while dragging."""
        if self._drag_start is None or self._zoom <= 1.0:
            super().mouseMoveEvent(event)
            return
        current_x = float(event.position().x())
        current_y = float(event.position().y())
        start_x, start_y = self._drag_start
        self._drag_start = (current_x, current_y)
        self._set_pan(
            self._pan_x + ((start_x - current_x) / max(1, self.width())),
            self._pan_y + ((start_y - current_y) / max(1, self.height())),
        )
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type:ignore[no-untyped-def] # noqa:N802
        """Stop image panning."""
        self._drag_start = None
        super().mouseReleaseEvent(event)

    def _set_zoom(self, zoom: float) -> None:
        """Clamp and apply image zoom."""
        self._zoom = max(1.0, min(50.0, zoom))
        if self._zoom == 1.0:
            self._pan_x = 0.0
            self._pan_y = 0.0
        self._render_pixmap()
        self.view_changed.emit()

    def _set_pan(self, pan_x: float, pan_y: float) -> None:
        """Clamp and apply image pan."""
        self._pan_x = max(0.0, min(1.0, pan_x))
        self._pan_y = max(0.0, min(1.0, pan_y))
        self._render_pixmap()
        self.view_changed.emit()

    def _render_pixmap(self) -> None:
        """Render the source pixmap scaled and cropped for zoom/pan."""
        if self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        target_width = max(1, int(self.width() * self._zoom))
        target_height = max(1, int(self.height() * self._zoom))
        scaled = self._source_pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        if self._zoom <= 1.0:
            self.setPixmap(scaled)
            return
        crop_width = min(self.width(), scaled.width())
        crop_height = min(self.height(), scaled.height())
        max_x = max(0, scaled.width() - crop_width)
        max_y = max(0, scaled.height() - crop_height)
        x_pos = int(round(max_x * self._pan_x))
        y_pos = int(round(max_y * self._pan_y))
        self.setPixmap(scaled.copy(x_pos, y_pos, crop_width, crop_height))


class PreviewPanel(QWidget):
    """Runtime Preview panel for loading and refreshing output/training images."""

    DEFAULT_REFRESH_MS = 1500

    def __init__(
        self,
        service: PreviewOutputService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service or PreviewOutputService()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(self.DEFAULT_REFRESH_MS)
        self._source_label = QLabel("No preview source configured")
        self._status_label = QLabel("No preview images loaded")
        self._image_label = PreviewImageView()
        self._image_list = QListWidget()
        self._open_button = QPushButton("Open")
        self._refresh_button = QPushButton("Refresh")
        self._clear_button = QPushButton("Clear")
        self._zoom_in_button = QPushButton("Zoom In")
        self._zoom_out_button = QPushButton("Zoom Out")
        self._reset_view_button = QPushButton("Reset View")
        self._build_ui()
        self._connect_signals()
        self._sync_actions()

    @property
    def service(self) -> PreviewOutputService:
        """Return the backing preview output service."""
        return self._service

    @property
    def source_path(self) -> str | None:
        """Return the configured preview source path, if any."""
        source = self._service.source
        return None if source is None else str(source)

    @property
    def is_live_refreshing(self) -> bool:
        """Return whether preview live refresh is active."""
        return self._refresh_timer.isActive()

    @property
    def zoom(self) -> float:
        """Return the image view zoom."""
        return self._image_label.zoom

    @property
    def pan(self) -> tuple[float, float]:
        """Return the image view pan."""
        return self._image_label.pan

    def start_live_refresh(self, interval_ms: int | None = None) -> None:
        """Start periodic preview refresh for running jobs."""
        if interval_ms is not None:
            self._refresh_timer.setInterval(max(250, interval_ms))
        if self._service.source is not None:
            self._refresh_timer.start()
            self._sync_actions()

    def stop_live_refresh(self) -> None:
        """Stop periodic preview refresh."""
        self._refresh_timer.stop()
        self._sync_actions()

    def restore_source(self, source: str | None) -> bool:
        """Restore a Preview source from saved UI state."""
        if not source:
            return False
        self.configure_output(source)
        return True

    def apply_context(self, context: CommandExecutionContext) -> bool:
        """Apply command-derived preview output or train preview context."""
        if context.preview_output_path is not None:
            self.configure_output(context.preview_output_path, batch_mode=context.batch_mode)
            return True
        if context.model_folder is not None or context.model_name is not None:
            self.configure_training_preview()
            return True
        return False

    def configure_training_preview(self, source: str | Path | None = None) -> None:
        """Configure the Tk-compatible training preview cache source."""
        self._service.configure_training(source)
        self._update_source_label()
        self.refresh_preview()

    def configure_output(self, source: str | Path, *, batch_mode: bool = False) -> None:
        """Configure a preview source path that may not exist yet."""
        self._service.configure(source, batch_mode=batch_mode)
        self._update_source_label()
        self.refresh_preview()

    def load_output(self, source: str | Path) -> bool:
        """Load preview images from an existing file or folder."""
        self._service.configure(source)
        return self.refresh_preview(validate=True)

    def refresh_preview(self, *, validate: bool = False) -> bool:
        """Refresh preview image discovery and preserve current selection where possible."""
        selected = self._selected_image_path()
        try:
            images = self._service.refresh(validate=validate)
        except (PreviewOutputError, OSError, ValueError) as err:
            self._status_label.setText(str(err))
            self._image_list.clear()
            self._image_label.clear_preview()
            self._sync_actions()
            return False
        self._render_images(images, selected_path=selected)
        self._update_source_label()
        self._set_status(len(images))
        self._sync_actions()
        return True

    def clear_preview(self) -> None:
        """Clear preview source and rendered image state."""
        self.cleanup_preview(clear_service=True, message="No preview images loaded")

    def cleanup_preview(
        self,
        *,
        clear_service: bool = True,
        message: str = "No preview images loaded",
    ) -> None:
        """Stop polling and clear UI/service state for terminal lifecycle paths."""
        self._refresh_timer.stop()
        if clear_service:
            self._service.clear()
        self._image_list.clear()
        self._source_label.setText("No preview source configured")
        self._status_label.setText(message)
        self._image_label.clear_preview()
        self._sync_actions()

    def zoom_in(self) -> None:
        """Zoom into the selected preview image."""
        self._image_label.zoom_in()
        self._sync_actions()

    def zoom_out(self) -> None:
        """Zoom out of the selected preview image."""
        self._image_label.zoom_out()
        self._sync_actions()

    def reset_view(self) -> None:
        """Reset preview zoom and pan."""
        self._image_label.reset_view()
        self._sync_actions()

    def _build_ui(self) -> None:
        """Build the panel layout."""
        self.setObjectName("qt-shell-preview-panel")
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        title = QLabel("Preview Output")
        title.setObjectName("qt-shell-preview-title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self._source_label.setObjectName("qt-shell-preview-source")
        self._source_label.setAlignment(Qt.AlignCenter)
        self._source_label.setWordWrap(True)
        layout.addWidget(self._source_label)

        content = QHBoxLayout()
        self._image_list.setObjectName("qt-shell-preview-list")
        self._image_list.setMinimumWidth(180)
        content.addWidget(self._image_list, 0)

        scroll = QScrollArea()
        scroll.setObjectName("qt-shell-preview-scroll")
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self._image_label)
        content.addWidget(scroll, 1)
        layout.addLayout(content, 1)

        controls = QHBoxLayout()
        for button, name in (
            (self._zoom_in_button, "zoom-in"),
            (self._zoom_out_button, "zoom-out"),
            (self._reset_view_button, "reset-view"),
        ):
            button.setObjectName(f"qt-shell-preview-{name}")
            controls.addWidget(button)
        controls.addStretch(1)
        layout.addLayout(controls)

        footer = QHBoxLayout()
        self._status_label.setObjectName("qt-shell-preview-status")
        footer.addWidget(self._status_label)
        footer.addStretch(1)
        for button in (self._open_button, self._refresh_button, self._clear_button):
            button.setObjectName(f"qt-shell-preview-{button.text().lower()}")
            footer.addWidget(button)
        layout.addLayout(footer)

    def _connect_signals(self) -> None:
        """Connect panel signals."""
        self._open_button.clicked.connect(lambda _checked=False: self._open_output_dialog())
        self._refresh_button.clicked.connect(lambda _checked=False: self.refresh_preview())
        self._clear_button.clicked.connect(lambda _checked=False: self.clear_preview())
        self._zoom_in_button.clicked.connect(lambda _checked=False: self.zoom_in())
        self._zoom_out_button.clicked.connect(lambda _checked=False: self.zoom_out())
        self._reset_view_button.clicked.connect(lambda _checked=False: self.reset_view())
        self._image_list.currentItemChanged.connect(self._current_image_changed)
        self._refresh_timer.timeout.connect(lambda: self.refresh_preview())

    def _open_output_dialog(self) -> None:
        """Prompt for preview output source and load it."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Preview Image",
            "",
            "Images (*.bmp *.gif *.jpeg *.jpg *.png *.webp);;All files (*)",
        )
        if filename:
            self.load_output(filename)
            return
        folder = QFileDialog.getExistingDirectory(self, "Open Preview Output Folder", "")
        if folder:
            self.load_output(folder)

    def _render_images(
        self,
        images: tuple[PreviewOutputImage, ...],
        *,
        selected_path: str | None = None,
    ) -> None:
        """Render image filenames into the list widget."""
        self._image_list.clear()
        self._image_label.clear_preview()
        if not images:
            return
        selected_row = 0
        for row, image in enumerate(images):
            item = QListWidgetItem(image.name)
            item.setData(Qt.UserRole, str(image.path))
            self._image_list.addItem(item)
            if selected_path == str(image.path):
                selected_row = row
        self._image_list.setCurrentRow(selected_row)

    def _selected_image_path(self) -> str | None:
        """Return the currently selected preview image path."""
        current = self._image_list.currentItem()
        return None if current is None else str(current.data(Qt.UserRole))

    def _current_image_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Update the image display when a preview list item is selected."""
        if current is None:
            self._image_label.clear_preview()
            self._sync_actions()
            return
        path = Path(str(current.data(Qt.UserRole)))
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._image_label.clear_preview(f"Unable to load image: {path.name}")
            self._sync_actions()
            return
        self._image_label.set_preview_pixmap(pixmap)
        self._sync_actions()

    def _set_status(self, image_count: int) -> None:
        """Update status text from current source and image count."""
        source = self._service.source
        live = " (live)" if self.is_live_refreshing else ""
        prefix = "Training preview" if self._service.mode == "train" else "Preview"
        if source is None:
            self._status_label.setText("No preview source configured")
        elif not source.exists():
            self._status_label.setText(f"Waiting for {prefix.lower()} output: {source}{live}")
        elif image_count == 0:
            self._status_label.setText(f"No {prefix.lower()} images found{live}")
        elif image_count == 1:
            self._status_label.setText(f"Loaded 1 {prefix.lower()} image{live}")
        else:
            self._status_label.setText(f"Loaded {image_count} {prefix.lower()} images{live}")

    def _update_source_label(self) -> None:
        """Update source label text."""
        source = self._service.source
        if source is None:
            text = "No preview source configured"
        elif self._service.mode == "train":
            text = f"Training preview source: {source}"
        elif self._service.mode == "batch":
            resolved = self._service.resolved_source
            suffix = f" (current batch: {resolved})" if resolved and resolved != source else ""
            text = f"Batch preview source: {source}{suffix}"
        else:
            text = f"Preview source: {source}"
        self._source_label.setText(text)

    def _sync_actions(self) -> None:
        """Enable actions based on configured preview state."""
        has_source = self._service.source is not None
        pixmap = self._image_label.pixmap()
        has_image = pixmap is not None and not pixmap.isNull()
        self._refresh_button.setEnabled(has_source)
        self._clear_button.setEnabled(has_source or self._image_list.count() > 0)
        self._zoom_in_button.setEnabled(has_image)
        self._zoom_out_button.setEnabled(has_image and self.zoom > 1.0)
        self._reset_view_button.setEnabled(has_image and (self.zoom > 1.0 or self.pan != (0.0, 0.0)))
