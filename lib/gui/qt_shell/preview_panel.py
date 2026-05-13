#!/usr/bin/env python3
"""Qt Preview panel backed by PreviewOutputService."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
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


class PreviewPanel(QWidget):
    """Runtime Preview panel for loading and refreshing extract/convert output images."""

    def __init__(
        self,
        service: PreviewOutputService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service or PreviewOutputService()
        self._source_label = QLabel("No preview source configured")
        self._status_label = QLabel("No preview images loaded")
        self._image_label = QLabel("No preview selected")
        self._image_list = QListWidget()
        self._open_button = QPushButton("Open")
        self._refresh_button = QPushButton("Refresh")
        self._clear_button = QPushButton("Clear")
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

    def restore_source(self, source: str | None) -> bool:
        """Restore a Preview source from saved UI state."""
        if not source:
            return False
        self.configure_output(source)
        return True

    def apply_context(self, context: CommandExecutionContext) -> bool:
        """Apply command-derived preview output context."""
        if context.preview_output_path is None:
            return False
        self.configure_output(context.preview_output_path)
        return True

    def configure_output(self, source: str | Path) -> None:
        """Configure a preview source path that may not exist yet."""
        self._service.configure(source)
        self._update_source_label()
        self.refresh_preview()

    def load_output(self, source: str | Path) -> bool:
        """Load preview images from an existing file or folder."""
        self._service.configure(source)
        return self.refresh_preview(validate=True)

    def refresh_preview(self, *, validate: bool = False) -> bool:
        """Refresh preview image discovery and render the first available image."""
        try:
            images = self._service.refresh(validate=validate)
        except (PreviewOutputError, OSError, ValueError) as err:
            self._status_label.setText(str(err))
            self._image_list.clear()
            self._image_label.setText("No preview selected")
            self._sync_actions()
            return False
        self._render_images(images)
        self._update_source_label()
        self._set_status(len(images))
        self._sync_actions()
        return True

    def clear_preview(self) -> None:
        """Clear preview source and rendered image state."""
        self._service.clear()
        self._image_list.clear()
        self._source_label.setText("No preview source configured")
        self._status_label.setText("No preview images loaded")
        self._image_label.setText("No preview selected")
        self._image_label.setPixmap(QPixmap())
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
        self._image_label.setObjectName("qt-shell-preview-image")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setWordWrap(True)
        self._image_label.setMinimumSize(240, 180)
        scroll.setWidget(self._image_label)
        content.addWidget(scroll, 1)
        layout.addLayout(content, 1)

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
        self._image_list.currentItemChanged.connect(self._current_image_changed)

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

    def _render_images(self, images: tuple[PreviewOutputImage, ...]) -> None:
        """Render image filenames into the list widget."""
        self._image_list.clear()
        self._image_label.setPixmap(QPixmap())
        if not images:
            self._image_label.setText("No preview selected")
            return
        for image in images:
            item = QListWidgetItem(image.name)
            item.setData(Qt.UserRole, str(image.path))
            self._image_list.addItem(item)
        self._image_list.setCurrentRow(0)

    def _current_image_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Update the image display when a preview list item is selected."""
        if current is None:
            return
        path = Path(str(current.data(Qt.UserRole)))
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._image_label.setPixmap(QPixmap())
            self._image_label.setText(f"Unable to load image: {path.name}")
            return
        self._image_label.setText("")
        self._image_label.setPixmap(pixmap)

    def _set_status(self, image_count: int) -> None:
        """Update status text from current source and image count."""
        source = self._service.source
        if source is None:
            self._status_label.setText("No preview source configured")
        elif not source.exists():
            self._status_label.setText(f"Waiting for preview output: {source}")
        elif image_count == 0:
            self._status_label.setText("No preview images found")
        elif image_count == 1:
            self._status_label.setText("Loaded 1 preview image")
        else:
            self._status_label.setText(f"Loaded {image_count} preview images")

    def _update_source_label(self) -> None:
        """Update source label text."""
        source = self._service.source
        self._source_label.setText(
            "No preview source configured" if source is None else f"Preview source: {source}"
        )

    def _sync_actions(self) -> None:
        """Enable actions based on configured preview state."""
        has_source = self._service.source is not None
        self._refresh_button.setEnabled(has_source)
        self._clear_button.setEnabled(has_source or self._image_list.count() > 0)
