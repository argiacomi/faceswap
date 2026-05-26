#!/usr/bin/env python3
"""Qt Manual Tool shell layout and widget composition helpers."""

from __future__ import annotations

import logging
import typing as T

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from lib.gui.qt_shell.theme import QtTheme, icon_for_action

from .actions import MANUAL_ACTIONS
from .face_viewer.viewport import _FACE_GRID_SIZES

logger = logging.getLogger(__name__)


class LayoutMixin:
    """Own root-window widget layout construction."""

    def _build_face_grid_panel(self) -> QWidget:
        """Return the face-strip + filtered-session face grid container."""
        container = QWidget()
        container.setObjectName("qt-manual-face-browser-panel")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        current_box = QWidget()
        current_layout = QVBoxLayout(current_box)
        current_layout.setContentsMargins(0, 0, 0, 0)
        current_layout.setSpacing(4)
        current_label = QLabel("Current frame faces")
        current_label.setObjectName("qt-manual-current-frame-faces-label")
        current_layout.addWidget(current_label)
        current_layout.addWidget(self._face_panel, 1)
        layout.addWidget(current_box, 1)

        grid_box = QWidget()
        grid_layout = QVBoxLayout(grid_box)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(4)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        session_label = QLabel("Filtered session faces")
        session_label.setObjectName("qt-manual-filtered-session-faces-label")
        controls.addWidget(session_label)
        controls.addStretch(1)
        controls.addWidget(QLabel("Size:"))
        self._face_grid_size_combo.setObjectName("qt-manual-face-grid-size")
        self._face_grid_size_combo.addItems(tuple(_FACE_GRID_SIZES))
        size_name = self._editor_state.faces_size or "Medium"
        if size_name not in _FACE_GRID_SIZES:
            size_name = "Medium"
        self._face_grid_size_combo.setCurrentText(size_name)
        self._face_grid_size_combo.currentTextChanged.connect(self._on_face_grid_size_changed)
        self._face_grid_panel.set_face_size(size_name)
        controls.addWidget(self._face_grid_size_combo)
        grid_layout.addLayout(controls)
        grid_layout.addWidget(self._face_grid_panel, 1)
        layout.addWidget(grid_box, 2)
        return container

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
        self._mask_controls = self._build_mask_controls()
        left_layout.addWidget(self._mask_controls)
        self._aligner_controls = self._build_aligner_controls()
        left_layout.addWidget(self._aligner_controls)
        left_layout.addWidget(self._frame_view, 1)
        self._filter_controls = self._build_filter_controls()
        left_layout.addWidget(self._filter_controls)
        left_layout.addWidget(self._transport_bar)
        left_layout.addWidget(self._status_label)

        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("qt-manual-main-splitter")
        splitter.addWidget(left)
        splitter.addWidget(self._build_face_grid_panel())
        splitter.addWidget(self._thumbnail_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([500, 140, 80])
        self._manual_splitter = splitter
        self.setCentralWidget(splitter)

    def _build_toolbar(self) -> None:
        """Build the Manual Tool action toolbar from :data:`MANUAL_ACTIONS`."""
        theme = QtTheme.default()
        toolbar = QToolBar("Manual Tool")
        toolbar.setObjectName("qt-manual-toolbar")
        toolbar.setIconSize(QSize(theme.icon_size, theme.icon_size))
        self.addToolBar(toolbar)
        for spec in MANUAL_ACTIONS:
            if spec.separator_before and spec.toolbar_visible:
                toolbar.addSeparator()
            owner: QWidget = self
            shortcut_context = Qt.WindowShortcut
            if spec.focus_scope == "frame_view":
                owner = self._frame_view
                shortcut_context = Qt.WidgetWithChildrenShortcut
                self._frame_view.setFocusPolicy(Qt.StrongFocus)
            action = QAction(spec.label, owner)
            action.setObjectName(f"qt-manual-action-{spec.key}")
            if spec.tooltip:
                action.setToolTip(spec.tooltip)
                action.setStatusTip(spec.tooltip)
            if spec.icon:
                icon = icon_for_action(theme, spec.icon)
                if not icon.isNull():
                    action.setIcon(icon)
            shortcuts = [QKeySequence(text) for text in spec.shortcut]
            if shortcuts:
                action.setShortcuts(shortcuts)
                action.setShortcutContext(shortcut_context)
            handler = getattr(self, spec.handler)
            action.triggered.connect(self._make_action_dispatch(spec.key, handler))
            owner.addAction(action)
            if spec.toolbar_visible:
                toolbar.addAction(action)
            self._actions[spec.key] = action
        self._sync_play_action_icon()

    def _make_action_dispatch(
        self, key: str, handler: T.Callable[[], object]
    ) -> T.Callable[[], None]:
        """Return a closure that invokes ``handler`` and emits :attr:`action_triggered`."""

        def _dispatch(_checked: bool = False) -> None:
            self.action_triggered.emit(key)
            try:
                handler()
            except Exception:  # pragma: no cover - defensive; surface in logs
                logger.exception("Manual Tool action %s raised", key)

        return _dispatch

    def _connect_signals(self) -> None:
        """Connect selection and dirty-state signals."""
        self._thumbnail_panel.currentItemChanged.connect(self._thumbnail_selected)
        self._thumbnail_panel.currentRowChanged.connect(self._on_thumbnail_row_changed)
        self._transport_bar.position_changed.connect(self._on_transport_position_changed)
        self._face_panel.face_selected.connect(self._on_face_selected)
        self._face_grid_panel.face_activated.connect(self._on_face_grid_activated)
        self._face_grid_panel.face_hovered.connect(self._on_face_grid_hovered)
        self._face_grid_panel.face_context_menu_requested.connect(
            self._on_face_grid_context_menu_requested
        )
        self._face_grid_panel.face_delete_requested.connect(self._delete_face_at)
        self._frame_view.clicked_at.connect(self._on_frame_clicked)
        self._frame_view.face_move_requested.connect(self._on_face_move_requested)
        self._frame_view.face_resize_requested.connect(self._on_face_resize_requested)
        self._frame_view.face_add_requested.connect(self._on_face_add_requested)
        self._frame_view.face_context_menu_requested.connect(self._on_frame_context_menu_requested)
        self._frame_view.landmark_move_requested.connect(self._on_landmark_move_requested)
        self._frame_view.landmarks_move_requested.connect(self._on_landmarks_move_requested)
        self._frame_view.landmarks_select_requested.connect(self._on_landmarks_select_requested)
        self._face_panel.face_context_menu_requested.connect(
            self._on_face_panel_context_menu_requested
        )
        self._frame_view.install_editor_seams(
            active_face_provider=self._active_face_index,
            active_bbox_provider=self._active_face_bbox,
            add_mode_provider=self._is_add_mode_active,
            face_hit_provider=self._face_at_source_point,
        )
        self._frame_view.install_landmark_seams(
            landmark_mode_provider=self._is_landmark_mode_active,
            landmark_provider=self._active_face_landmarks,
            landmark_selection_provider=lambda: self._overlay.selected_landmarks,
        )
        self._frame_view.install_extract_seams(
            extract_mode_provider=self._is_extract_mode_active,
        )
        self._frame_view.install_mask_seams(
            mask_mode_provider=self._is_mask_mode_active,
            brush_provider=self._current_brush_spec,
        )
        self._frame_view.face_scale_requested.connect(self._on_face_scale_requested)
        self._frame_view.face_rotate_requested.connect(self._on_face_rotate_requested)
        self._frame_view.mask_paint_requested.connect(self._on_mask_paint_requested)
        self.dirty_changed.connect(
            lambda dirty: self.statusBar().showMessage(
                "Manual Tool has unsaved changes" if dirty else "Manual Tool ready",
                5000,
            )
        )


__all__ = ["LayoutMixin"]
