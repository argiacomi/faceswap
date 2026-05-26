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
        """Return the default bottom face-grid container."""
        container = QWidget()
        container.setObjectName("qt-manual-face-browser-panel")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._face_panel.setObjectName("qt-manual-current-frame-face-panel")
        self._face_panel.hide()

        rail = QWidget()
        rail.setObjectName("qt-manual-face-grid-mini-rail")
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.setSpacing(4)
        for button, text, tooltip in (
            (self._face_grid_mesh_button, "M", "Toggle thumbnail Mesh overlay (F9)"),
            (self._face_grid_mask_button, "K", "Toggle thumbnail Mask overlay (F10)"),
        ):
            button.setText(text)
            button.setCheckable(True)
            button.setAutoRaise(False)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.setToolTip(tooltip)
            button.setFixedSize(28, 28)
            rail_layout.addWidget(button)
        self._face_grid_mesh_button.setObjectName("qt-manual-face-grid-mesh-toggle")
        self._face_grid_mask_button.setObjectName("qt-manual-face-grid-mask-toggle")
        self._face_grid_mesh_button.toggled.connect(
            lambda checked: self._editor_state.set("face_grid_mesh_visible", bool(checked))
        )
        self._face_grid_mask_button.toggled.connect(
            lambda checked: self._editor_state.set("face_grid_mask_visible", bool(checked))
        )
        rail_layout.addStretch(1)
        layout.addWidget(rail)

        grid_box = QWidget()
        grid_layout = QVBoxLayout(grid_box)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(2)
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addStretch(1)
        size_label = QLabel("Size:")
        size_label.setObjectName("qt-manual-face-grid-size-label")
        controls.addWidget(size_label)
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
        layout.addWidget(grid_box, 1)
        return container

    def _build_ui(self) -> None:
        """Build Manual Tool widgets."""
        self.resize(980, 680)
        self._build_toolbar()
        status = QStatusBar()
        status.setObjectName("qt-manual-status-bar")
        self.setStatusBar(status)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)
        self._metadata_label.setObjectName("qt-manual-session-metadata")
        self._metadata_label.setWordWrap(True)
        self._metadata_label.hide()
        left_layout.addWidget(self._metadata_label)
        self._mask_controls = self._build_mask_controls()
        self._mask_controls.hide()
        left_layout.addWidget(self._mask_controls)
        self._aligner_controls = self._build_aligner_controls()
        self._aligner_controls.hide()
        left_layout.addWidget(self._aligner_controls)
        left_layout.addWidget(self._frame_view, 1)
        self._filter_controls = self._build_filter_controls()
        left_layout.addWidget(self._filter_controls)
        left_layout.addWidget(self._transport_bar)
        self._status_label.hide()
        left_layout.addWidget(self._status_label)

        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("qt-manual-main-splitter")
        splitter.addWidget(left)
        splitter.addWidget(self._build_face_grid_panel())
        self._thumbnail_panel.setObjectName("qt-manual-hidden-frame-thumbnail-panel")
        self._thumbnail_panel.hide()
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([540, 140])
        self._manual_splitter = splitter
        self.setCentralWidget(splitter)

    def _build_toolbar(self) -> None:
        """Build the Manual Tool action toolbar from :data:`MANUAL_ACTIONS`."""
        theme = QtTheme.default()
        toolbar = QToolBar("Manual Tool")
        toolbar.setObjectName("qt-manual-toolbar")
        toolbar.setIconSize(QSize(theme.icon_size, theme.icon_size))
        toolbar.hide()
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
            if spec.key in {"cycle_annotation", "toggle_mask_annotation"}:
                action.setCheckable(True)
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
        self._face_grid_panel.faces_delete_requested.connect(self._delete_faces_at)
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
