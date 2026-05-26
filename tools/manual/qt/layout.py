#!/usr/bin/env python3
"""Qt Manual Tool shell layout and widget composition helpers."""

from __future__ import annotations

import logging
import typing as T

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lib.gui.qt_shell.theme import QtTheme, icon_for_action

from .actions import MANUAL_ACTIONS
from .face_viewer.viewport import _FACE_GRID_SIZES

logger = logging.getLogger(__name__)


class LayoutMixin:
    """Own root-window widget layout construction."""

    _EDITOR_RAIL_ACTIONS: T.ClassVar[tuple[str, ...]] = (
        "set_view_mode",
        "set_boundingbox_mode",
        "set_extractbox_mode",
        "set_landmarks_mode",
        "set_mask_mode",
    )
    _FRAME_ACTION_RAIL_ACTIONS: T.ClassVar[tuple[str, ...]] = (
        "copy_prev_face",
        "copy_next_face",
        "revert_frame",
    )
    _MODE_RAIL_ACTIONS: T.ClassVar[dict[str, tuple[str, ...]]] = {
        "View": ("zoom_in", "zoom_out", "reset_view", "magnify_active_face"),
        "BoundingBox": ("add_face", "delete_face", "undo_edit", "redo_edit"),
        "ExtractBox": ("delete_face", "undo_edit", "redo_edit"),
        "Landmarks": ("undo_edit", "redo_edit"),
        "Mask": ("mask_draw", "mask_erase", "brush_size_decrease", "brush_size_increase"),
    }

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

        layout.addWidget(self._face_grid_panel, 1)
        return container

    def _build_ui(self) -> None:
        """Build Manual Tool widgets."""
        self.resize(1120, 720)
        self._build_toolbar()
        status = QStatusBar()
        status.setObjectName("qt-manual-status-bar")
        self.setStatusBar(status)

        editor_area = QWidget()
        editor_area.setObjectName("qt-manual-editor-area")
        editor_layout = QHBoxLayout(editor_area)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        editor_layout.addWidget(self._build_editor_rail())

        center = QWidget()
        center.setObjectName("qt-manual-frame-center")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        canvas = QWidget()
        canvas.setObjectName("qt-manual-frame-canvas")
        canvas.setStyleSheet(
            "QWidget#qt-manual-frame-canvas { background-color: #000000; border: 0; }"
            "QWidget#qt-manual-frame-view { background-color: #000000; border: 0; }"
        )
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        canvas_layout.addWidget(self._frame_view, 1)
        center_layout.addWidget(canvas, 1)
        center_layout.addWidget(self._transport_bar)
        center_layout.addWidget(self._build_transport_button_row())
        editor_layout.addWidget(center, 1)
        editor_layout.addWidget(self._build_context_panel())

        self._metadata_label.setParent(editor_area)
        self._metadata_label.setObjectName("qt-manual-session-metadata")
        self._metadata_label.setWordWrap(True)
        self._metadata_label.hide()
        self._thumbnail_panel.setObjectName("qt-manual-hidden-frame-thumbnail-panel")
        self._thumbnail_panel.hide()
        self._status_label.hide()
        self._status_label.setParent(editor_area)

        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("qt-manual-main-splitter")
        splitter.addWidget(editor_area)
        splitter.addWidget(self._build_face_grid_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 160])
        self._manual_splitter = splitter
        self.setCentralWidget(splitter)

    def _build_editor_rail(self) -> QWidget:
        """Return the legacy-style left vertical editor/action rail."""
        rail = QWidget()
        rail.setObjectName("qt-manual-editor-rail")
        rail.setFixedWidth(42)
        rail.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._rail_action_buttons: dict[str, QToolButton] = {}
        for action_key in self._EDITOR_RAIL_ACTIONS:
            layout.addWidget(self._tool_button_for_action(action_key))
        layout.addWidget(self._rail_separator())
        for action_key in self._FRAME_ACTION_RAIL_ACTIONS:
            layout.addWidget(self._tool_button_for_action(action_key))
        self._mode_rail_separator = self._rail_separator()
        layout.addWidget(self._mode_rail_separator)
        for action_key in sorted(
            {key for keys in self._MODE_RAIL_ACTIONS.values() for key in keys}
        ):
            button = self._tool_button_for_action(action_key)
            button.hide()
            layout.addWidget(button)
        layout.addStretch(1)
        self._sync_rail_mode_actions()
        return rail

    def _build_context_panel(self) -> QWidget:
        """Return the fixed right-side contextual control panel."""
        panel = QWidget()
        panel.setObjectName("qt-manual-context-panel")
        panel.setMinimumWidth(240)
        panel.setMaximumWidth(300)
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._mask_controls = self._build_mask_controls()
        self._mask_controls.hide()
        layout.addWidget(self._mask_controls)
        self._aligner_controls = self._build_aligner_controls()
        self._aligner_controls.hide()
        layout.addWidget(self._aligner_controls)
        layout.addStretch(1)
        return panel

    def _build_transport_button_row(self) -> QWidget:
        """Return the legacy second transport row with actions and filters."""
        row = QWidget()
        row.setObjectName("qt-manual-transport-button-row")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(4)

        for action_key in (
            "play_pause",
            "first_frame",
            "previous_frame",
            "next_frame",
            "last_frame",
        ):
            layout.addWidget(self._tool_button_for_action(action_key, parent=row))
        layout.addStretch(1)
        self._filter_controls = self._build_filter_controls()
        layout.addWidget(self._filter_controls)
        layout.addWidget(self._tool_button_for_action("extract_faces", parent=row))
        layout.addWidget(self._tool_button_for_action("save", parent=row))
        self._configure_face_size_combo()
        layout.addWidget(self._face_grid_size_combo)
        return row

    def _configure_face_size_combo(self) -> None:
        """Move the face-grid size selector into the transport row."""
        self._face_grid_size_combo.setObjectName("qt-manual-face-grid-size")
        if self._face_grid_size_combo.count() == 0:
            self._face_grid_size_combo.addItems(tuple(_FACE_GRID_SIZES))
            self._face_grid_size_combo.currentTextChanged.connect(self._on_face_grid_size_changed)
        size_name = self._editor_state.faces_size or "Medium"
        if size_name not in _FACE_GRID_SIZES:
            size_name = "Medium"
        self._face_grid_size_combo.setCurrentText(size_name)
        self._face_grid_panel.set_face_size(size_name)

    def _tool_button_for_action(
        self,
        action_key: str,
        *,
        parent: QWidget | None = None,
    ) -> QToolButton:
        """Create an icon-only button bound to an existing action."""
        action = self._actions[action_key]
        button = QToolButton(parent or self)
        button.setObjectName(f"qt-manual-rail-action-{action_key}")
        button.setDefaultAction(action)
        button.setAutoRaise(False)
        button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        button.setFixedSize(32, 32)
        button.setIconSize(QSize(16, 16))
        if action.icon().isNull():
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.setText(action.text()[:1])
        self._rail_action_buttons[action_key] = button
        return button

    @staticmethod
    def _rail_separator() -> QFrame:
        """Return a compact separator for the vertical rail."""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setFixedHeight(8)
        return separator

    def _sync_rail_mode_actions(self) -> None:
        """Show optional rail actions only for the active editor mode."""
        buttons = getattr(self, "_rail_action_buttons", {})
        active_keys = set(self._MODE_RAIL_ACTIONS.get(self._editor_state.editor_mode, ()))
        all_mode_keys = {key for keys in self._MODE_RAIL_ACTIONS.values() for key in keys}
        for key in all_mode_keys:
            button = buttons.get(key)
            if button is not None:
                button.setVisible(key in active_keys)
        separator = getattr(self, "_mode_rail_separator", None)
        if separator is not None:
            separator.setVisible(bool(active_keys))

    def _build_toolbar(self) -> None:
        """Build the Manual Tool action toolbar from :data:`MANUAL_ACTIONS`."""
        theme = QtTheme.default()
        toolbar = QToolBar("Manual Tool")
        self._manual_toolbar = toolbar
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
        toolbar.hide()
        self._sync_play_action_icon()

    def _hide_default_parity_panels(self) -> None:
        """Keep non-legacy default panels hidden after window-state restore."""
        toolbar = getattr(self, "_manual_toolbar", None)
        if toolbar is not None:
            toolbar.hide()
        self._metadata_label.hide()
        self._thumbnail_panel.hide()
        self._face_panel.hide()
        self._status_label.hide()

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
            landmark_faces_provider=self._frame_landmark_faces,
            landmark_hover_callback=self._overlay.set_hovered_landmark,
        )
        self._frame_view.install_extract_seams(
            extract_mode_provider=self._is_extract_mode_active,
        )
        self._frame_view.install_mask_seams(
            mask_mode_provider=self._is_mask_mode_active,
            brush_provider=self._current_brush_spec,
            mask_roi_provider=self._active_mask_roi_contains,
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
