#!/usr/bin/env python3
"""Qt Manual Tool core window state and constructor."""

from __future__ import annotations

import contextlib
import typing as T

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import QComboBox, QLabel, QMainWindow, QProgressBar, QSplitter, QToolButton, QWidget

from lib.gui.services.command_builder import CommandBuilder
from tools.manual.session import (
    ManualEditableAlignments,
    ManualEditorState,
    ManualFrame,
    ManualSession,
    ManualVideoMetadata,
)

from .face_viewer import CrossFrameFaceGridPanel
from .face_viewer.thumbnails import FaceThumbnailPanel, ManualThumbnailPanel
from .face_viewer.viewport import FaceGridThumbnailRenderer
from .frame_viewer.editor.bounding_box import BoundingBoxWindowEditorMixin
from .frame_viewer.editor.extract_box import ExtractBoxWindowEditorMixin
from .frame_viewer.editor.landmarks import LandmarkWindowEditorMixin
from .frame_viewer.editor.mask import MaskWindowEditorMixin
from .frame_viewer.frame_view import ManualFrameView
from .frame_viewer.overlays import ManualFrameOverlay
from .transport import ManualTransportBar
from .video import VideoFrameProvider
from .workers import (
    ManualAlignerLoadWorker,
    ManualExtractFacesWorker,
    ManualSaveWorker,
    ManualStartupWorker,
)


class ManualToolWindow(
    BoundingBoxWindowEditorMixin,
    ExtractBoxWindowEditorMixin,
    LandmarkWindowEditorMixin,
    MaskWindowEditorMixin,
    QMainWindow,
):
    """Qt-native Manual Tool window base with shared state and construction."""

    _OVERLAY_COLOR_DEFAULTS: T.ClassVar[dict[str, QColor]] = {
        "bbox": QColor("#0000ff"),
        "extract": QColor("#00ff00"),
        "landmark": QColor("#ff00ff"),
        "landmark_selected": QColor("#ff00ff"),
        "mesh": QColor("#00ffff"),
        "mask": QColor("#ff0000"),
    }

    dirty_changed = Signal(bool)
    frame_changed = Signal(int)
    action_triggered = Signal(str)

    def __init__(
        self,
        session: ManualSession,
        *,
        legacy_args: list[str] | None = None,
        parent: QWidget | None = None,
        console_logger: T.Callable[[str], None] | None = None,
        aligner_service: T.Any = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("qt-manual-tool-window")
        self.setWindowTitle("Faceswap Manual Tool")
        self._session = session
        self._editor_state = session.create_editor_state()
        self._editor_state.mask_opacity = 40
        if aligner_service is None:
            from tools.manual.aligner_service import ManualAlignerService

            aligner_service = ManualAlignerService(status_callback=self._on_aligner_status)
        self._aligner_service = aligner_service
        self._aligner_load_worker: ManualAlignerLoadWorker | None = None
        self._aligner_load_target: tuple[str, str] | None = None
        self._aligner_loaded_targets: set[tuple[str, str]] = set()
        self._video_metadata: ManualVideoMetadata | None = None
        self._video_provider: VideoFrameProvider | None = None
        self._video_frames: list[ManualFrame] = []
        self._legacy_args = legacy_args or []
        self._alignments_handle = session.alignments_handle()
        self._editable = ManualEditableAlignments()
        self._editable.subscribe(self._on_editable_changed)
        self._current_frame: ManualFrame | None = None
        self._frame_view = ManualFrameView()
        self._thumbnail_panel = ManualThumbnailPanel()
        self._face_panel = FaceThumbnailPanel()
        self._face_grid_renderer = FaceGridThumbnailRenderer(self._editable)
        self._face_grid_panel = CrossFrameFaceGridPanel(self._face_grid_renderer)
        self._face_grid_size_combo = QComboBox()
        self._manual_splitter: QSplitter | None = None
        self._status_label = QLabel()
        self._metadata_label = QLabel()
        self._magnify_restore_state: dict[str, object] | None = None
        self._overlay_color_overrides: dict[str, dict[str, QColor]] = {}
        self._transport_bar = ManualTransportBar()
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(int(1000 / 24))
        self._play_timer.setTimerType(Qt.PreciseTimer)
        self._play_timer.timeout.connect(self._advance_during_playback)
        self._filtered_frame_indices: tuple[int, ...] = ()
        self._progress_bar: QProgressBar | None = None
        self._console_logger = console_logger
        self._startup_worker: ManualStartupWorker | None = None
        self._startup_complete = False
        self._thumb_progress_seen = False
        self._extract_worker: ManualExtractFacesWorker | None = None
        self._extract_total: int = 0
        self._extract_cancel_button: QToolButton | None = None
        self._actions: dict[str, QAction] = {}
        self._busy_operation: str | None = None
        self._save_in_flight: bool = False
        self._save_worker: ManualSaveWorker | None = None
        self._save_busy_stack: contextlib.ExitStack | None = None
        self._pending_extract_folder: str | None = None
        self._overlay = ManualFrameOverlay(
            self._editable,
            frame_index_provider=self._current_frame_index,
        )
        self._overlay.install_mask_render_seam(
            mask_type_provider=self.active_mask_type,
            mask_opacity_provider=lambda: self._editor_state.mask_opacity,
            mask_show_provider=self._should_render_mask,
        )
        self._overlay.install_color_provider(self._overlay_color)
        self._overlay.install_visibility_providers(
            editor_mode_provider=lambda: self._editor_state.editor_mode,
            annotation_mode_provider=lambda: self._editor_state.annotation_mode,
        )
        self._editor_state.subscribe("unsaved", self._on_unsaved_changed)
        self._editor_state.subscribe("face_index", lambda _v: self._sync_actions())
        self._editor_state.subscribe(
            "face_index", lambda value: self._overlay.set_active(int(value))
        )
        self._editor_state.subscribe("face_index", lambda _v: self._frame_view.update())
        self._editor_state.subscribe("face_index", lambda _v: self._refresh_face_grid_active())
        self._editor_state.subscribe(
            "face_index", lambda _v: self._refresh_mask_controls_visibility()
        )
        self._editor_state.subscribe(
            "face_index", lambda _v: self._refresh_aligner_controls_visibility()
        )
        self._editor_state.subscribe("frame_index", lambda _v: self._sync_actions())
        self._editor_state.subscribe(
            "frame_index", lambda _v: self._refresh_mask_controls_visibility()
        )
        self._editor_state.subscribe("frame_index", lambda _v: self._refresh_face_grid_active())
        self._editor_state.subscribe(
            "frame_index", lambda _v: self._refresh_aligner_controls_visibility()
        )
        self._editor_state.subscribe("editor_mode", self._on_editor_mode_changed)
        self._editor_state.subscribe("filter_mode", lambda _v: self._refresh_filter_results())
        self._editor_state.subscribe("filter_distance", lambda _v: self._refresh_filter_results())
        self._editor_state.subscribe("faces_size", self._on_face_grid_size_state_changed)
        self._editor_state.subscribe("annotation_mode", lambda _v: self._refresh_face_grid())
        self._editor_state.subscribe("annotation_mode", lambda _v: self._frame_view.update())
        self._editor_state.subscribe("mask_type", lambda _v: self._refresh_face_grid())
        self._editor_state.subscribe("mask_opacity", lambda _v: self._refresh_face_grid())
        self._build_ui()
        self._restore_manual_window_state()
        self._connect_signals()
        self._load_session()
        self._frame_view.add_overlay(self._overlay)
        self._start_background_startup()

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

    @property
    def frame_view(self) -> ManualFrameView:
        """Return the embedded frame view widget."""
        return self._frame_view

    @property
    def face_panel(self) -> FaceThumbnailPanel:
        """Return the per-frame face thumbnail panel widget."""
        return self._face_panel

    @property
    def face_grid_panel(self) -> CrossFrameFaceGridPanel:
        """Return the filtered-session cross-frame face grid widget."""
        return self._face_grid_panel

    @property
    def editable_alignments(self) -> ManualEditableAlignments:
        """Return the GUI-neutral editable alignment model."""
        return self._editable

    @property
    def frame_overlay(self) -> ManualFrameOverlay:
        """Return the editable-alignments overlay painter."""
        return self._overlay

    @property
    def actions_by_key(self) -> dict[str, QAction]:
        """Return the registered Manual Tool actions keyed by action key."""
        return dict(self._actions)

    @property
    def video_provider(self) -> VideoFrameProvider | None:
        """Return the async video frame provider for video inputs."""
        return self._video_provider

    @classmethod
    def from_command_values(
        cls,
        values: T.Mapping[str, object],
        *,
        builder: CommandBuilder,
        parent: QWidget | None = None,
        console_logger: T.Callable[[str], None] | None = None,
    ) -> ManualToolWindow:
        """Create a Manual Tool window from command-panel values."""
        session = ManualSession.from_cli_values(values)
        legacy_args = builder.build("tools", "manual", values, generate=False)
        return cls(
            session,
            legacy_args=legacy_args,
            parent=parent,
            console_logger=console_logger,
        )


__all__ = ["ManualToolWindow"]