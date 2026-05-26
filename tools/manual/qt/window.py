#!/usr/bin/env python3
"""Root window composition for the Qt Manual Tool."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import typing as T

from PySide6.QtCore import (
    QByteArray,
    QPointF,
    QSettings,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QImage,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from lib.gui.qt_shell.theme import QtTheme, icon_for_action
from lib.gui.services.command_builder import CommandBuilder
from tools.manual.session import (
    FaceThumbnail,
    ManualEditableAlignments,
    ManualEditorState,
    ManualFrame,
    ManualSession,
    ManualVideoMetadata,
)

from .actions import MANUAL_ACTIONS
from .face_grid_renderer import _FACE_GRID_SIZES, FaceGridEntry, FaceGridThumbnailRenderer
from .face_viewer import CrossFrameFaceGridPanel
from .frame_viewer.editor.bounding_box import BoundingBoxWindowEditorMixin
from .frame_viewer.editor.extract_box import ExtractBoxWindowEditorMixin
from .frame_viewer.editor.landmarks import LandmarkWindowEditorMixin
from .frame_viewer.editor.mask import MaskWindowEditorMixin
from .frame_viewer.frame_view import ManualFrameView
from .overlays import ManualFrameOverlay
from .thumbnails import FaceThumbnailPanel, ManualThumbnailPanel
from .transport import ManualTransportBar
from .video import VideoFrameProvider
from .workers import (
    ManualAlignerLoadWorker,
    ManualExtractFacesWorker,
    ManualSaveWorker,
    ManualStartupWorker,
    _ManualStartupTask,
)

logger = logging.getLogger(__name__)


class ManualToolWindow(
    BoundingBoxWindowEditorMixin,
    ExtractBoxWindowEditorMixin,
    LandmarkWindowEditorMixin,
    MaskWindowEditorMixin,
    QMainWindow,
):
    """Qt-native Manual Tool window with legacy fallback support."""

    _SETTINGS_ORG = "Faceswap"
    _SETTINGS_APP = "QtManualTool"
    _WINDOW_STATE_KEY = "manual_tool/window_state"
    _OVERLAY_COLOR_DEFAULTS: T.ClassVar[dict[str, QColor]] = {
        "bbox": QColor("#3aa0ff"),
        "active": QColor("#ffb000"),
        "landmark": QColor("#ffffff"),
        "landmark_selected": QColor("#ffb000"),
        "mask": QColor(255, 80, 80),
    }

    dirty_changed = Signal(bool)
    frame_changed = Signal(int)
    action_triggered = Signal(str)
    """Emitted with the :class:`ManualAction.key` whenever an action fires.

    Editor-mode and stubbed editing actions (copy/revert/delete) connect
    follow-up tickets through this signal without needing direct method hooks.
    """

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
        # Aligner integration (#104) — accept an optional injected service so
        # tests don't need to construct the production plugin pipeline.  The
        # real path lazily imports the default service so importing this
        # module stays cheap.
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
        # One cached alignments handle for the lifetime of this window so we
        # are not reopening the alignments file on every face refresh.
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
        # Frame-navigation widgets — driven from the *filtered* frame list
        # (#107).  The transport position is an index into
        # ``_filtered_frame_indices``; the thumbnail panel still shows every
        # frame but navigation skips frames the active filter rejects.
        self._transport_bar = ManualTransportBar()
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(int(1000 / 24))
        self._play_timer.setTimerType(Qt.PreciseTimer)
        self._play_timer.timeout.connect(self._advance_during_playback)
        # Filtered model (#107).  Empty until the thumbnail panel is
        # populated; ``_refresh_filter_results`` keeps it in lock-step with
        # ``editor_state.filter_mode`` / ``filter_distance`` and any
        # editable-model change that affects face count or alignment.
        self._filtered_frame_indices: tuple[int, ...] = ()
        self._progress_bar: QProgressBar | None = None
        self._console_logger = console_logger
        self._startup_worker: ManualStartupWorker | None = None
        self._startup_complete = False
        # #121: once a per-event thumb progress signal lands the named
        # ``"thumbs"`` stage must NOT reset the bar back to 66% — the
        # named-stage emit still fires alongside ``progress_percent`` so
        # console + status messaging stay per-frame, but the progress bar
        # only follows the live percent from that point on.
        self._thumb_progress_seen = False
        # Extract Faces worker — held so cancel/close can stop it cleanly.
        self._extract_worker: ManualExtractFacesWorker | None = None
        self._extract_total: int = 0
        # Visible Cancel control next to the extract progress bar (issue #118).
        self._extract_cancel_button: QToolButton | None = None
        self._actions: dict[str, QAction] = {}
        # Save/bulk-operation lock state.  ``_busy_operation`` carries a short
        # user-facing label for the in-flight job ("Saving alignments…", etc.)
        # and is set/unset by :meth:`_with_busy_lock`.  Mutating actions are
        # disabled while this is non-empty.
        self._busy_operation: str | None = None
        self._save_in_flight: bool = False
        # Async save state — when set, the ExitStack holds the busy-lock open
        # until the worker's completed/failed signal fires, and ``_save_worker``
        # is the live :class:`ManualSaveWorker` whose completion drains the
        # stack and finalises the save (clears dirty / shows status / dialog).
        self._save_worker: ManualSaveWorker | None = None
        self._save_busy_stack: contextlib.ExitStack | None = None
        # When extract is triggered against a dirty session, persistence is
        # scheduled first; this holds the output folder until save completes.
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
        # When the active face changes, refresh the Mask dropdown so the
        # option list reflects whatever the new face has persisted on disk.
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
        # Keep the filter-controls label + threshold-slider visibility in
        # lock-step with editor-state changes.  ``filter_mode`` already
        # gets ``_refresh_filter_results`` from the cycle action, but
        # programmatic flips (tests, future menu items) need this too.
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
        """Return the GUI-neutral editable alignment model.

        Exposed so callers (tests, follow-up integration code) can seed it
        from a real alignments file or assert edit/undo behavior without
        scraping internal state.
        """
        return self._editable

    @property
    def frame_overlay(self) -> ManualFrameOverlay:
        """Return the editable-alignments overlay painter."""
        return self._overlay

    @property
    def actions_by_key(self) -> dict[str, QAction]:
        """Return the registered Manual Tool actions keyed by :class:`ManualAction.key`.

        Exposed so that follow-up integrations and tests can drive specific
        actions (and assert their enabled state) without scraping the toolbar.
        """
        return dict(self._actions)

    def set_editor_overlay_color(self, editor_mode: str, role: str, color: str | QColor) -> None:
        """Override one overlay color role for one editor mode (#124)."""
        if role not in self._OVERLAY_COLOR_DEFAULTS:
            raise ValueError(f"Unknown overlay color role: {role}")
        qcolor = color if isinstance(color, QColor) else QColor(str(color))
        if not qcolor.isValid():
            raise ValueError(f"Invalid overlay color: {color}")
        self._overlay_color_overrides.setdefault(str(editor_mode), {})[role] = QColor(qcolor)
        self._frame_view.update()

    def editor_overlay_color(self, editor_mode: str, role: str) -> QColor:
        """Return the configured overlay color for ``editor_mode`` and ``role``."""
        override = self._overlay_color_overrides.get(str(editor_mode), {}).get(role)
        if override is not None:
            return QColor(override)
        return QColor(self._OVERLAY_COLOR_DEFAULTS.get(role, QColor("#3aa0ff")))

    def _overlay_color(self, role: str) -> QColor:
        """Return overlay color for the current editor mode."""
        return self.editor_overlay_color(self._editor_state.editor_mode, role)

    @property
    def video_provider(self) -> VideoFrameProvider | None:
        """Return the async video frame provider for video inputs (or ``None``)."""
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

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa:N802
        """Prompt before closing when the editor has unsaved changes."""
        if self._editor_state.unsaved:
            answer = QMessageBox.question(
                self,
                "Unsaved Manual Tool Changes",
                "Close the Manual Tool and discard unsaved changes?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._extract_worker is not None:
            # ``stop()`` returns True iff the worker thread actually
            # exited within the wait window.  If extraction is stuck in
            # a long per-frame op the thread keeps running — we must NOT
            # drop the reference (Python would then GC the QThread mid-
            # flight and SIGABRT in Qt) and we must NOT consume the
            # close event (so the user can try again once it does
            # finish).  See #119 task 2.
            stopped = self._extract_worker.stop()
            if not stopped:
                self.statusBar().showMessage(
                    "Extraction still running — please wait for it to finish, then close again.",
                    7000,
                )
                event.ignore()
                return
            self._extract_worker = None
        if self._save_worker is not None:
            # The save worker is uncancellable but short-lived; stop() blocks
            # only until the in-flight persist returns.  Drain the busy-lock
            # so close doesn't leak the disabled-actions / progress-bar state.
            self._save_worker.stop()
            self._save_worker = None
            self._drain_save_busy_stack()
        if self._aligner_load_worker is not None:
            stopped = self._aligner_load_worker.stop()
            if not stopped:
                self.statusBar().showMessage(
                    "Aligner is still loading — please wait, then close again.",
                    7000,
                )
                event.ignore()
                return
            self._aligner_load_worker = None
            self._aligner_load_target = None
        if self._video_provider is not None:
            self._video_provider.shutdown()
            self._video_provider = None
        if self._startup_worker is not None:
            self._startup_worker.stop()
            self._startup_worker = None
        self._save_manual_window_state()
        event.accept()

    def _settings(self) -> QSettings:
        """Return the persistent settings store for Manual Tool UI polish."""
        return QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)

    def _capture_manual_window_state(self) -> dict[str, object]:
        """Capture geometry, window state and Manual Tool splitter sizes."""
        geometry = bytes(self.saveGeometry().toBase64()).decode("ascii")
        window_state = bytes(self.saveState().toBase64()).decode("ascii")
        return {
            "geometry": geometry,
            "window_state": window_state,
            "maximized": self.isMaximized(),
            "fullscreen": self.isFullScreen(),
            "splitter_sizes": self._manual_splitter.sizes() if self._manual_splitter else [],
        }

    def _restore_manual_window_state_from(self, state: T.Mapping[str, object]) -> bool:
        """Restore a state payload captured by :meth:`_capture_manual_window_state`."""
        restored = False
        geometry = state.get("geometry")
        if isinstance(geometry, str) and geometry:
            restored = bool(self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii"))))
        window_state = state.get("window_state")
        if isinstance(window_state, str) and window_state:
            self.restoreState(QByteArray.fromBase64(window_state.encode("ascii")))
        sizes = state.get("splitter_sizes")
        if self._manual_splitter is not None and isinstance(sizes, list | tuple):
            try:
                int_sizes = [int(value) for value in sizes]
            except (TypeError, ValueError):
                int_sizes = []
            if int_sizes:
                self._manual_splitter.setSizes(int_sizes)
                restored = True
        if bool(state.get("fullscreen")):
            self.showFullScreen()
            restored = True
        elif bool(state.get("maximized")):
            self.showMaximized()
            restored = True
        return restored

    def _save_manual_window_state(self) -> None:
        """Persist Manual Tool window state."""
        settings = self._settings()
        settings.setValue(self._WINDOW_STATE_KEY, json.dumps(self._capture_manual_window_state()))
        settings.sync()

    def _restore_manual_window_state(self) -> bool:
        """Restore persisted Manual Tool window state, if present."""
        raw = self._settings().value(self._WINDOW_STATE_KEY, "")
        if not isinstance(raw, str) or not raw:
            return False
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return False
        if not isinstance(state, dict):
            return False
        return self._restore_manual_window_state_from(state)

    def mark_dirty(self, dirty: bool = True) -> None:
        """Set dirty state and update action availability."""
        self._editor_state.set("unsaved", dirty)

    def save(self) -> bool:
        """Schedule a non-blocking save of editable edits.

        Persistence runs on :class:`ManualSaveWorker` so the busy-lock
        progress bar and disabled mutating actions can repaint *before*
        ``Alignments.save`` blocks (issue #115). The host event loop is
        flushed once after the busy state is installed so the user sees
        the busy feedback no matter how brief the persist actually is.

        Returns:

        * ``True`` when a save was scheduled (or the no-op fast path was
          taken because there is nothing to persist).
        * ``False`` when a save is already in flight — duplicate ``Ctrl+S``
          presses cannot race the persist call.

        Completion arrives through :meth:`_on_save_completed` or
        :meth:`_on_save_failed`; those handlers drain the busy-lock,
        update dirty state and post the status message / dialog.
        """
        if self._save_in_flight:
            self.statusBar().showMessage("Save already in progress…", 3000)
            return False
        if self._current_frame_index() < 0 and not any(
            self._editable.face_count(i) for i in range(self._session.frame_count)
        ):
            self.statusBar().showMessage("Nothing to save", 3000)
            return True

        # Install the busy-lock through an ExitStack so the completion handler
        # can drain it asynchronously.  This keeps the visible busy state,
        # disabled-mutating-actions, and progress-bar lifecycle identical to
        # the old synchronous path while moving persistence onto a thread.
        #
        # Importantly we do NOT call ``QCoreApplication.processEvents`` here:
        # the busy lock has already installed the progress bar + disabled
        # mutating actions by the time we return to the caller, and the
        # worker's persist call runs on its own QThread — so the OS event
        # loop naturally paints the busy state on its next tick.  Calling
        # ``processEvents`` mid-schedule would let unrelated workers (e.g.
        # the startup thumbnail-cache scan) fire their completion and tear
        # the progress bar back down from underneath us.
        self._save_busy_stack = contextlib.ExitStack()
        self._save_busy_stack.enter_context(self._with_busy_lock("Saving alignments…", save=True))

        try:
            frame_names = self._frame_names_for_persist()
            worker = ManualSaveWorker(
                self._alignments_handle, self._editable, frame_names, parent=self
            )
        except Exception as err:  # noqa: BLE001 - schedule-time failure
            logger.exception("Manual Tool save: failed to schedule worker")
            self._drain_save_busy_stack()
            self.statusBar().showMessage(f"Manual Tool save failed: {err}", 7000)
            QMessageBox.critical(self, "Manual Tool Save", f"Manual Tool save failed: {err}")
            return False
        worker.completed.connect(self._on_save_completed)
        worker.failed.connect(self._on_save_failed)
        self._save_worker = worker
        worker.start()
        return True

    def _on_save_completed(self, modified: int) -> None:
        """Worker reported success — finalise dirty state and refresh views."""
        self._drain_save_busy_stack()
        self._teardown_save_worker()
        self._editable.clear_history()
        self.mark_dirty(False)
        self._editor_state.set("edited", False)
        self._editor_state.set("face_count_changed", False)
        self.refresh_faces()
        self._frame_view.update()
        self.statusBar().showMessage(f"Saved {modified} frame(s) to alignments file", 5000)
        self._emit_console(f"Manual Tool: saved {modified} frame(s)")
        # A pending extract was waiting for this save — resume now.
        self._resume_extract_after_save()

    def _on_save_failed(self, message: str) -> None:
        """Worker reported failure — leave dirty state intact, surface error."""
        self._drain_save_busy_stack()
        self._teardown_save_worker()
        full = f"Manual Tool save failed: {message}"
        logger.error(full)
        self.statusBar().showMessage(full, 7000)
        self._emit_console(full)
        QMessageBox.critical(self, "Manual Tool Save", full)
        # Abort any pending extract — extracting against a broken save would
        # write stale or partial output.
        self._pending_extract_folder = None

    def _drain_save_busy_stack(self) -> None:
        """Release the busy-lock context held for the in-flight save."""
        stack = self._save_busy_stack
        self._save_busy_stack = None
        if stack is not None:
            stack.close()

    def _teardown_save_worker(self) -> None:
        """Stop + drop the save worker, joining its QThread."""
        if self._save_worker is not None:
            self._save_worker.stop()
            self._save_worker.deleteLater()
            self._save_worker = None

    @contextlib.contextmanager
    def _with_busy_lock(self, label: str, *, save: bool = False) -> T.Iterator[None]:
        """Run a blocking operation with progress + action gating.

        While the block is active:

        * ``_busy_operation`` carries ``label`` so :meth:`_sync_actions` knows
          to disable mutating actions and prevent conflicting edits.
        * The status-bar progress bar is shown in indeterminate mode with
          ``label`` as the format string.
        * ``processEvents`` runs once so the busy state paints before the
          (potentially blocking) work begins.

        Always restores prior state in a ``finally``, including when an
        exception escapes — the caller is free to ``return``/``raise`` inside
        the ``with`` block.
        """
        prior_progress_format = None
        prior_progress_range: tuple[int, int] | None = None
        prior_progress_visible = False
        owns_progress_bar = False
        self._busy_operation = label
        if save:
            self._save_in_flight = True
        # The startup worker tears down ``_progress_bar`` once startup is
        # finished; for save/bulk ops we re-materialize it for the duration
        # of the operation so the user always gets visible busy feedback.
        if self._progress_bar is None:
            self._progress_bar = self._build_progress_bar()
            self.statusBar().addPermanentWidget(self._progress_bar)
            owns_progress_bar = True
        if self._progress_bar is not None:
            prior_progress_format = self._progress_bar.format()
            prior_progress_range = (
                self._progress_bar.minimum(),
                self._progress_bar.maximum(),
            )
            prior_progress_visible = self._progress_bar.isVisible()
            self._progress_bar.setRange(0, 0)  # indeterminate
            self._progress_bar.setFormat(label)
            self._progress_bar.show()
        self._sync_actions()
        self.statusBar().showMessage(label, 3000)
        # Intentionally do NOT call ``processEvents`` here — flushing the event
        # loop inside the lock can let unrelated workers (eg. the startup
        # thumbnail-cache scan) complete and tear down the progress bar we
        # just materialized, which then disappears from underneath the persist
        # callback.  The lock is already entered before any blocking work
        # begins, so the paint will happen on the next natural event-loop tick.
        try:
            yield
        finally:
            self._busy_operation = None
            if save:
                self._save_in_flight = False
            if self._progress_bar is not None:
                if owns_progress_bar:
                    self._hide_progress_bar()
                else:
                    if prior_progress_range is not None:
                        self._progress_bar.setRange(*prior_progress_range)
                    if prior_progress_format is not None:
                        self._progress_bar.setFormat(prior_progress_format)
                    if not prior_progress_visible:
                        self._progress_bar.hide()
            self._sync_actions()

    def _frame_name_for_index(self, frame_index: int) -> str | None:
        """Resolve one editable frame_index to its on-disk frame name.

        Mirrors :meth:`_frame_names_for_persist` but for a single index — used
        when refreshing the face panel where we just need the current frame's
        name for thumbnail lookup, not the whole mapping.
        """
        mapping = self._frame_names_for_persist()
        if callable(mapping):
            return mapping(frame_index)
        if 0 <= frame_index < len(mapping):
            return mapping[frame_index]
        return None

    def _frame_names_for_persist(
        self,
    ) -> list[str] | T.Callable[[int], str | None]:
        """Return a frame-name mapping ordered to match the editable model.

        Image-folder sessions return the source frame list (sparse alignments
        still align to the source ordering); video sessions return a callable
        that synthesizes the Faceswap-standard dummy frame name on demand, so
        edits on frames the alignments file has *never* seen before (the
        common case for fresh video input) still resolve to a writable name.
        Falls back to the alignments file's existing keys when neither source
        is available.
        """
        if self._session.has_images:
            return [frame.name for frame in self._session.frame_list]
        if self._session.is_video_input:
            return self._session.frame_name_for_index
        return list(self._alignments_handle.sorted_frame_names())

    def extract_faces(self) -> bool:
        """Prompt for an output folder and run Tk-parity Extract Faces.

        Returns ``True`` when the extraction was scheduled, ``False`` when
        the user cancelled the folder picker or there is nothing to extract.
        Mirrors the Tk Extract button: aligned 512-px PNGs per face, named
        ``{source_stem}_{face_index}.png``, with alignments + source
        metadata embedded in the PNG header.
        """
        if self._extract_worker is not None:
            self.statusBar().showMessage("Extraction already in progress…", 4000)
            return False
        # Extract the live editable state — unsaved adds/deletes/moves count.
        # Persisting bootstraps the alignments file on demand, so we
        # also persist any in-flight edits first to keep alignments.version
        # and source metadata accurate inside the PNG headers.
        if not self._editable_has_any_face():
            self.statusBar().showMessage(
                "Nothing to extract — no faces in the editable model", 4000
            )
            return False
        initial_dir = self._initial_extract_dir()
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select output folder for extracted faces",
            initial_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not chosen:
            return False
        self._start_extract_worker(chosen)
        return True

    def _editable_has_any_face(self) -> bool:
        """Return whether the live editable model has any face anywhere."""
        return any(
            self._editable.face_count(index) > 0 for index in self._editable_frame_indices()
        )

    def _editable_frame_indices(self) -> tuple[int, ...]:
        """Return every frame index that currently has editable state.

        Covers both ``has_images`` (where indices match the discovered frame
        list) and the video case (where frame_index is sparse) by combining
        the discovered range with whatever indices the editable model knows
        about.  The set is union-deduped and returned sorted.
        """
        known = set(self._editable.known_frame_indices())
        if self._session.has_images:
            known.update(range(len(self._session.frame_list)))
        return tuple(sorted(known))

    def _build_editable_extract_targets(self) -> tuple[tuple[str, tuple[T.Any, ...]], ...]:
        """Materialize the live editable model as ``EditableFaceSpec`` targets.

        Iterates every frame the editable model tracks, resolves each to a
        frame name through the same mapping ``save`` uses, and copies
        bbox + landmarks from the editable face.  ``mask`` / ``identity`` /
        ``metadata`` are sourced from the persisted alignments entry at the
        same ``face_index`` when available (so identity vectors and mask
        payloads survive an extract on a face the user has merely moved);
        if no persisted entry exists, they're empty — matching how Tk
        Manual treats a brand-new face.
        """
        import numpy as np

        from tools.manual.face_extraction import EditableFaceSpec

        resolver = self._frame_names_for_persist()
        try:
            alignments = self._alignments_handle.open()
            persisted_by_name = {
                name: tuple(entry.faces) for name, entry in alignments.data.items()
            }
        except Exception:  # noqa: BLE001 - extract still works against unsaved-only sessions
            persisted_by_name = {}

        targets: list[tuple[str, tuple[EditableFaceSpec, ...]]] = []
        for frame_index in self._editable_frame_indices():
            faces = self._editable.faces(frame_index)
            if not faces:
                continue
            if callable(resolver):
                frame_name = resolver(frame_index)
            elif 0 <= frame_index < len(resolver):
                frame_name = resolver[frame_index]
            else:
                frame_name = None
            if not frame_name:
                continue
            persisted = persisted_by_name.get(frame_name, ())
            specs: list[EditableFaceSpec] = []
            for face in faces:
                landmarks = (
                    np.asarray(face.landmarks, dtype=np.float32)
                    if face.landmarks
                    else np.zeros((0, 2), dtype=np.float32)
                )
                prev = persisted[face.face_index] if face.face_index < len(persisted) else None
                x, y, w, h = (int(round(value)) for value in face.bbox)
                specs.append(
                    EditableFaceSpec(
                        face_index=face.face_index,
                        bbox=(x, y, w, h),
                        landmarks_xy=landmarks,
                        mask=dict(getattr(prev, "mask", {}) or {}),
                        identity=dict(getattr(prev, "identity", {}) or {}),
                        metadata=dict(getattr(prev, "metadata", {}) or {}),
                    )
                )
            targets.append((frame_name, tuple(specs)))
        return tuple(targets)

    def _initial_extract_dir(self) -> str:
        """Default the folder picker to a sibling of the input source."""
        if self._session.is_video_input:
            return os.path.dirname(self._session.frames)
        return self._session.frames

    def _start_extract_worker(self, output_folder: str) -> None:
        """Spin up :class:`ManualExtractFacesWorker` and wire its signals.

        Persists in-flight edits first (which bootstraps the alignments file
        if it doesn't exist yet so ``alignments.version`` is well-defined
        inside the PNG header), then builds an editable snapshot to drive
        the extraction.  ``extract_faces`` iterates the snapshot directly
        rather than re-reading ``alignments.data``, so unsaved adds,
        deletes, moves, and resizes all appear in the extracted output —
        matching Tk Manual which extracts its live in-memory face list.

        Save is now asynchronous (#115), so when the session is dirty we
        chain the actual extract launch onto the save worker's
        ``completed`` signal — the user sees the "Saving alignments…" busy
        state, then "Extracting faces…" without either being clobbered.
        """
        if self._editor_state.unsaved:
            # Persist on the worker thread, then chain the extract launch off
            # the save completion.  ``_pending_extract_folder`` is consumed by
            # :meth:`_resume_extract_after_save` so the same code path runs
            # whether save was sync (legacy) or async.
            self._pending_extract_folder = output_folder
            self.save()
            return
        self._launch_extract_worker(output_folder)

    def _resume_extract_after_save(self) -> None:
        """Continue a pending extract once a chained save completes."""
        folder = self._pending_extract_folder
        self._pending_extract_folder = None
        if folder is None:
            return
        self._launch_extract_worker(folder)

    def _launch_extract_worker(self, output_folder: str) -> None:
        """Materialize the editable snapshot and start the extract worker."""
        editable_targets = self._build_editable_extract_targets()
        worker = ManualExtractFacesWorker(
            self._alignments_handle,
            self._session,
            output_folder,
            editable_targets=editable_targets,
            parent=self,
        )
        worker.progress.connect(self._on_extract_progress)
        worker.completed.connect(self._on_extract_completed)
        worker.failed.connect(self._on_extract_failed)
        self._extract_worker = worker
        self._extract_total = 0
        if self._progress_bar is None:
            self._progress_bar = self._build_progress_bar()
            self.statusBar().addPermanentWidget(self._progress_bar)
        if self._extract_cancel_button is None:
            self._extract_cancel_button = self._build_extract_cancel_button()
            self.statusBar().addPermanentWidget(self._extract_cancel_button)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat("Extracting faces…")
        self._progress_bar.show()
        self._extract_cancel_button.setEnabled(True)
        self._extract_cancel_button.show()
        self.statusBar().showMessage(f"Extracting faces to {output_folder}", 5000)
        self._busy_operation = "Extracting faces…"
        self._sync_actions()
        logger.info("Manual Tool extract started → %s", output_folder)
        self._emit_console(f"Manual Tool: extracting faces to {output_folder}")
        worker.start()

    def _on_extract_progress(self, done: int, total: int, message: str) -> None:
        """Update the determinate progress bar from the extract worker."""
        if total > 0 and self._extract_total != total:
            self._extract_total = total
            if self._progress_bar is not None:
                self._progress_bar.setRange(0, total)
                self._progress_bar.setFormat("Extracting %v / %m frame(s)")
        if self._progress_bar is not None and total > 0:
            self._progress_bar.setValue(min(done, total))
        self.statusBar().showMessage(message, 2000)

    def _on_extract_completed(self, result: object) -> None:
        """Surface a summary message and tear down the extract worker."""
        from tools.manual.face_extraction import ExtractFacesResult

        self._teardown_extract_worker()
        if not isinstance(result, ExtractFacesResult):  # pragma: no cover - defensive
            return
        if result.cancelled:
            self.statusBar().showMessage("Extract cancelled", 5000)
            return
        summary = (
            f"Extracted {result.faces_written} face(s) from {result.frames_processed} frame(s)"
        )
        skipped_bits: list[str] = []
        if result.skipped_frames:
            skipped_bits.append(f"{result.skipped_frames} frame(s)")
        if result.skipped_faces:
            skipped_bits.append(f"{result.skipped_faces} face(s)")
        if skipped_bits:
            summary += f" — skipped {', '.join(skipped_bits)}"
        self.statusBar().showMessage(summary, 7000)
        self._emit_console(f"Manual Tool: {summary}")
        logger.info("Manual Tool extract completed: %s", summary)
        if result.errors:
            joined_errors = chr(10).join(result.errors[:10])
            QMessageBox.warning(
                self,
                "Extract Faces",
                f"{summary}{chr(10)}{chr(10)}{joined_errors}",
            )

    def _on_extract_failed(self, message: str) -> None:
        """Report worker-level failure and tear down the extract worker."""
        self._teardown_extract_worker()
        full = f"Extract faces failed: {message}"
        logger.error(full)
        self._emit_console(full)
        self.statusBar().showMessage(full, 7000)
        QMessageBox.critical(self, "Extract Faces", full)

    def _teardown_extract_worker(self) -> None:
        """Hide progress + Cancel button, clear busy state, join the worker thread."""
        if self._progress_bar is not None:
            self._progress_bar.hide()
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setFormat("Loading %p%")
        if self._extract_cancel_button is not None:
            self._extract_cancel_button.hide()
            self._extract_cancel_button.setEnabled(False)
        self._busy_operation = None
        # ``_teardown_extract_worker`` fires from completed/failed handlers;
        # the underlying QThread has already received ``quit()`` via signal
        # connections, so ``stop()`` is just the join.  Still respect the
        # bool contract (#119 task 2): if the thread refuses to exit, keep
        # the reference alive so the QThread destructor doesn't fire on a
        # live thread.
        if self._extract_worker is not None and self._extract_worker.stop():
            self._extract_worker.deleteLater()
            self._extract_worker = None
        self._extract_total = 0
        self._sync_actions()

    def _build_extract_cancel_button(self) -> QToolButton:
        """Return a status-bar Cancel button bound to :meth:`cancel_extract`."""
        button = QToolButton(self)
        button.setObjectName("qt-manual-extract-cancel")
        button.setText("Cancel")
        button.setToolTip("Cancel Extract Faces (stops at the next frame boundary)")
        button.setAutoRaise(False)
        button.setEnabled(False)
        button.hide()
        button.clicked.connect(self.cancel_extract)
        return button

    def cancel_extract(self) -> bool:
        """Request an in-flight Extract Faces job to stop at the next frame.

        Returns ``True`` when a worker was actively cancelled, ``False`` when
        no extraction was running.  The Cancel button is disabled immediately
        so a double-click cannot fire the worker cancel path twice; the
        teardown later hides + re-disables it.
        """
        worker = self._extract_worker
        if worker is None:
            return False
        worker.cancel()
        if self._extract_cancel_button is not None:
            self._extract_cancel_button.setEnabled(False)
        self.statusBar().showMessage("Cancelling extract — stopping at next frame…", 4000)
        self._emit_console("Manual Tool: extract cancellation requested")
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

    # ---- #107 filtered navigation primitives ----

    def _all_frame_indices(self) -> tuple[int, ...]:
        """Return every known source-frame index (image folder or video).

        The thumbnail panel's row count is the authoritative count once a
        session is loaded — image-folder rows are ``frame.index`` (0..N-1)
        and video rows are likewise consecutive indices.  Used as the
        universe the active filter draws from.
        """
        return tuple(range(self._thumbnail_panel.count()))

    def _refresh_filter_results(self, *, preserve_current: bool = True) -> None:
        """Recompute ``_filtered_frame_indices`` from the active filter.

        Also keeps the transport bar's total in sync and clamps the current
        thumbnail row to a matching frame when ``preserve_current`` is
        ``True`` (the default).  When the current frame still matches the
        filter, it stays put — otherwise the first matching frame is
        selected.  An empty filter leaves the panel untouched and reports
        an empty filtered list so navigation actions disable cleanly.
        """
        from tools.manual.frame_filter import (
            DEFAULT_FILTER_MODE,
            filtered_frame_indices,
            misaligned_predicate_for_model,
        )

        mode = self._editor_state.filter_mode or DEFAULT_FILTER_MODE
        threshold = int(self._editor_state.filter_distance)
        predicate = misaligned_predicate_for_model(self._editable, threshold)
        self._filtered_frame_indices = filtered_frame_indices(
            self._all_frame_indices(),
            self._editable.face_count,
            mode,
            misaligned_predicate=predicate,
        )
        total = len(self._filtered_frame_indices)
        self._transport_bar.set_total(total)
        if total == 0:
            # Empty filter — no transport position to clamp.  Leave the
            # thumbnail panel as-is so the user still sees the source
            # frames; navigation actions disable via ``_sync_actions``.
            self._sync_actions()
            self._refresh_filter_controls()
            self._refresh_face_grid()
            return
        current_row = self._thumbnail_panel.currentRow()
        if preserve_current and current_row in self._filtered_frame_indices:
            position = self._filtered_frame_indices.index(current_row)
            self._transport_bar.set_position(position)
        else:
            new_row = self._filtered_frame_indices[0]
            self._thumbnail_panel.setCurrentRow(new_row)
        self._sync_actions()
        self._refresh_filter_controls()
        self._refresh_face_grid()

    def filtered_frame_indices(self) -> tuple[int, ...]:
        """Return the current filtered frame index list.

        Public read-only accessor — :class:`ManualFaceGrid` (the future
        cross-frame viewer landing in #108) consumes this so its visible
        set matches the active filter.
        """
        return self._filtered_frame_indices

    def _filtered_position(self) -> int:
        """Return the current frame's index in ``_filtered_frame_indices``.

        Returns ``-1`` when the current frame isn't in the filtered list
        (filter rejected it) or no frame is loaded.
        """
        row = self._thumbnail_panel.currentRow()
        if row < 0:
            return -1
        try:
            return self._filtered_frame_indices.index(row)
        except ValueError:
            return -1

    def goto_first_frame(self) -> None:
        """Select the first frame in the active filter (#107)."""
        if not self._filtered_frame_indices:
            self.statusBar().showMessage(self._no_filter_match_message(), 3000)
            return
        self._stop_playback()
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[0])

    def goto_last_frame(self) -> None:
        """Select the last frame in the active filter (#107)."""
        if not self._filtered_frame_indices:
            self.statusBar().showMessage(self._no_filter_match_message(), 3000)
            return
        self._stop_playback()
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[-1])

    def toggle_play(self) -> None:
        """Toggle playback through the *filtered* frames (#107)."""
        if self._editor_state.is_playing:
            self._stop_playback()
            return
        if not self._filtered_frame_indices:
            self.statusBar().showMessage(self._no_filter_match_message(), 3000)
            return
        position = self._filtered_position()
        if position < 0 or position >= len(self._filtered_frame_indices) - 1:
            # Rewind to the first filtered frame so Play does something
            # visible instead of being an immediate no-op.
            self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[0])
        self._editor_state.set("is_playing", True)
        self._play_timer.start()
        self._sync_play_action_icon()

    def _stop_playback(self) -> None:
        """Halt the auto-advance timer and reset the play-action icon."""
        if self._play_timer.isActive():
            self._play_timer.stop()
        if self._editor_state.is_playing:
            self._editor_state.set("is_playing", False)
        self._sync_play_action_icon()

    def _advance_during_playback(self) -> None:
        """QTimer slot: step forward through the filtered list (#107)."""
        if not self._filtered_frame_indices:
            self._stop_playback()
            return
        position = self._filtered_position()
        if position < 0 or position >= len(self._filtered_frame_indices) - 1:
            self._stop_playback()
            return
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[position + 1])

    def _no_filter_match_message(self) -> str:
        """Compose the empty-filter user message."""
        mode = self._editor_state.filter_mode or "All Frames"
        return f"No frames match filter: {mode}"

    def _sync_play_action_icon(self) -> None:
        """Switch the Play/Pause toolbar icon to match playback state."""
        action = self._actions.get("play_pause")
        if action is None:
            return
        theme = QtTheme.default()
        icon_key = "pause" if self._editor_state.is_playing else "play"
        icon = icon_for_action(theme, icon_key)
        if not icon.isNull():
            action.setIcon(icon)
        text = "Pause" if self._editor_state.is_playing else "Play"
        tooltip = (
            "Pause playback (Space)" if self._editor_state.is_playing else "Play playback (Space)"
        )
        action.setText(text)
        action.setToolTip(tooltip)
        action.setStatusTip(tooltip)

    def _on_thumbnail_row_changed(self, _row: int) -> None:
        """Refresh action availability + keep the transport bar in sync.

        The transport bar tracks the *filtered* position so its counter
        and slider reflect the active filter.  When the current frame
        isn't in the filtered list (e.g. the user is paused on a frame
        the filter would normally skip), the slider stays at the closest
        previously-painted position so it doesn't jitter.
        """
        self._sync_actions()
        if not self._filtered_frame_indices:
            return
        row = self._thumbnail_panel.currentRow()
        try:
            position = self._filtered_frame_indices.index(row)
        except ValueError:
            return
        self._transport_bar.set_position(position)

    def _on_transport_position_changed(self, position: int) -> None:
        """Apply a user-driven slider / jump-entry change (filtered position).

        ``position`` is an index into ``_filtered_frame_indices``; the
        actual thumbnail row is the frame index stored at that filtered
        position.  Out-of-range positions are clamped silently.
        """
        if not self._filtered_frame_indices:
            return
        if not 0 <= position < len(self._filtered_frame_indices):
            return
        target_row = self._filtered_frame_indices[position]
        if target_row == self._thumbnail_panel.currentRow():
            return
        self._stop_playback()
        self._thumbnail_panel.setCurrentRow(target_row)

    def revert_current_frame(self) -> None:
        """Revert editable edits recorded against the current frame only.

        Edits on other frames stay on the undo stack so unrelated work is
        not silently discarded.  Dirty state is recomputed from the
        remaining undo records rather than blanket-cleared.
        """
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("Nothing to revert", 3000)
            return
        reverted = self._editable.revert_frame(frame_index)
        self.mark_dirty(self._editable.can_undo)
        if not self._editable.can_undo:
            self._editor_state.set("edited", False)
        if reverted:
            self.statusBar().showMessage(f"Reverted {reverted} edit(s) in this frame", 5000)
        else:
            self.statusBar().showMessage("Nothing to revert in this frame", 3000)

    def copy_prev_face(self) -> bool:
        """Copy the editable faces from the previous frame onto the current one."""
        return self._copy_faces_from(self._current_frame_index() - 1, "previous")

    def copy_next_face(self) -> bool:
        """Copy the editable faces from the next frame onto the current one."""
        return self._copy_faces_from(self._current_frame_index() + 1, "next")

    def _copy_faces_from(self, source_frame_index: int, direction: str) -> bool:
        """Replace current frame faces with the source frame's editable faces.

        Operates entirely on :class:`ManualEditableAlignments` so each
        deletion/addition lands on the shared undo stack and the overlay
        repaints itself via the existing listener.  Returns ``False`` and
        surfaces a status message when there is nothing to copy.
        """
        current_index = self._current_frame_index()
        if current_index < 0:
            return False
        source_faces = self._editable.faces(source_frame_index)
        if not source_faces:
            self.statusBar().showMessage(f"No faces in the {direction} frame to copy", 5000)
            return False
        for face_index in reversed(range(self._editable.face_count(current_index))):
            self._editable.delete_face(current_index, face_index)
        for face in source_faces:
            self._editable.add_face(current_index, face.bbox, landmarks=face.landmarks)
        self._editor_state.set("face_count_changed", True)
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        if self._editable.face_count(current_index):
            self._editor_state.set("face_index", 0)
        self.statusBar().showMessage(
            f"Copied {len(source_faces)} face(s) from the {direction} frame", 5000
        )
        return True

    def delete_active_face(self) -> None:
        """Delete the active face from the editable alignment model."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return
        face_index = self._editor_state.face_index
        if not self._editable.delete_face(frame_index, face_index):
            self.statusBar().showMessage("No face selected to delete", 5000)
            return
        self._editor_state.set("face_count_changed", True)
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        new_count = self._editable.face_count(frame_index)
        if new_count == 0:
            # No face left to operate on; flag "no active selection".
            self._editor_state.set("face_index", -1)
        else:
            self._editor_state.set("face_index", min(face_index, new_count - 1))
        self.statusBar().showMessage(f"Deleted face index {face_index}", 5000)

    def add_face_at_center(
        self,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> int | None:
        """Add a new face. Defaults to a centered fixed-size bbox.

        ``bbox`` is provided in source-image coordinates. The convenience
        default places a 64x64 box centred in the source frame so the action
        is usable without a pointer-driven add gesture.
        """
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return None
        if bbox is None:
            src_w, src_h = self._frame_view.source_size
            if src_w <= 0 or src_h <= 0:
                self.statusBar().showMessage("Cannot add face: frame not loaded", 5000)
                return None
            size = float(min(64, max(8, min(src_w, src_h) // 4)))
            bbox = (src_w / 2 - size / 2, src_h / 2 - size / 2, size, size)
        try:
            new_index = self._editable.add_face(frame_index, bbox)
        except ValueError as err:
            self.statusBar().showMessage(str(err), 5000)
            return None
        self._editor_state.set("face_count_changed", True)
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        self._editor_state.set("face_index", new_index)
        self.statusBar().showMessage(f"Added face index {new_index}", 5000)
        # Aligner integration (#104): initialise landmarks for the new face
        # through the configured aligner when auto-run is on.  Failures are
        # surfaced through ``rerun_aligner_for_face`` and leave the new face
        # in place with empty landmarks — Tk's behaviour when the aligner
        # refuses to produce points.
        self._maybe_run_aligner(int(new_index))
        return new_index

    def nudge_active_face(self, dx: float, dy: float) -> bool:
        """Translate the active face's bbox + landmarks by ``(dx, dy)`` pixels."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to edit", 3000)
            return False
        face_index = self._editor_state.face_index
        if face_index < 0:
            self.statusBar().showMessage("No active face to nudge", 3000)
            return False
        if not self._editable.move_face(frame_index, face_index, dx, dy):
            self.statusBar().showMessage("Nudge failed (no active face)", 3000)
            return False
        self._editor_state.set("edited", True)
        self.mark_dirty(True)
        self.refresh_faces()
        self._frame_view.update()
        return True

    def nudge_up_one(self) -> bool:
        """Nudge active face up by 1 source pixel."""
        return self.nudge_active_face(0.0, -1.0)

    def nudge_down_one(self) -> bool:
        """Nudge active face down by 1 source pixel."""
        return self.nudge_active_face(0.0, 1.0)

    def nudge_left_one(self) -> bool:
        """Nudge active face left by 1 source pixel."""
        return self.nudge_active_face(-1.0, 0.0)

    def nudge_right_one(self) -> bool:
        """Nudge active face right by 1 source pixel."""
        return self.nudge_active_face(1.0, 0.0)

    def nudge_up_fast(self) -> bool:
        """Nudge active face up by 10 source pixels (Shift modifier)."""
        return self.nudge_active_face(0.0, -10.0)

    def nudge_down_fast(self) -> bool:
        """Nudge active face down by 10 source pixels (Shift modifier)."""
        return self.nudge_active_face(0.0, 10.0)

    def nudge_left_fast(self) -> bool:
        """Nudge active face left by 10 source pixels (Shift modifier)."""
        return self.nudge_active_face(-10.0, 0.0)

    def nudge_right_fast(self) -> bool:
        """Nudge active face right by 10 source pixels (Shift modifier)."""
        return self.nudge_active_face(10.0, 0.0)

    def undo_edit(self) -> bool:
        """Reverse the last edit (if any)."""
        if not self._editable.undo():
            self.statusBar().showMessage("Nothing to undo", 3000)
            return False
        self.statusBar().showMessage("Undid last edit", 3000)
        return True

    def redo_edit(self) -> bool:
        """Re-apply the most recently undone edit (if any)."""
        if not self._editable.redo():
            self.statusBar().showMessage("Nothing to redo", 3000)
            return False
        self.statusBar().showMessage("Redid last edit", 3000)
        return True

    def cycle_filter_mode(self) -> None:
        """Advance to the next filter mode in the legacy rotation order.

        Recomputes the filtered frame list immediately so navigation +
        transport bar update in lock-step (#107).
        """
        from tools.manual.frame_filter import FILTER_MODES

        current = self._editor_state.filter_mode or FILTER_MODES[0]
        try:
            index = FILTER_MODES.index(current)
        except ValueError:
            index = -1
        new_mode = FILTER_MODES[(index + 1) % len(FILTER_MODES)]
        self._editor_state.set("filter_mode", new_mode)
        self._refresh_filter_results()
        filtered_count = len(self._filtered_frame_indices)
        suffix = f" ({filtered_count} match)" if filtered_count else " (no matches)"
        self.statusBar().showMessage(f"Filter: {new_mode}{suffix}", 5000)

    def cycle_annotation_display(self) -> None:
        """Cycle annotation overlays in the legacy rotation order."""
        order = ("None", "Mesh", "Mask", "Landmarks")
        current = self._editor_state.annotation_mode or order[0]
        try:
            index = order.index(current)
        except ValueError:
            index = -1
        self._editor_state.set("annotation_mode", order[(index + 1) % len(order)])
        self.statusBar().showMessage(f"Annotation: {self._editor_state.annotation_mode}", 5000)

    def set_editor_view(self) -> None:
        """Activate the View editor mode."""
        self._editor_state.set("editor_mode", "View")

    def set_editor_boundingbox(self) -> None:
        """Activate the Bounding Box editor mode."""
        self._editor_state.set("editor_mode", "BoundingBox")

    def set_editor_extractbox(self) -> None:
        """Activate the Extract Box editor mode."""
        self._editor_state.set("editor_mode", "ExtractBox")

    def set_editor_landmarks(self) -> None:
        """Activate the Landmarks editor mode."""
        self._editor_state.set("editor_mode", "Landmarks")

    def set_editor_mask(self) -> None:
        """Activate the Mask editor mode."""
        self._editor_state.set("editor_mode", "Mask")

    def zoom_in(self) -> None:
        """Zoom into the embedded frame view (action handler)."""
        self._frame_view.zoom_in()

    def zoom_out(self) -> None:
        """Zoom out of the embedded frame view (action handler)."""
        self._frame_view.zoom_out()

    def reset_view(self) -> None:
        """Reset the embedded frame view (action handler)."""
        self._magnify_restore_state = None
        self._frame_view.reset_view()

    def magnify_active_face(self) -> bool:
        """Toggle fitting the active face's bbox to the viewport (Landmark editor M).

        Returns ``False`` and surfaces a status message when no face is
        active — Tk's Magnify is a no-op in that case too, so parity holds.
        """
        if self._magnify_restore_state is not None:
            return self._restore_magnified_view()
        bbox = self._active_face_bbox()
        if bbox is None:
            self.statusBar().showMessage("No active face to magnify", 3000)
            return False
        self._magnify_restore_state = self._frame_view.view_state()
        if not self._frame_view.magnify_to_source_rect(bbox):
            self._magnify_restore_state = None
            self.statusBar().showMessage("Could not magnify active face", 3000)
            return False
        return True

    def _auto_magnify_active_face(self) -> bool:
        """Fit the active face when entering detail editors without losing view state."""
        if self._magnify_restore_state is not None:
            return True
        bbox = self._active_face_bbox()
        if bbox is None:
            return False
        self._magnify_restore_state = self._frame_view.view_state()
        if self._frame_view.magnify_to_source_rect(bbox):
            return True
        self._magnify_restore_state = None
        return False

    def _restore_magnified_view(self) -> bool:
        """Restore the zoom/pan state captured before magnification."""
        state = self._magnify_restore_state
        if state is None:
            return False
        self._magnify_restore_state = None
        return self._frame_view.restore_view_state(state)

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
        # Mask editor (#101): inline dropdown that lets the user pick which
        # mask type to edit.  Hidden by default; surfaced only when the Mask
        # editor (F5) is active so it doesn't clutter the View mode.
        self._mask_controls = self._build_mask_controls()
        left_layout.addWidget(self._mask_controls)
        # Bounding Box editor (#104): aligner plugin selection, normalization,
        # auto-run, explicit rerun, and visible load progress.  Hidden except
        # while F2 / BoundingBox mode is active.
        self._aligner_controls = self._build_aligner_controls()
        left_layout.addWidget(self._aligner_controls)
        left_layout.addWidget(self._frame_view, 1)
        # Filter awareness (#107): a slim row that surfaces filter state and
        # the Misaligned threshold control.  Always visible (so the user can
        # see the active filter + match count); the threshold slider inside
        # only appears for the Misaligned Faces filter.
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
            # Owner registration wires the shortcut: ``self`` for window-scope,
            # ``self._frame_view`` for frame-view scope so the face panel can
            # claim arrow keys for its own navigation when it has focus.
            owner.addAction(action)
            if spec.toolbar_visible:
                toolbar.addAction(action)
            self._actions[spec.key] = action
        # Initial play/pause icon + text mirrors the editor state's not-playing
        # state — actions are built fresh so the default label/icon land here.
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

    def _load_session(self) -> None:
        """Populate the Qt Manual Tool from a neutral session.

        Only renders cheap session metadata + frame discovery here.  The
        expensive alignments-file parse and thumbnail-cache check run in
        :class:`ManualStartupWorker` and refresh the metadata label once
        they complete.
        """
        frame_summary: str
        if self._session.has_images:
            frame_summary = str(self._session.frame_count)
        elif self._session.is_video_input:
            frame_summary = "video input (loading…)"
        else:
            frame_summary = "video input"
        thumbs_state = "regenerate forced" if self._session.thumb_regenerate else "loading…"
        self._metadata_label.setText(
            "\n".join(
                (
                    f"Input: {self._session.frames}",
                    f"Alignments: {self._alignments_handle.path}",
                    f"Frames: {frame_summary}",
                    f"Thumbnails: {thumbs_state}",
                )
            )
        )
        # Seed the editable model from any existing on-disk alignments
        # before the first frame is selected so the overlay + face panel show
        # the persisted faces immediately.  This open() blocks the UI for the
        # duration of an Alignments load; that is acceptable because the
        # alternative (empty panel until the worker completes) breaks core
        # Manual Tool behavior on an existing alignments file.
        try:
            # Image-folder sessions seed against their source filenames so
            # sparse alignments (an entry only for ``frame_010.png``) attach
            # to the matching frame index — not lexicographic position 0.
            # Video sessions fall back to the alignment-keys ordering since
            # those keys *are* the canonical frame order.
            if self._session.has_images:
                self._editable.seed_from_handle(
                    self._alignments_handle,
                    frame_names=[frame.name for frame in self._session.frame_list],
                )
            else:
                self._editable.seed_from_handle(self._alignments_handle)
        except Exception:  # noqa: BLE001 - re-surfaced by the startup worker
            logger.exception("Manual Tool synchronous seed failed")
        if self._session.has_images:
            self._thumbnail_panel.set_frames(self._session.frame_list)
            self._refresh_filter_results(preserve_current=False)
            self._thumbnail_panel.setCurrentRow(0)
        elif self._session.is_video_input:
            # Load video metadata synchronously before the provider starts
            # so the worker (a) does not race the provider for the same
            # alignments-file open and (b) the provider sees any persisted
            # pts_time / keyframes payload from disk on first launch.
            try:
                self._video_metadata = self._alignments_handle.video_metadata()
            except Exception:  # noqa: BLE001 - re-surfaced by the startup worker
                logger.exception("Manual Tool video metadata load failed")
                self._video_metadata = None
            self._start_video_provider()
        else:
            self._frame_view.clear_frame(
                "Video input detected. Frame extraction will be wired in a follow-up."
            )
        self._status_label.setText("Manual Tool starting…")
        self._sync_actions()

    def _start_background_startup(self) -> None:
        """Kick off ``ManualStartupWorker`` for async alignments preparation."""
        # Reset the per-startup #121 flag so a retry can paint the static
        # ``thumbs`` anchor again before any live percent fires.
        self._thumb_progress_seen = False
        self._progress_bar = self._build_progress_bar()
        if self._progress_bar is not None:
            self.statusBar().addPermanentWidget(self._progress_bar)
            self._progress_bar.show()
        self._startup_worker = ManualStartupWorker(
            self._alignments_handle, self._editable, self._session, parent=self
        )
        self._startup_worker.progress.connect(self._on_startup_progress)
        self._startup_worker.progress_percent.connect(self._on_startup_progress_percent)
        self._startup_worker.completed.connect(self._on_startup_completed)
        self._startup_worker.failed.connect(self._on_startup_failed)
        logger.info("Manual Tool startup worker scheduled")
        self._emit_console("Manual Tool: preparing session…")
        self._startup_worker.start()

    def _build_progress_bar(self) -> QProgressBar:
        """Return a Qt-native determinate progress bar for the status bar.

        Stages reported by :class:`_ManualStartupTask` map to discrete
        percentages (``open=33``, ``thumbs=66``, ``complete=100``) so users
        see real forward motion instead of a spinning indeterminate bar.
        """
        bar = QProgressBar()
        bar.setObjectName("qt-manual-startup-progress")
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setMaximumWidth(180)
        bar.setTextVisible(True)
        bar.setFormat("Loading %p%")
        bar.hide()
        return bar

    def _on_startup_progress(self, stage: str, message: str) -> None:
        """Surface intermediate startup messages to the status bar + console.

        Static stage percents (open / thumbs / complete) paint a fresh value
        on the determinate bar.  The named ``"thumbs"`` stage in particular
        fires alongside every per-event ``progress_percent`` emit so console
        + status messaging stay per-frame — but once live thumbnail progress
        has started, painting the static 66% anchor would visibly *reset*
        the bar (see issues #121 / #119 task 1).  Skip that one repaint.
        """
        logger.debug("Manual Tool startup [%s]: %s", stage, message)
        self.statusBar().showMessage(message)
        self._emit_console(f"Manual Tool [{stage}]: {message}")
        percent = _ManualStartupTask.STAGE_PERCENT.get(stage)
        if (
            percent is not None
            and self._progress_bar is not None
            and not (stage == "thumbs" and self._thumb_progress_seen)
        ):
            self._progress_bar.setValue(percent)

    def _on_startup_progress_percent(self, percent: int, _message: str) -> None:
        """Paint the determinate progress bar from a per-event thumb percent.

        The percent is carried by the signal payload itself, so queued
        deliveries cannot disagree with the value computed at emit time —
        unlike the previous ``STAGE_PERCENT["thumbs_progress"]`` trick.

        Also marks ``_thumb_progress_seen`` so the named-stage handler
        stops resetting the bar to the 66% static ``thumbs`` anchor for
        the rest of this startup.
        """
        self._thumb_progress_seen = True
        if self._progress_bar is None:
            return
        clamped = max(0, min(100, int(percent)))
        self._progress_bar.setValue(clamped)

    def _on_startup_completed(self, has_thumbnails: bool, summary: str) -> None:
        """Finalize UI state once the startup worker reports success.

        The worker's ``completed`` signal fires once; after the handler
        runs there's no further use for the worker.  Drain it explicitly
        (mirrors the aligner preload + extract-worker patterns) so the
        QThread doesn't linger past the test's qtbot assertions.
        """
        self._startup_complete = True
        thumbs_state = (
            "regenerate forced"
            if self._session.thumb_regenerate
            else ("cached" if has_thumbnails else "needs generation")
        )
        self._metadata_label.setText(
            "\n".join(
                (
                    f"Input: {self._session.frames}",
                    f"Alignments: {self._alignments_handle.path}",
                    f"Frames: {self._frame_summary_text()}",
                    f"Thumbnails: {thumbs_state}",
                )
            )
        )
        if self._session.is_video_input:
            self._video_metadata = self._alignments_handle.video_metadata()
        self._status_label.setText("Manual Tool ready")
        self.statusBar().showMessage(summary, 5000)
        self._hide_progress_bar()
        logger.info("Manual Tool startup complete: %s", summary)
        self._emit_console(f"Manual Tool ready — {summary}")
        # The editable model may now hold seeded faces; repaint + refresh.
        self.refresh_faces()
        self._refresh_face_grid()
        self._frame_view.update()
        self._sync_actions()
        self._teardown_startup_worker()

    def _on_startup_failed(self, message: str) -> None:
        """Surface startup failures through status, console, log and dialog."""
        self._startup_complete = False
        self._status_label.setText("Manual Tool startup failed")
        self.statusBar().showMessage(message, 7000)
        self._hide_progress_bar()
        logger.error("Manual Tool startup failed: %s", message)
        self._emit_console(f"Manual Tool startup failed: {message}")
        QMessageBox.critical(self, "Manual Tool Startup", message)
        self._teardown_startup_worker()

    def _teardown_startup_worker(self) -> None:
        """Drain + drop the startup worker after its terminal signal fires.

        Mirrors the aligner-preload and extract-worker teardown patterns:
        after the worker's ``completed`` or ``failed`` signal fires there's
        no more work for it to do, so we tell the QThread to quit, wait
        briefly for it to exit, then mark it for Qt deletion and clear
        ``_startup_worker``.  Without this every Manual Tool test left a
        QThread alive until widget teardown, which is the closest remaining
        cross-test signal-leak candidate.
        """
        worker = self._startup_worker
        if worker is None:
            return
        worker.stop()
        worker.deleteLater()
        self._startup_worker = None

    def _hide_progress_bar(self) -> None:
        """Hide and detach the indeterminate startup progress widget.

        Yields the bar back to a busy-lock when one is currently active —
        the lock's ``finally`` restores its prior state, so we mustn't yank
        the widget out from underneath the in-flight operation.  Without
        this guard the startup worker's completion can tear down the bar
        that an async save just materialised.
        """
        if self._busy_operation:
            # Hide is still safe; only avoid the detach/null-out.
            if self._progress_bar is not None:
                self._progress_bar.hide()
            return
        if self._progress_bar is not None:
            self._progress_bar.hide()
            self.statusBar().removeWidget(self._progress_bar)
            self._progress_bar = None

    def _emit_console(self, message: str) -> None:
        """Forward a user-facing message to the host shell console, if any."""
        if self._console_logger is None:
            return
        try:
            self._console_logger(message)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Manual Tool console logger raised")

    def _frame_summary_text(self) -> str:
        """Return the metadata summary text for the loaded frame source."""
        if self._session.has_images:
            return str(self._session.frame_count)
        if self._video_metadata is not None and self._video_metadata.is_valid:
            return f"{self._video_metadata.frame_count} (video)"
        return "video input"

    def _start_video_provider(self) -> None:
        """Initialize the async video frame provider for video sessions."""
        self._frame_view.clear_frame("Loading video frames…")
        meta_dict: dict[str, list[int]] | None
        if self._video_metadata is not None and self._video_metadata.is_valid:
            meta_dict = {
                "pts_time": list(self._video_metadata.pts_time),
                "keyframes": list(self._video_metadata.keyframes),
            }
        else:
            meta_dict = None
        self._video_provider = VideoFrameProvider(
            self._session.frames,
            video_meta_data=meta_dict,
            parent=self,
        )
        self._video_provider.count_ready.connect(self._on_video_count_ready)
        self._video_provider.frame_ready.connect(self._on_video_frame_ready)
        self._video_provider.load_failed.connect(self._on_video_load_failed)
        self._video_provider.start()

    def _on_video_count_ready(self, count: int) -> None:
        """Populate the thumbnail list with video frame placeholders."""
        if count <= 0:
            self._frame_view.clear_frame("Video reported zero frames")
            return
        self._video_frames = [
            ManualFrame(index=idx, name=f"frame_{idx:06d}", path=self._session.frames)
            for idx in range(count)
        ]
        self._thumbnail_panel.set_frames(tuple(self._video_frames))
        self._refresh_filter_results(preserve_current=False)
        self._thumbnail_panel.setCurrentRow(0)

    def _on_video_frame_ready(self, index: int, filename: str, image: QImage) -> None:
        """Display a video frame that was decoded on the worker thread."""
        if (
            self._current_frame is not None
            and self._current_frame.index == index
            and self._session.is_video_input
        ):
            self._frame_view.set_image(image, self._current_frame)
            self._status_label.setText(f"Frame {index + 1}: {filename}")

    def _on_video_load_failed(self, index: int, message: str) -> None:
        """Surface a video load failure in the status label and console."""
        logger.warning("Manual Tool video frame %s failed: %s", index, message)
        self._status_label.setText(f"Failed to load frame {index}: {message}")

    def _on_unsaved_changed(self, dirty: bool) -> None:
        """Forward editor-state unsaved changes to UI signals."""
        self.dirty_changed.emit(bool(dirty))
        self._sync_actions()

    def _on_face_selected(self, face_index: int) -> None:
        """Propagate active face selection to the shared editor state.

        A negative ``face_index`` (emitted when the panel is cleared) is
        propagated as ``-1`` so editor state and the overlay do not keep
        pointing at a face that no longer exists.
        """
        self._editor_state.set("face_index", face_index if face_index >= 0 else -1)
        self._sync_actions()

    def _current_frame_index(self) -> int:
        """Return the sorted-frame index of the active frame, or -1 if none."""
        return -1 if self._current_frame is None else self._current_frame.index

    def _on_editable_changed(self, frame_index: int) -> None:
        """React to any change in the editable alignment model.

        The face panel mirrors the alignments-file thumbnails (read-only) and
        emits ``face_selected(-1)`` whenever it is empty.  Refresh it first,
        then resync ``editor_state.face_index`` against the editable model so
        a programmatic add does not leave us pointed at ``-1``.  Dirty state
        is derived from the model's undo stack so any edit (including ones
        made on a non-current frame) marks the session unsaved, and undoing
        back to the saved snapshot drops the dirty flag automatically.
        """
        self.mark_dirty(self._editable.can_undo)
        self._refresh_filter_results()
        if frame_index != self._current_frame_index():
            return
        self.refresh_faces()
        self._frame_view.update()
        face_count = self._editable.face_count(frame_index)
        active = self._editor_state.face_index
        if face_count > 0 and active < 0:
            self._editor_state.set("face_index", 0)
        elif face_count == 0 and active >= 0:
            self._editor_state.set("face_index", -1)
        self._sync_actions()

    def _on_frame_clicked(self, point: QPointF) -> None:
        """Hit-test the editable model at the clicked source-image point."""
        frame_index = self._current_frame_index()
        if frame_index < 0:
            return
        face_index = self._editable.hit_test(frame_index, point.x(), point.y())
        if face_index is None:
            return
        self._editor_state.set("face_index", face_index)
        self._face_panel.select_face(face_index)

    def _active_face_index(self) -> int | None:
        """Return the currently selected ``face_index`` or ``None``."""
        index = self._editor_state.face_index
        if not isinstance(index, int) or index < 0:
            return None
        return index

    # ---- Mask editor (#101) public action handlers ----

    # ---- Aligner integration (#104) ----

    def _on_aligner_status(self, status: T.Any) -> None:
        """Forward :class:`AlignerStatus` events to status, console and controls."""
        message = getattr(status, "message", "") or ""
        kind = getattr(status, "kind", "") or ""
        aligner = getattr(status, "aligner", "") or self._active_aligner_name()
        if not message:
            return
        self._set_aligner_load_status(kind, aligner, message)
        timeout = 5000 if kind == "failed" else 3000
        self.statusBar().showMessage(message, timeout)
        self._emit_console(f"Manual Tool aligner: {message}")
        if kind == "failed":
            logger.error("Manual Tool aligner: %s", message)

    def available_aligners(self) -> tuple[str, ...]:
        """Return aligner plugin display names from the cached service."""
        return self._aligner_service.available_aligners

    def available_normalizations(self) -> tuple[str, ...]:
        """Return normalization methods supported by the aligner service."""
        return self._aligner_service.available_normalizations()

    def set_aligner_name(self, name: str) -> None:
        """Select the active aligner plugin (Bounding Box dropdown handler)."""
        value = str(name).strip()
        if not value or value.startswith("No aligners"):
            return
        self._editor_state.set("aligner_name", value)
        self._sync_aligner_controls()
        self._schedule_aligner_preload()

    def set_aligner_normalization(self, method: str) -> None:
        """Apply a normalization method to the aligner service (radio handler)."""
        value = str(method)
        self._editor_state.set("aligner_normalization", value)
        self._aligner_service.set_normalization(value)
        self._sync_aligner_controls()
        self._schedule_aligner_preload()

    def _active_aligner_name(self) -> str:
        """Return the editor-state aligner name, falling back to the service default."""
        return self._editor_state.aligner_name or self._aligner_service.default_aligner()

    def _build_filter_controls(self) -> QWidget:
        """Return the filter-awareness control row (#107).

        Always visible: shows the active filter + match count.  The
        Misaligned threshold slider inside is shown only when
        ``filter_mode == "Misaligned Faces"`` (Tk parity) so the row
        stays slim for the other five filter modes.
        """
        from tools.manual.frame_filter import (
            MISALIGNED_THRESHOLD_MAX,
            MISALIGNED_THRESHOLD_MIN,
        )

        container = QWidget()
        container.setObjectName("qt-manual-filter-controls")
        # Pin to its sizeHint so the always-visible filter row doesn't steal
        # vertical space from the frame view when BoundingBox controls and
        # the transport bar are also visible.
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(8)

        self._filter_label = QLabel()
        self._filter_label.setObjectName("qt-manual-filter-label")
        layout.addWidget(self._filter_label)
        layout.addStretch(1)

        # Misaligned threshold sub-group — visible only for that filter mode.
        self._filter_threshold_label = QLabel("Threshold:")
        self._filter_threshold_label.setObjectName("qt-manual-filter-threshold-label")
        layout.addWidget(self._filter_threshold_label)
        self._filter_threshold_slider = QSlider(Qt.Horizontal)
        self._filter_threshold_slider.setObjectName("qt-manual-filter-threshold-slider")
        self._filter_threshold_slider.setRange(MISALIGNED_THRESHOLD_MIN, MISALIGNED_THRESHOLD_MAX)
        self._filter_threshold_slider.setValue(int(self._editor_state.filter_distance))
        self._filter_threshold_slider.setFixedWidth(120)
        self._filter_threshold_slider.valueChanged.connect(self._on_filter_threshold_changed)
        layout.addWidget(self._filter_threshold_slider)
        self._filter_threshold_value = QLabel(str(self._editor_state.filter_distance))
        self._filter_threshold_value.setObjectName("qt-manual-filter-threshold-value")
        layout.addWidget(self._filter_threshold_value)

        self._refresh_filter_controls()
        return container

    def _refresh_filter_controls(self) -> None:
        """Mirror the active filter mode + match count into the control row."""
        label = getattr(self, "_filter_label", None)
        if label is None:
            return
        mode = self._editor_state.filter_mode or "All Frames"
        total = len(self._filtered_frame_indices)
        if mode == "All Frames":
            text = f"Filter: All Frames ({total})"
        else:
            text = f"Filter: {mode} ({total} match)"
        label.setText(text)

        misaligned = mode == "Misaligned Faces"
        if hasattr(self, "_filter_threshold_slider"):
            self._filter_threshold_label.setVisible(misaligned)
            self._filter_threshold_slider.setVisible(misaligned)
            self._filter_threshold_value.setVisible(misaligned)
            current = int(self._editor_state.filter_distance)
            self._filter_threshold_value.setText(str(current))
            self._filter_threshold_slider.blockSignals(True)
            try:
                self._filter_threshold_slider.setValue(current)
            finally:
                self._filter_threshold_slider.blockSignals(False)

    def _on_filter_threshold_changed(self, value: int) -> None:
        """Persist threshold + refresh the filter when the user moves the slider."""
        self._editor_state.set("filter_distance", int(value))
        self._refresh_filter_results()

    def _build_aligner_controls(self) -> QWidget:
        """Return the Bounding Box aligner control panel (#104).

        Hidden by default; surfaced only while ``editor_mode == "BoundingBox"``.
        The selected aligner, normalization and auto-run values are stored on
        :class:`ManualEditorState` so they persist across editor-mode switches.
        """
        container = QWidget()
        container.setObjectName("qt-manual-aligner-controls")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addWidget(QLabel("Aligner:"))

        self._aligner_combo = QComboBox()
        self._aligner_combo.setObjectName("qt-manual-aligner-combo")
        self._aligner_combo.setMinimumWidth(150)
        self._aligner_combo.currentTextChanged.connect(self._on_aligner_combo_changed)
        top.addWidget(self._aligner_combo)

        self._aligner_auto_run_checkbox = QCheckBox("Auto-run")
        self._aligner_auto_run_checkbox.setObjectName("qt-manual-aligner-auto-run")
        self._aligner_auto_run_checkbox.setChecked(bool(self._editor_state.aligner_auto_run))
        self._aligner_auto_run_checkbox.toggled.connect(self._on_aligner_auto_run_toggled)
        top.addWidget(self._aligner_auto_run_checkbox)

        self._aligner_rerun_button = QToolButton()
        self._aligner_rerun_button.setObjectName("qt-manual-aligner-rerun")
        self._aligner_rerun_button.setText("Run")
        self._aligner_rerun_button.setToolTip("Run the selected aligner for the active face")
        self._aligner_rerun_button.clicked.connect(self._on_aligner_rerun_clicked)
        top.addWidget(self._aligner_rerun_button)
        top.addStretch(1)
        outer.addLayout(top)

        norm_row = QHBoxLayout()
        norm_row.setContentsMargins(0, 0, 0, 0)
        norm_row.setSpacing(8)
        norm_row.addWidget(QLabel("Normalization:"))
        self._aligner_normalization_group = QButtonGroup(container)
        self._aligner_normalization_group.setExclusive(True)
        self._aligner_normalization_buttons: dict[str, QRadioButton] = {}
        for method in self.available_normalizations():
            button = QRadioButton(str(method))
            button.setObjectName(f"qt-manual-aligner-normalization-{method}")
            button.toggled.connect(
                lambda checked, value=str(method): (
                    self.set_aligner_normalization(value) if checked else None
                )
            )
            self._aligner_normalization_group.addButton(button)
            self._aligner_normalization_buttons[str(method)] = button
            norm_row.addWidget(button)
        norm_row.addStretch(1)
        outer.addLayout(norm_row)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(8)
        self._aligner_load_progress = QProgressBar()
        self._aligner_load_progress.setObjectName("qt-manual-aligner-load-progress")
        self._aligner_load_progress.setRange(0, 100)
        self._aligner_load_progress.setValue(0)
        self._aligner_load_progress.setMaximumWidth(160)
        self._aligner_load_progress.setTextVisible(True)
        self._aligner_load_progress.hide()
        progress_row.addWidget(self._aligner_load_progress)

        self._aligner_status_label = QLabel("Aligner idle")
        self._aligner_status_label.setObjectName("qt-manual-aligner-status-label")
        self._aligner_status_label.setWordWrap(True)
        progress_row.addWidget(self._aligner_status_label, 1)
        outer.addLayout(progress_row)

        container.setVisible(self._editor_state.editor_mode == "BoundingBox")
        self._sync_aligner_controls()
        return container

    def _on_aligner_combo_changed(self, name: str) -> None:
        """Handle a user-selected aligner plugin from the dropdown."""
        self.set_aligner_name(name)

    def _on_aligner_auto_run_toggled(self, checked: bool) -> None:
        """Persist the auto-run checkbox value on editor state."""
        self._editor_state.set("aligner_auto_run", bool(checked))
        self._sync_aligner_controls()

    def _on_aligner_rerun_clicked(self) -> None:
        """Run the active aligner against the selected face."""
        face_index = self._active_face_index()
        if face_index is None:
            self.statusBar().showMessage("No active face to align", 3000)
            return
        self.rerun_aligner_for_face(int(face_index))

    def _refresh_aligner_controls_visibility(self) -> None:
        """Show or hide Bounding Box aligner controls to match editor mode.

        Merely entering BoundingBox mode must not preload the production
        aligner.  Passive preload makes unrelated GUI tests and ordinary mode
        switches touch plugin/model loading.  Explicit user changes to the
        aligner dropdown or normalization radios still call
        :meth:`_schedule_aligner_preload`, and actual alignment runs continue
        to load on demand through :meth:`rerun_aligner_for_face`.
        """
        controls = getattr(self, "_aligner_controls", None)
        if controls is None:
            return
        active = self._editor_state.editor_mode == "BoundingBox"
        controls.setVisible(active)
        if active:
            self._sync_aligner_controls()

    def _sync_aligner_controls(self) -> None:
        """Mirror editor-state aligner settings into the visible controls."""
        combo = getattr(self, "_aligner_combo", None)
        if combo is None:
            return
        aligners = list(self.available_aligners())
        current = self._editor_state.aligner_name or (
            self._aligner_service.default_aligner() if aligners else ""
        )
        if current and current not in aligners:
            aligners.insert(0, current)

        combo.blockSignals(True)
        combo.clear()
        if aligners:
            combo.addItems(aligners)
            combo.setEnabled(True)
            index = combo.findText(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
        else:
            combo.addItem("No aligners available")
            combo.setEnabled(False)
        combo.blockSignals(False)

        normalization = str(self._editor_state.aligner_normalization)
        for method, button in self._aligner_normalization_buttons.items():
            button.blockSignals(True)
            button.setChecked(method == normalization)
            button.blockSignals(False)
        if (
            normalization not in self._aligner_normalization_buttons
            and self._aligner_normalization_buttons
        ):
            first = next(iter(self._aligner_normalization_buttons.values()))
            first.blockSignals(True)
            first.setChecked(True)
            first.blockSignals(False)

        auto_run = getattr(self, "_aligner_auto_run_checkbox", None)
        if auto_run is not None:
            auto_run.blockSignals(True)
            auto_run.setChecked(bool(self._editor_state.aligner_auto_run))
            auto_run.blockSignals(False)

        rerun = getattr(self, "_aligner_rerun_button", None)
        if rerun is not None:
            rerun.setEnabled(bool(aligners) and self._active_face_index() is not None)

    def _schedule_aligner_preload(self) -> None:
        """Preload the selected aligner/backend while BBox controls are visible."""
        controls = getattr(self, "_aligner_controls", None)
        if controls is None or not controls.isVisible():
            return
        aligner = self._active_aligner_name()
        normalization = str(self._editor_state.aligner_normalization)
        if not aligner:
            self._set_aligner_load_status("failed", "", "No aligner plugins available")
            return
        target = (aligner, normalization)
        if target in self._aligner_loaded_targets:
            self._set_aligner_load_status("ready", aligner, f"Aligner '{aligner}' ready")
            return
        if self._aligner_load_worker is not None and self._aligner_load_target == target:
            return
        if self._aligner_load_worker is not None:
            if not self._aligner_load_worker.stop():
                self.statusBar().showMessage(
                    "Aligner is still loading — wait for it to finish before changing selection.",
                    5000,
                )
                return
            self._aligner_load_worker.deleteLater()
            self._aligner_load_worker = None

        worker = ManualAlignerLoadWorker(
            self._aligner_service,
            aligner,
            normalization,
            parent=self,
        )
        worker.status.connect(self._on_aligner_load_status)
        worker.completed.connect(self._on_aligner_load_completed)
        self._aligner_load_worker = worker
        self._aligner_load_target = target
        self._set_aligner_load_status("loading", aligner, f"Loading aligner '{aligner}'…")
        worker.start()

    def _on_aligner_load_status(self, kind: str, aligner: str, message: str) -> None:
        """Handle status emitted by the background aligner preload worker."""
        self._set_aligner_load_status(kind, aligner, message)
        timeout = 5000 if kind == "failed" else 3000
        self.statusBar().showMessage(message, timeout)
        self._emit_console(f"Manual Tool aligner: {message}")

    def _on_aligner_load_completed(
        self, aligner: str, normalization: str, ok: bool, message: str
    ) -> None:
        """Finalize the visible aligner preload state."""
        target = (str(aligner), str(normalization))
        if self._aligner_load_target is not None and target != self._aligner_load_target:
            return
        if ok:
            self._aligner_loaded_targets.add(target)
            self._set_aligner_load_status("ready", aligner, message)
        else:
            self._set_aligner_load_status("failed", aligner, message)

        worker = self._aligner_load_worker
        if worker is not None:
            worker.stop(wait_ms=1000)
            worker.deleteLater()
            self._aligner_load_worker = None

        self._aligner_load_target = None
        self._sync_aligner_controls()

    def _set_aligner_load_status(self, kind: str, aligner: str, message: str) -> None:
        """Paint the inline aligner load-progress widget."""
        progress = getattr(self, "_aligner_load_progress", None)
        label = getattr(self, "_aligner_status_label", None)
        if label is not None and message:
            label.setText(message)
        if progress is None:
            return
        if kind in ("loading", "aligning"):
            progress.setRange(0, 0)
            progress.setFormat("Loading aligner…")
            progress.show()
            rerun = getattr(self, "_aligner_rerun_button", None)
            if rerun is not None:
                rerun.setEnabled(False)
        elif kind in ("ready", "aligned"):
            progress.hide()
            progress.setRange(0, 100)
            progress.setValue(100)
        elif kind == "failed":
            progress.hide()
            progress.setRange(0, 100)
            progress.setValue(0)

    def rerun_aligner_for_face(self, face_index: int) -> bool:
        """Rerun the configured aligner against ``face_index`` on the current frame.

        Returns ``True`` if landmarks were refreshed. Failures (no image,
        empty bbox, aligner unavailable, model error) surface a status
        message + console line and never corrupt the editable model — the
        existing landmark cloud stays put.
        """
        frame_index = self._current_frame_index()
        if frame_index < 0:
            self.statusBar().showMessage("No frame to align", 3000)
            return False
        faces = self._editable.faces(frame_index)
        if face_index < 0 or face_index >= len(faces):
            self.statusBar().showMessage("No active face to align", 3000)
            return False
        image = self._frame_view.current_frame_array()
        if image is None:
            self.statusBar().showMessage("Cannot align: frame image not loaded", 4000)
            return False
        face = faces[face_index]
        try:
            landmarks = self._aligner_service.align(
                image,
                face.bbox,
                aligner=self._active_aligner_name() or None,
                normalization=self._editor_state.aligner_normalization,
            )
        except Exception as err:  # noqa: BLE001 - surface to the user; model untouched
            logger.exception("Manual Tool: aligner run failed")
            self.statusBar().showMessage(f"Aligner failed: {err}", 5000)
            self._emit_console(f"Manual Tool aligner failed: {err}")
            return False
        new_points = [(float(point[0]), float(point[1])) for point in landmarks]
        if not self._editable.set_landmarks(frame_index, face_index, new_points):
            return False
        self._editor_state.set("edited", True)
        self.refresh_faces()
        self._frame_view.update()
        return True

    def refresh_faces(self) -> None:
        """Rebuild the face panel from the editable model + persisted thumbs.

        The editable model is the source of truth for what's currently on
        screen — add/delete/move edits must appear in the panel before
        Save runs.  Thumbnails come from the alignments file when the face
        is unchanged from its persisted bbox; otherwise we surface a
        placeholder so a stale JPEG is never shown next to a moved bbox.
        """
        if self._current_frame is None:
            self._face_panel.set_faces(())
            return
        frame_index = self._current_frame.index
        entries = self._face_thumbnail_entries_for_frame(frame_index)
        if not entries:
            self._face_panel.set_faces(())
            return
        self._face_panel.set_faces(entries)

    def _face_thumbnail_entries_for_frame(self, frame_index: int) -> tuple[FaceThumbnail, ...]:
        """Return face thumbnail entries for one editable frame index."""
        editable_faces = self._editable.faces(frame_index)
        if not editable_faces:
            return ()
        frame_name = self._frame_name_for_index(frame_index)
        if frame_name is None:
            frame_name = (
                self._current_frame.name
                if self._current_frame is not None and self._current_frame.index == frame_index
                else f"frame_{frame_index:06d}"
            )
        # Look up persisted thumbnails by frame *name* — the editable model is
        # anchored to the source frame list, but the alignments file may be
        # sparse, so sorted-index lookups (``faces_for_frame``) can attach the
        # wrong frame's thumbnail to a given editable index.
        persisted = {
            entry.face_index: entry
            for entry in self._alignments_handle.faces_for_frame_name(
                frame_name, frame_index=frame_index
            )
        }
        entries = []
        for face in editable_faces:
            previous = persisted.get(face.face_index)
            entries.append(
                FaceThumbnail(
                    frame_index=frame_index,
                    frame_name=frame_name,
                    face_index=face.face_index,
                    thumbnail_jpeg=previous.thumbnail_jpeg if previous else b"",
                )
            )
        return tuple(entries)

    def _face_grid_entries(self) -> tuple[FaceGridEntry, ...]:
        """Build one grid entry per visible face in the active filtered session."""
        entries: list[FaceGridEntry] = []
        for frame_index in self.filtered_frame_indices():
            frame_name = self._frame_name_for_index(frame_index) or f"frame_{frame_index:06d}"
            thumbs = {
                thumb.face_index: thumb
                for thumb in self._face_thumbnail_entries_for_frame(frame_index)
            }
            for face in self._editable.faces(frame_index):
                entries.append(
                    FaceGridEntry(
                        frame_index=frame_index,
                        frame_name=frame_name,
                        face_index=int(face.face_index),
                        thumbnail=thumbs.get(int(face.face_index)),
                        bbox=face.bbox,
                        landmarks=face.landmarks,
                    )
                )
        return tuple(entries)

    def _refresh_face_grid(self) -> None:
        """Rebuild the cross-frame face grid from filters, state and annotations."""
        panel = getattr(self, "_face_grid_panel", None)
        if panel is None:
            return
        panel.set_overlay_state(
            show_mesh=self._editor_state.annotation_mode == "Mesh",
            show_mask=self._should_render_mask(),
            mask_type=self.active_mask_type(),
            mask_opacity=int(self._editor_state.mask_opacity),
        )
        panel.set_entries(self._face_grid_entries())
        self._refresh_face_grid_active()

    def _refresh_face_grid_active(self) -> None:
        """Update active frame/face styling without rebuilding thumbnail icons."""
        panel = getattr(self, "_face_grid_panel", None)
        if panel is None:
            return
        panel.set_active(self._current_frame_index(), int(self._editor_state.face_index))

    def _on_face_grid_activated(self, frame_index: int, face_index: int) -> None:
        """Navigate to the clicked grid face and make it the active face."""
        if frame_index < 0:
            return
        self._stop_playback()
        if self._thumbnail_panel.currentRow() != frame_index:
            self._thumbnail_panel.setCurrentRow(frame_index)
        self._editor_state.set("face_index", int(face_index))
        self._face_panel.select_face(int(face_index))
        self._face_grid_panel.set_active(int(frame_index), int(face_index))
        self._frame_view.update()
        self._sync_actions()

    def _on_face_grid_hovered(self, frame_index: int, face_index: int) -> None:
        """Surface simple hover feedback for cross-frame thumbnails."""
        self.statusBar().showMessage(
            f"Frame {frame_index + 1} / Face {face_index + 1}",
            2000,
        )

    def _on_face_grid_context_menu_requested(
        self,
        frame_index: int,
        face_index: int,
        global_pos: QPointF,
    ) -> None:
        """Open a Delete Face context menu for a cross-frame grid item."""
        self._show_face_context_menu(face_index, global_pos, frame_index=frame_index)

    def _on_face_grid_size_changed(self, size_name: str) -> None:
        """Persist a user-selected face-grid size."""
        if size_name not in _FACE_GRID_SIZES:
            return
        self._editor_state.set("faces_size", size_name)

    def _on_face_grid_size_state_changed(self, size_name: object) -> None:
        """Apply persisted face-grid size state and relayout immediately."""
        name = str(size_name or "Medium")
        if name not in _FACE_GRID_SIZES:
            name = "Medium"
        combo = getattr(self, "_face_grid_size_combo", None)
        if combo is not None and combo.currentText() != name:
            combo.blockSignals(True)
            try:
                combo.setCurrentText(name)
            finally:
                combo.blockSignals(False)
        self._face_grid_panel.set_face_size(name)
        self._refresh_face_grid()

    def _thumbnail_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Display the selected source frame."""
        if current is None:
            return
        index = current.data(Qt.UserRole)
        if not isinstance(index, int):
            return
        if index != self._current_frame_index():
            self._magnify_restore_state = None
        if self._session.has_images:
            if index >= len(self._session.frame_list):
                return
            frame = self._session.frame_list[index]
            self._current_frame = frame
            self._editor_state.set("frame_index", frame.index)
            self.frame_changed.emit(frame.index)
            if self._frame_view.load_frame(frame):
                self._status_label.setText(
                    f"Frame {frame.index + 1} of {self._session.frame_count}: {frame.name}"
                )
            self.refresh_faces()
        elif self._video_frames:
            if index >= len(self._video_frames):
                return
            frame = self._video_frames[index]
            self._current_frame = frame
            self._editor_state.set("frame_index", frame.index)
            self.frame_changed.emit(frame.index)
            if self._video_provider is not None:
                self._video_provider.request_frame(frame.index)
                self._status_label.setText(
                    f"Loading frame {frame.index + 1} of {len(self._video_frames)}"
                )
            self.refresh_faces()

    def _previous_frame(self) -> None:
        """Select previous frame from the active filter (#107)."""
        self._stop_playback()
        if not self._filtered_frame_indices:
            return
        position = self._filtered_position()
        if position <= 0:
            return
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[position - 1])

    def _next_frame(self) -> None:
        """Select next frame from the active filter (#107)."""
        self._stop_playback()
        if not self._filtered_frame_indices:
            return
        position = self._filtered_position()
        if position < 0 or position >= len(self._filtered_frame_indices) - 1:
            return
        self._thumbnail_panel.setCurrentRow(self._filtered_frame_indices[position + 1])

    _MUTATING_ACTION_KEYS: T.ClassVar[tuple[str, ...]] = (
        "save",
        "revert_frame",
        "copy_prev_face",
        "copy_next_face",
        "delete_face",
        "add_face",
        "undo_edit",
        "redo_edit",
        "extract_faces",
    )

    def _sync_actions(self) -> None:
        """Update action availability from session, selection and dirty state.

        Navigation gates run off the *filtered* frame list (#107): first /
        previous / next / last / play are disabled when no frame matches
        the active filter, and enabled bounds follow the filtered position
        rather than the raw thumbnail row.  ``copy_prev_face`` /
        ``copy_next_face`` use the same bounds so a filter that hides
        adjacent frames also hides the copy-from-neighbor affordance.
        """
        if not self._actions:
            return
        filtered = self._filtered_frame_indices
        filtered_total = len(filtered)
        has_frames = filtered_total > 0
        if has_frames:
            filtered_position = self._filtered_position()
            not_first = filtered_position > 0
            not_last = 0 <= filtered_position < filtered_total - 1
        else:
            not_first = False
            not_last = False
        frame_index = self._current_frame_index()
        editable_count = self._editable.face_count(frame_index) if frame_index >= 0 else 0
        face_selected = self._editor_state.face_index >= 0 and (
            editable_count > 0 or bool(self._face_panel.faces)
        )
        has_frame_loaded = self._frame_view.has_frame
        editor_mode = self._editor_state.editor_mode

        availability: dict[str, bool] = {
            "save": self._editor_state.unsaved,
            "revert_frame": self._editor_state.edited
            or self._editor_state.unsaved
            or self._editable.can_undo,
            "first_frame": not_first,
            "previous_frame": not_first,
            "next_frame": not_last,
            "last_frame": not_last,
            "play_pause": has_frames,
            "copy_prev_face": not_first,
            "copy_next_face": not_last,
            "delete_face": face_selected and editable_count > 0,
            "add_face": has_frame_loaded,
            "undo_edit": self._editable.can_undo,
            "redo_edit": self._editable.can_redo,
            "cycle_filter": has_frames,
            "cycle_annotation": has_frames,
            "set_view_mode": editor_mode != "View",
            "set_boundingbox_mode": editor_mode != "BoundingBox",
            "set_extractbox_mode": editor_mode != "ExtractBox",
            "set_landmarks_mode": editor_mode != "Landmarks",
            "set_mask_mode": editor_mode != "Mask",
            "zoom_in": has_frames,
            "zoom_out": has_frames,
            "reset_view": has_frames,
            "legacy_tool": bool(self._legacy_args),
            "extract_faces": has_frames and self._alignments_handle.exists,
        }
        # When a blocking operation is in flight, disable every mutating /
        # save-conflicting action regardless of the underlying availability so
        # the user cannot race the save (or future bulk job).  Read-only
        # navigation, mode switches and view zoom remain enabled so they don't
        # feel unresponsive while the worker runs.
        if self._busy_operation is not None:
            for key in self._MUTATING_ACTION_KEYS:
                if key in availability:
                    availability[key] = False
        for key, enabled in availability.items():
            action = self._actions.get(key)
            if action is not None:
                action.setEnabled(enabled)

    def _on_editor_mode_changed(self, _mode: object) -> None:
        """Refresh action availability when the editor mode flips."""
        self._sync_actions()
        self._refresh_mask_controls_visibility()
        self._refresh_aligner_controls_visibility()
        mode = str(self._editor_state.editor_mode)
        if mode in {"Landmarks", "Mask"}:
            self._auto_magnify_active_face()
        else:
            self._restore_magnified_view()
        self._frame_view.update()
