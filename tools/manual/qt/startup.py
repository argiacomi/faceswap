#!/usr/bin/env python3
"""Qt Manual Tool session startup and video bootstrap helpers."""

from __future__ import annotations

import logging

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QMessageBox, QProgressBar

from tools.manual.session import ManualFrame

from .video import VideoFrameProvider
from .workers import ManualStartupWorker, _ManualStartupTask

logger = logging.getLogger(__name__)


class StartupMixin:
    """Own session loading, startup worker callbacks and video provider setup."""

    def _load_session(self) -> None:
        """Populate the Qt Manual Tool from a neutral session."""
        if self._session.has_images:  # type: ignore[attr-defined]
            frame_summary = str(self._session.frame_count)  # type: ignore[attr-defined]
        elif self._session.is_video_input:  # type: ignore[attr-defined]
            frame_summary = "video input (loading…)"
        else:
            frame_summary = "video input"
        thumbs_state = "regenerate forced" if self._session.thumb_regenerate else "loading…"  # type: ignore[attr-defined]
        self._metadata_label.setText(  # type: ignore[attr-defined]
            "\n".join(
                (
                    f"Input: {self._session.frames}",  # type: ignore[attr-defined]
                    f"Alignments: {self._alignments_handle.path}",  # type: ignore[attr-defined]
                    f"Frames: {frame_summary}",
                    f"Thumbnails: {thumbs_state}",
                )
            )
        )
        try:
            if self._session.has_images:  # type: ignore[attr-defined]
                self._editable.seed_from_handle(  # type: ignore[attr-defined]
                    self._alignments_handle,  # type: ignore[attr-defined]
                    frame_names=[frame.name for frame in self._session.frame_list],  # type: ignore[attr-defined]
                )
            else:
                self._editable.seed_from_handle(self._alignments_handle)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - re-surfaced by the startup worker
            logger.exception("Manual Tool synchronous seed failed")
        if self._session.has_images:  # type: ignore[attr-defined]
            self._thumbnail_panel.set_frames(self._session.frame_list)  # type: ignore[attr-defined]
            self._refresh_filter_results(preserve_current=False)  # type: ignore[attr-defined]
            self._thumbnail_panel.setCurrentRow(0)  # type: ignore[attr-defined]
        elif self._session.is_video_input:  # type: ignore[attr-defined]
            try:
                self._video_metadata = self._alignments_handle.video_metadata()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - re-surfaced by the startup worker
                logger.exception("Manual Tool video metadata load failed")
                self._video_metadata = None
            self._start_video_provider()
        else:
            self._frame_view.clear_frame(  # type: ignore[attr-defined]
                "Video input detected. Frame extraction will be wired in a follow-up."
            )
        self._status_label.setText("Manual Tool starting…")  # type: ignore[attr-defined]
        self._sync_actions()  # type: ignore[attr-defined]

    def _start_background_startup(self) -> None:
        """Kick off ``ManualStartupWorker`` for async alignments preparation."""
        self._thumb_progress_seen = False
        self._progress_bar = self._build_progress_bar()
        if self._progress_bar is not None:
            self.statusBar().addPermanentWidget(self._progress_bar)  # type: ignore[attr-defined]
            self._progress_bar.show()
        self._startup_worker = ManualStartupWorker(
            self._alignments_handle,  # type: ignore[attr-defined]
            self._editable,  # type: ignore[attr-defined]
            self._session,  # type: ignore[attr-defined]
            parent=self,  # type: ignore[arg-type]
        )
        self._startup_worker.progress.connect(self._on_startup_progress)
        self._startup_worker.progress_percent.connect(self._on_startup_progress_percent)
        self._startup_worker.completed.connect(self._on_startup_completed)
        self._startup_worker.failed.connect(self._on_startup_failed)
        logger.info("Manual Tool startup worker scheduled")
        self._emit_console("Manual Tool: preparing session…")  # type: ignore[attr-defined]
        self._startup_worker.start()

    def _build_progress_bar(self) -> QProgressBar:
        """Return a determinate progress bar for the status bar."""
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
        """Surface intermediate startup messages to status and console."""
        logger.debug("Manual Tool startup [%s]: %s", stage, message)
        self.statusBar().showMessage(message)  # type: ignore[attr-defined]
        self._emit_console(f"Manual Tool [{stage}]: {message}")  # type: ignore[attr-defined]
        percent = _ManualStartupTask.STAGE_PERCENT.get(stage)
        if (
            percent is not None
            and self._progress_bar is not None
            and not (stage == "thumbs" and self._thumb_progress_seen)
        ):
            self._progress_bar.setValue(percent)

    def _on_startup_progress_percent(self, percent: int, _message: str) -> None:
        """Paint the determinate progress bar from per-event thumbnail progress."""
        self._thumb_progress_seen = True
        if self._progress_bar is None:
            return
        self._progress_bar.setValue(max(0, min(100, int(percent))))

    def _on_startup_completed(self, has_thumbnails: bool, summary: str) -> None:
        """Finalize UI state once the startup worker reports success."""
        self._startup_complete = True
        thumbs_state = (
            "regenerate forced"
            if self._session.thumb_regenerate  # type: ignore[attr-defined]
            else ("cached" if has_thumbnails else "needs generation")
        )
        self._metadata_label.setText(  # type: ignore[attr-defined]
            "\n".join(
                (
                    f"Input: {self._session.frames}",  # type: ignore[attr-defined]
                    f"Alignments: {self._alignments_handle.path}",  # type: ignore[attr-defined]
                    f"Frames: {self._frame_summary_text()}",
                    f"Thumbnails: {thumbs_state}",
                )
            )
        )
        if self._session.is_video_input:  # type: ignore[attr-defined]
            self._video_metadata = self._alignments_handle.video_metadata()  # type: ignore[attr-defined]
        self._status_label.setText("Manual Tool ready")  # type: ignore[attr-defined]
        self.statusBar().showMessage(summary, 5000)  # type: ignore[attr-defined]
        self._hide_progress_bar()  # type: ignore[attr-defined]
        logger.info("Manual Tool startup complete: %s", summary)
        self._emit_console(f"Manual Tool ready: {summary}")  # type: ignore[attr-defined]
        self.refresh_faces()  # type: ignore[attr-defined]
        self._refresh_face_grid()  # type: ignore[attr-defined]
        self._frame_view.update()  # type: ignore[attr-defined]
        self._sync_actions()  # type: ignore[attr-defined]
        self._teardown_startup_worker()

    def _on_startup_failed(self, message: str) -> None:
        """Surface startup failures through status, console, log and dialog."""
        self._startup_complete = False
        self._status_label.setText("Manual Tool startup failed")  # type: ignore[attr-defined]
        self.statusBar().showMessage(message, 7000)  # type: ignore[attr-defined]
        self._hide_progress_bar()  # type: ignore[attr-defined]
        logger.error("Manual Tool startup failed: %s", message)
        self._emit_console(f"Manual Tool startup failed: {message}")  # type: ignore[attr-defined]
        QMessageBox.critical(self, "Manual Tool Startup", message)
        self._teardown_startup_worker()

    def _teardown_startup_worker(self) -> None:
        """Drain and drop the startup worker after its terminal signal fires."""
        worker = self._startup_worker
        if worker is None:
            return
        worker.stop()
        worker.deleteLater()
        self._startup_worker = None  # type: ignore[assignment]

    def _frame_summary_text(self) -> str:
        """Return the metadata summary text for the loaded frame source."""
        if self._session.has_images:  # type: ignore[attr-defined]
            return str(self._session.frame_count)  # type: ignore[attr-defined]
        if self._video_metadata is not None and self._video_metadata.is_valid:
            return f"{self._video_metadata.frame_count} (video)"
        return "video input"

    def _start_video_provider(self) -> None:
        """Initialize the async video frame provider for video sessions."""
        self._frame_view.clear_frame("Loading video frames…")  # type: ignore[attr-defined]
        if self._video_metadata is not None and self._video_metadata.is_valid:
            meta_dict = {
                "pts_time": list(self._video_metadata.pts_time),
                "keyframes": list(self._video_metadata.keyframes),
            }
        else:
            meta_dict = None
        self._video_provider = VideoFrameProvider(
            self._session.frames,  # type: ignore[attr-defined]
            video_meta_data=meta_dict,
            parent=self,  # type: ignore[arg-type]
        )
        self._video_provider.count_ready.connect(self._on_video_count_ready)
        self._video_provider.frame_ready.connect(self._on_video_frame_ready)
        self._video_provider.load_failed.connect(self._on_video_load_failed)
        self._video_provider.start()

    def _on_video_count_ready(self, count: int) -> None:
        """Populate the thumbnail list with video frame placeholders."""
        if count <= 0:
            self._frame_view.clear_frame("Video reported zero frames")  # type: ignore[attr-defined]
            return
        self._video_frames = [
            ManualFrame(index=idx, name=f"frame_{idx:06d}", path=self._session.frames)  # type: ignore[attr-defined]
            for idx in range(count)
        ]
        self._thumbnail_panel.set_frames(tuple(self._video_frames))  # type: ignore[attr-defined]
        self._refresh_filter_results(preserve_current=False)  # type: ignore[attr-defined]
        self._thumbnail_panel.setCurrentRow(0)  # type: ignore[attr-defined]

    def _on_video_frame_ready(self, index: int, filename: str, image: QImage) -> None:
        """Display a video frame that was decoded on the worker thread."""
        if (
            self._current_frame is not None  # type: ignore[attr-defined]
            and self._current_frame.index == index  # type: ignore[attr-defined]
            and self._session.is_video_input  # type: ignore[attr-defined]
        ):
            self._frame_view.set_image(image, self._current_frame)  # type: ignore[attr-defined]
            self._status_label.setText(f"Frame {index + 1}: {filename}")  # type: ignore[attr-defined]

    def _on_video_load_failed(self, index: int, message: str) -> None:
        """Surface a video load failure in the status label and console."""
        logger.warning("Manual Tool video frame %s failed: %s", index, message)
        self._status_label.setText(f"Failed to load frame {index}: {message}")  # type: ignore[attr-defined]


__all__ = ["StartupMixin"]
