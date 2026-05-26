#!/usr/bin/env python3
"""Qt Manual Tool implementation module."""

from __future__ import annotations

import logging
import typing as T

from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
)
from PySide6.QtGui import (
    QImage,
)

logger = logging.getLogger(__name__)


class _VideoFrameWorker(QObject):
    """Worker that lives on a QThread and serves video frames on demand."""

    count_ready = Signal(int)
    frame_ready = Signal(int, str, QImage)
    load_failed = Signal(int, str)

    def __init__(
        self,
        path: str,
        video_meta_data: dict[str, list[int]] | None,
    ) -> None:
        super().__init__()
        self._path = path
        self._video_meta_data = video_meta_data
        self._loader: T.Any | None = None

    def initialize(self) -> None:
        """Open the underlying SingleFrameLoader and emit the frame count."""
        from lib.image import SingleFrameLoader

        try:
            self._loader = SingleFrameLoader(self._path, video_meta_data=self._video_meta_data)
        except Exception as err:  # pragma: no cover - exercised through integration
            self.load_failed.emit(-1, f"Could not open video: {err}")
            return
        self.count_ready.emit(int(self._loader.count))

    def fetch(self, index: int) -> None:
        """Decode and emit the requested frame index as a QImage."""
        if self._loader is None:
            self.load_failed.emit(index, "Video frame loader not initialized")
            return
        try:
            filename, frame = self._loader.image_from_index(index)
        except Exception as err:  # pragma: no cover - exercised through integration
            self.load_failed.emit(index, str(err))
            return
        image = _bgr_array_to_qimage(frame)
        if image.isNull():
            self.load_failed.emit(index, "Decoded frame produced an empty image")
            return
        self.frame_ready.emit(index, filename, image)


def _bgr_array_to_qimage(frame: T.Any) -> QImage:
    """Convert a numpy BGR uint8 array (from SingleFrameLoader) to a QImage."""
    if frame is None or frame.size == 0:
        return QImage()
    height, width = frame.shape[:2]
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    if channels == 1:
        image = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
    elif channels == 3:
        # SingleFrameLoader returns BGR; Qt expects RGB, so swap into Format_BGR888.
        image = QImage(frame.data, width, height, width * 3, QImage.Format_BGR888)
    else:
        # 4-channel BGRA fallback.
        image = QImage(frame.data, width, height, width * 4, QImage.Format_ARGB32)
    # The numpy buffer is owned by the caller; copy to detach.
    return image.copy()


class VideoFrameProvider(QObject):
    """Async video frame provider for the native Qt Manual Tool.

    The provider owns a worker QObject moved onto a private QThread.  Consumers
    call :meth:`request_frame` and listen on :attr:`frame_ready` (or
    :attr:`load_failed`) for results, keeping the UI thread responsive while
    individual frames decode.
    """

    count_ready = Signal(int)
    frame_ready = Signal(int, str, QImage)
    load_failed = Signal(int, str)
    _request = Signal(int)
    _start_init = Signal()

    def __init__(
        self,
        path: str,
        video_meta_data: dict[str, list[int]] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._worker = _VideoFrameWorker(path, video_meta_data)
        self._worker.moveToThread(self._thread)
        self._worker.count_ready.connect(self.count_ready)
        self._worker.frame_ready.connect(self.frame_ready)
        self._worker.load_failed.connect(self.load_failed)
        self._start_init.connect(self._worker.initialize)
        self._request.connect(self._worker.fetch)
        self._thread.start()

    def start(self) -> None:
        """Open the video on the worker thread."""
        self._start_init.emit()

    def request_frame(self, index: int) -> None:
        """Ask the worker thread to decode and emit the given frame index."""
        self._request.emit(int(index))

    def shutdown(self) -> None:
        """Stop the worker thread and release resources."""
        if not self._thread.isRunning():
            return
        self._thread.quit()
        self._thread.wait(2000)
