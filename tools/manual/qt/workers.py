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

from tools.manual.session import (
    ManualAlignmentsHandle,
    ManualEditableAlignments,
    ManualSession,
)

logger = logging.getLogger(__name__)


class _ManualStartupTask(QObject):
    """Worker that performs Manual Tool session preparation off the UI thread.

    Stages reported via :attr:`progress`:
        * ``"open"``      – Open the alignments file (parses on first access).
        * ``"seed"``      – Seed the editable model from the alignments file.
        * ``"thumbs"``    – Check the thumbnail cache state.
        * ``"complete"``  – All stages finished.

    The worker is intentionally tiny and synchronous inside :meth:`run` — Qt
    moves it to a dedicated QThread, so blocking IO inside this object does
    not freeze the window.
    """

    progress = Signal(str, str)
    """Emitted with ``(stage, message)`` as each named stage starts."""
    progress_percent = Signal(int, str)
    """Emitted with ``(percent, message)`` for live sub-stage progress (eg.
    per-frame thumbnail regeneration).  Each event carries its own immutable
    percent value so queued signals always paint the correct bar position."""
    completed = Signal(bool, str)
    """Emitted with ``(has_thumbnails, summary)`` on success."""
    failed = Signal(str)
    """Emitted with a user-facing error message on failure."""
    _start = Signal()

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
        session: ManualSession,
    ) -> None:
        super().__init__()
        self._handle = handle
        self._editable = editable
        self._session = session
        self._start.connect(self.run)

    def kick_off(self) -> None:
        """Schedule the worker on its owning thread."""
        self._start.emit()

    STAGE_PERCENT: T.ClassVar[dict[str, int]] = {
        "open": 33,
        "thumbs": 66,
        "complete": 100,
    }
    """Static stage anchors used by :meth:`progress` events.

    Live thumbnail-regeneration percentages are *not* stored here — they're
    delivered through the dedicated :attr:`progress_percent` signal so each
    event carries its own immutable percent and cannot bleed between
    Manual Tool windows (see issue #114).
    """

    def run(self) -> None:  # noqa: D401 - Qt slot
        """Execute the staged startup work."""
        try:
            if not self._handle.exists:
                self.progress.emit("complete", "No alignments file yet — ready")
                self.completed.emit(False, "No alignments file yet")
                return

            self.progress.emit("open", "Opening alignments file…")
            alignments = self._handle.open()
            face_count = sum(len(entry.faces) for entry in alignments.data.values())

            self.progress.emit("thumbs", "Checking thumbnail cache…")
            has_thumbnails = self._handle.has_thumbnails()
            needs_regen = self._session.thumb_regenerate or not has_thumbnails
            if needs_regen and face_count:
                regenerated = self._handle.regenerate_thumbnails(
                    self._session, progress=self._emit_thumb_progress
                )
                if regenerated:
                    has_thumbnails = True
                    logger.info("Manual Tool generated thumbnails for %d frame(s)", regenerated)

            summary = f"Loaded {face_count} face(s) across {len(alignments.data)} frame(s)"
            self.progress.emit("complete", summary)
            self.completed.emit(has_thumbnails, summary)
        except Exception as err:  # noqa: BLE001 - surface any startup failure
            logger.exception("Manual Tool startup failed")
            self.failed.emit(str(err))

    def _emit_thumb_progress(self, done: int, total: int, message: str) -> None:
        """Bridge :mod:`thumbnail_generation` progress to immutable per-event percent.

        Maps the (done, total) pair into the 66 → 99 % band of the startup bar
        and emits it on the dedicated :attr:`progress_percent` signal.  Also
        fires the named ``progress`` signal so console fanout and status-bar
        message paths continue to receive each step.
        """
        denominator = max(1, total)
        percent = min(99, 66 + round((done / denominator) * 33))
        self.progress_percent.emit(int(percent), message)
        self.progress.emit("thumbs", message)


class ManualStartupWorker(QObject):
    """Owns the QThread that runs :class:`_ManualStartupTask`.

    The window subscribes to :attr:`progress`, :attr:`completed` and
    :attr:`failed` for user-facing feedback.  Calling :meth:`stop` is safe at
    any time — it gracefully terminates the thread even if the user closes
    the window mid-load.
    """

    progress = Signal(str, str)
    progress_percent = Signal(int, str)
    completed = Signal(bool, str)
    failed = Signal(str)

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
        session: ManualSession,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._task = _ManualStartupTask(handle, editable, session)
        self._task.moveToThread(self._thread)
        self._task.progress.connect(self.progress)
        self._task.progress_percent.connect(self.progress_percent)
        self._task.completed.connect(self.completed)
        self._task.failed.connect(self.failed)
        # Auto-quit the QThread as soon as the task reports a terminal
        # state so the thread is fully stopped before the parent window
        # destructor runs.  Without this, Qt aborts on the QThread
        # destructor under pytest teardown when stop() was not called.
        self._task.completed.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._task.deleteLater)
        self._thread.start()

    def start(self) -> None:
        """Begin background processing on the worker thread."""
        self._task.kick_off()

    def stop(self) -> None:
        """Stop the worker thread and wait for it to exit."""
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)


class _ManualExtractFacesTask(QObject):
    """Worker that runs Tk-parity Extract Faces off the UI thread.

    Construct with the alignments handle, the live session, and the user-
    picked output folder; connect to :attr:`progress` / :attr:`completed` /
    :attr:`failed`; then call :meth:`kick_off`.  ``cancel`` is thread-safe
    and surfaces to the next ``is_cancelled()`` poll between frames.
    """

    progress = Signal(int, int, str)
    """``(done, total, message)`` — total is the number of source frames."""
    completed = Signal(object)
    """Emits the :class:`ExtractFacesResult` from a finished run."""
    failed = Signal(str)
    """Emits a user-facing error string when extraction raises."""
    _start = Signal()

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        session: ManualSession,
        output_folder: str,
        editable_targets: T.Any = None,
    ) -> None:
        super().__init__()
        self._handle = handle
        self._session = session
        self._output_folder = output_folder
        # Editable snapshot — when supplied, ``extract_faces`` uses it instead
        # of the persisted alignments file so unsaved Manual Tool edits are
        # included in the extracted PNG output (parity with Tk Manual which
        # extracts from its live in-memory face list).
        self._editable_targets = editable_targets
        self._cancelled = False
        self._start.connect(self.run)

    def kick_off(self) -> None:
        """Schedule :meth:`run` on this object's owning thread."""
        self._start.emit()

    def cancel(self) -> None:
        """Request cancellation between frames.  Safe to call from any thread."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Return whether :meth:`cancel` has been called."""
        return self._cancelled

    def run(self) -> None:  # noqa: D401 - Qt slot
        """Execute the extraction and emit terminal signals."""
        try:
            from tools.manual.face_extraction import (
                FaceExtractionRequest,
                extract_faces,
            )

            request = FaceExtractionRequest(
                handle=self._handle,
                session=self._session,
                output_folder=self._output_folder,
                editable_targets=self._editable_targets,
            )
            result = extract_faces(
                request,
                progress=lambda done, total, message: self.progress.emit(done, total, message),
                is_cancelled=self.is_cancelled,
            )
        except Exception as err:  # noqa: BLE001 - surface any failure
            logger.exception("Manual Tool extract faces failed")
            self.failed.emit(str(err))
            return
        self.completed.emit(result)


class ManualExtractFacesWorker(QObject):
    """Owns the QThread that drives :class:`_ManualExtractFacesTask`."""

    progress = Signal(int, int, str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        session: ManualSession,
        output_folder: str,
        editable_targets: T.Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._task = _ManualExtractFacesTask(
            handle, session, output_folder, editable_targets=editable_targets
        )
        self._task.moveToThread(self._thread)
        self._task.progress.connect(self.progress)
        self._task.completed.connect(self.completed)
        self._task.failed.connect(self.failed)
        self._task.completed.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._task.deleteLater)
        self._thread.start()

    def start(self) -> None:
        """Begin background extraction on the worker thread."""
        self._task.kick_off()

    def cancel(self) -> None:
        """Forward a cancel request to the underlying task."""
        self._task.cancel()

    _STOP_WAIT_MS = 3000
    """How long ``stop()`` waits for the QThread to exit before giving up."""

    def stop(self, *, wait_ms: int | None = None) -> bool:
        """Try to shut down the worker thread; report whether it exited.

        Cancellation is polled by ``_ManualExtractFacesTask`` between
        frames, so a long-running per-frame operation (image read, mask
        encode, alignment) may not finish within the wait window.  Returns
        ``True`` iff the QThread has actually exited by the deadline; the
        caller can then drop its reference safely.  ``False`` means the
        worker is still in-flight and must NOT have its reference cleared
        — see :meth:`ManualToolWindow.closeEvent` for the reentrant-safe
        cleanup path (issue #119 task 2).
        """
        if not self._thread.isRunning():
            return True
        self.cancel()
        self._thread.quit()
        timeout = self._STOP_WAIT_MS if wait_ms is None else int(wait_ms)
        # ``QThread.wait`` returns True iff the thread terminated within
        # the timeout — that's our return contract.
        return bool(self._thread.wait(timeout))


class _ManualAlignerLoadTask(QObject):
    """Worker task that preloads one aligner backend off the UI thread."""

    status = Signal(str, str, str)
    """``(kind, aligner, message)`` for visible load-progress updates."""
    completed = Signal(str, str, bool, str)
    """``(aligner, normalization, ok, message)`` once preload finishes."""
    _start = Signal()

    def __init__(self, service: T.Any, aligner: str, normalization: str) -> None:
        super().__init__()
        self._service = service
        self._aligner = str(aligner)
        self._normalization = str(normalization)
        self._start.connect(self.run)

    def kick_off(self) -> None:
        """Schedule :meth:`run` on this object's owning thread."""
        self._start.emit()

    def run(self) -> None:  # noqa: D401 - Qt slot
        """Preload the requested aligner backend and emit terminal status."""
        loading = f"Loading aligner '{self._aligner}'…"
        self.status.emit("loading", self._aligner, loading)
        try:
            # Keep GUI updates on Qt signals by avoiding ManualAlignerService.preload(),
            # whose status callback may call directly from this worker thread.
            # The private method is part of the same service boundary this Qt
            # host already depends on for cached backend lifetime.
            get_backend = getattr(self._service, "_get_backend", None)
            if callable(get_backend):
                get_backend(self._aligner, self._normalization)
            else:  # pragma: no cover - compatibility fallback for non-standard stubs
                ok = bool(self._service.preload(self._aligner, self._normalization))
                if not ok:
                    raise RuntimeError(f"Aligner '{self._aligner}' failed to load")
        except Exception as err:  # noqa: BLE001 - surface to the host
            logger.exception("Manual Tool: aligner preload failed (%s)", self._aligner)
            message = f"Aligner '{self._aligner}' failed to load: {err}"
            self.status.emit("failed", self._aligner, message)
            self.completed.emit(self._aligner, self._normalization, False, message)
            return
        message = f"Aligner '{self._aligner}' ready"
        self.status.emit("ready", self._aligner, message)
        self.completed.emit(self._aligner, self._normalization, True, message)


class ManualAlignerLoadWorker(QObject):
    """Owns the QThread that preloads an aligner backend."""

    status = Signal(str, str, str)
    completed = Signal(str, str, bool, str)

    def __init__(
        self,
        service: T.Any,
        aligner: str,
        normalization: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._task = _ManualAlignerLoadTask(service, aligner, normalization)
        self._task.moveToThread(self._thread)
        self._task.status.connect(self.status)
        self._task.completed.connect(self.completed)
        self._task.completed.connect(self._thread.quit)
        self._thread.finished.connect(self._task.deleteLater)
        self._thread.start()

    def start(self) -> None:
        """Begin background aligner preload."""
        self._task.kick_off()

    def stop(self, *, wait_ms: int = 1000) -> bool:
        """Stop the loader thread and report whether it exited."""
        if not self._thread.isRunning():
            return True
        self._thread.quit()
        return bool(self._thread.wait(int(wait_ms)))


class _ManualSaveTask(QObject):
    """Worker that runs alignment persistence off the UI thread.

    Mirrors :class:`_ManualExtractFacesTask`: the host installs progress /
    completion / failure signal handlers and calls :meth:`kick_off`.  The
    task is single-shot — re-entry is prevented by ``_save_in_flight`` on
    the host before this is ever constructed.

    The editable model and frame-name resolver are captured at schedule time
    by the host; mutating actions are gated through ``_busy_operation`` so
    no caller thread can race the persist with concurrent edits.
    """

    completed = Signal(int)
    """Emits the number of frames modified on a successful save."""
    failed = Signal(str)
    """Emits a user-facing error string when persistence raises."""
    _start = Signal()

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
        frame_names: T.Any,
    ) -> None:
        super().__init__()
        self._handle = handle
        self._editable = editable
        self._frame_names = frame_names
        self._start.connect(self.run)

    def kick_off(self) -> None:
        """Schedule :meth:`run` on this object's owning thread."""
        self._start.emit()

    def run(self) -> None:  # noqa: D401 - Qt slot
        """Execute the persist call and emit the terminal signal."""
        try:
            modified = self._handle.persist(self._editable, frame_names=self._frame_names)
        except Exception as err:  # noqa: BLE001 - surface any persist failure
            logger.exception("Manual Tool save worker failed")
            self.failed.emit(str(err))
            return
        self.completed.emit(int(modified))


class ManualSaveWorker(QObject):
    """Owns the QThread that drives :class:`_ManualSaveTask`.

    Save is short-lived and uncancellable — the worker exists purely to keep
    persistence off the UI thread so the busy progress bar and disabled
    mutating-action state can repaint before ``Alignments.save`` blocks.
    """

    completed = Signal(int)
    failed = Signal(str)

    def __init__(
        self,
        handle: ManualAlignmentsHandle,
        editable: ManualEditableAlignments,
        frame_names: T.Any,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._thread = QThread(self)
        self._task = _ManualSaveTask(handle, editable, frame_names)
        self._task.moveToThread(self._thread)
        self._task.completed.connect(self.completed)
        self._task.failed.connect(self.failed)
        self._task.completed.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._task.deleteLater)
        self._thread.start()

    def start(self) -> None:
        """Begin background persistence on the worker thread."""
        self._task.kick_off()

    def stop(self) -> None:
        """Stop the worker thread (after any in-flight persist completes)."""
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
