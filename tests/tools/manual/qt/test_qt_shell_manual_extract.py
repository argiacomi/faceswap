#!/usr/bin/env python3
# mypy: disable-error-code="misc"
"""Native Qt Manual Tool Extract Faces tests.

Covers:
* #116 — extract uses the live editable model (unsaved adds + deletes).
* #118 — Cancel control is visible during extraction and cancels safely.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QFileDialog

from tools.manual.qt import ManualToolWindow
from tools.manual.session import ManualSession


def _write_real_frame(path: Path, value: int = 90) -> None:
    """Write a small valid PNG with cv2 (Pixmap fixtures aren't decoded by read_image)."""
    image = np.full((96, 96, 3), value, dtype=np.uint8)  # type: ignore[var-annotated]
    assert cv2.imwrite(str(path), image)


def _session_with_real_frames(folder: Path, count: int = 2) -> ManualSession:
    """Write ``count`` cv2-readable PNG fixtures and return a session."""
    for index in range(count):
        _write_real_frame(folder / f"frame_{index:03d}.png", value=80 + index * 20)
    # Use QPixmap fixtures to keep the Qt frame loader happy for the GUI side too.
    for index in range(count):
        pixmap = QPixmap(96, 96)
        pixmap.fill(QColor("#3366ff"))
        # Do *not* overwrite the cv2-written frame — keep both surfaces happy by
        # writing the pixmap copy to a sibling that the Qt frame view can use.
        pixmap.save(str(folder / f"frame_{index:03d}_qt.png"), "PNG")
    return ManualSession.create(frames=str(folder))


def _wait_for_startup(qtbot, window: ManualToolWindow, *, timeout: int = 5000) -> None:
    """Wait until ManualToolWindow's startup worker has fully drained.

    Extract tests write the same ``alignments.fsa`` that startup refresh may
    read when ``_on_startup_completed`` calls ``refresh_faces``.  Starting an
    async save/extract while startup is still finishing can make startup read a
    partially-written compressed file and raise from the Qt event loop.  The
    tests are not about startup/save interleaving, so drain startup first.
    """
    qtbot.waitUntil(lambda: window._startup_worker is None, timeout=timeout)
    qtbot.wait(0)


def _seed_face(window: ManualToolWindow, frame_index: int, bbox: tuple) -> None:
    """Add an editable face with non-degenerate landmarks for AlignedFace."""
    landmarks = tuple((30.0 + i * 0.5, 40.0 + i * 0.4) for i in range(68))
    window.editable_alignments.add_face(frame_index, bbox, landmarks=landmarks)


def test_extract_uses_editable_state_after_unsaved_add(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsaved add appears in extracted output (#116)."""
    session = _session_with_real_frames(tmp_path)
    # Pre-existing PNG fixtures from QPixmap are filtered out by ManualSession
    # when their suffixes don't match — recreate to use just the cv2-written
    # frames so frame_list[0] is frame_000.png.
    for stray in tmp_path.glob("*_qt.png"):
        stray.unlink()
    session = ManualSession.create(frames=str(tmp_path))

    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    _wait_for_startup(qtbot, window)

    output_folder = tmp_path / "out"
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **kw: str(output_folder))
    )

    _seed_face(window, 0, (10.0, 10.0, 60.0, 60.0))
    assert window.editor_state.unsaved is True

    # extract_faces() schedules a save first (because the session is dirty)
    # then chains extract on completion.  Wait for the extracted PNG to land
    # on disk and the dirty flag to clear.
    assert window.extract_faces() is True
    qtbot.waitUntil(
        lambda: (
            (output_folder / "frame_000_0.png").is_file()
            and window.editor_state.unsaved is False
            and window._extract_worker is None
        ),
        timeout=10000,
    )

    # The added face must show up in the output folder.
    assert (output_folder / "frame_000_0.png").is_file()
    # And the alignments file was persisted as a side-effect of the unsaved add.
    assert window.editor_state.unsaved is False


def test_extract_uses_editable_state_after_unsaved_delete(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsaved delete *removes* the face from extracted output (#116)."""
    # Seed an alignments file with one face, persisted to disk.
    _write_real_frame(tmp_path / "frame_000.png", value=80)
    session = ManualSession.create(frames=str(tmp_path))
    window_seed = ManualToolWindow(session)
    qtbot.addWidget(window_seed)
    _wait_for_startup(qtbot, window_seed)
    _seed_face(window_seed, 0, (10.0, 10.0, 60.0, 60.0))
    assert window_seed.save() is True
    qtbot.waitUntil(lambda: window_seed._save_worker is None, timeout=5000)
    window_seed.close()

    # Re-open the session, delete the face in the editable model, do NOT save,
    # then trigger extract.  The on-disk alignments file still has the face;
    # the editable model says it's gone — extract must follow the editable
    # model (per #116) so the output is empty.
    session2 = ManualSession.create(frames=str(tmp_path))
    window = ManualToolWindow(session2)
    qtbot.addWidget(window)
    _wait_for_startup(qtbot, window)
    # Editable model is seeded from the alignments file on startup.
    qtbot.waitUntil(lambda: window.editable_alignments.face_count(0) == 1, timeout=3000)
    window.editable_alignments.delete_face(0, 0)
    # delete_face leaves the frame index "known" but with zero faces — the
    # editable model reports this as an empty target set, which we expect to
    # short-circuit extract_faces() with a status message.
    output_folder = tmp_path / "out_delete"
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **kw: str(output_folder))
    )

    scheduled = window.extract_faces()
    # Either the pre-flight rejected the empty model (returns False) or the
    # worker ran with empty targets and wrote nothing. Both satisfy #116.
    if scheduled:
        qtbot.waitUntil(lambda: window._extract_worker is None, timeout=5000)
    assert not any(output_folder.glob("*.png")) if output_folder.exists() else True


# ---------------------------------------------------------------------------
# #118 — visible Cancel button
# ---------------------------------------------------------------------------


def test_cancel_button_appears_while_extract_runs(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Cancel button becomes visible + enabled while extract is in flight."""
    _write_real_frame(tmp_path / "frame_000.png")
    session = ManualSession.create(frames=str(tmp_path))
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    # ``isVisible()`` is False until the top-level window is on screen, so show
    # it explicitly for the visibility-state assertions below.
    window.show()
    qtbot.waitExposed(window)
    _wait_for_startup(qtbot, window)
    _seed_face(window, 0, (10.0, 10.0, 60.0, 60.0))

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **kw: str(tmp_path / "out"))
    )

    # Save first so extract_faces does not detour through the save→extract
    # chain — this test is about the Cancel button visibility, not chaining.
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    visible_during_run: list[bool] = []
    enabled_during_run: list[bool] = []

    # Patch the worker's start to capture the cancel button state before the
    # event loop drains and the worker finishes.
    real_start = None

    def _capture_then_start(self):
        button = window._extract_cancel_button
        visible_during_run.append(bool(button is not None and button.isVisible()))
        enabled_during_run.append(bool(button is not None and button.isEnabled()))
        real_start(self)  # noqa: F821 - bound below

    from tools.manual import qt as mt

    real_start = mt.ManualExtractFacesWorker.start
    monkeypatch.setattr(
        mt.ManualExtractFacesWorker, "start", lambda self: _capture_then_start(self)
    )

    assert window.extract_faces() is True
    # Drain to completion so teardown runs and the button hides again.
    qtbot.waitUntil(lambda: window._extract_worker is None, timeout=5000)

    assert visible_during_run == [True]
    assert enabled_during_run == [True]
    # After teardown the button is hidden + disabled.
    button = window._extract_cancel_button
    assert button is not None
    assert button.isVisible() is False
    assert button.isEnabled() is False


def test_cancel_extract_returns_false_when_idle(qtbot, tmp_path: Path) -> None:
    """``cancel_extract`` is a no-op when no extraction is running."""
    _write_real_frame(tmp_path / "frame_000.png")
    session = ManualSession.create(frames=str(tmp_path))
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    assert window.cancel_extract() is False


def test_cancel_extract_forwards_to_worker_and_disables_button(
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cancel_extract`` calls ``worker.cancel`` and disables the button immediately."""
    _write_real_frame(tmp_path / "frame_000.png")
    session = ManualSession.create(frames=str(tmp_path))
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    _wait_for_startup(qtbot, window)
    _seed_face(window, 0, (10.0, 10.0, 60.0, 60.0))

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **kw: str(tmp_path / "out"))
    )

    # Save first so extract_faces does not detour through the save→extract chain.
    assert window.save() is True
    qtbot.waitUntil(lambda: window._save_worker is None, timeout=5000)

    cancel_calls: list[int] = []
    from tools.manual import qt as mt

    real_cancel = mt.ManualExtractFacesWorker.cancel

    def _spy_cancel(self) -> None:
        cancel_calls.append(1)
        real_cancel(self)

    monkeypatch.setattr(mt.ManualExtractFacesWorker, "cancel", _spy_cancel)

    # Block start so the worker is alive long enough for cancel_extract to
    # observe the button state.  We capture the worker but never schedule the
    # task, then manually cancel.
    monkeypatch.setattr(mt.ManualExtractFacesWorker, "start", lambda self: None)

    assert window.extract_faces() is True
    assert window._extract_worker is not None
    assert window._extract_cancel_button is not None
    assert window._extract_cancel_button.isEnabled() is True

    assert window.cancel_extract() is True
    assert cancel_calls == [1]
    assert window._extract_cancel_button.isEnabled() is False

    # Clean up so qtbot doesn't complain about a dangling worker thread.
    window._teardown_extract_worker()


# ---------------------------------------------------------------------------
# #119 task 2 — extract worker shutdown contract
# ---------------------------------------------------------------------------


def test_extract_worker_stop_returns_true_when_thread_already_finished(
    qtbot, tmp_path: Path
) -> None:
    """A no-op stop on a worker whose thread already exited returns True."""
    from tools.manual.qt import ManualExtractFacesWorker

    session = _session_with_real_frames(tmp_path, count=1)
    worker = ManualExtractFacesWorker(
        session.alignments_handle(),
        session,
        str(tmp_path / "out"),
    )
    qtbot.addWidget = lambda *_args, **_kwargs: None
    # Quit the QThread before we call ``stop`` — it should report success.
    worker._thread.quit()  # noqa: SLF001
    worker._thread.wait(1000)  # noqa: SLF001
    assert worker._thread.isRunning() is False  # noqa: SLF001
    assert worker.stop() is True


def test_extract_worker_stop_returns_false_when_thread_refuses_to_exit(
    qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker whose thread is stuck mid-frame reports stop() == False."""
    import threading

    from tools.manual.qt import ManualExtractFacesWorker

    session = _session_with_real_frames(tmp_path, count=1)
    release = threading.Event()
    in_run = threading.Event()

    def slow_run(self) -> None:
        in_run.set()
        # Block until the test releases us.  This simulates a frame
        # operation that ignores the cancel poll between frames.
        release.wait(timeout=5.0)
        self.completed.emit(object())

    monkeypatch.setattr("tools.manual.qt._ManualExtractFacesTask.run", slow_run)
    worker = ManualExtractFacesWorker(
        session.alignments_handle(),
        session,
        str(tmp_path / "out"),
    )
    try:
        worker.start()
        assert in_run.wait(timeout=2.0)
        # The run slot is blocked; stop() with a small timeout must return False.
        assert worker.stop(wait_ms=100) is False
        # ``cancel()`` was called as part of ``stop()`` before ``quit``.
        assert worker._task.is_cancelled() is True  # noqa: SLF001
    finally:
        release.set()
        worker._thread.wait(2000)  # noqa: SLF001


def test_close_event_ignores_close_while_extract_thread_alive(
    qtbot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Closing the window during a stuck extract is rejected; ref stays alive."""
    import threading

    from PySide6.QtGui import QCloseEvent

    session = _session_with_real_frames(tmp_path, count=1)
    release = threading.Event()
    in_run = threading.Event()

    def slow_run(self) -> None:
        in_run.set()
        release.wait(timeout=5.0)
        self.completed.emit(object())

    monkeypatch.setattr("tools.manual.qt._ManualExtractFacesTask.run", slow_run)
    # Shorten the worker's stop wait so the test doesn't sit for 3 s.
    from tools.manual.qt import ManualExtractFacesWorker

    monkeypatch.setattr(ManualExtractFacesWorker, "_STOP_WAIT_MS", 100)

    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    _wait_for_startup(qtbot, window)

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *_a, **_k: str(tmp_path / "out"))
    )
    _seed_face(window, 0, (20.0, 20.0, 50.0, 50.0))
    assert window.extract_faces() is True
    qtbot.waitUntil(lambda: in_run.is_set(), timeout=2000)

    # Simulate the user closing the window while extraction is stuck.
    event = QCloseEvent()
    window.closeEvent(event)
    try:
        assert event.isAccepted() is False
        # The reference is preserved so the QThread destructor does not
        # fire on a still-running thread.
        assert window._extract_worker is not None  # noqa: SLF001
    finally:
        # Let the worker finish so cleanup can complete.
        release.set()
        worker = window._extract_worker  # noqa: SLF001
        if worker is not None:
            worker._thread.wait(2000)  # noqa: SLF001
