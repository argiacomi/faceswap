#!/usr/bin/env python3
"""Native Qt Manual Tool startup feedback tests.

Covers the async startup worker (progress + completion + failure), the
status-bar progress indicator, the QMessageBox failure dialog and the
host-window console-logger fanout for runtime messages.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PySide6.QtGui import QColor, QPixmap

from lib.gui.qt_shell.manual_tool import (
    ManualStartupWorker,
    ManualToolWindow,
    _ManualStartupTask,
)
from tools.manual.session import (
    ManualAlignmentsHandle,
    ManualEditableAlignments,
    ManualSession,
)


def _session_with_frames(folder: Path, count: int = 2) -> ManualSession:
    """Write ``count`` small PNG fixtures and return a session."""
    for index in range(count):
        path = folder / f"frame_{index:03d}.png"
        pixmap = QPixmap(48, 32)
        pixmap.fill(QColor("#3366ff"))
        assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _drain_qt_events(qtbot, *, timeout_ms: int = 1500) -> None:  # type:ignore[no-untyped-def]
    """Pump Qt events until the predicate fires or timeout — used to await signals."""
    qtbot.wait(timeout_ms)


def test_startup_task_completes_quickly_for_missing_alignments(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """A session with no on-disk alignments file still completes startup cleanly."""
    session = _session_with_frames(tmp_path)
    handle = session.alignments_handle()
    editable = ManualEditableAlignments()
    task = _ManualStartupTask(handle, editable, session)
    completed: list[tuple[bool, str]] = []
    failed: list[str] = []
    progress: list[tuple[str, str]] = []
    task.completed.connect(lambda has_thumbs, msg: completed.append((has_thumbs, msg)))
    task.failed.connect(failed.append)
    task.progress.connect(lambda stage, msg: progress.append((stage, msg)))

    task.run()

    assert failed == []
    assert completed == [(False, "No alignments file yet")]
    assert progress == [("complete", "No alignments file yet — ready")]


def test_startup_task_open_and_thumb_stages_complete(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """The worker walks open + thumbs stages and reports completion.

    Seeding the editable model is performed synchronously by
    :class:`ManualToolWindow` itself before the worker starts (so the first
    frame view shows persisted faces immediately); the worker is responsible
    for the alignments-file parse and the thumbnail cache check.
    """
    from lib.align import Alignments

    session = _session_with_frames(tmp_path)
    handle = session.alignments_handle()
    seed = ManualEditableAlignments()
    seed.add_face(0, (10.0, 12.0, 30.0, 30.0))
    handle.persist(seed, frame_names=[f.name for f in session.frame_list])
    assert handle.exists

    fresh_handle = ManualAlignmentsHandle(handle.folder, handle.filename, is_video=False)
    editable = ManualEditableAlignments()
    task = _ManualStartupTask(fresh_handle, editable, session)
    completed: list[tuple[bool, str]] = []
    stages: list[str] = []
    task.completed.connect(lambda has_thumbs, msg: completed.append((has_thumbs, msg)))
    task.progress.connect(lambda stage, _msg: stages.append(stage))

    task.run()

    assert completed, "completed signal must fire"
    assert "open" in stages
    assert "thumbs" in stages
    assert "complete" in stages
    # Sanity-check Alignments file integrity after the round trip.
    reloaded = Alignments(handle.folder, handle.filename)
    assert "frame_000.png" in reloaded.data


def test_manual_tool_window_seeds_editable_model_synchronously(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """Opening an existing alignments file populates the editable model up-front."""
    session = _session_with_frames(tmp_path)
    handle = session.alignments_handle()
    seed = ManualEditableAlignments()
    seed.add_face(0, (15.0, 20.0, 32.0, 32.0))
    handle.persist(seed, frame_names=[f.name for f in session.frame_list])

    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    # The synchronous seed must run before the first frame selection so the
    # face panel + overlay reflect the persisted faces immediately.
    assert window.editable_alignments.face_count(0) == 1
    assert window.editable_alignments.faces(0)[0].bbox == (15.0, 20.0, 32.0, 32.0)
    assert len(window.face_panel.faces) == 1


def test_manual_tool_window_seeds_sparse_alignments_to_filename_index(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """An alignments file with only ``frame_002.png`` seeds onto frame index 2."""
    session = _session_with_frames(tmp_path, count=5)
    handle = session.alignments_handle()
    # Persist a face onto the third frame only (frame_002.png).
    seed = ManualEditableAlignments()
    seed.add_face(2, (10.0, 10.0, 30.0, 30.0))
    handle.persist(seed, frame_names=[f.name for f in session.frame_list])

    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    # The face must attach to frame_index 2 (which corresponds to
    # ``frame_002.png`` in the source frame list) not 0.
    assert window.editable_alignments.face_count(0) == 0
    assert window.editable_alignments.face_count(2) == 1
    assert window.editable_alignments.faces(2)[0].bbox == (10.0, 10.0, 30.0, 30.0)


def test_startup_task_emits_failed_signal_on_exception(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the underlying open() raises, the worker reports a clear error."""
    session = _session_with_frames(tmp_path)
    handle = session.alignments_handle()
    editable = ManualEditableAlignments()

    # Force exists to True so the worker reaches the open() call we patched.
    monkeypatch.setattr(type(handle), "exists", property(lambda _self: True))

    def _boom(_self: object) -> object:
        raise RuntimeError("simulated open() failure")

    monkeypatch.setattr(type(handle), "open", _boom)
    task = _ManualStartupTask(handle, editable, session)
    completed: list[tuple[bool, str]] = []
    failed: list[str] = []
    task.completed.connect(lambda *args: completed.append(args))
    task.failed.connect(failed.append)

    task.run()

    assert completed == []
    assert failed and "simulated open()" in failed[0]


def test_manual_tool_window_runs_startup_worker(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """ManualToolWindow constructs and drives a ManualStartupWorker."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    assert window._startup_worker is not None  # noqa: SLF001
    # Wait for the worker to complete; the missing-alignments path is fast.
    qtbot.waitUntil(lambda: window._startup_complete, timeout=2000)  # noqa: SLF001
    assert "ready" in window._status_label.text().lower()  # noqa: SLF001


def test_manual_tool_window_pipes_messages_to_console_logger(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
) -> None:
    """A console_logger receives startup progress + ready messages."""
    session = _session_with_frames(tmp_path)
    captured: list[str] = []
    window = ManualToolWindow(session, console_logger=captured.append)
    qtbot.addWidget(window)

    qtbot.waitUntil(lambda: window._startup_complete, timeout=2000)  # noqa: SLF001

    assert any("preparing session" in line for line in captured)
    assert any("Manual Tool [" in line for line in captured)
    assert any("ready" in line.lower() for line in captured)


def test_manual_tool_window_shows_failure_dialog_on_startup_error(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A startup failure ends in an error status, log entry and console line."""
    session = _session_with_frames(tmp_path)
    captured: list[str] = []
    caplog.set_level(logging.ERROR, logger="lib.gui.qt_shell.manual_tool")
    window = ManualToolWindow(session, console_logger=captured.append)
    qtbot.addWidget(window)
    # Force the worker into the failure path. Triggering the slot directly is
    # the deterministic way to drive the UI feedback without racing the
    # background thread.
    window._on_startup_failed("boom")  # noqa: SLF001

    assert "boom" in window.statusBar().currentMessage()
    assert "failed" in window._status_label.text().lower()  # noqa: SLF001
    assert any("boom" in line for line in captured)
    assert any("boom" in record.message for record in caplog.records)


def test_video_metadata_loaded_before_video_provider_starts(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video sessions must populate _video_metadata before the provider starts.

    Builds a minimal video-input session via monkeypatching so the test does
    not require a real ffmpeg-decodable file, then asserts that
    ``_start_video_provider`` runs only after ``self._video_metadata`` has
    been assigned.
    """
    from tools.manual.session import ManualSession, ManualVideoMetadata

    fake_video = tmp_path / "input.mp4"
    fake_video.write_bytes(b"\x00")

    # Build a session whose ``is_video_input`` flag is True without
    # exercising the real video detection path (which requires ffmpeg).
    session = ManualSession(
        frames=str(fake_video),
        alignments_path=str(tmp_path / "alignments.fsa"),
        is_video_input=True,
        frame_list=(),
    )

    fake_meta = ManualVideoMetadata(pts_time=(0, 1, 2), keyframes=(0,))

    def _fake_video_metadata(_self: object) -> ManualVideoMetadata:
        return fake_meta

    monkeypatch.setattr(
        "tools.manual.session.ManualAlignmentsHandle.video_metadata",
        _fake_video_metadata,
    )

    observed: dict[str, object] = {}

    def _capture_start_video_provider(self: object) -> None:
        observed["video_metadata"] = self._video_metadata  # noqa: SLF001

    monkeypatch.setattr(
        ManualToolWindow,
        "_start_video_provider",
        _capture_start_video_provider,
    )

    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    assert observed.get("video_metadata") is fake_meta, (
        "Video provider must see populated metadata before it starts; "
        f"observed={observed.get('video_metadata')!r}"
    )


def test_manual_startup_worker_stop_quits_thread(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """ManualStartupWorker.stop() shuts the QThread down cleanly."""
    session = _session_with_frames(tmp_path)
    handle = session.alignments_handle()
    editable = ManualEditableAlignments()
    worker = ManualStartupWorker(handle, editable, session)
    worker.start()
    _drain_qt_events(qtbot, timeout_ms=200)
    worker.stop()
    assert worker._thread.isRunning() is False  # noqa: SLF001


def test_manual_startup_progress_bar_reports_determinate_percent(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Each startup stage advances the progress bar to its mapped percent."""
    from lib.gui.qt_shell.manual_tool import ManualToolWindow, _ManualStartupTask

    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    assert window._progress_bar.minimum() == 0
    assert window._progress_bar.maximum() == 100

    # Drive the staged progress slot directly to assert per-stage percentages.
    window._on_startup_progress("open", "Opening alignments file…")
    assert window._progress_bar.value() == _ManualStartupTask.STAGE_PERCENT["open"]

    window._on_startup_progress("thumbs", "Checking thumbnail cache…")
    assert window._progress_bar.value() == _ManualStartupTask.STAGE_PERCENT["thumbs"]

    window._on_startup_progress("complete", "Done")
    assert window._progress_bar.value() == _ManualStartupTask.STAGE_PERCENT["complete"]


def test_manual_startup_task_stage_percent_is_monotonic() -> None:
    """Stage percentages must monotonically advance toward 100."""
    from lib.gui.qt_shell.manual_tool import _ManualStartupTask

    percentages = [
        _ManualStartupTask.STAGE_PERCENT["open"],
        _ManualStartupTask.STAGE_PERCENT["thumbs"],
        _ManualStartupTask.STAGE_PERCENT["complete"],
    ]
    assert percentages == sorted(percentages)
    assert percentages[-1] == 100
