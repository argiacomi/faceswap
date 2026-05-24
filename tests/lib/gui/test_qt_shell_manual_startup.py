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


# ---------------------------------------------------------------------------
# #114 — per-event thumbnail progress percent
# ---------------------------------------------------------------------------


def test_emit_thumb_progress_carries_per_event_percent(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Multi-step thumbnail progress emits the right percent for every step.

    The previous implementation mutated ``STAGE_PERCENT["thumbs_progress"]``
    before emitting on a named-stage signal, which could race queued Qt
    deliveries and paint the wrong percent.  The new design emits
    ``(percent, message)`` directly so each event carries an immutable
    percent value computed at emit time.
    """
    session = _session_with_frames(tmp_path)
    handle = session.alignments_handle()
    editable = ManualEditableAlignments()
    task = _ManualStartupTask(handle, editable, session)
    received: list[tuple[int, str]] = []
    task.progress_percent.connect(lambda pct, msg: received.append((pct, msg)))

    task._emit_thumb_progress(1, 4, "frame 1")  # noqa: SLF001
    task._emit_thumb_progress(2, 4, "frame 2")  # noqa: SLF001
    task._emit_thumb_progress(4, 4, "frame 4")  # noqa: SLF001

    # 66 + round(done/total * 33) clamped to 99.
    assert received == [
        (66 + round(1 / 4 * 33), "frame 1"),
        (66 + round(2 / 4 * 33), "frame 2"),
        (66 + round(4 / 4 * 33), "frame 4"),
    ]
    # Clamped at 99 so the bar never paints "complete" until the worker
    # explicitly emits the "complete" named stage.
    assert received[-1][0] == 99


def test_thumb_progress_state_does_not_bleed_between_tasks(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Two startup task instances must not influence each other's percent."""
    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    session_a = _session_with_frames(folder_a, count=2)
    session_b = _session_with_frames(folder_b, count=2)
    handle_a = session_a.alignments_handle()
    handle_b = session_b.alignments_handle()
    editable_a = ManualEditableAlignments()
    editable_b = ManualEditableAlignments()
    task_a = _ManualStartupTask(handle_a, editable_a, session_a)
    task_b = _ManualStartupTask(handle_b, editable_b, session_b)

    received_a: list[int] = []
    received_b: list[int] = []
    task_a.progress_percent.connect(lambda pct, _msg: received_a.append(pct))
    task_b.progress_percent.connect(lambda pct, _msg: received_b.append(pct))

    # Task A advances 50 %; Task B then advances 25 %.  The 25 % event must
    # not be influenced by Task A's previously-emitted 50 % value (which
    # the old implementation would have stashed onto the shared class
    # attribute).
    task_a._emit_thumb_progress(1, 2, "a1")  # noqa: SLF001
    task_b._emit_thumb_progress(1, 4, "b1")  # noqa: SLF001
    task_a._emit_thumb_progress(2, 2, "a2")  # noqa: SLF001

    assert received_a == [66 + round(1 / 2 * 33), 99]
    assert received_b == [66 + round(1 / 4 * 33)]


def test_stage_percent_no_longer_contains_thumbs_progress() -> None:
    """``thumbs_progress`` is no longer a static stage in the class table."""
    assert "thumbs_progress" not in _ManualStartupTask.STAGE_PERCENT


# ---------------------------------------------------------------------------
# #121 / #119 task 1 — thumb stage anchor must not reset live percent
# ---------------------------------------------------------------------------


def test_thumbs_named_stage_does_not_reset_live_thumb_percent(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """After a per-event percent fires, the named ``thumbs`` stage must not
    repaint the bar back to the 66% anchor.

    ``_ManualStartupTask._emit_thumb_progress`` emits the per-event percent
    first and then the named stage signal so console / status messaging
    stays per-frame.  Without this guard the named-stage handler reads
    ``STAGE_PERCENT["thumbs"] == 66`` and clobbers the live value.
    """
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    # Sanity: pre-condition matches the new initialiser.
    assert window._thumb_progress_seen is False

    # Simulate first-frame live progress: bar should track the live value.
    window._on_startup_progress_percent(74, "frame 1")
    assert window._progress_bar is not None
    assert window._progress_bar.value() == 74
    assert window._thumb_progress_seen is True

    # Now the named ``thumbs`` signal arrives for the same frame — it must
    # NOT reset the bar back to 66.
    window._on_startup_progress("thumbs", "frame 1")
    assert window._progress_bar.value() == 74

    # A subsequent live percent still paints normally.
    window._on_startup_progress_percent(88, "frame 2")
    assert window._progress_bar.value() == 88
    window._on_startup_progress("thumbs", "frame 2")
    assert window._progress_bar.value() == 88


def test_thumbs_named_stage_still_paints_anchor_before_live_progress(
    qtbot, tmp_path: Path
) -> None:  # type:ignore[no-untyped-def]
    """The initial ``thumbs`` stage *does* still paint 66% — the guard only
    suppresses the *reset* once live progress has started."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    assert window._progress_bar is not None

    # No live percent has fired yet — anchor paints normally.
    window._on_startup_progress("thumbs", "Checking thumbnail cache…")
    assert window._progress_bar.value() == 66
    assert window._thumb_progress_seen is False


def test_startup_retry_resets_thumb_progress_seen_flag(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A second startup (e.g. after a failure + retry) clears the guard flag."""
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    window._on_startup_progress_percent(74, "live")
    assert window._thumb_progress_seen is True

    # Re-launching the startup worker resets the flag so the named stage
    # can paint the anchor once more.
    window._start_background_startup()
    assert window._thumb_progress_seen is False


# ---------------------------------------------------------------------------
# Startup-worker lifecycle (#107 follow-up — cross-test signal leak)
# ---------------------------------------------------------------------------


def test_startup_worker_is_torn_down_after_completion(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``_on_startup_completed`` must drain + drop ``_startup_worker``.

    Before this fix every Manual Tool test left the startup ``QThread``
    alive past the assertion phase (the worker only got stopped at
    ``closeEvent`` / pytest-qt widget teardown).  That left queued
    Qt signals available to fire on whichever widget the next test
    constructed — the closest remaining cross-test leak candidate.
    """
    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window._startup_complete is True, timeout=3000)
    # Terminal startup signal → ``_teardown_startup_worker`` ran.
    assert window._startup_worker is None


def test_startup_worker_is_torn_down_after_failure(qtbot, tmp_path: Path, monkeypatch) -> None:  # type:ignore[no-untyped-def]
    """``_on_startup_failed`` also drains + drops the worker.

    Without the drain a failed startup leaves the QThread running until
    the user closes the window.
    """
    from PySide6.QtWidgets import QMessageBox

    # Force the startup task to fail by making the alignments file unreadable.
    def _broken_open(self) -> object:  # type:ignore[no-untyped-def]
        raise RuntimeError("simulated open failure")

    from tools.manual.session import ManualAlignmentsHandle

    monkeypatch.setattr(ManualAlignmentsHandle, "exists", property(lambda _self: True))
    monkeypatch.setattr(ManualAlignmentsHandle, "open", _broken_open)
    # Swallow the modal dialog the failure path opens.
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *args, **kwargs: 0))

    session = _session_with_frames(tmp_path)
    window = ManualToolWindow(session)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window._startup_worker is None, timeout=3000)
    assert window._startup_complete is False
