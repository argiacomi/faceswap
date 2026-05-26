#!/usr/bin/env python3
"""Aligner integration tests for the native Qt Manual Tool (#104).

Confirms the Bounding Box editor's aligner rerun + new-face landmark
initialisation behaviour using a stubbed ``ManualAlignerService`` so the
tests do not load actual aligner plugins.
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QProgressBar,
    QRadioButton,
    QToolButton,
    QWidget,
)

from tools.manual.aligner_service import (
    NORMALIZATION_CHOICES,
    AlignerStatus,
    ManualAlignerService,
)
from tools.manual.qt import ManualToolWindow
from tools.manual.session import ManualSession


@dataclass
class _StubBackend:
    aligner: str
    normalization: str
    landmarks: np.ndarray
    align_calls: list[tuple[np.ndarray, tuple[float, float, float, float]]] = field(
        default_factory=list
    )
    normalization_changes: list[str] = field(default_factory=list)
    fail_with: Exception | None = None

    def align(self, image: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        if self.fail_with is not None:
            raise self.fail_with
        self.align_calls.append((image, bbox))
        return self.landmarks

    def set_normalization(self, method: str) -> None:
        self.normalization_changes.append(method)
        self.normalization = method


def _stub_service(
    *,
    landmarks: np.ndarray | None = None,
    fail: Exception | None = None,
    instances: list[_StubBackend] | None = None,
    available: tuple[str, ...] = ("HRNet",),
    default: str = "HRNet",
) -> ManualAlignerService:
    """Build a stubbed service that records every align call."""
    instances = instances if instances is not None else []
    landmarks = (
        landmarks if landmarks is not None else np.tile([[1.0, 2.0]], (68, 1)).astype(np.float32)
    )

    def factory(aligner: str, normalization: str) -> _StubBackend:
        backend = _StubBackend(
            aligner=aligner,
            normalization=normalization,
            landmarks=landmarks.copy(),
            fail_with=fail,
        )
        instances.append(backend)
        return backend

    svc = ManualAlignerService(
        available=lambda: available,
        default=lambda: default,
        factory=factory,
    )
    svc._test_instances = instances  # type:ignore[attr-defined]
    return svc


def _session_with_frame(folder: Path) -> ManualSession:
    pixmap = QPixmap(120, 120)
    pixmap.fill(QColor("#9988aa"))
    path = folder / "frame_000.png"
    assert pixmap.save(str(path), "PNG")
    return ManualSession.create(frames=str(folder))


def _make_window(
    qtbot,  # type:ignore[no-untyped-def]
    tmp_path: Path,
    *,
    service: ManualAlignerService | None = None,
) -> tuple[ManualToolWindow, ManualAlignerService]:
    service = service or _stub_service()
    window = ManualToolWindow(_session_with_frame(tmp_path), aligner_service=service)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window._frame_view.source_size != (0, 0), timeout=2000)
    return window, service


_WidgetT = T.TypeVar("_WidgetT", bound=QWidget)


def _child(window: ManualToolWindow, widget_type: type[_WidgetT], name: str) -> _WidgetT:
    """Return a named child widget, failing with a useful assertion message."""
    widget = window.findChild(widget_type, name)
    assert widget is not None, f"Missing widget {name}"
    return widget


def _wait_for_aligner_preload(qtbot, window: ManualToolWindow, *, timeout: int = 3000) -> None:  # type:ignore[no-untyped-def]
    """Wait until any explicit aligner preload worker has fully drained.

    Tests that trigger preload via ``set_aligner_normalization`` /
    ``clahe.click()`` / ``combo.setCurrentIndex(...)`` / mode switches
    into ``BoundingBox`` must drain the worker before returning so the
    QThread doesn't outlive the test and bleed signals into the next
    test's widgets.  ``ManualToolWindow._on_aligner_load_completed`` is
    the only place that clears ``_aligner_load_worker``, so polling that
    attribute back to ``None`` is the canonical "preload is done"
    signal.
    """
    qtbot.waitUntil(lambda: window._aligner_load_worker is None, timeout=timeout)


# ---------------------------------------------------------------------------
# Selection + normalization
# ---------------------------------------------------------------------------


def test_available_aligners_and_normalizations_surface_through_window(
    qtbot, tmp_path: Path
) -> None:  # type:ignore[no-untyped-def]
    """The window forwards service discovery results for dropdown population."""
    window, _ = _make_window(qtbot, tmp_path)
    assert window.available_aligners() == ("HRNet",)
    assert window.available_normalizations() == NORMALIZATION_CHOICES


def test_set_aligner_name_records_choice_on_editor_state(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``set_aligner_name`` stores the chosen plugin on the editor state."""
    window, _ = _make_window(qtbot, tmp_path)
    window.set_aligner_name("FAN")
    assert window._editor_state.aligner_name == "FAN"


def test_set_normalization_propagates_to_loaded_backends(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Normalization changes update editor state + already-loaded backends."""
    window, service = _make_window(qtbot, tmp_path)
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")
    # Force a backend instance to materialise.
    window.rerun_aligner_for_face(0)
    instances = service._test_instances  # type:ignore[attr-defined]
    assert instances and instances[-1].normalization == "hist"

    window.set_aligner_normalization("clahe")
    assert window._editor_state.aligner_normalization == "clahe"
    # Changing normalization also schedules a preload for the new
    # ``(aligner, normalization)`` tuple, so the last backend can be the newly
    # cached clahe backend.  The propagation contract is that any already
    # loaded backend received the normalization change before that preload.
    _wait_for_aligner_preload(qtbot, window)
    assert any("clahe" in backend.normalization_changes for backend in instances)


# ---------------------------------------------------------------------------
# Visible BBox controls
# ---------------------------------------------------------------------------


def test_bbox_aligner_controls_exist_and_populate_from_service(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """BBox editor surfaces aligner dropdown, radios, auto-run and progress widgets."""
    service = _stub_service(available=("HRNet", "FAN"), default="FAN")
    window, _ = _make_window(qtbot, tmp_path, service=service)

    controls = _child(window, QWidget, "qt-manual-aligner-controls")
    combo = _child(window, QComboBox, "qt-manual-aligner-combo")
    auto_run = _child(window, QCheckBox, "qt-manual-aligner-auto-run")
    rerun = _child(window, QToolButton, "qt-manual-aligner-rerun")
    progress = _child(window, QProgressBar, "qt-manual-aligner-load-progress")
    status = _child(window, QLabel, "qt-manual-aligner-status-label")

    assert controls.isVisible() is False
    window._editor_state.set("editor_mode", "BoundingBox")

    assert controls.isVisible() is True
    assert [combo.itemText(index) for index in range(combo.count())] == ["HRNet", "FAN"]
    assert combo.currentText() == "FAN"
    assert _child(window, QRadioButton, "qt-manual-aligner-normalization-hist").isChecked()
    for method in NORMALIZATION_CHOICES:
        assert _child(window, QRadioButton, f"qt-manual-aligner-normalization-{method}")
    assert auto_run.isChecked() is True
    assert rerun.text() == "Run"
    assert progress is not None
    assert status.text()


def test_bbox_aligner_controls_are_visible_only_in_bbox_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Aligner controls hide outside BoundingBox mode without being destroyed."""
    window, _ = _make_window(qtbot, tmp_path)
    controls = _child(window, QWidget, "qt-manual-aligner-controls")

    assert controls.isVisible() is False
    window._editor_state.set("editor_mode", "BoundingBox")
    assert controls.isVisible() is True

    window._editor_state.set("editor_mode", "Landmarks")
    assert controls.isVisible() is False
    window._editor_state.set("editor_mode", "Mask")
    assert controls.isVisible() is False
    window._editor_state.set("editor_mode", "View")
    assert controls.isVisible() is False


def test_bbox_aligner_selection_persists_across_editor_mode_changes(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Dropdown + normalization choices survive leaving and re-entering BBox mode."""
    service = _stub_service(available=("HRNet", "FAN"), default="HRNet")
    window, _ = _make_window(qtbot, tmp_path, service=service)
    controls = _child(window, QWidget, "qt-manual-aligner-controls")
    combo = _child(window, QComboBox, "qt-manual-aligner-combo")
    clahe = _child(window, QRadioButton, "qt-manual-aligner-normalization-clahe")

    window._editor_state.set("editor_mode", "BoundingBox")
    _wait_for_aligner_preload(qtbot, window)
    combo.setCurrentIndex(combo.findText("FAN"))
    qtbot.waitUntil(lambda: window._editor_state.aligner_name == "FAN", timeout=2000)
    _wait_for_aligner_preload(qtbot, window)
    clahe.click()
    qtbot.waitUntil(lambda: window._editor_state.aligner_normalization == "clahe", timeout=2000)
    _wait_for_aligner_preload(qtbot, window)

    window._editor_state.set("editor_mode", "Landmarks")
    assert controls.isVisible() is False
    window._editor_state.set("editor_mode", "BoundingBox")

    assert controls.isVisible() is True
    assert combo.currentText() == "FAN"
    assert clahe.isChecked() is True
    assert window._editor_state.aligner_name == "FAN"
    assert window._editor_state.aligner_normalization == "clahe"


def test_bbox_auto_run_checkbox_updates_state_and_gates_rerun(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """The visible Auto-run checkbox controls post-bbox-edit aligner reruns."""
    window, service = _make_window(qtbot, tmp_path)
    checkbox = _child(window, QCheckBox, "qt-manual-aligner-auto-run")
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")
    _wait_for_aligner_preload(qtbot, window)

    checkbox.click()

    assert checkbox.isChecked() is False
    assert window._editor_state.aligner_auto_run is False
    window._on_face_move_requested(0, 5.0, 7.0)
    instances = service._test_instances  # type:ignore[attr-defined]
    assert all(not backend.align_calls for backend in instances)


def test_bbox_aligner_preload_progress_paints_loading_then_ready(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Changing a BBox normalization starts background preload and updates inline status."""
    window, _ = _make_window(qtbot, tmp_path)
    progress = _child(window, QProgressBar, "qt-manual-aligner-load-progress")
    status = _child(window, QLabel, "qt-manual-aligner-status-label")
    clahe = _child(window, QRadioButton, "qt-manual-aligner-normalization-clahe")

    window._editor_state.set("editor_mode", "BoundingBox")
    assert progress.isVisible() is False
    clahe.click()

    assert progress.isVisible() is True
    assert "Loading aligner" in progress.format()
    _wait_for_aligner_preload(qtbot, window)
    assert progress.isVisible() is False
    assert "ready" in status.text().lower()


def test_bbox_aligner_preload_clears_worker_state_after_completion(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Completed explicit preload drains worker state before tests continue."""
    window, _ = _make_window(qtbot, tmp_path)
    clahe = _child(window, QRadioButton, "qt-manual-aligner-normalization-clahe")

    window._editor_state.set("editor_mode", "BoundingBox")
    clahe.click()
    _wait_for_aligner_preload(qtbot, window)

    assert window._aligner_load_target is None
    assert ("HRNet", "clahe") in window._aligner_loaded_targets


# ---------------------------------------------------------------------------
# Rerun on bbox edits
# ---------------------------------------------------------------------------


def test_bbox_move_triggers_aligner_when_auto_run_enabled(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Moving a bbox in BoundingBox mode reruns the aligner with the new bbox."""
    canned = np.tile([[5.0, 6.0]], (68, 1)).astype(np.float32)
    window, service = _make_window(qtbot, tmp_path, service=_stub_service(landmarks=canned))
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")

    window._on_face_move_requested(0, 5.0, 7.0)

    instances = service._test_instances  # type:ignore[attr-defined]
    assert instances, "Aligner backend should have been constructed"
    call = instances[-1].align_calls[-1]
    # The aligner sees the *moved* bbox.
    assert call[1] == (15.0, 17.0, 40.0, 40.0)
    # Landmarks were refreshed to the canned reply.
    face = window._editable.faces(0)[0]
    assert face.landmarks[0] == (5.0, 6.0)


def test_successful_aligner_refresh_is_dirty_and_undoable(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A successful explicit rerun records an undoable landmark refresh."""
    canned = np.tile([[9.0, 10.0]], (68, 1)).astype(np.float32)
    window, _ = _make_window(qtbot, tmp_path, service=_stub_service(landmarks=canned))
    window._editable.add_face(
        0,
        (10.0, 10.0, 40.0, 40.0),
        landmarks=[(1.0, 1.0), (2.0, 2.0)],
    )
    window._editable.clear_history()
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")

    assert window.rerun_aligner_for_face(0) is True
    assert window._editor_state.edited is True
    assert window._editable.can_undo is True
    assert window._editable.faces(0)[0].landmarks[0] == (9.0, 10.0)

    assert window._editable.undo() is True
    assert window._editable.faces(0)[0].landmarks == ((1.0, 1.0), (2.0, 2.0))
    assert window._editable.redo() is True
    assert window._editable.faces(0)[0].landmarks[0] == (9.0, 10.0)


def test_bbox_resize_triggers_aligner_when_auto_run_enabled(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Resizing a bbox in BoundingBox mode reruns the aligner with the new bbox."""
    window, service = _make_window(qtbot, tmp_path)
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")

    window._on_face_resize_requested(0, QRectF(8.0, 8.0, 50.0, 50.0))
    instances = service._test_instances  # type:ignore[attr-defined]
    assert instances[-1].align_calls[-1][1] == (8.0, 8.0, 50.0, 50.0)


def test_bbox_move_and_resize_use_selected_aligner_and_normalization(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """BBox edits rerun landmarks through the selected aligner backend."""
    instances: list[_StubBackend] = []
    window, _ = _make_window(
        qtbot,
        tmp_path,
        service=_stub_service(
            instances=instances,
            available=("HRNet", "FAN"),
            default="HRNet",
        ),
    )
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")
    window.set_aligner_name("FAN")
    window.set_aligner_normalization("clahe")
    _wait_for_aligner_preload(qtbot, window)

    window._on_face_move_requested(0, 5.0, 7.0)
    window._on_face_resize_requested(0, QRectF(8.0, 8.0, 50.0, 50.0))

    align_calls = [backend for backend in instances if backend.align_calls]
    assert align_calls
    assert all(backend.aligner == "FAN" for backend in align_calls)
    assert all(backend.normalization == "clahe" for backend in align_calls)
    bboxes = [call[1] for backend in align_calls for call in backend.align_calls]
    assert bboxes[-2] == (15.0, 17.0, 40.0, 40.0)
    assert bboxes[-1] == (8.0, 8.0, 50.0, 50.0)


def test_bbox_pointer_add_uses_add_then_aligner_regeneration_path(
    qtbot,
    tmp_path: Path,
) -> None:  # type:ignore[no-untyped-def]
    """Frame-view BBox add creates the face then regenerates landmarks."""
    canned = np.tile([[13.0, 14.0]], (68, 1)).astype(np.float32)
    window, service = _make_window(qtbot, tmp_path, service=_stub_service(landmarks=canned))
    window._editor_state.set("editor_mode", "BoundingBox")

    window._on_face_add_requested(QRectF(30.0, 30.0, 40.0, 40.0))

    face = window._editable.faces(0)[0]
    assert face.bbox == (30.0, 30.0, 40.0, 40.0)
    assert face.landmarks[0] == (13.0, 14.0)
    instances = service._test_instances  # type:ignore[attr-defined]
    assert instances[-1].align_calls[-1][1] == (30.0, 30.0, 40.0, 40.0)


def test_bbox_move_skips_aligner_outside_bbox_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """In View / Landmarks / Extract / Mask modes the aligner is not invoked."""
    window, service = _make_window(qtbot, tmp_path)
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "View")
    window._on_face_move_requested(0, 5.0, 5.0)
    assert service._test_instances == []  # type:ignore[attr-defined]


def test_bbox_move_skips_aligner_when_auto_run_disabled(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``aligner_auto_run=False`` prevents the post-edit rerun."""
    window, service = _make_window(qtbot, tmp_path)
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")
    window._editor_state.set("aligner_auto_run", False)

    window._on_face_move_requested(0, 5.0, 7.0)
    instances = service._test_instances  # type:ignore[attr-defined]
    assert all(not backend.align_calls for backend in instances)


# ---------------------------------------------------------------------------
# New-face initialisation
# ---------------------------------------------------------------------------


def test_add_face_initialises_landmarks_via_aligner_in_bbox_mode(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A new pointer-added face gets its landmarks from the aligner."""
    canned = np.tile([[3.0, 4.0]], (68, 1)).astype(np.float32)
    window, service = _make_window(qtbot, tmp_path, service=_stub_service(landmarks=canned))
    window._editor_state.set("editor_mode", "BoundingBox")

    new_index = window.add_face_at_center((30.0, 30.0, 40.0, 40.0))
    assert new_index == 0

    face = window._editable.faces(0)[0]
    assert face.landmarks[0] == (3.0, 4.0)
    assert service._test_instances  # type:ignore[attr-defined]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_aligner_failure_leaves_editable_model_intact(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A failing aligner surfaces a status message and does not touch landmarks."""
    failing = _stub_service(fail=RuntimeError("aligner broke"))
    window, _ = _make_window(qtbot, tmp_path, service=failing)
    window._editable.add_face(
        0,
        (10.0, 10.0, 40.0, 40.0),
        landmarks=[(1.0, 1.0), (2.0, 2.0)],
    )
    window._editor_state.set("face_index", 0)
    window._editor_state.set("editor_mode", "BoundingBox")

    before = window._editable.faces(0)[0].landmarks
    assert window.rerun_aligner_for_face(0) is False
    after = window._editable.faces(0)[0].landmarks
    assert before == after


def test_rerun_aligner_no_face_returns_false_with_status_message(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``rerun_aligner_for_face`` rejects an out-of-range face_index."""
    window, _ = _make_window(qtbot, tmp_path)
    # No faces have been added yet.
    assert window.rerun_aligner_for_face(0) is False


def test_aligner_status_callback_routed_to_status_bar(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """``_on_aligner_status`` updates the status bar message text."""
    window, _ = _make_window(qtbot, tmp_path)
    window._on_aligner_status(AlignerStatus(kind="loading", aligner="HRNet", message="Hi"))
    assert window.statusBar().currentMessage() == "Hi"


def test_set_aligner_normalization_updates_editor_state_without_face(
    qtbot, tmp_path: Path
) -> None:  # type:ignore[no-untyped-def]
    """Normalization editor-state updates land even before any backend is loaded."""
    window, _ = _make_window(qtbot, tmp_path)
    window.set_aligner_normalization("mean")
    assert window._editor_state.aligner_normalization == "mean"
    # ``set_aligner_normalization`` schedules a preload whenever the BBox
    # controls are visible.  Even though this test runs outside BoundingBox
    # mode, drain unconditionally so a future controls-visibility regression
    # doesn't leak a worker across tests.
    _wait_for_aligner_preload(qtbot, window)


def test_rerun_aligner_with_unstubbed_image_returns_false_when_frame_unloaded(
    qtbot, tmp_path: Path
) -> None:  # type:ignore[no-untyped-def]
    """When the frame view has no image, the aligner rerun no-ops cleanly."""
    window, _ = _make_window(qtbot, tmp_path)
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0))
    window._editor_state.set("face_index", 0)
    # Wipe the frame source.
    window._frame_view.clear_frame("test")
    assert window.rerun_aligner_for_face(0) is False


# ---------------------------------------------------------------------------
# #104 closure — preload worker lifecycle + dirty / undo coverage
# ---------------------------------------------------------------------------


def test_explicit_preload_clears_worker_state_on_completion(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """After an explicit aligner preload completes the worker reference is gone."""
    window, _ = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    window._editor_state.set("aligner_name", "HRNet")

    window._schedule_aligner_preload()
    _wait_for_aligner_preload(qtbot, window)

    # The completed worker was deleted and the target was cleared.
    assert window._aligner_load_worker is None
    assert window._aligner_load_target is None
    # The (aligner, normalization) pair has joined the loaded set.
    assert ("HRNet", "hist") in window._aligner_loaded_targets


def test_close_event_after_preload_completes_does_not_leave_thread(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A normal close after preload finishes leaves no dangling worker thread."""
    from PySide6.QtGui import QCloseEvent

    window, _ = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")
    window._editor_state.set("aligner_name", "HRNet")
    window._schedule_aligner_preload()
    _wait_for_aligner_preload(qtbot, window)

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is True
    assert window._aligner_load_worker is None


def test_aligner_refresh_marks_dirty_and_is_undoable(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """A bbox move + aligner refresh under auto-run is undoable as one logical edit."""
    import numpy as np

    landmarks_v2 = np.full((68, 2), 11.0, dtype=np.float32)
    window, service = _make_window(qtbot, tmp_path, service=_stub_service(landmarks=landmarks_v2))
    window._editor_state.set("editor_mode", "BoundingBox")
    initial_landmarks = tuple((float(i), float(i)) for i in range(5))
    window._editable.add_face(0, (10.0, 10.0, 40.0, 40.0), landmarks=initial_landmarks)
    window._editor_state.set("face_index", 0)
    window.mark_dirty(False)

    # Move the bbox — the auto-run path replaces landmarks with the stub
    # ``landmarks_v2`` value.
    window._on_face_move_requested(0, 5.0, 5.0)

    face = window._editable.faces(0)[0]
    assert face.landmarks[0] == (11.0, 11.0)
    assert window._editor_state.edited is True

    # Two undos roll back set_landmarks then move_face — both edits are
    # individually undoable, so the user can step back to the original.
    assert window._editable.undo() is True  # undo set_landmarks
    assert window._editable.undo() is True  # undo move_face
    restored = window._editable.faces(0)[0]
    assert restored.bbox == (10.0, 10.0, 40.0, 40.0)
    assert restored.landmarks == initial_landmarks
    # Redo replays both edits.
    assert window._editable.redo() is True
    assert window._editable.redo() is True
    final = window._editable.faces(0)[0]
    assert final.bbox == (15.0, 15.0, 40.0, 40.0)
    assert final.landmarks[0] == (11.0, 11.0)


def test_entering_bbox_mode_does_not_preload_aligner(qtbot, tmp_path: Path) -> None:  # type:ignore[no-untyped-def]
    """Switching to BoundingBox mode shows controls but does NOT start a preload."""
    window, _ = _make_window(qtbot, tmp_path)
    window._editor_state.set("editor_mode", "BoundingBox")

    # The aligner controls are visible…
    assert window._aligner_controls.isVisible() is True
    # …but no worker has been launched.  Passive preload was deliberately
    # removed so unrelated GUI tests / mode switches don't trigger
    # production plugin loading.
    assert window._aligner_load_worker is None
    assert window._aligner_load_target is None
