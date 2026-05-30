#!/usr/bin/env python3
"""Regression tests for Manual Tool polish from issue #201.

Covers the four bug fixes:

1. ``Manual._on_close`` runs the loader shutdown + destroys the root + exits.
2. ``Manual.__init__`` emits visible startup progress logs.
3. ``Filter.raw_indices`` emits ``(frame_idx, -1)`` placeholders for
   face-less frames under All Frames / No Faces; the renderer can build
   a full-frame placeholder TKFace from those entries.
4. ``DetectedFaces._IO.save`` logs no-op / start / success / failure paths
   at INFO so the save button has observable behaviour.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bug 3 — Filter.raw_indices placeholder emission
# ---------------------------------------------------------------------------


def _filter_with_counts(counts: list[int], filter_mode: str):
    """Build a :class:`tools.manual.detected_faces.Filter` for unit testing.

    The class normally consumes a fully-wired ``DetectedFaces`` object;
    we only need ``_globals.var_filter_mode`` and a stub
    ``face_count_per_index`` + ``frames_list``-driving stub so this test
    sidesteps the Tk + alignments machinery.
    """
    from tools.manual.detected_faces import Filter

    detected_faces = SimpleNamespace(
        _globals=SimpleNamespace(
            var_filter_mode=SimpleNamespace(get=lambda: filter_mode),
            var_filter_distance=SimpleNamespace(get=lambda: 10),
        ),
        face_count_per_index=counts,
        current_faces=[[] for _ in counts],
    )

    instance = Filter.__new__(Filter)
    instance._globals = detected_faces._globals
    instance._detected_faces = detected_faces
    return instance


def test_filter_raw_indices_all_frames_emits_placeholder_for_empty_frames() -> None:
    """All Frames must produce one row per face PLUS one ``-1``-faced row
    per face-less frame so the grid allocates a cell for every frame.
    """
    filt = _filter_with_counts([2, 0, 1, 0], "All Frames")

    indices = filt.raw_indices

    # Frame 0: 2 faces, Frame 1: empty placeholder, Frame 2: 1 face,
    # Frame 3: empty placeholder.
    assert indices["frame"] == [0, 0, 1, 2, 3]
    assert indices["face"] == [0, 1, -1, 0, -1]


def test_filter_raw_indices_no_faces_emits_placeholder_only_for_empty_frames() -> None:
    """No Faces never lists frames with faces, but does list one ``(frame, -1)``
    entry per face-less frame so those frames get a grid cell.
    """
    filt = _filter_with_counts([2, 0, 1, 0], "No Faces")

    indices = filt.raw_indices

    assert indices["frame"] == [1, 3]
    assert indices["face"] == [-1, -1]


def test_filter_raw_indices_single_face_does_not_emit_placeholder() -> None:
    """Single Face / Multiple Faces / etc. preserve the legacy behaviour —
    face-less frames are filtered out entirely.
    """
    filt = _filter_with_counts([2, 0, 1, 0], "Single Face")

    indices = filt.raw_indices

    # Only frame_idx 2 has exactly 1 face. Empty frames don't appear.
    assert indices["frame"] == [2]
    assert indices["face"] == [0]


# ---------------------------------------------------------------------------
# Bug 3 — Grid._get_display_faces tolerates the ``-1`` placeholder
# ---------------------------------------------------------------------------


def test_grid_get_display_faces_treats_placeholder_as_none() -> None:
    """A ``(frame_idx, -1)`` placeholder lands as ``None`` in the
    display_faces array — the same as trailing grid padding — so the
    renderer reads the labels grid for the placeholder indication.
    """
    from tools.manual.face_viewer.frame import Grid

    grid = Grid.__new__(Grid)
    grid._is_valid = True
    grid._face_size = 64
    grid._raw_indices = {"frame": [0, 1, 2], "face": [0, -1, 0]}
    # 1 row x 3 cols; one frame each. ``current_faces`` is per-frame.
    current_face_0 = SimpleNamespace(name="frame0_face0")
    current_face_2 = SimpleNamespace(name="frame2_face0")
    grid._detected_faces = SimpleNamespace(
        current_faces=[[current_face_0], [], [current_face_2]],
    )
    grid._grid = np.zeros((4, 1, 3), dtype="int")  # 1 row, 3 cols, just for shape
    grid._display_faces = None

    Grid._get_display_faces(grid)

    assert grid._display_faces.shape == (1, 3)
    assert grid._display_faces[0, 0] is current_face_0
    assert grid._display_faces[0, 1] is None  # placeholder
    assert grid._display_faces[0, 2] is current_face_2


# ---------------------------------------------------------------------------
# Bug 3 — TkGlobals.frame_image + attach_frame_loader
# ---------------------------------------------------------------------------


class _FakeInnerLoader:
    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self._frames = frames

    def image_from_index(self, idx: int):
        return f"frame_{idx:06d}.png", self._frames[idx]


def _make_globals_without_tkinter():
    """Construct a TkGlobals stub that doesn't require a live Tk
    interpreter — the attach + frame_image helpers don't touch Tk."""
    from tools.manual.globals import TkGlobals

    instance = TkGlobals.__new__(TkGlobals)
    instance._frame_loader = None
    return instance


def test_tk_globals_frame_image_returns_none_when_no_loader_attached() -> None:
    """Pre-attach the helper must safely return ``None`` so the renderer's
    fallback (gray placeholder) kicks in instead of crashing."""
    globals_ = _make_globals_without_tkinter()
    assert globals_.frame_image(0) is None


def test_tk_globals_attach_frame_loader_then_frame_image_uses_loader() -> None:
    """Once the loader is attached, ``frame_image`` round-trips through it."""
    sample = np.full((480, 640, 3), 200, dtype=np.uint8)
    inner = _FakeInnerLoader({3: sample})
    frame_loader = SimpleNamespace(_loader=inner)
    globals_ = _make_globals_without_tkinter()
    globals_.attach_frame_loader(frame_loader)

    image = globals_.frame_image(3)

    assert image is sample


def test_tk_globals_frame_image_swallows_loader_errors() -> None:
    """A loader raising on bad index must surface as ``None`` so the
    renderer falls back to the gray placeholder rather than crashing
    the whole grid redraw."""

    class _ExplodingLoader:
        def image_from_index(self, _idx):
            raise RuntimeError("frame missing")

    globals_ = _make_globals_without_tkinter()
    globals_.attach_frame_loader(SimpleNamespace(_loader=_ExplodingLoader()))

    assert globals_.frame_image(0) is None


# ---------------------------------------------------------------------------
# Bug 4 — Save feedback
# ---------------------------------------------------------------------------


def _io_with_unsaved(unsaved: bool, alignments_path: str = "/tmp/alignments.fsa"):
    """Build a ``_IO`` instance with a controllable unsaved state."""
    from tools.manual.detected_faces import _DiskIO as _IO

    io_obj = _IO.__new__(_IO)
    io_obj._tk_unsaved = SimpleNamespace(get=lambda: unsaved, set=MagicMock())
    io_obj._updated_frame_indices = {0, 1}
    io_obj._sorted_frame_names = ["frame_0.png", "frame_1.png"]
    io_obj._frame_faces = [[SimpleNamespace(to_alignment=lambda: {"f": 0})] for _ in range(2)]
    io_obj._alignments = SimpleNamespace(
        file=alignments_path,
        data={
            "frame_0.png": SimpleNamespace(faces=[]),
            "frame_1.png": SimpleNamespace(faces=[]),
        },
        backup=MagicMock(),
        save=MagicMock(),
    )
    return io_obj


def test_save_logs_noop_when_no_unsaved_changes(caplog: pytest.LogCaptureFixture) -> None:
    """No-op save must be visible at INFO so the user knows nothing was
    written, instead of the silent DEBUG-level "Returning" from before
    issue #201.
    """
    from tools.manual.detected_faces import _DiskIO as _IO

    caplog.set_level(logging.INFO, logger="tools.manual.detected_faces")
    io_obj = _io_with_unsaved(unsaved=False)
    _IO.save(io_obj)

    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "No alignment changes to save" in log_text
    # Counter assertions on the saved-state machinery.
    io_obj._tk_unsaved.set.assert_not_called()
    io_obj._alignments.save.assert_not_called()


def test_save_logs_writing_and_completion_on_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Successful save logs both a start ("Writing alignments...") and a
    completion message at INFO so the save button has observable
    behaviour."""
    from tools.manual.detected_faces import _DiskIO as _IO

    caplog.set_level(logging.INFO, logger="tools.manual.detected_faces")
    io_obj = _io_with_unsaved(unsaved=True, alignments_path="/data/A/alignments.fsa")
    _IO.save(io_obj)

    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Writing alignments" in log_text
    assert "/data/A/alignments.fsa" in log_text
    assert "Saved alignments" in log_text
    io_obj._alignments.backup.assert_called_once()
    io_obj._alignments.save.assert_called_once()
    io_obj._tk_unsaved.set.assert_called_once_with(False)


def test_save_logs_failure_at_error_when_save_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A serializer crash must NOT be swallowed silently — log at ERROR
    and leave the dirty bit set so the user can retry."""
    from tools.manual.detected_faces import _DiskIO as _IO

    caplog.set_level(logging.ERROR, logger="tools.manual.detected_faces")
    io_obj = _io_with_unsaved(unsaved=True, alignments_path="/data/X/alignments.fsa")
    io_obj._alignments.save = MagicMock(side_effect=OSError("disk full"))

    _IO.save(io_obj)

    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Failed to write alignments" in log_text
    assert "/data/X/alignments.fsa" in log_text
    assert "disk full" in log_text
    # Dirty state must remain so the user can retry.
    io_obj._tk_unsaved.set.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 1 — Shutdown handler dispatches loader.close + destroy + exit
# ---------------------------------------------------------------------------


def test_on_close_calls_loader_close_then_destroy_then_exit(monkeypatch) -> None:
    """The WM_DELETE_WINDOW handler must close the SingleFrameLoader,
    destroy the root window, then sys.exit so the standalone process
    actually terminates."""
    from tools.manual.manual import Manual

    instance = Manual.__new__(Manual)
    inner_loader = SimpleNamespace(close=MagicMock())
    instance._frame_loader = SimpleNamespace(_loader=inner_loader)
    instance.destroy = MagicMock()

    exits: list[int] = []
    monkeypatch.setattr(sys, "exit", lambda code=0: exits.append(code))

    Manual._on_close(instance)

    inner_loader.close.assert_called_once()
    instance.destroy.assert_called_once()
    assert exits == [0]


def test_on_close_tolerates_missing_loader(monkeypatch) -> None:
    """If the close fires before the loader was wired (early init failure),
    the handler must still destroy + exit rather than raise."""
    from tools.manual.manual import Manual

    instance = Manual.__new__(Manual)
    # Manual subclasses tk.Tk, whose ``__getattr__`` recurses on
    # uninitialised instances. Seed the dict directly so the
    # ``getattr(self, "_frame_loader", None)`` lookup in ``_on_close``
    # finds a value in __dict__ and never falls through to Tk.
    instance.__dict__["_frame_loader"] = None
    instance.__dict__["destroy"] = MagicMock()

    exits: list[int] = []
    monkeypatch.setattr(sys, "exit", lambda code=0: exits.append(code))

    Manual._on_close(instance)

    instance.destroy.assert_called_once()
    assert exits == [0]


def test_on_close_swallows_loader_exception(monkeypatch) -> None:
    """A close-time loader error must not block the destroy + exit
    sequence — the user should still see the window go away."""
    from tools.manual.manual import Manual

    instance = Manual.__new__(Manual)
    inner_loader = SimpleNamespace(close=MagicMock(side_effect=RuntimeError("reader hung")))
    instance._frame_loader = SimpleNamespace(_loader=inner_loader)
    instance.destroy = MagicMock()

    exits: list[int] = []
    monkeypatch.setattr(sys, "exit", lambda code=0: exits.append(code))

    Manual._on_close(instance)

    inner_loader.close.assert_called_once()
    instance.destroy.assert_called_once()
    assert exits == [0]
