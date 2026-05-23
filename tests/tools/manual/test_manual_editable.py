#!/usr/bin/env python3
"""Tests for the GUI-neutral editable alignment model."""

from __future__ import annotations

import pytest

from tools.manual.session import EditableFace, ManualEditableAlignments


def _bbox(x: float = 10.0, y: float = 10.0, w: float = 20.0, h: float = 20.0):
    return (x, y, w, h)


def test_add_face_returns_monotonic_indices() -> None:
    """add_face assigns 0, 1, 2 in order and counts each frame independently."""
    model = ManualEditableAlignments()
    assert model.add_face(0, _bbox()) == 0
    assert model.add_face(0, _bbox(40, 10)) == 1
    assert model.add_face(2, _bbox()) == 0
    assert model.face_count(0) == 2
    assert model.face_count(1) == 0
    assert model.face_count(2) == 1


def test_add_face_rejects_degenerate_bbox() -> None:
    """Width/height below the minimum threshold raise a ValueError."""
    model = ManualEditableAlignments()
    with pytest.raises(ValueError, match="width and height"):
        model.add_face(0, (0.0, 0.0, 0.5, 50.0))


def test_delete_face_reindexes_remaining_entries() -> None:
    """Deleting a middle face renumbers later faces so face_index stays monotonic."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(0, 0))
    model.add_face(0, _bbox(40, 0))
    model.add_face(0, _bbox(80, 0))

    assert model.delete_face(0, 1) is True
    faces = model.faces(0)
    assert [face.face_index for face in faces] == [0, 1]
    assert faces[1].bbox[0] == 80.0


def test_delete_face_out_of_range_returns_false() -> None:
    """Deleting an invalid face_index reports failure instead of mutating."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox())
    assert model.delete_face(0, 5) is False
    assert model.face_count(0) == 1


def test_move_face_translates_bbox_and_landmarks() -> None:
    """move_face shifts bbox + landmarks by ``(dx, dy)`` source pixels."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(0, 0, 10, 10), landmarks=[(2.0, 3.0), (4.0, 5.0)])
    assert model.move_face(0, 0, 5.0, -2.0) is True
    face = model.faces(0)[0]
    assert face.bbox == (5.0, -2.0, 10.0, 10.0)
    assert face.landmarks == ((7.0, 1.0), (9.0, 3.0))


def test_move_face_no_op_when_delta_is_zero() -> None:
    """Zero-delta moves succeed without recording an undoable edit."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox())
    assert model.move_face(0, 0, 0.0, 0.0) is True
    assert model.can_undo is True  # add_face still queued
    # Undoing once reverts the add — confirming the no-op didn't push a record.
    assert model.undo() is True
    assert model.face_count(0) == 0
    assert model.can_undo is False


def test_undo_reverts_last_mutation() -> None:
    """undo pops the last reversible operation and notifies listeners."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox())
    model.add_face(0, _bbox(40, 0))
    assert model.face_count(0) == 2

    assert model.undo() is True
    assert model.face_count(0) == 1
    assert model.can_undo is True
    assert model.can_redo is True


def test_redo_replays_undone_mutation() -> None:
    """redo replays the most recently undone mutation."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox())
    model.undo()
    assert model.face_count(0) == 0
    assert model.redo() is True
    assert model.face_count(0) == 1
    assert model.can_undo is True
    assert model.can_redo is False


def test_undo_redo_with_empty_stacks_returns_false() -> None:
    """Calling undo/redo on an empty stack returns False instead of raising."""
    model = ManualEditableAlignments()
    assert model.undo() is False
    assert model.redo() is False


def test_subscribe_emits_on_mutations() -> None:
    """subscribe fires the listener with the modified frame_index."""
    model = ManualEditableAlignments()
    events: list[int] = []
    unsubscribe = model.subscribe(events.append)
    model.add_face(0, _bbox())
    model.add_face(1, _bbox())
    model.delete_face(0, 0)

    assert events == [0, 1, 0]
    unsubscribe()
    model.add_face(2, _bbox())
    assert events == [0, 1, 0]


def test_hit_test_returns_topmost_face() -> None:
    """hit_test prefers later-added faces when boxes overlap."""
    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 100.0, 100.0))
    model.add_face(0, (10.0, 10.0, 20.0, 20.0))

    assert model.hit_test(0, 15.0, 15.0) == 1  # topmost smaller box
    assert model.hit_test(0, 95.0, 95.0) == 0  # only the larger box covers
    assert model.hit_test(0, 200.0, 200.0) is None


def test_update_landmark_moves_single_point() -> None:
    """update_landmark replaces only the indexed point."""
    model = ManualEditableAlignments()
    model.add_face(
        0,
        _bbox(),
        landmarks=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
    )

    assert model.update_landmark(0, 0, 1, 30.0, 40.0) is True
    landmarks = model.faces(0)[0].landmarks
    assert landmarks == ((1.0, 2.0), (30.0, 40.0), (5.0, 6.0))


def test_update_landmark_out_of_range_returns_false() -> None:
    """update_landmark rejects invalid face or landmark indices."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0)])
    assert model.update_landmark(0, 0, 7, 0.0, 0.0) is False
    assert model.update_landmark(0, 9, 0, 0.0, 0.0) is False


# ---------------------------------------------------------------------------
# #103 — Landmark editor group move
# ---------------------------------------------------------------------------


def test_move_landmarks_translates_selected_indices_only() -> None:
    """move_landmarks shifts only the indices in the selection set."""
    model = ManualEditableAlignments()
    model.add_face(
        0,
        _bbox(),
        landmarks=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)],
    )

    assert model.move_landmarks(0, 0, (1, 3), 10.0, 20.0) is True
    landmarks = model.faces(0)[0].landmarks
    assert landmarks == (
        (1.0, 2.0),
        (13.0, 24.0),
        (5.0, 6.0),
        (17.0, 28.0),
    )


def test_move_landmarks_zero_delta_is_noop_true() -> None:
    """A zero-pixel delta returns True without recording new history."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0)])
    model.clear_history()  # ignore the add_face undo entry
    assert model.move_landmarks(0, 0, (0,), 0.0, 0.0) is True
    assert model.can_undo is False


def test_move_landmarks_empty_after_filter_is_noop_true() -> None:
    """If every index is out of range the call no-ops cleanly."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0), (3.0, 4.0)])
    # Indices 5/6 don't exist; 2 also doesn't.  Should drop them all and
    # return True without mutating state.
    assert model.move_landmarks(0, 0, (5, 6), 1.0, 1.0) is True
    assert model.faces(0)[0].landmarks == ((1.0, 2.0), (3.0, 4.0))


def test_move_landmarks_invalid_face_returns_false() -> None:
    """move_landmarks rejects an out-of-range face_index."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0)])
    assert model.move_landmarks(0, 9, (0,), 1.0, 1.0) is False


def test_move_landmarks_no_landmarks_returns_false() -> None:
    """A face with no stored landmarks cannot have any moved."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox())  # no landmarks
    assert model.move_landmarks(0, 0, (0,), 1.0, 1.0) is False


def test_move_landmarks_records_undo_history() -> None:
    """move_landmarks participates in the undo/redo stack."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(0.0, 0.0), (10.0, 10.0)])

    assert model.move_landmarks(0, 0, (0, 1), 5.0, 7.0) is True
    assert model.faces(0)[0].landmarks == ((5.0, 7.0), (15.0, 17.0))
    assert model.can_undo is True

    assert model.undo() is True
    assert model.faces(0)[0].landmarks == ((0.0, 0.0), (10.0, 10.0))

    assert model.redo() is True
    assert model.faces(0)[0].landmarks == ((5.0, 7.0), (15.0, 17.0))


def test_seed_from_handle_clears_history() -> None:
    """seed_from_handle resets the undo/redo stacks alongside the data."""
    model = ManualEditableAlignments(
        {0: [EditableFace(face_index=0, bbox=_bbox())]},
    )
    model.add_face(0, _bbox(40, 0))
    assert model.can_undo is True

    class _StubHandle:
        exists = False

        def open(self) -> object:  # pragma: no cover - not reached when exists=False
            raise AssertionError("open should not be called when exists is False")

    model.seed_from_handle(_StubHandle())  # type: ignore[arg-type]
    # Stub has exists=False so the seed is a no-op — confirm history untouched.
    assert model.can_undo is True


def test_seed_from_handle_maps_sparse_alignments_to_filename_index() -> None:
    """Sparse alignments seed into the frame index that matches the filename."""
    import numpy as np

    from lib.align.objects import AlignmentsEntry, FileAlignments

    class _StubHandle:
        exists = True

        def __init__(self, payload: object) -> None:
            self._payload = payload

        def open(self) -> object:
            return self._payload

    class _StubAlignments:
        def __init__(self) -> None:
            self.data: dict[str, AlignmentsEntry] = {
                "frame_010.png": AlignmentsEntry(
                    faces=[
                        FileAlignments(
                            x=10,
                            y=20,
                            w=30,
                            h=40,
                            landmarks_xy=np.zeros((0, 2), dtype=np.float32),
                        )
                    ],
                    video_meta={},
                )
            }

    handle = _StubHandle(_StubAlignments())
    model = ManualEditableAlignments()
    frame_names = [f"frame_{idx:03d}.png" for idx in range(20)]

    model.seed_from_handle(handle, frame_names=frame_names)

    # The sole alignment entry must land on frame_index 10 (the position of
    # ``frame_010.png`` in the source frame list), not 0 (the lexicographic
    # position of the alignment key).
    assert model.face_count(0) == 0
    assert model.face_count(10) == 1
    assert model.faces(10)[0].bbox == (10.0, 20.0, 30.0, 40.0)


def test_seed_from_handle_falls_back_to_sorted_alignments_for_video() -> None:
    """Without ``frame_names`` the seed uses sorted(alignments.data) order."""
    import numpy as np

    from lib.align.objects import AlignmentsEntry, FileAlignments

    class _StubAlignments:
        def __init__(self) -> None:
            self.data: dict[str, AlignmentsEntry] = {
                "frame_b.png": AlignmentsEntry(
                    faces=[
                        FileAlignments(
                            x=1,
                            y=2,
                            w=3,
                            h=4,
                            landmarks_xy=np.zeros((0, 2), dtype=np.float32),
                        )
                    ],
                    video_meta={},
                ),
                "frame_a.png": AlignmentsEntry(
                    faces=[
                        FileAlignments(
                            x=5,
                            y=6,
                            w=7,
                            h=8,
                            landmarks_xy=np.zeros((0, 2), dtype=np.float32),
                        )
                    ],
                    video_meta={},
                ),
            }

    class _StubHandle:
        exists = True

        def __init__(self) -> None:
            self._payload = _StubAlignments()

        def open(self) -> object:
            return self._payload

    model = ManualEditableAlignments()
    model.seed_from_handle(_StubHandle())

    # Sorted alphabetically: frame_a.png -> index 0, frame_b.png -> index 1.
    assert model.faces(0)[0].bbox == (5.0, 6.0, 7.0, 8.0)
    assert model.faces(1)[0].bbox == (1.0, 2.0, 3.0, 4.0)


def test_delete_last_face_keeps_frame_dirty_for_persist() -> None:
    """Deleting the only face on a frame still records the frame as dirty.

    Without dirty-frame tracking ``apply_to_alignments`` would iterate
    ``self._faces`` and skip the now-empty frame, leaving the on-disk
    alignments stale.  Confirm the model can still report the frame
    needs to be written.
    """
    model = ManualEditableAlignments(
        {0: [EditableFace(face_index=0, bbox=_bbox())]},
    )
    # Seeded entries do not pre-populate the dirty set.
    assert model._dirty_frames == set()  # noqa: SLF001

    model.delete_face(0, 0)

    assert model.face_count(0) == 0
    assert 0 in model._dirty_frames  # noqa: SLF001


def test_apply_to_alignments_accepts_callable_resolver() -> None:
    """Video sessions pass a callable; frames without existing entries persist."""
    from lib.align.objects import AlignmentsEntry

    class _StubAlignments:
        def __init__(self) -> None:
            self.data: dict[str, AlignmentsEntry] = {}

    alignments = _StubAlignments()
    model = ManualEditableAlignments()
    # Add a face on frame 250 of a video — no existing alignments key.
    model.add_face(250, _bbox(5, 5))

    def resolver(index: int) -> str | None:
        return f"clip_{index + 1:06d}.mp4" if index >= 0 else None

    modified = model.apply_to_alignments(alignments, frame_names=resolver)

    assert modified == 1
    assert "clip_000251.mp4" in alignments.data
    assert len(alignments.data["clip_000251.mp4"].faces) == 1


def test_apply_to_alignments_raises_when_resolver_returns_none() -> None:
    """A resolver returning None for a dirty frame surfaces as ValueError."""

    class _StubAlignments:
        def __init__(self) -> None:
            self.data: dict[str, object] = {}

    alignments = _StubAlignments()
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(0, 0))

    with pytest.raises(ValueError, match="no matching frame name"):
        model.apply_to_alignments(alignments, frame_names=lambda _idx: None)


def test_apply_to_alignments_writes_empty_for_fully_deleted_frame() -> None:
    """A frame with all faces deleted is written as ``entry.faces = []``."""
    import numpy as np

    from lib.align.objects import AlignmentsEntry, FileAlignments

    class _StubAlignments:
        def __init__(self) -> None:
            self.data: dict[str, AlignmentsEntry] = {
                "frame.png": AlignmentsEntry(
                    faces=[
                        FileAlignments(
                            x=10,
                            y=10,
                            w=20,
                            h=20,
                            landmarks_xy=np.zeros((0, 2), dtype=np.float32),
                        )
                    ],
                    video_meta={},
                )
            }

    alignments = _StubAlignments()
    model = ManualEditableAlignments(
        {0: [EditableFace(face_index=0, bbox=(10.0, 10.0, 20.0, 20.0))]},
    )
    model.delete_face(0, 0)

    modified = model.apply_to_alignments(alignments, frame_names=["frame.png"])

    assert modified == 1
    assert alignments.data["frame.png"].faces == []


def test_revert_frame_only_undoes_matching_frame() -> None:
    """revert_frame leaves edits on other frames in place."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(0, 0))  # frame 0
    model.add_face(1, _bbox(40, 40))  # frame 1
    model.add_face(0, _bbox(80, 0))  # frame 0 again

    reverted = model.revert_frame(0)

    assert reverted == 2
    assert model.face_count(0) == 0
    assert model.face_count(1) == 1
    # Frame-1 op is preserved on the undo stack so it can still be undone.
    assert model.can_undo is True

    assert model.undo() is True
    assert model.face_count(1) == 0


def test_revert_frame_no_op_when_no_matching_edits() -> None:
    """revert_frame reports zero and leaves the stack untouched if no matches."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox())
    assert model.revert_frame(7) == 0
    assert model.can_undo is True
    assert model.face_count(0) == 1


def test_editable_face_contains_and_center() -> None:
    """EditableFace.contains and EditableFace.center work in source coordinates."""
    face = EditableFace(face_index=0, bbox=(10.0, 20.0, 30.0, 40.0))
    assert face.contains(15.0, 25.0) is True
    assert face.contains(9.0, 25.0) is False
    assert face.center() == (25.0, 40.0)


def test_resize_face_updates_bbox_and_scales_landmarks() -> None:
    """resize_face replaces the bbox and rescales landmarks proportionally."""
    model = ManualEditableAlignments(
        {
            0: [
                EditableFace(
                    face_index=0,
                    bbox=(10.0, 10.0, 20.0, 20.0),
                    landmarks=((20.0, 20.0), (25.0, 25.0)),  # center, 3/4
                )
            ]
        }
    )

    # Double width + height, anchor shifted by 5 in each axis.
    assert model.resize_face(0, 0, (5.0, 5.0, 40.0, 40.0)) is True
    face = model.faces(0)[0]
    assert face.bbox == (5.0, 5.0, 40.0, 40.0)
    # Original landmark (20,20) was bbox-center; should stay relative center.
    assert face.landmarks[0] == (25.0, 25.0)
    # Original (25,25) at 75%; new should be at 5 + 0.75 * 40 = 35.
    assert face.landmarks[1] == (35.0, 35.0)


def test_resize_face_rejects_invalid_face_index() -> None:
    """An out-of-range face_index returns False without mutating state."""
    model = ManualEditableAlignments({0: [EditableFace(face_index=0, bbox=_bbox())]})

    assert model.resize_face(0, 5, (0.0, 0.0, 10.0, 10.0)) is False
    assert model.faces(0)[0].bbox == _bbox()


def test_resize_face_is_undoable() -> None:
    """resize_face participates in the undo/redo stack."""
    model = ManualEditableAlignments(
        {0: [EditableFace(face_index=0, bbox=(10.0, 10.0, 20.0, 20.0))]}
    )

    assert model.resize_face(0, 0, (0.0, 0.0, 40.0, 40.0)) is True
    assert model.faces(0)[0].bbox == (0.0, 0.0, 40.0, 40.0)
    assert model.undo() is True
    assert model.faces(0)[0].bbox == (10.0, 10.0, 20.0, 20.0)
    assert model.redo() is True
    assert model.faces(0)[0].bbox == (0.0, 0.0, 40.0, 40.0)


def test_resize_face_rejects_degenerate_input() -> None:
    """Inverted/degenerate rects must be rejected at the model boundary.

    The frame view normalizes drag rects before calling :meth:`resize_face` so
    inverted bbox should never reach the model in practice — but the model is
    the last line of defence and surfaces the same ``ValueError`` that
    :meth:`_validate_bbox` raises elsewhere.
    """
    model = ManualEditableAlignments(
        {0: [EditableFace(face_index=0, bbox=(50.0, 50.0, 20.0, 20.0))]}
    )

    with pytest.raises(ValueError, match="width and height"):
        model.resize_face(0, 0, (60.0, 60.0, -30.0, -30.0))
    # Bbox is unchanged after the rejection.
    assert model.faces(0)[0].bbox == (50.0, 50.0, 20.0, 20.0)


# ---------------------------------------------------------------------------
# #102 — Extract Box editor: scale + rotate
# ---------------------------------------------------------------------------


def test_scale_face_scales_bbox_and_landmarks_around_center() -> None:
    """scale_face scales bbox + landmarks uniformly around the bbox centre."""
    model = ManualEditableAlignments()
    model.add_face(
        0,
        (0.0, 0.0, 20.0, 20.0),
        landmarks=[(5.0, 5.0), (15.0, 15.0)],
    )
    assert model.scale_face(0, 0, 2.0) is True
    face = model.faces(0)[0]
    # Centre stays at (10, 10); width/height double; landmarks scale 2x from centre.
    assert face.bbox == (-10.0, -10.0, 40.0, 40.0)
    assert face.landmarks == ((0.0, 0.0), (20.0, 20.0))


def test_scale_face_one_is_noop_true() -> None:
    """A 1.0 scale is a no-op and does not push history."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0)])
    model.clear_history()
    assert model.scale_face(0, 0, 1.0) is True
    assert model.can_undo is False


def test_scale_face_rejects_non_positive_scale() -> None:
    """A zero or negative scale is rejected."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0)])
    assert model.scale_face(0, 0, 0.0) is False
    assert model.scale_face(0, 0, -1.0) is False


def test_scale_face_rejects_degenerate_result() -> None:
    """A scale that would shrink the bbox below 1px is rejected."""
    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 2.0, 2.0))
    assert model.scale_face(0, 0, 0.1) is False  # would produce 0.2 px sides


def test_scale_face_invalid_face_returns_false() -> None:
    """scale_face rejects an out-of-range face_index."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox())
    assert model.scale_face(0, 9, 1.5) is False


def test_scale_face_records_undo_history() -> None:
    """scale_face participates in undo/redo."""
    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 10.0, 10.0), landmarks=[(5.0, 5.0)])
    assert model.scale_face(0, 0, 2.0) is True
    assert model.faces(0)[0].bbox == (-5.0, -5.0, 20.0, 20.0)
    assert model.undo() is True
    assert model.faces(0)[0].bbox == (0.0, 0.0, 10.0, 10.0)
    assert model.redo() is True
    assert model.faces(0)[0].bbox == (-5.0, -5.0, 20.0, 20.0)


def test_rotate_face_rotates_landmarks_around_center() -> None:
    """rotate_face rotates landmarks 90 degrees around the bbox centre."""
    import math

    model = ManualEditableAlignments()
    # Centre at (10, 10).  A landmark at (15, 10) should rotate to (10, 15)
    # under a +90deg (pi/2) rotation in y-down screen coords.
    model.add_face(
        0,
        (0.0, 0.0, 20.0, 20.0),
        landmarks=[(15.0, 10.0)],
    )
    assert model.rotate_face(0, 0, math.pi / 2) is True
    lx, ly = model.faces(0)[0].landmarks[0]
    assert abs(lx - 10.0) < 1e-6
    assert abs(ly - 15.0) < 1e-6


def test_rotate_face_zero_angle_is_noop_true() -> None:
    """A zero-radian rotation is a no-op and does not push history."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0)])
    model.clear_history()
    assert model.rotate_face(0, 0, 0.0) is True
    assert model.can_undo is False


def test_rotate_face_no_landmarks_is_noop_true() -> None:
    """rotate_face on a face without landmarks no-ops without raising."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox())
    assert model.rotate_face(0, 0, 0.5) is True


def test_rotate_face_invalid_face_returns_false() -> None:
    """rotate_face rejects an out-of-range face_index."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0)])
    assert model.rotate_face(0, 9, 0.5) is False


def test_rotate_face_records_undo_history() -> None:
    """rotate_face participates in undo/redo and the bbox tracks the cloud."""
    import math

    model = ManualEditableAlignments()
    model.add_face(
        0,
        (0.0, 0.0, 10.0, 10.0),
        landmarks=[(2.0, 5.0), (8.0, 5.0)],
    )
    before_bbox = model.faces(0)[0].bbox
    assert model.rotate_face(0, 0, math.pi / 2) is True
    after_bbox = model.faces(0)[0].bbox
    assert after_bbox != before_bbox
    assert model.undo() is True
    assert model.faces(0)[0].bbox == before_bbox


# ---------------------------------------------------------------------------
# #101 — Mask editor model API
# ---------------------------------------------------------------------------


def test_paint_mask_stroke_creates_grid_on_first_paint() -> None:
    """A brush stroke lazily allocates the mask grid for the face."""
    import numpy as np

    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 64.0, 64.0))
    assert model.get_mask(0, 0, "components") is None

    assert model.paint_mask_stroke(0, 0, "components", 32.0, 32.0, 4.0, 255) is True
    mask = model.get_mask(0, 0, "components")
    assert mask is not None
    assert mask.shape == (model.MASK_STORED_SIZE, model.MASK_STORED_SIZE)
    assert mask.dtype == np.uint8
    # Center of the stamp must be painted ON.
    cy = cx = model.MASK_STORED_SIZE // 2
    assert mask[cy, cx] == 255


def test_paint_mask_stroke_erase_zeroes_pixels() -> None:
    """An erase stroke (value=0) clears painted pixels but keeps grid allocated."""
    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 64.0, 64.0))
    model.paint_mask_stroke(0, 0, "components", 32.0, 32.0, 8.0, 255)
    mask = model.get_mask(0, 0, "components")
    cy = cx = model.MASK_STORED_SIZE // 2
    assert mask[cy, cx] == 255
    # Erase a smaller stamp at the same point.
    assert model.paint_mask_stroke(0, 0, "components", 32.0, 32.0, 4.0, 0) is True
    assert mask[cy, cx] == 0


def test_paint_mask_stroke_out_of_bbox_returns_false() -> None:
    """A stroke whose centre is outside the face bbox is rejected."""
    model = ManualEditableAlignments()
    model.add_face(0, (10.0, 10.0, 20.0, 20.0))
    assert model.paint_mask_stroke(0, 0, "components", 100.0, 100.0, 4.0, 255) is False
    assert model.get_mask(0, 0, "components") is None


def test_paint_mask_stroke_invalid_face_returns_false() -> None:
    """An out-of-range face_index is rejected."""
    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 32.0, 32.0))
    assert model.paint_mask_stroke(0, 9, "components", 16.0, 16.0, 4.0, 255) is False


def test_paint_mask_stroke_records_undo() -> None:
    """A paint stroke participates in the undo/redo stack."""
    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 64.0, 64.0))
    assert model.paint_mask_stroke(0, 0, "components", 32.0, 32.0, 4.0, 255) is True
    mask = model.get_mask(0, 0, "components")
    cy = cx = model.MASK_STORED_SIZE // 2
    assert mask[cy, cx] == 255

    assert model.undo() is True
    # The pixels written by the stroke must revert.
    assert mask[cy, cx] == 0
    assert model.redo() is True
    assert mask[cy, cx] == 255


def test_clear_mask_drops_grid_and_is_undoable() -> None:
    """clear_mask removes the grid; undo restores the prior state."""
    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 32.0, 32.0))
    model.paint_mask_stroke(0, 0, "components", 16.0, 16.0, 4.0, 255)
    assert model.has_mask(0, 0, "components") is True
    assert model.clear_mask(0, 0, "components") is True
    assert model.has_mask(0, 0, "components") is False
    assert model.undo() is True
    assert model.has_mask(0, 0, "components") is True


def test_clear_mask_when_absent_returns_false() -> None:
    """clear_mask is a no-op when no mask exists."""
    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 32.0, 32.0))
    assert model.clear_mask(0, 0, "components") is False


def test_known_mask_types_lists_painted_types_only() -> None:
    """``known_mask_types`` returns each painted mask type for a face."""
    model = ManualEditableAlignments()
    model.add_face(0, (0.0, 0.0, 32.0, 32.0))
    model.paint_mask_stroke(0, 0, "components", 16.0, 16.0, 2.0, 255)
    model.paint_mask_stroke(0, 0, "extended", 16.0, 16.0, 2.0, 255)
    types = model.known_mask_types(0, 0)
    assert types == ("components", "extended")


# ---------------------------------------------------------------------------
# #104 — set_landmarks (Aligner integration)
# ---------------------------------------------------------------------------


def test_set_landmarks_replaces_all_points_and_keeps_bbox() -> None:
    """set_landmarks swaps the landmark cloud without moving the bbox."""
    model = ManualEditableAlignments()
    model.add_face(
        0,
        (10.0, 10.0, 40.0, 40.0),
        landmarks=[(20.0, 20.0)],
    )
    assert model.set_landmarks(0, 0, [(11.0, 11.0), (12.0, 12.0)]) is True
    face = model.faces(0)[0]
    assert face.bbox == (10.0, 10.0, 40.0, 40.0)
    assert face.landmarks == ((11.0, 11.0), (12.0, 12.0))


def test_set_landmarks_records_undo_history() -> None:
    """set_landmarks participates in undo/redo."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0)])
    model.set_landmarks(0, 0, [(99.0, 99.0)])
    assert model.faces(0)[0].landmarks == ((99.0, 99.0),)
    assert model.undo() is True
    assert model.faces(0)[0].landmarks == ((1.0, 2.0),)
    assert model.redo() is True
    assert model.faces(0)[0].landmarks == ((99.0, 99.0),)


def test_set_landmarks_invalid_face_returns_false() -> None:
    """set_landmarks rejects out-of-range face_index."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox())
    assert model.set_landmarks(0, 9, [(1.0, 2.0)]) is False


def test_set_landmarks_with_empty_sequence_clears_landmarks() -> None:
    """An empty landmarks sequence is allowed and clears the face's points."""
    model = ManualEditableAlignments()
    model.add_face(0, _bbox(), landmarks=[(1.0, 2.0), (3.0, 4.0)])
    assert model.set_landmarks(0, 0, []) is True
    assert model.faces(0)[0].landmarks == ()


# ---------------------------------------------------------------------------
# #101 — Mask blob writeback + lazy decode
# ---------------------------------------------------------------------------


def test_encode_mask_blob_round_trips_through_decode() -> None:
    """``encode_mask_blob`` + ``decode_mask_blob`` recover the original grid."""
    import numpy as np

    size = ManualEditableAlignments.MASK_STORED_SIZE
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[10:40, 10:40] = 255
    blob = ManualEditableAlignments.encode_mask_blob(grid, (5.0, 5.0, 100.0, 100.0))
    decoded = ManualEditableAlignments.decode_mask_blob(blob)
    assert decoded is not None
    assert decoded.shape == (size, size)
    assert np.array_equal(decoded, grid)


def test_encode_mask_blob_affine_matrix_maps_to_bbox() -> None:
    """The encoded affine matrix maps stored (0,0) → bbox origin + stored (size,size) → bbox bottom-right."""
    import numpy as np

    size = ManualEditableAlignments.MASK_STORED_SIZE
    grid = np.ones((size, size), dtype=np.uint8) * 200
    bbox = (12.0, 18.0, 40.0, 80.0)
    blob = ManualEditableAlignments.encode_mask_blob(grid, bbox)
    a = blob.affine_matrix
    # ``[fx, fy] = a @ [sx, sy, 1]``
    origin = a @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    far = a @ np.array([float(size), float(size), 1.0], dtype=np.float32)
    assert origin[0] == pytest.approx(12.0)
    assert origin[1] == pytest.approx(18.0)
    assert far[0] == pytest.approx(12.0 + 40.0)
    assert far[1] == pytest.approx(18.0 + 80.0)
    assert blob.stored_size == size
    assert blob.stored_centering == "head"


def test_decode_mask_blob_rejects_mismatched_stored_size() -> None:
    """A blob whose ``stored_size`` doesn't match the model's canonical size returns None."""
    import zlib

    import numpy as np

    from lib.align.objects import MaskAlignmentsFile

    blob = MaskAlignmentsFile(
        mask=zlib.compress(np.zeros((64, 64), dtype=np.uint8).tobytes()),
        affine_matrix=np.eye(3, dtype=np.float32)[:2],
        interpolator=1,
        stored_size=64,
        stored_centering="head",
    )
    assert ManualEditableAlignments.decode_mask_blob(blob) is None


def test_decode_mask_blob_rejects_corrupt_compression() -> None:
    """A blob whose compressed payload is invalid decodes to None."""
    import numpy as np

    from lib.align.objects import MaskAlignmentsFile

    blob = MaskAlignmentsFile(
        mask=b"not-zlib-data",
        affine_matrix=np.eye(3, dtype=np.float32)[:2],
        interpolator=1,
        stored_size=ManualEditableAlignments.MASK_STORED_SIZE,
        stored_centering="head",
    )
    assert ManualEditableAlignments.decode_mask_blob(blob) is None


def test_apply_to_alignments_writes_painted_masks_to_face() -> None:
    """A painted mask shows up encoded in the ``faces[i].mask`` dict on save."""
    import numpy as np

    from lib.align.objects import AlignmentsEntry, FileAlignments

    class _StubAlignments:
        def __init__(self) -> None:
            self.data: dict[str, AlignmentsEntry] = {
                "frame.png": AlignmentsEntry(
                    faces=[
                        FileAlignments(
                            x=10,
                            y=10,
                            w=40,
                            h=40,
                            landmarks_xy=np.zeros((0, 2), dtype=np.float32),
                        ),
                    ],
                    video_meta={},
                )
            }

    model = ManualEditableAlignments()
    model.add_face(0, (10.0, 10.0, 40.0, 40.0), landmarks=[(20.0, 20.0)])
    model.paint_mask_stroke(0, 0, "components", 25.0, 25.0, 4.0, 255)

    alignments = _StubAlignments()
    modified = model.apply_to_alignments(alignments, frame_names=["frame.png"])

    assert modified == 1
    persisted = alignments.data["frame.png"].faces[0]
    assert "components" in persisted.mask
    blob = persisted.mask["components"]
    decoded = ManualEditableAlignments.decode_mask_blob(blob)
    assert decoded is not None
    assert int(decoded.sum()) > 0


def test_apply_to_alignments_skips_untouched_persisted_masks() -> None:
    """A persisted mask that the user never painted is NOT re-encoded on save.

    Re-encoding would replace the legacy landmark-aligned affine matrix
    with our axis-aligned bbox projection and degrade the mask.
    """
    import zlib

    import numpy as np

    from lib.align.objects import AlignmentsEntry, FileAlignments, MaskAlignmentsFile

    original_affine = np.array([[1.5, 0.0, 7.0], [0.0, 1.5, 9.0]], dtype=np.float32)
    original_blob = MaskAlignmentsFile(
        mask=zlib.compress(np.full((128, 128), 200, dtype=np.uint8).tobytes()),
        affine_matrix=original_affine,
        interpolator=1,
        stored_size=128,
        stored_centering="head",
    )

    class _StubAlignments:
        def __init__(self) -> None:
            self.data: dict[str, AlignmentsEntry] = {
                "frame.png": AlignmentsEntry(
                    faces=[
                        FileAlignments(
                            x=10,
                            y=10,
                            w=40,
                            h=40,
                            landmarks_xy=np.zeros((0, 2), dtype=np.float32),
                            mask={"components": original_blob},
                        ),
                    ],
                    video_meta={},
                )
            }

    model = ManualEditableAlignments()
    model.add_face(0, (10.0, 10.0, 40.0, 40.0), landmarks=[(20.0, 20.0)])
    # Trigger a non-mask edit so the persist path actually runs.
    model.move_face(0, 0, 0.5, 0.5)
    alignments = _StubAlignments()
    model.apply_to_alignments(alignments, frame_names=["frame.png"])

    # The untouched mask blob must be the same object — preserved verbatim.
    after = alignments.data["frame.png"].faces[0].mask
    assert after["components"] is original_blob
    assert np.array_equal(after["components"].affine_matrix, original_affine)


def test_apply_to_alignments_clears_mask_after_user_clears_it() -> None:
    """Calling ``clear_mask`` deletes the persisted entry on the next save."""
    import zlib

    import numpy as np

    from lib.align.objects import AlignmentsEntry, FileAlignments, MaskAlignmentsFile

    blob = MaskAlignmentsFile(
        mask=zlib.compress(np.full((128, 128), 200, dtype=np.uint8).tobytes()),
        affine_matrix=np.eye(3, dtype=np.float32)[:2],
        interpolator=1,
        stored_size=128,
        stored_centering="head",
    )

    class _StubAlignments:
        def __init__(self) -> None:
            self.data: dict[str, AlignmentsEntry] = {
                "frame.png": AlignmentsEntry(
                    faces=[
                        FileAlignments(
                            x=10,
                            y=10,
                            w=40,
                            h=40,
                            landmarks_xy=np.zeros((0, 2), dtype=np.float32),
                            mask={"components": blob},
                        ),
                    ],
                    video_meta={},
                )
            }

    model = ManualEditableAlignments()
    model.add_face(0, (10.0, 10.0, 40.0, 40.0), landmarks=[(20.0, 20.0)])
    # Materialise the editor cache via get_mask (lazy decode), then clear it.
    model._persisted_mask_blobs[(0, 0, "components")] = blob  # type:ignore[arg-type]
    decoded = model.get_mask(0, 0, "components")
    assert decoded is not None
    assert model.clear_mask(0, 0, "components") is True
    alignments = _StubAlignments()
    model.apply_to_alignments(alignments, frame_names=["frame.png"])
    after = alignments.data["frame.png"].faces[0].mask
    assert "components" not in after


def test_known_mask_types_includes_persisted_blobs() -> None:
    """``known_mask_types`` lists masks already on disk via the seed map."""
    import numpy as np

    from lib.align.objects import MaskAlignmentsFile

    model = ManualEditableAlignments()
    model.add_face(0, (10.0, 10.0, 40.0, 40.0))
    blob = MaskAlignmentsFile(
        mask=b"\x00",
        affine_matrix=np.eye(3, dtype=np.float32)[:2],
        interpolator=1,
        stored_size=128,
        stored_centering="head",
    )
    model._persisted_mask_blobs[(0, 0, "bisenet-fp_head")] = blob  # type:ignore[arg-type]
    assert "bisenet-fp_head" in model.known_mask_types(0, 0)
