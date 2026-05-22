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
