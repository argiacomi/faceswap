#!/usr/bin/env python3
"""Tests for Tk manual detected-face update helpers."""

from __future__ import annotations

import typing as T
from types import SimpleNamespace

from tools.manual.detected_faces import FaceUpdate


class _Var:
    """Small stand-in for ``tkinter.Variable`` in update helper tests."""

    def __init__(self, value: bool = False) -> None:
        self.value = value

    def get(self) -> bool:
        """Return the stored value."""
        return self.value

    def set(self, value: bool) -> None:
        """Set the stored value."""
        self.value = value


class _Globals:
    """Minimal globals object for ``FaceUpdate`` tests."""

    def __init__(self) -> None:
        self.var_full_update = _Var()
        self.face_index = 0
        self.selected_face_indices: tuple[int, ...] = ()

    def set_face_index(self, index: int) -> None:
        """Store the active face index."""
        self.face_index = index

    def clear_selected_face_indices(self) -> None:
        """Clear selected faces."""
        self.selected_face_indices = ()


def _face_update(
    mask: dict[str, T.Any], faces: list[T.Any] | None = None
) -> tuple[FaceUpdate, T.Any]:
    """Create a ``FaceUpdate`` backed by a minimal detected-faces stub."""
    faces = [SimpleNamespace(mask=mask)] if faces is None else faces
    detected_faces = SimpleNamespace(
        _globals=_Globals(),
        _frame_faces=[faces],
        _updated_frame_indices=set(),
        tk_unsaved=_Var(),
        tk_edited=_Var(),
        tk_face_count_changed=_Var(),
        extractor=None,
    )
    return FaceUpdate(detected_faces), detected_faces  # type: ignore[arg-type]


def test_delete_mask_removes_mask_key_and_marks_frame_dirty() -> None:
    """Deleting a present mask removes the metadata key, not just pixels."""
    update, detected_faces = _face_update({"bisenet-fp": object()})

    assert update.delete_mask(0, 0, "bisenet-fp") is True

    assert "bisenet-fp" not in detected_faces._frame_faces[0][0].mask
    assert detected_faces._updated_frame_indices == {0}
    assert detected_faces.tk_unsaved.get() is True
    assert detected_faces.tk_edited.get() is True
    assert detected_faces._globals.var_full_update.get() is True


def test_delete_mask_missing_key_is_noop() -> None:
    """Deleting an absent mask does not create an unsaved edit."""
    existing_mask = object()
    update, detected_faces = _face_update({"components": existing_mask})

    assert update.delete_mask(0, 0, "bisenet-fp") is False

    assert detected_faces._frame_faces[0][0].mask == {"components": existing_mask}
    assert detected_faces._updated_frame_indices == set()
    assert detected_faces.tk_unsaved.get() is False
    assert detected_faces.tk_edited.get() is False
    assert detected_faces._globals.var_full_update.get() is False


def test_delete_many_removes_selected_faces_in_descending_order() -> None:
    """Deleting multiple faces handles shifted indices correctly."""
    faces = [SimpleNamespace(mask={}, marker=idx) for idx in range(4)]
    update, detected_faces = _face_update({}, faces=faces)

    update.delete_many(0, (1, 3))

    assert [face.marker for face in detected_faces._frame_faces[0]] == [0, 2]
    assert detected_faces._updated_frame_indices == {0}
    assert detected_faces.tk_unsaved.get() is True
    assert detected_faces.tk_face_count_changed.get() is True
    assert detected_faces._globals.var_full_update.get() is True
