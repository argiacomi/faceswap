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


def _face_update(mask: dict[str, T.Any]) -> tuple[FaceUpdate, T.Any]:
    """Create a ``FaceUpdate`` backed by a minimal detected-faces stub."""
    detected_faces = SimpleNamespace(
        _globals=SimpleNamespace(var_full_update=_Var()),
        _frame_faces=[[SimpleNamespace(mask=mask)]],
        _updated_frame_indices=set(),
        tk_unsaved=_Var(),
        tk_edited=_Var(),
        extractor=None,
    )
    return FaceUpdate(detected_faces), detected_faces


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
