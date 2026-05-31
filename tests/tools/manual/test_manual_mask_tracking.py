#!/usr/bin/env python3
"""Regression tests for mask touched-state undo/redo bookkeeping."""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from lib.align.objects import AlignmentsEntry, FileAlignments, MaskAlignmentsFile
from tools.manual.session import ManualEditableAlignments

_MASK_KEY = (0, 0, "components")
_BBOX = (0.0, 0.0, 64.0, 64.0)


class _StubAlignments:
    """Small alignments payload containing one face with one persisted mask."""

    def __init__(self, blob: MaskAlignmentsFile) -> None:
        self.data: dict[str, AlignmentsEntry] = {
            "frame.png": AlignmentsEntry(
                faces=[
                    FileAlignments(
                        x=0,
                        y=0,
                        w=64,
                        h=64,
                        landmarks_xy=np.zeros((0, 2), dtype=np.float32),
                        mask={"components": blob},
                    ),
                ],
                video_meta={},
            )
        }


def _sentinel_blob() -> MaskAlignmentsFile:
    """Return a persisted mask blob with a non-Qt affine sentinel."""
    grid = np.zeros(  # type: ignore[var-annotated]
        (
            ManualEditableAlignments.MASK_STORED_SIZE,
            ManualEditableAlignments.MASK_STORED_SIZE,
        ),
        dtype=np.uint8,
    )
    grid[2:16, 3:17] = 123
    affine = np.array([[1.25, 0.125, 7.0], [0.25, 1.5, 11.0]], dtype=np.float32)
    return MaskAlignmentsFile(
        mask=zlib.compress(grid.tobytes()),
        affine_matrix=affine,
        interpolator=1,
        stored_size=ManualEditableAlignments.MASK_STORED_SIZE,
        stored_centering="head",
    )


def _model_with_persisted_mask(blob: MaskAlignmentsFile) -> ManualEditableAlignments:
    """Return a clean editable model seeded with a persisted mask blob."""
    model = ManualEditableAlignments()
    model.add_face(0, _BBOX, landmarks=[(32.0, 32.0)])
    model.clear_history()
    model._persisted_mask_blobs[_MASK_KEY] = blob  # noqa: SLF001
    return model


def test_paint_undo_save_preserves_original_persisted_mask_blob() -> None:
    """Paint -> undo -> save preserves the original persisted blob/affine."""
    blob = _sentinel_blob()
    model = _model_with_persisted_mask(blob)

    assert model.paint_mask_stroke(0, 0, "components", 50.0, 50.0, 4.0, 255) is True
    assert model.undo() is True
    assert _MASK_KEY not in model._painted_masks  # noqa: SLF001

    alignments = _StubAlignments(blob)
    model.apply_to_alignments(alignments, frame_names=["frame.png"])

    after = alignments.data["frame.png"].faces[0].mask["components"]
    assert after is blob
    assert np.array_equal(after.affine_matrix, blob.affine_matrix)


def test_clear_undo_save_preserves_original_persisted_mask_blob() -> None:
    """Clear -> undo -> save preserves the original persisted blob/affine."""
    blob = _sentinel_blob()
    model = _model_with_persisted_mask(blob)

    assert model.clear_mask(0, 0, "components") is True
    assert model.undo() is True
    assert _MASK_KEY not in model._painted_masks  # noqa: SLF001

    alignments = _StubAlignments(blob)
    model.apply_to_alignments(alignments, frame_names=["frame.png"])

    after = alignments.data["frame.png"].faces[0].mask["components"]
    assert after is blob
    assert np.array_equal(after.affine_matrix, blob.affine_matrix)


def test_paint_redo_save_applies_redone_mask_edit() -> None:
    """Redoing paint marks the mask touched so save writes the redone edit."""
    blob = _sentinel_blob()
    model = _model_with_persisted_mask(blob)

    model.paint_mask_stroke(0, 0, "components", 50.0, 50.0, 4.0, 255)
    assert model.undo() is True
    assert model.redo() is True
    assert _MASK_KEY in model._painted_masks  # noqa: SLF001

    matrix = model.mask_affine_matrix(0, 0, "components")
    assert matrix is not None
    stored_point = cv2.transform(
        np.asarray([[[50.0, 50.0]]], dtype=np.float32),
        np.asarray(matrix, dtype=np.float32)[:2],
    ).reshape(2)

    alignments = _StubAlignments(blob)
    model.apply_to_alignments(alignments, frame_names=["frame.png"])

    after = alignments.data["frame.png"].faces[0].mask["components"]
    assert after is not blob
    decoded = ManualEditableAlignments.decode_mask_blob(after)
    assert decoded is not None
    stamp_x, stamp_y = np.rint(stored_point).astype("int32")
    assert int(decoded[int(stamp_y), int(stamp_x)]) == 255


def test_clear_redo_save_deletes_redone_mask_edit() -> None:
    """Redoing clear marks the mask touched so save deletes the mask key."""
    blob = _sentinel_blob()
    model = _model_with_persisted_mask(blob)

    assert model.clear_mask(0, 0, "components") is True
    assert model.undo() is True
    assert model.redo() is True
    assert _MASK_KEY in model._painted_masks  # noqa: SLF001

    alignments = _StubAlignments(blob)
    model.apply_to_alignments(alignments, frame_names=["frame.png"])

    assert "components" not in alignments.data["frame.png"].faces[0].mask


def test_clear_history_drops_stale_mask_touched_tracking() -> None:
    """Successful-save history reset drops stale touched-mask state too."""
    blob = _sentinel_blob()
    model = _model_with_persisted_mask(blob)

    assert model.paint_mask_stroke(0, 0, "components", 50.0, 50.0, 4.0, 255) is True
    assert _MASK_KEY in model._painted_masks  # noqa: SLF001

    model.clear_history()

    assert _MASK_KEY not in model._painted_masks  # noqa: SLF001
