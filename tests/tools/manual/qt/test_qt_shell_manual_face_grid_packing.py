#!/usr/bin/env python3
"""Dense face-grid packing tests for the Qt Manual Tool (#132)."""

from __future__ import annotations

from tools.manual.qt.face_viewer import CrossFrameFaceGridPanel
from tools.manual.qt.face_viewer.viewport import (
    _FACE_GRID_SIZES,
    FaceGridEntry,
    FaceGridThumbnailRenderer,
)
from tools.manual.session import ManualEditableAlignments


def _panel(qtbot) -> CrossFrameFaceGridPanel:  # type:ignore[no-untyped-def]
    panel = CrossFrameFaceGridPanel(FaceGridThumbnailRenderer(ManualEditableAlignments()))
    qtbot.addWidget(panel)
    return panel


def test_face_grid_uses_legacy_thumbnail_size_table(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Qt face-grid sizes match the legacy Tk table."""
    panel = _panel(qtbot)

    assert _FACE_GRID_SIZES == {
        "Tiny": 32,
        "Small": 64,
        "Medium": 96,
        "Large": 128,
        "Extra Large": 192,
    }

    for name, size in _FACE_GRID_SIZES.items():
        panel.set_face_size(name)
        assert panel.iconSize().width() == size
        assert panel.iconSize().height() == size
        assert panel.gridSize().width() == size + 4
        assert panel.gridSize().height() == size + 4


def test_face_grid_omits_per_thumbnail_labels_and_packs_tightly(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Dense grid items have no default text label and minimal spacing."""
    panel = _panel(qtbot)
    panel.set_face_size("Tiny")
    panel.set_entries(
        (
            FaceGridEntry(0, "frame_000.png", 0, None, (0.0, 0.0, 16.0, 16.0)),
            FaceGridEntry(0, "frame_000.png", 1, None, (16.0, 0.0, 16.0, 16.0)),
            FaceGridEntry(1, "frame_001.png", 0, None, (0.0, 0.0, 16.0, 16.0)),
        )
    )

    assert panel.spacing() == 1
    assert panel.gridSize().width() == panel.iconSize().width() + 4
    assert panel.gridSize().height() == panel.iconSize().height() + 4
    for row in range(panel.count()):
        item = panel.item(row)
        assert item is not None
        assert item.text() == ""
        assert "Frame" in item.toolTip()
