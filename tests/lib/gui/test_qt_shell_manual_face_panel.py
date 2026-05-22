#!/usr/bin/env python3
"""Native Qt Manual Tool face thumbnail panel tests."""

from __future__ import annotations

from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent

from lib.gui.qt_shell.manual_tool import FaceThumbnailPanel, _decode_jpeg_to_qimage
from tools.manual.session import FaceThumbnail


def _make_jpeg_bytes(color: str = "#3366ff", size: int = 24) -> bytes:
    """Create a small JPEG byte string for fixture data."""
    image = QImage(size, size, QImage.Format_RGB32)
    image.fill(QColor(color))
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    assert image.save(buffer, "JPG", 90)
    payload = bytes(buffer.data())
    buffer.close()
    # Sanity check: should look like a JPEG.
    assert payload.startswith(b"\xff\xd8")
    return payload


def _face(face_index: int, payload: bytes = b"") -> FaceThumbnail:
    """Build a fixture FaceThumbnail entry."""
    return FaceThumbnail(
        frame_index=0,
        frame_name="frame_000001.png",
        face_index=face_index,
        thumbnail_jpeg=payload,
    )


def test_decode_jpeg_to_qimage_returns_null_for_empty_payload() -> None:
    """The JPEG decoder gracefully handles empty payloads."""
    assert _decode_jpeg_to_qimage(b"").isNull()


def test_decode_jpeg_to_qimage_decodes_valid_payload() -> None:
    """A JPEG byte string round-trips to a non-null QImage of the expected size."""
    payload = _make_jpeg_bytes(size=32)
    image = _decode_jpeg_to_qimage(payload)
    assert image.isNull() is False
    assert image.width() == 32
    assert image.height() == 32


def test_face_panel_set_faces_populates_items(qtbot) -> None:  # type:ignore[no-untyped-def]
    """set_faces populates one item per face with the matching face_index."""
    panel = FaceThumbnailPanel()
    qtbot.addWidget(panel)
    payload = _make_jpeg_bytes()
    panel.set_faces([_face(0, payload), _face(1, payload), _face(2, payload)])

    assert panel.count() == 3
    assert panel.faces == (_face(0, payload), _face(1, payload), _face(2, payload))
    assert panel.currentRow() == 0
    assert panel.active_face() is not None
    assert panel.active_face().face_index == 0


def test_face_panel_empty_state_shows_placeholder_item(qtbot) -> None:  # type:ignore[no-untyped-def]
    """An empty frame still renders a non-selectable placeholder row."""
    panel = FaceThumbnailPanel()
    qtbot.addWidget(panel)
    panel.set_faces(())
    assert panel.count() == 1
    item = panel.item(0)
    assert item is not None
    assert item.text() == "No faces in frame"
    assert item.flags() == Qt.NoItemFlags
    assert panel.active_face() is None


def test_face_panel_emits_face_selected_on_row_change(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Selecting a row emits the face_selected signal with the face_index."""
    panel = FaceThumbnailPanel()
    qtbot.addWidget(panel)
    captured: list[int] = []
    panel.face_selected.connect(captured.append)
    payload = _make_jpeg_bytes()
    panel.set_faces([_face(0, payload), _face(1, payload), _face(2, payload)])

    captured.clear()
    panel.setCurrentRow(2)
    panel.setCurrentRow(1)
    assert captured == [2, 1]


def test_face_panel_keyboard_navigation_changes_selection(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Right-arrow keypress advances selection and emits the matching signal."""
    panel = FaceThumbnailPanel()
    qtbot.addWidget(panel)
    payload = _make_jpeg_bytes()
    panel.set_faces([_face(0, payload), _face(1, payload), _face(2, payload)])
    captured: list[int] = []
    panel.face_selected.connect(captured.append)

    panel.setFocus(Qt.OtherFocusReason)
    key = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
    panel.keyPressEvent(key)
    key = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
    panel.keyPressEvent(key)
    key = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Left, Qt.NoModifier)
    panel.keyPressEvent(key)

    assert panel.currentRow() == 1
    assert captured == [1, 2, 1]


def test_face_panel_select_face_resolves_face_index(qtbot) -> None:  # type:ignore[no-untyped-def]
    """select_face honors face_index regardless of row order changes."""
    panel = FaceThumbnailPanel()
    qtbot.addWidget(panel)
    payload = _make_jpeg_bytes()
    panel.set_faces([_face(0, payload), _face(1, payload), _face(2, payload)])

    assert panel.select_face(2) is True
    assert panel.currentRow() == 2
    assert panel.select_face(99) is False


def test_face_panel_refreshes_on_set_faces(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Calling set_faces a second time fully refreshes the panel state."""
    panel = FaceThumbnailPanel()
    qtbot.addWidget(panel)
    payload = _make_jpeg_bytes()
    panel.set_faces([_face(0, payload), _face(1, payload)])
    panel.set_faces([_face(0, payload)])  # Simulate a face removal.
    assert panel.count() == 1
    assert panel.faces == (_face(0, payload),)
    assert panel.currentRow() == 0

    panel.set_faces(())  # Simulate clearing the active frame.
    assert panel.count() == 1
    assert panel.item(0).text() == "No faces in frame"


def test_face_thumbnail_has_image_flag() -> None:
    """FaceThumbnail.has_image reflects whether a thumbnail payload is present."""
    assert _face(0, b"").has_image is False
    assert _face(0, b"\xff\xd8payload").has_image is True


def test_face_panel_uses_placeholder_for_undecodable_payload(qtbot) -> None:  # type:ignore[no-untyped-def]
    """Items with empty/undecodable payloads still get an icon."""
    panel = FaceThumbnailPanel()
    qtbot.addWidget(panel)
    panel.set_faces([_face(0, b""), _face(1, b"not-an-image")])
    assert panel.count() == 2
    for row in range(2):
        item = panel.item(row)
        assert item is not None
        assert item.icon().isNull() is False


def test_manual_tool_window_updates_editor_face_index_on_panel_selection(  # type:ignore[no-untyped-def]
    qtbot,
    tmp_path,
) -> None:
    """Selecting a face in the panel updates the shared editor state."""
    from pathlib import Path

    from PySide6.QtGui import QColor as _QColor
    from PySide6.QtGui import QPixmap as _QPixmap

    from lib.gui.qt_shell.manual_tool import ManualToolWindow
    from tools.manual.session import ManualSession

    folder = Path(tmp_path)
    pixmap = _QPixmap(32, 32)
    pixmap.fill(_QColor("#3366ff"))
    pixmap.save(str(folder / "frame_000.png"), "PNG")
    session = ManualSession.create(frames=str(folder))
    window = ManualToolWindow(session)
    qtbot.addWidget(window)

    payload = _make_jpeg_bytes()
    window.face_panel.set_faces([_face(0, payload), _face(1, payload), _face(2, payload)])

    captured: list[int] = []
    window.editor_state.subscribe("face_index", captured.append)

    window.face_panel.setCurrentRow(2)
    window.face_panel.setCurrentRow(1)

    assert captured == [2, 1]
    assert window.editor_state.face_index == 1
