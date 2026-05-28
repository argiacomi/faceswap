#!/usr/bin python3
"""Pytest unit tests for :mod:`tools.preview.viewer`"""

from __future__ import annotations

import tkinter as tk
import typing as T
from tkinter import ttk
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_mock
from PIL import ImageTk

from lib.logger import log_setup

# Need to setup logging to avoid trace/verbose errors
log_setup("DEBUG", "pytest_viewer.log", "PyTest, False")

from tools.preview.viewer import _Faces, FacesDisplay, ImagesCanvas  # pylint:disable=wrong-import-position  # noqa

if T.TYPE_CHECKING:
    from lib.align.aligned_face import CenteringType


# pylint:disable=protected-access


_PREVIEW_COLUMNS = 4
_PREVIEW_FACE_SIZE = 128


def test__faces():
    """Test the :class:`~tools.preview.viewer._Faces dataclass initializes correctly"""
    faces = _Faces(5, 64)
    assert isinstance(faces.filenames, list) and not faces.filenames
    assert faces.matrix.shape == (5, 2, 3)
    assert faces.src.shape == (5, 64, 64, 3)
    assert faces.dst.shape == (5, 64, 64, 3)


class TestFacesDisplay:
    """Test :class:`~tools.preview.viewer.FacesDisplay`"""

    _padding = 64

    def get_faces_display_instance(
        self, columns: int = _PREVIEW_COLUMNS, face_size: int = _PREVIEW_FACE_SIZE
    ) -> FacesDisplay:
        """Obtain an instance of :class:`~tools.preview.viewer.FacesDisplay`."""
        app = MagicMock()
        retval = FacesDisplay(app, face_size, self._padding, columns)
        retval._faces = _Faces(columns, face_size)
        return retval

    def test_init(self) -> None:
        """Test :class:`~tools.preview.viewer.FacesDisplay` __init__ method"""
        f_display = self.get_faces_display_instance(face_size=256)
        assert f_display._size == 256
        assert f_display._padding == self._padding
        assert isinstance(f_display._app, MagicMock)

        assert f_display._display_dims == (1, 1)
        assert isinstance(f_display._faces, _Faces)

        assert f_display._centering is None
        assert f_display._faces_source.size == 0
        assert f_display._faces_dest.size == 0
        assert f_display._tk_image is None
        assert f_display.update_source is False
        assert not f_display.source and isinstance(f_display.source, list)
        assert not f_display.destination and isinstance(f_display.destination, list)

    def test_public_display_state_setters(self) -> None:
        """Public setters update the layout and source-face centering contract."""
        f_display = self.get_faces_display_instance()
        centering: CenteringType = "legacy"

        f_display.source = [None for _ in range(_PREVIEW_COLUMNS)]  # type:ignore
        f_display.set_centering_offset(centering, 0.80)
        f_display.set_display_dimensions((800, 600))

        assert f_display._total_columns == _PREVIEW_COLUMNS
        assert f_display._centering == centering
        assert f_display._y_offset == 0.80
        assert f_display._display_dims == (800, 600)

    # TODO remove the next line that suppresses a weird pytest bug when it tears down the tempdir
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_update_tk_image_builds_scaled_public_image(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        """update_tk_image builds, scales and exposes the public tk_image."""
        f_display = self.get_faces_display_instance()
        f_display._build_faces_image = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
        f_display._get_scale_size = T.cast(
            MagicMock,  # type:ignore
            mocker.MagicMock(return_value=(128, 128)),
        )
        f_display._faces_source = np.zeros(
            (_PREVIEW_FACE_SIZE, _PREVIEW_FACE_SIZE, 3), dtype=np.uint8
        )
        f_display._faces_dest = np.zeros(
            (_PREVIEW_FACE_SIZE, _PREVIEW_FACE_SIZE, 3), dtype=np.uint8
        )

        class MockPhotoImage:
            """Mock PIL ImageTk image to avoid creating a real Tk interpreter."""

            def __init__(self, image) -> None:
                self._image = image

            def width(self) -> int:
                """Return the mocked image width."""
                return self._image.width

            def height(self) -> int:
                """Return the mocked image height."""
                return self._image.height

        mocker.patch("tools.preview.viewer.ImageTk.PhotoImage", MockPhotoImage)
        f_display.update_tk_image()

        f_display._build_faces_image.assert_called_once()
        f_display._get_scale_size.assert_called_once()
        assert isinstance(f_display._tk_image, ImageTk.PhotoImage)
        assert f_display._tk_image.width() == 128
        assert f_display._tk_image.height() == 128
        assert f_display.tk_image == f_display._tk_image

    def test_get_scale_size_fits_image_to_display_contract(self) -> None:
        """Scale size preserves aspect ratio and fits inside the display dimensions."""
        f_display = self.get_faces_display_instance()
        f_display.set_display_dimensions((800, 600))

        tall_image = np.zeros((_PREVIEW_FACE_SIZE * 2, _PREVIEW_FACE_SIZE, 3), dtype=np.uint8)
        wide_image = np.zeros((_PREVIEW_FACE_SIZE, _PREVIEW_FACE_SIZE * 4, 3), dtype=np.uint8)

        assert f_display._get_scale_size(tall_image) == (300, 600)
        assert f_display._get_scale_size(wide_image) == (800, 200)

    def test_build_faces_image_outputs_expected_source_and_destination_shapes(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Building preview rows preserves public image shape contracts."""
        header_size = 32

        f_display = self.get_faces_display_instance(face_size=256)
        f_display._faces_from_frames = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
        f_display._header_text = T.cast(  # type:ignore
            MagicMock,
            mocker.MagicMock(return_value=np.random.rand(header_size, 256 * _PREVIEW_COLUMNS, 3)),
        )
        f_display._draw_rect = T.cast(
            MagicMock,  # type:ignore
            mocker.MagicMock(side_effect=lambda x: x),
        )

        f_display.update_source = True
        f_display._build_faces_image()

        f_display._faces_from_frames.assert_called_once()
        f_display._header_text.assert_called_once()
        assert f_display._draw_rect.call_count == _PREVIEW_COLUMNS * 2
        assert f_display._faces_source.shape == (256 + header_size, 256 * _PREVIEW_COLUMNS, 3)
        assert f_display._faces_dest.shape == (256, 256 * _PREVIEW_COLUMNS, 3)

    def test_faces_from_frames_routes_source_updates_and_destination_refresh(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Frame extraction refreshes sources only when requested, and destinations every time."""
        f_display = self.get_faces_display_instance()
        f_display.source = [mocker.MagicMock() for _ in range(3)]
        f_display.destination = [
            np.random.rand(_PREVIEW_FACE_SIZE, _PREVIEW_FACE_SIZE, 3) for _ in range(3)
        ]
        f_display._crop_source_faces = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
        f_display._crop_destination_faces = T.cast(MagicMock, mocker.MagicMock())  # type:ignore

        f_display.update_source = True
        f_display._faces_from_frames()
        f_display._crop_source_faces.assert_called_once()
        f_display._crop_destination_faces.assert_called_once()

        f_display._crop_source_faces.reset_mock()
        f_display._crop_destination_faces.reset_mock()

        f_display.update_source = False
        f_display._faces_from_frames()
        f_display._crop_source_faces.assert_not_called()
        f_display._crop_destination_faces.assert_called_once()

    def test_crop_source_and_destination_faces_preserve_image_contracts(
        self, monkeypatch: pytest.MonkeyPatch, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Representative crop path captures filenames, matrices and output face shapes."""
        columns = _PREVIEW_COLUMNS
        face_size = _PREVIEW_FACE_SIZE
        f_display = self.get_faces_display_instance(columns, face_size)
        f_display._centering = "face"
        f_display.update_source = True

        transform_image_mock = mocker.MagicMock(
            return_value=np.zeros((face_size, face_size, 3), dtype=np.uint8)
        )
        monkeypatch.setattr("tools.preview.viewer.transform_image", transform_image_mock)

        mats = np.random.random((columns, 2, 3)).astype(np.float32)
        f_display.source = [mocker.MagicMock() for _ in range(columns)]
        for idx, mock in enumerate(f_display.source):
            assert isinstance(mock, MagicMock)
            mock.inbound.detected_faces.__getitem__ = lambda self, x, y=mock: y
            mock.aligned.matrix = mats[idx]
            mock.inbound.filename = f"test_filename_{idx}.txt"
            mock.inbound.image = np.random.rand(1280, 720, 3)

        f_display.destination = [np.random.rand(1280, 720, 3) for _ in range(columns)]

        f_display._crop_source_faces()
        f_display._crop_destination_faces()

        assert f_display._faces.filenames == [f"test_filename_{idx}" for idx in range(columns)]
        assert np.array_equal(f_display._faces.matrix, mats)
        assert not f_display.update_source
        assert transform_image_mock.call_count == columns * 2
        assert f_display._faces.src.shape == (columns, face_size, face_size, 3)
        assert f_display._faces.dst.shape == (columns, face_size, face_size, 3)

    def test_header_text_builds_one_label_per_column(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        """Header text renders a label area that matches the face grid width."""
        f_display = self.get_faces_display_instance()
        f_display.source = [None for _ in range(_PREVIEW_COLUMNS)]  # type:ignore
        f_display._faces.filenames = [f"filename_{idx}.png" for idx in range(_PREVIEW_COLUMNS)]

        cv2_mock = mocker.patch("tools.preview.viewer.cv2")
        text_width, text_height = (100, 32)
        cv2_mock.getTextSize.return_value = [
            (text_width, text_height),
        ]

        header_box = f_display._header_text()
        assert cv2_mock.getTextSize.call_count == _PREVIEW_COLUMNS
        assert cv2_mock.putText.call_count == _PREVIEW_COLUMNS
        assert header_box.shape == (
            _PREVIEW_FACE_SIZE // 8,
            _PREVIEW_FACE_SIZE * _PREVIEW_COLUMNS,
            3,
        )

    def test_draw_rect_clips_and_returns_uint8_image(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        """The face-border helper preserves the image contract expected by the preview rows."""
        f_display = self.get_faces_display_instance()
        cv2_mock = mocker.patch("tools.preview.viewer.cv2")

        image = (np.random.rand(_PREVIEW_FACE_SIZE, _PREVIEW_FACE_SIZE, 3) * 255.0) + 50
        assert image.max() > 255.0
        output = f_display._draw_rect(image)
        cv2_mock.rectangle.assert_called_once()
        assert output.max() == 255
        assert output.dtype == np.uint8


class TestImagesCanvas:
    """Test :class:`~tools.preview.viewer.ImagesCanvas`"""

    @pytest.fixture
    def parent(self) -> MagicMock:
        """Mock object to act as the parent widget to the ImagesCanvas."""
        retval = MagicMock(spec=ttk.PanedWindow)
        retval.tk = retval
        retval._w = "mock_ttkPanedWindow"
        retval.children = {}
        retval.call = retval
        retval.createcommand = retval
        retval.preview_display = MagicMock(spec=FacesDisplay)
        return retval

    @pytest.fixture(name="images_canvas_instance")
    def images_canvas_fixture(self, parent) -> ImagesCanvas:
        """Fixture for creating a testing :class:`~tools.preview.viewer.ImagesCanvas` instance."""
        app = MagicMock()
        return ImagesCanvas(app, parent)

    def test_init(self, images_canvas_instance: ImagesCanvas, parent: MagicMock) -> None:
        """Test :class:`~tools.preview.viewer.ImagesCanvas` __init__ method"""
        assert images_canvas_instance._display == parent.preview_display
        assert isinstance(images_canvas_instance._canvas, tk.Canvas)
        assert images_canvas_instance._canvas.master == images_canvas_instance
        assert images_canvas_instance._canvas.winfo_ismapped()

    def test_resize(
        self,
        images_canvas_instance: ImagesCanvas,
        parent: MagicMock,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test :class:`~tools.preview.viewer.ImagesCanvas` resize method"""
        event_mock = mocker.MagicMock(spec=tk.Event, width=100, height=200)
        images_canvas_instance.reload = T.cast(MagicMock, mocker.MagicMock())  # type:ignore

        images_canvas_instance._resize(event_mock)

        parent.preview_display.set_display_dimensions.assert_called_once_with((100, 200))
        images_canvas_instance.reload.assert_called_once()

    def test_reload(
        self,
        images_canvas_instance: ImagesCanvas,
        parent: MagicMock,
        mocker: pytest_mock.MockerFixture,
    ) -> None:
        """Test :class:`~tools.preview.viewer.ImagesCanvas` reload method"""
        itemconfig_mock = mocker.patch.object(tk.Canvas, "itemconfig")

        images_canvas_instance.reload()

        parent.preview_display.update_tk_image.assert_called_once()
        itemconfig_mock.assert_called_once()
