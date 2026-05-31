#!/usr/bin/env python3
"""Unit tests for ``tools/mask`` covering the #198 fixes:

* Fatal validation paths exit with non-zero status (not the previous
  silent ``sys.exit(0)``).
* ``Import._map_images`` uses O(N) stem-indexed dict lookup instead of
  the previous O(N^2) ``next(... in file_list)`` + ``file_list.pop``
  pattern, AND still reports extra masks via ``_warn_extra_masks``.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from tools.mask import mask as mask_module
from tools.mask import mask_output as mask_output_module
from tools.mask.mask_import import Import

# ---------------------------------------------------------------------------
# #198 P1 — fatal exit codes
# ---------------------------------------------------------------------------


def test_fail_helper_logs_and_exits_non_zero(caplog) -> None:
    """``_fail`` logs at ERROR and raises SystemExit(1) — the new contract
    for every fatal user-input branch in the mask tool (issue #198)."""
    caplog.set_level(logging.ERROR, logger=mask_module.logger.name)

    with pytest.raises(SystemExit) as excinfo:
        mask_module._fail("boom: %s", "value")

    assert excinfo.value.code == 1
    assert "boom: value" in caplog.text


def test_check_input_missing_path_exits_non_zero(tmp_path) -> None:
    """``_check_input`` must exit 1 (not 0) when the input path is missing."""
    instance = mask_module._Mask.__new__(mask_module._Mask)
    instance._input_is_faces = False  # noqa: SLF001

    missing = tmp_path / "does_not_exist"
    with pytest.raises(SystemExit) as excinfo:
        instance._check_input(str(missing))  # noqa: SLF001
    assert excinfo.value.code == 1


def test_check_input_faces_on_file_exits_non_zero(tmp_path) -> None:
    """``_check_input`` must exit 1 when ``faces`` was requested but the
    input is a file rather than a folder."""
    instance = mask_module._Mask.__new__(mask_module._Mask)
    instance._input_is_faces = True  # noqa: SLF001

    file_input = tmp_path / "video.mp4"
    file_input.write_bytes(b"")
    with pytest.raises(SystemExit) as excinfo:
        instance._check_input(str(file_input))  # noqa: SLF001
    assert excinfo.value.code == 1


def test_get_input_locations_batch_on_file_exits_non_zero(tmp_path) -> None:
    """Batch mode against a file input must exit 1 (was 1 already — keeps
    that contract under the _fail-helper refactor)."""
    instance = mask_module.Mask.__new__(mask_module.Mask)
    instance._args = MagicMock(input=str(tmp_path / "nope"), batch_mode=True)

    with pytest.raises(SystemExit) as excinfo:
        instance._get_input_locations()  # noqa: SLF001
    assert excinfo.value.code == 1


def test_output_processing_missing_folder_exits_non_zero() -> None:
    """``mask_output.Output._get_saver`` must exit 1 when ``output`` was
    requested without a folder, not silently succeed via exit 0."""
    instance = mask_output_module.Output.__new__(mask_output_module.Output)

    with pytest.raises(SystemExit) as excinfo:
        instance._set_saver(None, "output")  # noqa: SLF001
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# #198 P2 high — O(N) dict mapping in _map_images
# ---------------------------------------------------------------------------


def _make_import_instance() -> Import:
    """Build a minimally-initialised ``Import`` so per-method tests can
    invoke its helpers without running the full constructor."""
    instance = Import.__new__(Import)
    return instance


def test_map_images_uses_dict_lookup_for_known_pairs(tmp_path) -> None:
    """Source filenames whose stems match a mask filename map to that
    mask. Mask paths are returned in full (not basename) so downstream
    consumers can open them."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    masks = [str(mask_dir / f"frame_{i:06d}.png") for i in range(5)]

    source_files = [
        str(tmp_path / "frame_000000.jpg"),
        str(tmp_path / "frame_000002.jpg"),
        str(tmp_path / "frame_000004.jpg"),
    ]

    instance = _make_import_instance()
    with patch.object(instance, "_warn_extra_masks") as warn:
        mapping = instance._map_images(list(masks), source_files)  # noqa: SLF001

    assert mapping == {
        "frame_000000.jpg": masks[0],
        "frame_000002.jpg": masks[2],
        "frame_000004.jpg": masks[4],
    }
    # Unmatched mask stems are still reported as extras (regression
    # against the pre-#198 ``file_list.pop(...)`` set-difference behaviour).
    warn.assert_called_once()
    extras = warn.call_args.args[0]
    assert sorted(extras) == sorted([masks[1], masks[3]])


def test_map_images_returns_only_unmapped_when_no_overlap(tmp_path, caplog) -> None:
    """When NO source matches any mask, the function logs an error and
    exits non-zero (existing contract preserved under the new lookup)."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    masks = [str(mask_dir / "mask_a.png"), str(mask_dir / "mask_b.png")]
    source_files = [str(tmp_path / "unrelated_1.jpg"), str(tmp_path / "unrelated_2.jpg")]

    instance = _make_import_instance()
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as excinfo:
        instance._map_images(masks, source_files)  # noqa: SLF001
    assert excinfo.value.code == 1
    assert "No masks map between the source data and the mask folder" in caplog.text


def test_map_images_is_linear_in_mask_count() -> None:
    """At 5,000 masks the new dict-based map MUST run in well under one
    second; the previous O(N^2) shape was the motivating bug in #198.

    This isn't a hard perf test — just a coarse upper-bound that would
    fail loudly if anyone reverted to the quadratic pattern.
    """
    import time

    n = 5000
    masks = [f"/masks/frame_{i:06d}.png" for i in range(n)]
    sources = [f"/src/frame_{i:06d}.jpg" for i in range(n)]

    instance = _make_import_instance()
    with patch.object(instance, "_warn_extra_masks"):
        start = time.perf_counter()
        mapping = instance._map_images(list(masks), list(sources))  # noqa: SLF001
        elapsed = time.perf_counter() - start

    assert len(mapping) == n
    # Generous bound — the dict shape on 5k pairs is sub-100ms on
    # modern hardware; the previous O(N^2) shape took >10s.
    assert elapsed < 2.0, f"_map_images took {elapsed:.2f}s on n={n}"


# ---------------------------------------------------------------------------
# #198 P3 — _get_input_locations path bind + Path.stem cleanup
# ---------------------------------------------------------------------------


def test_get_input_locations_lists_subdirs_and_video_files(tmp_path) -> None:
    """The rewritten ``_get_input_locations`` must still pick up
    folders + video-extension files under the batch root."""
    (tmp_path / "subdir_a").mkdir()
    (tmp_path / "subdir_b").mkdir()
    (tmp_path / "movie.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    instance = mask_module.Mask.__new__(mask_module.Mask)
    instance._args = MagicMock(input=str(tmp_path), batch_mode=True)

    locations = instance._get_input_locations()  # noqa: SLF001

    assert sorted(os.path.basename(loc) for loc in locations) == [
        "movie.mp4",
        "subdir_a",
        "subdir_b",
    ]


def test_get_output_location_uses_stem(tmp_path) -> None:
    """The output sub-folder name is the stem of the input location —
    ``project_v1.mp4`` -> ``<out>/project_v1``."""
    instance = mask_module.Mask.__new__(mask_module.Mask)
    instance._args = MagicMock(output=str(tmp_path / "out"))

    result = instance._get_output_location(str(tmp_path / "project_v1.mp4"))  # noqa: SLF001
    assert result == os.path.join(str(tmp_path / "out"), "project_v1")
