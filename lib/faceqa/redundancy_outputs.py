#!/usr/bin/env python3
"""File-organisation and visual outputs for the FaceQA redundancy pipeline.

Given a :class:`lib.faceqa.redundancy.RedundancyReport` and a directory of
aligned faces, this module organises the faces into ``keep`` / ``review`` /
``prune_candidate`` folders, writes a mapping log + CSV / JSONL keep and
prune manifests, and renders one contact sheet per multi-face redundancy
cluster.
"""

from __future__ import annotations

import logging
import shutil
import typing as T
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lib.faceqa.redundancy import KEEP, PRUNE, REVIEW, RedundancyRecord, RedundancyReport
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


CONTACT_SHEET_TILE_SIZE = 384
CONTACT_SHEET_COLS = 12
CONTACT_SHEET_FORMAT = "png"


@dataclass
class RedundancyLayout:
    """Paths produced when laying out redundancy outputs."""

    keep_dir: Path
    review_dir: Path
    prune_dir: Path
    contact_sheets_dir: Path
    mapping_log: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "keep_dir": str(self.keep_dir),
            "review_dir": str(self.review_dir),
            "prune_candidate_dir": str(self.prune_dir),
            "contact_sheets_dir": str(self.contact_sheets_dir),
            "mapping_log": str(self.mapping_log),
        }


# ---------------------------------------------------------------------------
# Sorted folders
# ---------------------------------------------------------------------------


def _face_filename(record: RedundancyRecord, *, faces_dir: Path) -> Path | None:
    stem = Path(record.frame).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = faces_dir / f"{stem}_{record.face_index}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_sorted_folders(
    report: RedundancyReport,
    *,
    faces_dir: str | Path,
    output_dir: str | Path,
    copy: bool = True,
    progress_callback: T.Callable[[int], None] | None = None,
) -> RedundancyLayout:
    """Sort aligned faces into ``keep``/``review``/``prune_candidate`` folders.

    ``copy=True`` (default, safe): duplicate each face from ``faces_dir`` into
    a bucket subfolder under ``output_dir``. Originals are preserved.

    ``copy=False`` (destructive): MOVE the originals into bucket subfolders
    that get created inside ``output_dir``. Callers wanting to keep their
    extracted faces folder intact should leave the default. A ``redundancy_mapping.csv``
    is always written next to the buckets for repeatability.
    """
    faces_dir_p = Path(faces_dir)
    out = Path(output_dir)
    layout = RedundancyLayout(
        keep_dir=_ensure_dir(out / "keep"),
        review_dir=_ensure_dir(out / "review"),
        prune_dir=_ensure_dir(out / "prune_candidate"),
        contact_sheets_dir=_ensure_dir(out / "contact_sheets"),
        mapping_log=out / "redundancy_mapping.csv",
    )
    rec_to_dir = {
        KEEP: layout.keep_dir,
        REVIEW: layout.review_dir,
        PRUNE: layout.prune_dir,
    }
    operation = "copy" if copy else "move"
    mapping_lines = ["source,destination,recommendation,cluster_id,representative,operation"]
    for record in report.records:
        source = _face_filename(record, faces_dir=faces_dir_p)
        target_dir = rec_to_dir.get(record.recommendation)
        if source is None or target_dir is None:
            mapping_lines.append(
                f"{record.frame}#face{record.face_index},,"
                f"{record.recommendation},{record.cluster_id},{record.representative},"
                f"{operation}"
            )
            if progress_callback is not None:
                progress_callback(1)
            continue
        destination = target_dir / source.name
        if copy:
            shutil.copy2(source, destination)
        else:
            shutil.move(str(source), str(destination))
        mapping_lines.append(
            f"{source},{destination},{record.recommendation},"
            f"{record.cluster_id},{record.representative},{operation}"
        )
        if progress_callback is not None:
            progress_callback(1)
    layout.mapping_log.write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")
    return layout


# ---------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_REC_COLOURS: dict[str, tuple[int, int, int]] = {
    KEEP: (60, 180, 75),
    REVIEW: (60, 180, 200),
    PRUNE: (60, 60, 200),
}


def _read_face(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return np.full((size, size, 3), 50, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    canvas: np.ndarray = np.full((size, size, 3), 30, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _tile_for_record(
    record: RedundancyRecord,
    *,
    faces_dir: Path,
    tile_size: int,
) -> np.ndarray:
    source = _face_filename(record, faces_dir=faces_dir)
    image = (
        _read_face(source, tile_size)
        if source is not None
        else np.full((tile_size, tile_size, 3), 30, dtype=np.uint8)
    )
    colour = _REC_COLOURS.get(record.recommendation, (200, 200, 200))
    border = max(6, tile_size // 64)
    cv2.rectangle(image, (0, 0), (tile_size - 1, tile_size - 1), colour, thickness=border)
    if record.representative:
        cv2.rectangle(
            image,
            (border, border),
            (tile_size - border, tile_size - border),
            (255, 255, 255),
            thickness=2,
        )
    panel_height = max(48, tile_size // 6)
    panel: np.ndarray = np.full((panel_height, tile_size, 3), 20, dtype=np.uint8)
    name = Path(record.frame).name
    if len(name) > 28:
        name = name[:25] + "…"
    line1 = f"{name}#{record.face_index} c{record.cluster_id}"
    role = "REP" if record.representative else record.recommendation.upper()
    distance = record.redundancy_distance_to_representative
    distance_label = "n/a" if distance is None else f"{distance:.2f}"
    line2 = f"{role} q={record.quality_score:.2f} d={distance_label}"
    cv2.putText(
        panel, line1, (8, panel_height // 2 - 4), _FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA
    )
    cv2.putText(panel, line2, (8, panel_height - 8), _FONT, 0.5, colour, 1, cv2.LINE_AA)
    return T.cast(np.ndarray, np.vstack([image, panel]))


def render_contact_sheets(
    report: RedundancyReport,
    *,
    faces_dir: str | Path,
    output_dir: str | Path,
    progress_callback: T.Callable[[int], None] | None = None,
) -> list[Path]:
    """Render one contact sheet per multi-face redundancy cluster.

    ``progress_callback`` (when supplied) is invoked with ``1`` after each
    cluster sheet is written.
    """
    faces_dir_p = Path(faces_dir)
    output_dir_p = _ensure_dir(Path(output_dir))
    clusters: dict[int, list[RedundancyRecord]] = {}
    for record in report.records:
        if record.cluster_size <= 1:
            continue
        clusters.setdefault(record.cluster_id, []).append(record)
    triage_by_cluster = {
        int(item["cluster_id"]): str(item.get("triage_group", "normal_review"))
        for item in report.cluster_pruning_stats
    }
    written: list[Path] = []
    for cluster_id, records in sorted(clusters.items()):
        # Issue #208 follow-up 2: within each cluster, order tiles so
        # the contact sheet is auditable at a glance:
        #
        #   1. Representative first.
        #   2. Then by recommendation (keep -> review -> prune) so the
        #      reviewer can see the keep candidates immediately and
        #      verify the prune candidates against them without
        #      hunting.
        #   3. Then by appearance mode (lighting / makeup / event
        #      fingerprint) so visually-similar sub-groups appear
        #      together within each recommendation bucket.
        #   4. Then by fine pose band and expression band so faces
        #      with similar pose / mouth shape sit side-by-side.
        #   5. Finally by frame + face index for stable ordering.
        recommendation_rank = {KEEP: 0, REVIEW: 1, PRUNE: 2}

        def _none_last(value):
            return (value is None, value)

        records.sort(
            key=lambda r: (
                0 if r.representative else 1,
                recommendation_rank.get(r.recommendation, 3),
                (r.appearance_mode is None, r.appearance_mode or ()),
                _none_last(r.yaw_micro_band),
                _none_last(r.pitch_micro_band),
                _none_last(r.mouth_openness_band),
                _none_last(r.smile_proxy_band),
                -r.quality_score,
                r.frame,
                r.face_index,
            )
        )
        tiles = [
            _tile_for_record(record, faces_dir=faces_dir_p, tile_size=CONTACT_SHEET_TILE_SIZE)
            for record in records
        ]
        rows = []
        pad_tile = np.full_like(tiles[0], 20) if tiles else None
        for start in range(0, len(tiles), CONTACT_SHEET_COLS):
            row_tiles = tiles[start : start + CONTACT_SHEET_COLS]
            if len(row_tiles) < CONTACT_SHEET_COLS and pad_tile is not None:
                while len(row_tiles) < CONTACT_SHEET_COLS:
                    row_tiles.append(pad_tile)
            rows.append(np.hstack(row_tiles))
        sheet = np.vstack(rows)
        sheet = _annotate_sheet_header(sheet, cluster_id, records)
        triage_group = triage_by_cluster.get(cluster_id, "normal_review")
        path = _ensure_dir(output_dir_p / triage_group) / f"cluster_{cluster_id:04d}.png"
        cv2.imwrite(str(path), sheet)
        written.append(path)
        if progress_callback is not None:
            progress_callback(1)
    return written


def _annotate_sheet_header(
    sheet: np.ndarray,
    cluster_id: int,
    records: list[RedundancyRecord],
) -> np.ndarray:
    header_height = 64
    width = sheet.shape[1]
    header: np.ndarray = np.full((header_height, width, 3), 12, dtype=np.uint8)
    rep = next((r for r in records if r.representative), records[0])
    title = (
        f"Cluster {cluster_id}  |  size={len(records)}  "
        f"|  representative: {Path(rep.frame).name}#{rep.face_index}"
    )
    cv2.putText(header, title, (16, 26), _FONT, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    sub = (
        f"keep={sum(1 for r in records if r.recommendation == KEEP)}  "
        f"review={sum(1 for r in records if r.recommendation == REVIEW)}  "
        f"prune={sum(1 for r in records if r.recommendation == PRUNE)}"
    )
    cv2.putText(header, sub, (16, 52), _FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return T.cast(np.ndarray, np.vstack([header, sheet]))


__all__ = get_module_objects(__name__)
