#!/usr/bin/env python3
"""Tests for the FaceQA duplicate-cluster pipeline."""

from __future__ import annotations

import json

import numpy as np

from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.faceqa.duplicate_outputs import render_contact_sheets, write_sorted_folders
from lib.faceqa.duplicates import (
    KEEP,
    PRUNE,
    REVIEW,
    cluster_duplicates,
    write_manifests,
)
from lib.serializer import get_serializer


def _face(
    *,
    width: int = 256,
    height: int = 256,
    embedding: np.ndarray | None = None,
    model: str = "insightface",
    pose: dict | None = None,
    faceqa: dict | None = None,
) -> FileAlignments:
    landmarks: np.ndarray = np.zeros((68, 2), dtype="float32")
    face = FileAlignments(x=0, y=0, w=width, h=height, landmarks_xy=landmarks)
    if embedding is not None:
        face.identity = {model: embedding.astype(np.float32)}
    if pose is not None or faceqa is not None:
        metadata: dict[str, object] = {}
        if pose is not None:
            metadata["pose"] = pose
        if faceqa is not None:
            metadata["faceqa"] = faceqa
        face.metadata = metadata
    return face


def _save_alignments(path, frames: dict[str, list[FileAlignments]]) -> None:
    get_serializer("compressed").save(
        str(path),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                frame: AlignmentsEntry(faces=faces).to_dict() for frame, faces in frames.items()
            },
        },
    )


def test_cluster_duplicates_groups_high_similarity_embeddings(tmp_path) -> None:
    """Faces with near-identical embeddings should land in the same cluster."""
    base = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    near_dup = base + np.array([0.01, 0.01, 0.0, 0.0], dtype="float32")
    different = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [_face(embedding=base, faceqa={"blur_score": 10.0})],
            "frame_000002.png": [_face(embedding=near_dup, faceqa={"blur_score": 4.0})],
            "frame_000003.png": [_face(embedding=different, faceqa={"blur_score": 5.0})],
        },
    )

    report = cluster_duplicates(alignments)

    assert report.total_faces == 3
    # frame 1 + 2 → same cluster; frame 3 alone.
    cluster_ids = {(r.frame, r.cluster_id) for r in report.records}
    cluster_of = {frame: cid for frame, cid in cluster_ids}
    assert cluster_of["frame_000001.png"] == cluster_of["frame_000002.png"]
    assert cluster_of["frame_000003.png"] != cluster_of["frame_000001.png"]
    # Singleton cluster gets `keep`.
    singleton = next(r for r in report.records if r.frame == "frame_000003.png")
    assert singleton.recommendation == KEEP
    assert singleton.representative is True


def test_cluster_duplicates_is_deterministic(tmp_path) -> None:
    """Repeated runs against the same alignments yield identical reports."""
    embedding = np.array([0.5, 0.5, 0.5, 0.5], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [_face(embedding=embedding)],
            "frame_000002.png": [_face(embedding=embedding + 0.01)],
        },
    )

    first = cluster_duplicates(alignments).to_dict()
    second = cluster_duplicates(alignments).to_dict()

    assert first == second


def test_representative_selection_prefers_higher_quality(tmp_path) -> None:
    """Within a cluster the highest-quality face becomes the representative."""
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [
                _face(
                    embedding=embedding,
                    width=64,
                    height=64,
                    faceqa={"blur_score": 0.5, "black_pixel_ratio": 0.6},
                )
            ],
            "frame_000002.png": [
                _face(
                    embedding=embedding + 0.005,
                    width=512,
                    height=512,
                    faceqa={"blur_score": 12.0, "black_pixel_ratio": 0.0},
                )
            ],
            "frame_000003.png": [
                _face(
                    embedding=embedding + 0.01,
                    width=256,
                    height=256,
                    faceqa={"blur_score": 3.0, "black_pixel_ratio": 0.0},
                )
            ],
        },
    )

    report = cluster_duplicates(alignments)
    rep = next(r for r in report.records if r.representative)
    assert rep.frame == "frame_000002.png"
    assert rep.recommendation == KEEP
    # The low-quality face should be a prune candidate.
    low = next(r for r in report.records if r.frame == "frame_000001.png")
    assert low.recommendation == PRUNE


def test_cluster_duplicates_routes_missing_embedding_to_review(tmp_path) -> None:
    """Faces missing the selected model embedding are routed to ``review``."""
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [_face(embedding=embedding, model="insightface")],
            "frame_000002.png": [_face(model="adaface")],  # No insightface embedding.
        },
    )

    report = cluster_duplicates(alignments, identity_model="insightface")

    missing = next(r for r in report.records if r.frame == "frame_000002.png")
    assert missing.recommendation == REVIEW
    assert "embedding" in missing.reason


def test_cluster_duplicates_handles_no_embeddings_at_all(tmp_path) -> None:
    """When no faces have embeddings each face becomes a review singleton."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [_face()],
            "frame_000002.png": [_face()],
        },
    )

    report = cluster_duplicates(alignments)

    assert report.identity_model is None
    assert report.total_faces == 2
    assert report.skipped_no_embedding == 2
    assert all(r.recommendation == REVIEW for r in report.records)


def test_cluster_duplicates_respects_temporal_window(tmp_path) -> None:
    """Temporally-distant duplicates can be excluded from the same cluster."""
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [_face(embedding=embedding)],
            "frame_000200.png": [_face(embedding=embedding + 0.01)],
        },
    )

    no_window = cluster_duplicates(alignments, temporal_window=None)
    constrained = cluster_duplicates(alignments, temporal_window=10)

    no_window_clusters = {r.cluster_id for r in no_window.records}
    constrained_clusters = {r.cluster_id for r in constrained.records}
    assert len(no_window_clusters) == 1
    assert len(constrained_clusters) == 2


def test_write_manifests_emits_csv_and_jsonl(tmp_path) -> None:
    """write_manifests should produce keep + prune CSV and JSONL files."""
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [
                _face(embedding=embedding, width=512, height=512, faceqa={"blur_score": 8.0})
            ],
            "frame_000002.png": [
                _face(embedding=embedding + 0.01, width=64, height=64, faceqa={"blur_score": 0.5})
            ],
        },
    )

    report = cluster_duplicates(alignments)
    out = write_manifests(report, tmp_path / "manifests")

    keep_csv = out["keep_csv"].read_text(encoding="utf-8")
    keep_jsonl = out["keep_jsonl"].read_text(encoding="utf-8")
    prune_csv = out["prune_csv"].read_text(encoding="utf-8")
    prune_jsonl = out["prune_jsonl"].read_text(encoding="utf-8")

    assert "frame_000001.png" in keep_csv
    assert any(json.loads(line)["frame"] == "frame_000001.png" for line in keep_jsonl.splitlines())
    assert "frame_000002.png" in prune_csv
    assert any(
        json.loads(line)["frame"] == "frame_000002.png" for line in prune_jsonl.splitlines()
    )


def test_write_sorted_folders_copies_aligned_faces(tmp_path) -> None:
    """Sorted folders should mirror keep / review / prune recommendations."""
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [
                _face(embedding=embedding, width=512, height=512, faceqa={"blur_score": 8.0})
            ],
            "frame_000002.png": [
                _face(embedding=embedding + 0.01, width=64, height=64, faceqa={"blur_score": 0.5})
            ],
        },
    )
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    blank = np.full((128, 128, 3), 80, dtype=np.uint8)
    import cv2

    cv2.imwrite(str(faces_dir / "frame_000001_0.png"), blank)
    cv2.imwrite(str(faces_dir / "frame_000002_0.png"), blank)

    report = cluster_duplicates(alignments)
    layout = write_sorted_folders(report, faces_dir=faces_dir, output_dir=tmp_path / "sorted")

    keep_files = list(layout.sorted_keep_dir.iterdir())
    prune_files = list(layout.sorted_prune_dir.iterdir())
    assert any(p.name == "frame_000001_0.png" for p in keep_files)
    assert any(p.name == "frame_000002_0.png" for p in prune_files)
    mapping = layout.mapping_log.read_text(encoding="utf-8")
    assert "frame_000001_0.png" in mapping
    assert "frame_000002_0.png" in mapping


def test_render_contact_sheets_emits_one_per_multi_face_cluster(tmp_path) -> None:
    """Multi-face clusters should each yield an inspectable contact sheet."""
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [_face(embedding=embedding)],
            "frame_000002.png": [_face(embedding=embedding + 0.005)],
            "frame_000050.png": [_face(embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"))],
        },
    )
    faces_dir = tmp_path / "faces"
    faces_dir.mkdir()
    blank = np.full((128, 128, 3), 80, dtype=np.uint8)
    import cv2

    cv2.imwrite(str(faces_dir / "frame_000001_0.png"), blank)
    cv2.imwrite(str(faces_dir / "frame_000002_0.png"), blank)
    cv2.imwrite(str(faces_dir / "frame_000050_0.png"), blank)

    report = cluster_duplicates(alignments)
    paths = render_contact_sheets(
        report,
        faces_dir=faces_dir,
        output_dir=tmp_path / "contact_sheets",
        tile_size=128,
        columns=2,
    )

    # Only the multi-face cluster (frames 1 + 2) produces a sheet.
    assert len(paths) == 1
    assert paths[0].suffix == ".png"
    assert paths[0].is_file()


def test_report_summary_counts_are_consistent(tmp_path) -> None:
    """Report counts (keep/review/prune) should match the record-level recommendations."""
    embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [
                _face(embedding=embedding, width=512, height=512, faceqa={"blur_score": 8.0})
            ],
            "frame_000002.png": [
                _face(embedding=embedding + 0.01, width=64, height=64, faceqa={"blur_score": 0.5})
            ],
            "frame_000050.png": [_face(embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"))],
        },
    )

    report = cluster_duplicates(alignments)

    assert report.keep_count == sum(1 for r in report.records if r.recommendation == KEEP)
    assert report.review_count == sum(1 for r in report.records if r.recommendation == REVIEW)
    assert report.prune_candidate_count == sum(
        1 for r in report.records if r.recommendation == PRUNE
    )
    assert report.cluster_count == len({r.cluster_id for r in report.records})
