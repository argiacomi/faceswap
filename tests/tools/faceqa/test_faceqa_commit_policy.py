#!/usr/bin/env python3
"""Source-level guardrails for FaceQA alignments commit policy."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tools.faceqa.faceqa as faceqa_module
from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.serializer import get_serializer
from tools.faceqa.faceqa import Faceqa

ROOT = Path(__file__).resolve().parents[3]
FACEQA = ROOT / "tools" / "faceqa" / "faceqa.py"
COVERAGE = ROOT / "lib" / "faceqa" / "coverage.py"


def _minimal_face() -> FileAlignments:
    """Return a minimal alignment face for coverage-tool integration tests."""
    landmarks = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    return FileAlignments(x=0, y=0, w=80, h=90, landmarks_xy=landmarks)


def _coverage_args(alignments: Path, output_dir: Path) -> Namespace:
    """Return minimal FaceQA coverage args."""
    return Namespace(
        mode="coverage",
        alignments=str(alignments),
        frames_dir=str(output_dir),
        faces_dir=None,
        output_dir=str(output_dir),
        exclude_duplicates=False,
        exclude_outliers=False,
        min_bucket_pct=5.0,
        suggest_pruning=False,
        sort_prune=False,
        contact_sheets=False,
        keep_originals=True,
        prune_aggressiveness="balanced",
    )


def test_faceqa_backfills_identity_against_loaded_entries():
    text = FACEQA.read_text(encoding="utf-8")

    assert "self._run_identity_backfill(alignments)" not in text
    assert "identity_backfill = self._run_identity_backfill(entries)" in text
    assert "backfill_identity_entries(" in text


def test_faceqa_commit_condition_uses_real_image_metric_changes():
    text = FACEQA.read_text(encoding="utf-8")

    assert "image_metrics_changed = False" in text
    assert "metadata_change_callback=_track_image_metrics_change" in text
    assert "or image_metrics_changed" in text
    assert "or metrics_backfiller is not None" not in text


def test_coverage_exposes_entries_based_identity_backfill():
    text = COVERAGE.read_text(encoding="utf-8")

    assert "def backfill_identity_entries(" in text
    assert "save_alignments_envelope(path, raw, entries)" in text
    assert "report.persisted = True" in text


def test_faceqa_coverage_commits_all_enrichment_with_one_alignments_save(
    tmp_path,
    monkeypatch,
):
    """Identity, pose, and image metrics enrichment should share one commit."""
    alignments = tmp_path / "alignments.fsa"
    get_serializer("compressed").save(
        str(alignments),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                "frame_000001.png": AlignmentsEntry(faces=[_minimal_face()]).to_dict(),
            },
        },
    )

    def _fake_identity_backfill(entries, *, frames_loader, model=None, progress_callback=None):
        face = next(iter(entries.values())).faces[0]
        face.identity["arcface"] = np.ones(4, dtype="float32")
        if progress_callback is not None:
            progress_callback(1)
        return SimpleNamespace(
            model="arcface",
            total_faces=1,
            already_present=0,
            backfilled=1,
            skipped_no_frame=0,
            skipped_failed=0,
            disabled_reason=None,
        )

    def _fake_pose_backfill(_record, _face):
        return {
            "yaw": 10.0,
            "pitch": 0.0,
            "roll": 0.0,
            "source": "spiga_backfill",
            "model": "spiga",
        }

    def _fake_metrics_backfill(_face, _frame, _face_index):
        return {
            "blur_score": 1.0,
            "black_pixel_ratio": 0.0,
            "provenance": "frame_aligned_crop",
        }

    _fake_metrics_backfill.disabled_reason = None

    save_calls = []

    def _save_once(path, raw, entries):
        save_calls.append((path, raw, entries))

    monkeypatch.setattr(Faceqa, "_frames_loader", lambda _self: object())
    monkeypatch.setattr(Faceqa, "_pose_backfiller", lambda _self: _fake_pose_backfill)
    monkeypatch.setattr(Faceqa, "_metrics_backfiller", lambda _self: _fake_metrics_backfill)
    monkeypatch.setattr(faceqa_module, "backfill_identity_entries", _fake_identity_backfill)
    monkeypatch.setattr(faceqa_module, "save_alignments_envelope", _save_once)

    Faceqa(_coverage_args(alignments, tmp_path)).process()

    assert len(save_calls) == 1
    saved_path, _raw, entries = save_calls[0]
    assert Path(saved_path) == alignments

    face = next(iter(entries.values())).faces[0]
    assert "arcface" in face.identity
    assert face.metadata["faceqa"]["pose"]["source"] == "spiga_backfill"
    assert face.metadata["faceqa"]["image_metrics"]["provenance"] == "frame_aligned_crop"
