#!/usr/bin/env python3
"""Source-level guardrails for FaceQA alignments commit policy."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FACEQA = ROOT / "tools" / "faceqa" / "faceqa.py"
COVERAGE = ROOT / "lib" / "faceqa" / "coverage.py"


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
