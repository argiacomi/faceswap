#!/usr/bin/env python3
"""Tests for landmark E2E pipeline default path behavior."""

from __future__ import annotations

import json
from pathlib import Path

from lib.landmarks.datasets.sources import DEFAULT_CACHE_DIR
from tools.landmarks import run_static_weight_pipeline


def test_static_weight_pipeline_dry_run_uses_default_output_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dry-run can execute without explicit path arguments."""
    monkeypatch.chdir(tmp_path)

    rc = run_static_weight_pipeline.main(["--dry-run"])

    summary_path = tmp_path / DEFAULT_CACHE_DIR / "runs" / "static_weight_validation" / "run_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert summary_path.is_file()
    assert payload["output_root"] == str(DEFAULT_CACHE_DIR / "runs" / "static_weight_validation")
    assert payload["args"]["dataset_cache_dir"] == str(DEFAULT_CACHE_DIR)
    assert "dataset:auto-build-or-reuse" in payload["planned_stages"]
    assert "dataset:wflw" in payload["planned_stages"]
    assert "dataset:cofw" in payload["planned_stages"]


def test_static_weight_pipeline_skip_dataset_build_dry_run_reuses_default_manifest_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Skipping dataset build keeps the same standard run-root convention."""
    monkeypatch.chdir(tmp_path)

    rc = run_static_weight_pipeline.main(["--dry-run", "--skip-dataset-build"])

    summary_path = tmp_path / DEFAULT_CACHE_DIR / "runs" / "static_weight_validation" / "run_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["planned_stages"][0] == "dataset:reuse-existing"
    assert payload["args"]["output_root"] == str(DEFAULT_CACHE_DIR / "runs" / "static_weight_validation")
