#!/usr/bin/env python3
"""Tests for the end-to-end landmark resolver pipeline wrapper."""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path

import pytest

from tools.landmarks.run_landmark_resolver_pipeline import (
    STAGES,
    PipelineContractError,
    PipelinePaths,
    _apply_config,
    _contract_for,
    _freeze_metadata,
    _promotion_check,
    _validate_stage_outputs,
    run_pipeline,
)


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "run_root": tmp_path / "static_run",
        "production_root": tmp_path / "production_run",
        "output_root": tmp_path / "resolver_run",
        "promotion_scope": "production",
        "dry_run": False,
        "resume": False,
        "stop_after": None,
        "start_at": None,
        "write_config": False,
        "config_path": tmp_path / "extract.ini",
        "config_section": "landmark_ensemble",
        "python_executable": "python",
        "models": "hrnet,spiga,orformer",
        "candidates": "hrnet,spiga,orformer,static_weighted_downweight",
        "gt_hard_resolver_metadata": None,
        "overwrite_frozen_metadata": False,
        "hard_source_manifest": None,
        "force_static_pipeline": False,
        "static_pipeline_arg": [],
        "candidate_search_arg": [],
        "hard_validation_arg": [],
        "scorer_train_arg": [],
        "scorer_eval_arg": [],
        "log_level": "INFO",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _touch_pipeline_outputs(paths: PipelinePaths) -> None:
    for path in (
        paths.run_manifest,
        paths.run_splits,
        paths.run_static_weights,
        paths.run_summary,
        paths.production_manifest,
        paths.best_setup,
        paths.best_weights,
        paths.hard_manifest,
        paths.frozen_gt_metadata,
        paths.scorer_artifact,
        paths.scorer_report,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.endswith(".json"):
            payload = {
                "status": "pass",
                "promotion_status": "pass",
                "failed_gates": [],
                "fallback_count": 3,
                "safe_fallback_count": 2,
                "hard_slice_fallback_count": 1,
                "consensus_collapse_fallback_count": 1,
            }
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        else:
            path.write_text("{}\n", encoding="utf-8")
    paths.run_cache.mkdir(parents=True, exist_ok=True)
    paths.production_cache.mkdir(parents=True, exist_ok=True)


def test_pipeline_runner_dry_run_writes_summaries_and_contracts(tmp_path: Path) -> None:
    args = _args(tmp_path, dry_run=True)

    summary = run_pipeline(args)

    assert summary["status"] == "planned"
    assert len(summary["stages"]) == len(STAGES)
    assert {stage["status"] for stage in summary["stages"]} <= {"planned", "ok"}
    assert all(stage["contract"]["inputs"] is not None for stage in summary["stages"])
    assert all(stage["contract"]["outputs"] for stage in summary["stages"])
    assert all(stage["contract"]["cache_behavior"] for stage in summary["stages"])
    assert all(stage["contract"]["success_criteria"] for stage in summary["stages"])
    assert summary["best_weights_path"].endswith("best_weights.json")
    assert summary["scorer_path"].endswith("runtime_resolver_scorer.json")
    assert summary["eval_report_path"].endswith("scorer_policy_report.json")
    assert summary["selected_production_policy"] == "learned_quality_v1"
    assert "resolver_scorer_path" in summary["config_fields_changed"]
    assert (args.output_root / "pipeline_summary.json").is_file()
    assert (args.output_root / "pipeline_summary.md").is_file()


def test_stage_contract_declares_required_files(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    contract = _contract_for("scorer_training", args, paths)

    assert str(paths.hard_manifest) in contract.required_files
    assert str(paths.frozen_gt_metadata) in contract.required_files
    assert str(paths.scorer_artifact) in contract.outputs


def test_stage_output_validation_reports_missing_outputs(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    with pytest.raises(PipelineContractError, match="candidate_search.*best_setup.json"):
        _validate_stage_outputs("candidate_search", paths)


def test_freeze_metadata_requires_explicit_source(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    with pytest.raises(FileNotFoundError, match="will not silently regenerate"):
        _freeze_metadata(args, paths)


def test_freeze_metadata_reuses_existing_on_resume(tmp_path: Path) -> None:
    args = _args(tmp_path, resume=True)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    paths.frozen_gt_metadata.parent.mkdir(parents=True, exist_ok=True)
    paths.frozen_gt_metadata.write_text("existing\n", encoding="utf-8")

    notes = _freeze_metadata(args, paths)

    assert "reused existing frozen metadata" in notes[0]
    assert paths.frozen_gt_metadata.read_text(encoding="utf-8") == "existing\n"


def test_promotion_check_fails_on_failed_gate(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    paths.scorer_report.parent.mkdir(parents=True, exist_ok=True)
    paths.scorer_report.write_text(
        json.dumps({"promotion_status": "fail", "failed_gates": ["x"]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="production promotion check failed"):
        _promotion_check(paths)


def test_config_preview_and_write(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    notes = _apply_config(args, paths)
    preview = json.loads((args.output_root / "config_update_preview.json").read_text(encoding="utf-8"))
    assert "preview" in notes[0]
    assert preview["updates"]["resolver_policy"] == "learned_quality_v1"

    args.config_path.write_text("[global]\n", encoding="utf-8")
    args.write_config = True
    _apply_config(args, paths)
    parser = configparser.ConfigParser()
    parser.read(args.config_path, encoding="utf-8")
    assert parser.get("landmark_ensemble", "use_alignment_resolver") == "true"
    assert parser.get("landmark_ensemble", "resolver_policy") == "learned_quality_v1"


def test_resume_skips_completed_stages_and_records_validated_outputs(tmp_path: Path) -> None:
    args = _args(tmp_path, resume=True, start_at="production_promotion_check", stop_after="config_update")
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "artifacts_manifest.json").write_text("{}\n", encoding="utf-8")
    (paths.output_root / "config_update_preview.json").write_text("{}\n", encoding="utf-8")

    summary = run_pipeline(args)

    assert summary["status"] == "pass"
    assert summary["promotion_status"] == "pass"
    assert summary["fallback_counts"] == {
        "fallback_count": 3,
        "safe_fallback_count": 2,
        "hard_slice_fallback_count": 1,
        "consensus_collapse_fallback_count": 1,
    }
    assert [stage["status"] for stage in summary["stages"]] == ["skipped", "skipped", "skipped"]
    assert all(stage["validated_outputs"] for stage in summary["stages"])
