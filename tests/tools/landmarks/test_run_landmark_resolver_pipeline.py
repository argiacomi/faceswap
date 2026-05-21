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
    _command_scorer_eval,
    _command_scorer_training,
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
        "print_config_patch": False,
        "progress": False,
        "no_progress": True,
        "config_path": tmp_path / "extract.ini",
        "config_section": "align.ensemble",
        "python_executable": "python",
        "dataset": "wflw",
        "models": "hrnet,spiga,orformer",
        "candidates": "hrnet,spiga,orformer,static_weighted_downweight",
        "production_images": None,
        "production_alignments": None,
        "gt_hard_resolver_metadata": None,
        "overwrite_frozen_metadata": False,
        "hard_source_manifest": None,
        "split_mode": "scenario-stratified",
        "split_fit": 0.6,
        "split_select": 0.2,
        "split_report": 0.2,
        "split_seed": 1337,
        "scorer_target": "selection_cost",
        "allow_image_backfill": False,
        "dataset_build_arg": [],
        "cache_prediction_arg": [],
        "production_manifest_arg": [],
        "candidate_search_arg": [],
        "hard_validation_arg": [],
        "scorer_train_arg": [],
        "scorer_eval_arg": [],
        "log_level": "INFO",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _touch_pipeline_outputs(paths: PipelinePaths, *, promotion_status: str = "pass") -> None:
    for path in (
        paths.run_manifest,
        paths.run_splits,
        paths.run_static_weights,
        paths.run_summary,
        paths.production_manifest,
        paths.production_root / "resolver_metadata.jsonl",
        paths.production_root / "audit.json",
        paths.best_setup,
        paths.best_weights,
        paths.hard_manifest,
        paths.frozen_gt_metadata,
        paths.binary_scorer_artifact,
        paths.scorer_artifact,
        paths.continuous_scorer_eval_rows,
        paths.scorer_report,
        paths.exported_best_setup,
        paths.exported_best_weights,
        paths.exported_scorer_artifact,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.endswith(".json"):
            payload = {
                "status": promotion_status,
                "promotion_status": promotion_status,
                "failed_gates": [] if promotion_status == "pass" else ["gate_failed"],
                "production_gate_status": promotion_status,
                "production_failed_gates": [] if promotion_status == "pass" else ["gate_failed"],
                "gt_hard_gate_status": "pass",
                "gt_hard_failed_gates": [],
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


def test_pipeline_runner_dry_run_writes_summaries_contracts_and_progress(tmp_path: Path) -> None:
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
    assert "weights_path" in summary["config_fields_changed"]
    progress_log = Path(summary["progress_log_path"])
    assert progress_log.is_file()
    progress_rows = [
        json.loads(line) for line in progress_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(progress_rows) == len(STAGES) * 2
    assert {row["event"] for row in progress_rows} == {"start", "finish"}
    assert progress_rows[0]["stage"] == STAGES[0]
    assert progress_rows[-1]["stage"] == STAGES[-1]
    assert (args.output_root / "pipeline_summary.json").is_file()
    assert (args.output_root / "pipeline_summary.md").is_file()


def test_stage_contract_declares_required_files(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    contract = _contract_for("continuous_scorer_training", args, paths)

    assert str(paths.hard_manifest) in contract.required_files
    assert str(paths.frozen_gt_metadata) in contract.required_files
    assert str(paths.scorer_artifact) in contract.outputs
    assert str(paths.continuous_scorer_eval_rows) in contract.outputs


def test_scorer_train_and_eval_commands_allow_image_backfill_when_enabled(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, allow_image_backfill=True)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    train_command = _command_scorer_training(
        args, paths, output_dir=paths.continuous_scorer_train_dir, target=args.scorer_target
    )
    eval_command = _command_scorer_eval(args, paths)

    assert "--allow-image-backfill" in train_command
    assert "--allow-image-backfill" in eval_command


def test_scorer_train_and_eval_commands_do_not_allow_image_backfill_by_default(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    train_command = _command_scorer_training(
        args, paths, output_dir=paths.continuous_scorer_train_dir, target=args.scorer_target
    )
    eval_command = _command_scorer_eval(args, paths)

    assert "--allow-image-backfill" not in train_command
    assert "--allow-image-backfill" not in eval_command


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
    paths.hard_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.hard_manifest.write_text(
        json.dumps(
            {
                "dataset": "gt_hard",
                "samples": [
                    {
                        "sample_id": "s1",
                        "image": "s1.png",
                        "landmarks": "s1.npy",
                        "metadata": {"face_index": 0},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths.frozen_gt_metadata.parent.mkdir(parents=True, exist_ok=True)
    paths.frozen_gt_metadata.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "face_index": 0,
                "landmark_ensemble": {"runtime_bucket": "frontal"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    notes = _freeze_metadata(args, paths)

    assert "reused existing frozen metadata" in notes[0]
    assert "runtime_bucket" in paths.frozen_gt_metadata.read_text(encoding="utf-8")


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


def test_config_preview_and_write_preserves_unmanaged_config_entries(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)

    notes = _apply_config(args, paths)
    preview = json.loads(
        (args.output_root / "config_update_preview.json").read_text(encoding="utf-8")
    )
    patch = (args.output_root / "config_update_patch.ini").read_text(encoding="utf-8")

    assert "preview" in notes[0]
    assert preview["section"] == "align.ensemble"
    assert preview["updates"]["resolver_policy"] == "learned_quality_v1"
    assert "production_weights_path" not in preview["updates"]
    assert "coordinate_space" not in preview["updates"]
    assert patch.startswith("[align.ensemble]\n")
    assert "weights_path = " in patch

    args.config_path.write_text(
        "\n".join(
            [
                "# global comment survives",
                "[global]",
                "aligner_min_scale = 0.03",
                "",
                "[align.ensemble]",
                "# align ensemble intro comment survives",
                "stale_experimental = true",
                "resolver_policy = roll_aware_veto",
                "# align ensemble key comment survives",
                "; align ensemble footer comment survives",
                "",
                "[align.fan]",
                "# align fan comment survives",
                "batch_size = 16",
                "",
                "[detect.scrfd]",
                "model = 10g",
                "",
            ]
        ),
        encoding="utf-8",
    )
    args.write_config = True
    _apply_config(args, paths)

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(args.config_path, encoding="utf-8")

    assert parser.get("global", "aligner_min_scale") == "0.03"
    assert parser.get("align.fan", "batch_size") == "16"
    assert parser.get("detect.scrfd", "model") == "10g"
    assert parser.get("align.ensemble", "stale_experimental") == "true"
    assert parser.get("align.ensemble", "use_alignment_resolver") == "True"
    assert parser.get("align.ensemble", "resolver_policy") == "learned_quality_v1"
    assert parser.get("align.ensemble", "resolver_scorer_path") == str(
        paths.exported_scorer_artifact
    )
    assert parser.get("align.ensemble", "weights_path") == str(paths.exported_best_weights)

    config_text = args.config_path.read_text(encoding="utf-8")
    assert "# global comment survives" in config_text
    assert "# align ensemble intro comment survives" in config_text
    assert "# align ensemble key comment survives" in config_text
    assert "; align ensemble footer comment survives" in config_text
    assert "# align fan comment survives" in config_text


def test_config_write_requires_artifacts_and_passing_promotion(tmp_path: Path) -> None:
    args = _args(tmp_path, write_config=True)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    args.config_path.write_text("[global]\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="promoted setup artifact"):
        _apply_config(args, paths)

    _touch_pipeline_outputs(paths, promotion_status="fail")
    with pytest.raises(RuntimeError, match="production promotion check failed"):
        _apply_config(args, paths)


def test_resume_skips_completed_stages_records_validated_outputs_and_progress(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path, resume=True, start_at="production_promotion_check", stop_after="config_update"
    )
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "artifacts_manifest.json").write_text("{}\n", encoding="utf-8")
    (paths.output_root / "config_update_preview.json").write_text("{}\n", encoding="utf-8")
    (paths.output_root / "config_update_patch.ini").write_text(
        "[align.ensemble]\n", encoding="utf-8"
    )

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
    progress_rows = [
        json.loads(line)
        for line in Path(summary["progress_log_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["stage"] for row in progress_rows if row["event"] == "start"] == [
        "production_promotion_check",
        "artifact_export",
        "config_update",
    ]


def test_resume_write_config_does_not_skip_config_update(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        resume=True,
        write_config=True,
        start_at="config_update",
        stop_after="config_update",
    )
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)
    (paths.output_root / "config_update_preview.json").write_text("{}\n", encoding="utf-8")
    (paths.output_root / "config_update_patch.ini").write_text(
        "[align.ensemble]\n", encoding="utf-8"
    )
    args.config_path.write_text(
        "\n".join(
            [
                "# config comment survives",
                "[align.ensemble]",
                "resolver_policy = stale_policy",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = run_pipeline(args)

    assert summary["stages"][0]["name"] == "config_update"
    assert summary["stages"][0]["status"] == "ok"
    config_text = args.config_path.read_text(encoding="utf-8")
    assert "resolver_policy = learned_quality_v1" in config_text
    assert "# config comment survives" in config_text
