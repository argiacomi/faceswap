#!/usr/bin/env python3
"""Tests for the end-to-end landmark resolver pipeline wrapper."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.landmarks.run_landmark_resolver_pipeline import (
    STAGES,
    PipelineContractError,
    PipelinePaths,
    _apply_config,
    _build_splits,
    _command_candidate_search,
    _command_dataset_build,
    _command_gt_hard_resolver_metadata,
    _command_hard_source_manifest,
    _command_hard_source_prediction_cache,
    _command_hard_validation,
    _command_production_prediction_cache,
    _command_production_resolver_metadata,
    _command_scorer_eval,
    _command_scorer_training,
    _command_v2_scorer_training,
    _commands_dataset_build,
    _contract_for,
    _export_artifacts,
    _freeze_metadata,
    _promotion_check,
    _stage_complete,
    _stage_forced,
    _validate_stage_outputs,
    _write_scorer_training_sentinel,
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
        "overwrite_from": None,
        "force_downstream_of": None,
        "stop_after": None,
        "start_at": None,
        "write_config": False,
        "print_config_patch": False,
        "write_summary_md": False,
        "progress": False,
        "no_progress": True,
        "config_path": tmp_path / "extract.ini",
        "config_section": "align.ensemble",
        "python_executable": "python",
        "dataset": "wflw",
        "source_dir": None,
        "dataset_source": [],
        "hard_source_dataset": "aflw2000-3d",
        "hard_source_dir": None,
        "models": "hrnet,spiga,orformer",
        "candidates": "hrnet,spiga,orformer,static_weighted_downweight",
        "production_images": None,
        "production_alignments": None,
        "gt_hard_resolver_metadata": None,
        "overwrite_frozen_metadata": False,
        "generate_gt_hard_resolver_metadata": True,
        "hard_source_manifest": None,
        "split_mode": "scenario-stratified",
        "split_fit": 0.6,
        "split_select": 0.2,
        "split_report": 0.2,
        "split_seed": 1337,
        "scorer_target": "selection_cost",
        "promoted_scorer_version": "continuous_regret_v1_1",
        "v2_eval_fraction": 0.20,
        "v2_learning_rate": 0.05,
        "v2_iterations": 150,
        "v2_num_leaves": 31,
        "prediction_cache_mode": "run-models",
        "allow_image_backfill": True,
        "dataset_build_arg": [],
        "hard_source_dataset_build_arg": [],
        "cache_prediction_arg": [],
        "hard_source_cache_prediction_arg": [],
        "production_manifest_arg": [],
        "production_cache_prediction_arg": [],
        "production_resolver_metadata_arg": [],
        "candidate_search_arg": [],
        "hard_validation_arg": [],
        "gt_hard_resolver_metadata_arg": [],
        "scorer_train_arg": [],
        "scorer_eval_arg": [],
        "log_level": "INFO",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _touch_pipeline_outputs(paths: PipelinePaths, *, promotion_status: str = "pass") -> None:
    json_payload = {
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

    paths.run_cache.mkdir(parents=True, exist_ok=True)
    paths.production_cache.mkdir(parents=True, exist_ok=True)

    for output_path in (
        paths.run_manifest,
        paths.run_manifest_sentinel,
        paths.hard_source_manifest,
        paths.run_cache_sentinel,
        paths.hard_source_cache_sentinel,
        paths.run_splits,
        paths.run_static_weights,
        paths.run_summary,
        paths.production_manifest,
        paths.production_cache_sentinel,
        paths.production_root / "resolver_metadata.jsonl",
        paths.production_resolver_metadata_sentinel,
        paths.production_root / "audit.json",
        paths.best_setup,
        paths.best_weights,
        paths.gt_runtime_bucket_candidate_table,
        paths.gt_runtime_bucket_metrics_json,
        paths.gt_runtime_bucket_metrics_csv,
        paths.hard_manifest,
        paths.frozen_gt_metadata,
        paths.canonical_binary_scorer_artifact,
        paths.canonical_continuous_scorer_artifact,
        paths.canonical_v2_scorer_artifact,
        paths.scorer_rows_csv,
        paths.scorer_dataset_manifest,
        paths.scorer_suite_metrics,
        paths.scorer_training_sentinel,
        paths.scorer_report,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".json":
            output_path.write_text(json.dumps(json_payload) + chr(10), encoding="utf-8")
        elif output_path.suffix == ".csv":
            output_path.write_text(
                "sample_id,split" + chr(10) + "s1,eval" + chr(10),
                encoding="utf-8",
            )
        else:
            output_path.write_text("{}" + chr(10), encoding="utf-8")

    scorer_payload = {
        "artifact_schema_version": 1,
        "model_type": "linear_regression",
        "target": "selection_cost",
        "score_semantics": "predicted_cost",
        "higher_is_better": False,
        "features": ["candidate_name=hrnet"],
        "coefficients": [1.0],
        "intercept": 0.0,
        "version": "continuous_regret_v1_1",
        "scorer_version": "continuous_regret_v1_1",
        "runtime_policy": "learned_quality_v1_1",
    }
    paths.canonical_continuous_scorer_artifact.write_text(
        json.dumps(scorer_payload) + chr(10),
        encoding="utf-8",
    )

    binary_payload = dict(scorer_payload)
    binary_payload.update(
        {
            "model_type": "logistic_regression",
            "target": "candidate_failure_or_high_gap",
            "score_semantics": "predicted_risk",
            "version": "learned_quality_v1",
            "scorer_version": "learned_quality_v1",
            "runtime_policy": "learned_quality_v1",
        }
    )
    paths.canonical_binary_scorer_artifact.write_text(
        json.dumps(binary_payload) + chr(10),
        encoding="utf-8",
    )

    v2_payload = {
        "artifact_schema_version": 2,
        "model_type": "lightgbm_lambdarank",
        "target": "selection_cost",
        "score_semantics": "predicted_cost",
        "higher_is_better": False,
        "features": ["candidate_name=hrnet"],
        "model_data": "fake-model",
        "version": "learned_quality_v2",
        "scorer_version": "learned_quality_v2",
        "runtime_policy": "learned_quality_v2",
    }
    paths.canonical_v2_scorer_artifact.write_text(
        json.dumps(v2_payload) + chr(10),
        encoding="utf-8",
    )

    paths.promotion_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.promotion_manifest.write_text(
        json.dumps(
            {
                "active_policy": "learned_quality_v1_1",
                "best_setup": str(paths.best_setup),
                "best_weights": str(paths.best_weights),
                "runtime_resolver_scorer_source": str(paths.canonical_continuous_scorer_artifact),
                "runtime_resolver_scorer_source_sha256": __import__("hashlib")
                .sha256(paths.canonical_continuous_scorer_artifact.read_bytes())
                .hexdigest(),
                "runtime_resolver_scorer_version": "continuous_regret_v1_1",
                "promotion": {
                    "active_policy": "learned_quality_v1_1",
                    "source": str(paths.canonical_continuous_scorer_artifact),
                    "runtime_export": "production_bundle",
                    "immutable_scorer_artifact": True,
                },
            }
        )
        + chr(10),
        encoding="utf-8",
    )


def _write_v2_promotion_report(
    paths: PipelinePaths,
    *,
    v2_failure_rate: float = 0.0,
    v2_mean_nme: float = 0.015,
    v2_p90_nme: float = 0.024,
    static_failure_rate: float = 0.0,
    static_mean_nme: float = 0.016,
    static_p90_nme: float = 0.025,
) -> None:
    paths.scorer_report.parent.mkdir(parents=True, exist_ok=True)
    paths.scorer_report.write_text(
        json.dumps(
            {
                "status": "pass",
                "promotion_status": "pass",
                "failed_gates": [],
                "production_gate_status": "pass",
                "production_failed_gates": [],
                "gt_hard_gate_status": "pass",
                "gt_hard_failed_gates": [],
                "gate_config": {
                    "failure_rate_epsilon": 0.0,
                    "mean_epsilon_nme": 0.001,
                    "p90_epsilon_nme": 0.003,
                },
                "production_only_policy_metrics": {
                    "learned_quality_v2": {
                        "failure_rate": v2_failure_rate,
                        "mean_nme": v2_mean_nme,
                        "p90_nme": v2_p90_nme,
                    },
                    "static_weighted_downweight": {
                        "failure_rate": static_failure_rate,
                        "mean_nme": static_mean_nme,
                        "p90_nme": static_p90_nme,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_pipeline_runner_dry_run_writes_summaries_contracts_and_progress(tmp_path: Path) -> None:
    args = _args(tmp_path, dry_run=True, write_summary_md=True)

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
    assert summary["selected_runtime_policy"] == "learned_quality_v1_1"
    assert summary["selected_production_policy"] == "learned_quality_v1_1"
    assert summary["promoted_scorer_version"] == "continuous_regret_v1_1"
    assert summary["promoted_scorer_target"] == "selection_cost"
    # Path-bearing keys are no longer written into extract.ini — runtime
    # resolves artifacts from the installed production bundle instead.
    assert "resolver_scorer_path" not in summary["config_fields_changed"]
    assert "weights_path" not in summary["config_fields_changed"]
    assert "setup_path" not in summary["config_fields_changed"]
    assert "setup_mode" not in summary["config_fields_changed"]
    assert "resolver_policy" in summary["config_fields_changed"]
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


def test_overwrite_from_forces_stage_and_downstream_resume_skip(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, resume=True, overwrite_from="candidate_search")

    assert not _stage_forced("build_production_prediction_cache", args)
    assert _stage_forced("candidate_search", args)
    assert _stage_forced("scorer_evaluation", args)


def test_force_downstream_of_excludes_named_stage(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, resume=True, force_downstream_of="build_production_manifest")

    assert not _stage_forced("build_production_manifest", args)
    assert _stage_forced("build_production_prediction_cache", args)
    assert _stage_forced("candidate_search", args)


def test_pipeline_includes_production_resolver_metadata_before_candidate_search() -> None:
    assert (
        STAGES.index("build_production_prediction_cache")
        < STAGES.index("build_production_resolver_metadata")
        < STAGES.index("candidate_search")
    )


def test_overwrite_from_production_manifest_bypasses_existing_artifact_reuse(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        dry_run=True,
        resume=True,
        overwrite_from="build_production_manifest",
        start_at="build_production_manifest",
        stop_after="build_production_manifest",
        production_images=tmp_path / "images",
        production_alignments=tmp_path / "alignments.fsa",
    )
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    paths.production_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.production_manifest.write_text(json.dumps({"samples": []}) + "\n", encoding="utf-8")
    (paths.production_root / "resolver_metadata.jsonl").write_text("{}", encoding="utf-8")
    (paths.production_root / "audit.json").write_text("{}", encoding="utf-8")

    summary = run_pipeline(args)

    assert summary["stages"][0]["name"] == "build_production_manifest"
    assert summary["stages"][0]["status"] == "planned"
    assert summary["overwrite_from"] == "build_production_manifest"


def test_base_dataset_build_uses_replace_then_merge_for_multiple_datasets(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        dataset="wflw,cofw,300w",
        dataset_source=[
            f"wflw={tmp_path / 'wflw'}",
            f"cofw={tmp_path / 'cofw'}",
            f"300w={tmp_path / '300w'}",
        ],
    )
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    commands = _commands_dataset_build(args, paths)

    assert [command[command.index("--dataset") + 1] for command in commands] == [
        "wflw",
        "cofw",
        "300w",
    ]
    assert [command[command.index("--manifest-mode") + 1] for command in commands] == [
        "replace",
        "merge",
        "merge",
    ]
    assert commands[0][commands[0].index("--source-dir") + 1] == str(tmp_path / "wflw")
    assert commands[1][commands[1].index("--source-dir") + 1] == str(tmp_path / "cofw")
    assert commands[2][commands[2].index("--source-dir") + 1] == str(tmp_path / "300w")


def test_base_dataset_build_resume_requires_completion_sentinel(tmp_path: Path) -> None:
    args = _args(tmp_path, dataset="wflw,cofw")
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    paths.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.run_manifest.write_text(json.dumps({"samples": []}) + "\n", encoding="utf-8")

    assert not _stage_complete("build_dataset_manifest", args, paths)


def test_single_dataset_source_dir_is_passed_to_base_build(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, dataset="wflw", source_dir=tmp_path / "wflw_source")
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    command = _command_dataset_build(args, paths)

    assert command[command.index("--dataset") + 1] == "wflw"
    assert command[command.index("--source-dir") + 1] == str(tmp_path / "wflw_source")
    assert command[command.index("--manifest-mode") + 1] == "replace"


def test_hard_source_manifest_and_cache_default_to_aflw2000_3d(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        hard_source_dir=tmp_path / "aflw2000_source",
        hard_source_dataset_build_arg=["--hard-build-extra"],
        hard_source_cache_prediction_arg=["--hard-cache-extra"],
    )
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    manifest_command = _command_hard_source_manifest(args, paths)
    cache_command = _command_hard_source_prediction_cache(args, paths)

    assert manifest_command[manifest_command.index("--dataset") + 1] == "aflw2000-3d"
    assert manifest_command[manifest_command.index("--output-dir") + 1] == str(
        paths.hard_source_dataset_dir
    )
    assert manifest_command[manifest_command.index("--source-dir") + 1] == str(
        tmp_path / "aflw2000_source"
    )
    assert manifest_command[manifest_command.index("--manifest-mode") + 1] == "replace"
    assert "--hard-build-extra" in manifest_command
    assert cache_command[cache_command.index("--manifest") + 1] == str(paths.hard_source_manifest)
    assert cache_command[cache_command.index("--cache-dir") + 1] == str(paths.run_cache)
    assert "--run-models" in cache_command
    assert "--hard-cache-extra" in cache_command


def test_hard_source_manifest_override_is_used_by_cache_validation_and_summary(
    tmp_path: Path,
) -> None:
    custom_manifest = tmp_path / "custom_hard_manifest.json"
    args = _args(
        tmp_path,
        hard_source_manifest=custom_manifest,
        dry_run=True,
        start_at="build_hard_source_prediction_cache",
        stop_after="hard_alignment_validation",
    )
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    cache_command = _command_hard_source_prediction_cache(args, paths)
    validation_command = _command_hard_validation(args, paths)
    cache_contract = _contract_for("build_hard_source_prediction_cache", args, paths)
    validation_contract = _contract_for("hard_alignment_validation", args, paths)
    summary = run_pipeline(args)

    assert cache_command[cache_command.index("--manifest") + 1] == str(custom_manifest)
    assert validation_command[validation_command.index("--manifest") + 1] == str(custom_manifest)
    assert str(custom_manifest) in cache_contract.required_files
    assert str(custom_manifest) in validation_contract.required_files
    assert summary["sources"]["hard_source"]["manifest"] == str(custom_manifest)


def test_hard_source_manifest_override_skips_default_build_stage(tmp_path: Path) -> None:
    custom_manifest = tmp_path / "custom_hard_manifest.json"
    custom_manifest.write_text(json.dumps({"samples": []}) + "\n", encoding="utf-8")
    args = _args(
        tmp_path,
        hard_source_manifest=custom_manifest,
        start_at="build_hard_source_manifest",
        stop_after="build_hard_source_manifest",
    )

    summary = run_pipeline(args)

    assert summary["stages"][0]["status"] == "skipped"
    assert summary["stages"][0]["validated_outputs"] == [str(custom_manifest)]
    assert "caller-supplied hard-source manifest" in summary["stages"][0]["notes"][0]


def test_hard_source_cache_resume_rejects_sentinel_for_different_manifest(
    tmp_path: Path,
) -> None:
    custom_manifest = tmp_path / "custom_hard_manifest.json"
    args = _args(tmp_path, hard_source_manifest=custom_manifest)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    paths.hard_source_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.hard_source_manifest.write_text(json.dumps({"samples": []}) + "\n", encoding="utf-8")
    custom_manifest.write_text(json.dumps({"samples": []}) + "\n", encoding="utf-8")
    paths.run_cache.mkdir(parents=True, exist_ok=True)
    paths.hard_source_cache_sentinel.write_text(
        json.dumps(
            {
                "manifest": str(paths.hard_source_manifest),
                "manifest_sha256": hashlib.sha256(
                    paths.hard_source_manifest.read_bytes()
                ).hexdigest(),
                "models": args.models,
                "mode": args.prediction_cache_mode,
                "extra_args": args.hard_source_cache_prediction_arg,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert not _stage_complete("build_hard_source_prediction_cache", args, paths)


def test_hard_validation_defaults_to_aflw2000_hard_source_manifest(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    command = _command_hard_validation(args, paths)
    contract = _contract_for("hard_alignment_validation", args, paths)

    assert command[command.index("--manifest") + 1] == str(paths.hard_source_manifest)
    assert str(paths.hard_source_manifest) in contract.required_files
    assert str(paths.hard_source_cache_sentinel) in contract.required_files


def test_split_mode_accepts_uploaded_workflow_alias(tmp_path: Path) -> None:
    args = _args(tmp_path, dry_run=True, start_at="build_splits", stop_after="build_splits")
    args.split_mode = "scenario_stratified"

    summary = run_pipeline(args)

    assert summary["stages"][0]["name"] == "build_splits"
    assert summary["stages"][0]["status"] == "planned"


def test_split_mode_alias_is_persisted_as_canonical_mode(tmp_path: Path) -> None:
    args = _args(tmp_path, split_mode="scenario_stratified")
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    paths.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.run_manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"sample_id": f"s{i}", "dataset": "wflw", "condition": "frontal"}
                    for i in range(6)
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _build_splits(args, paths)

    payload = json.loads(paths.run_splits.read_text(encoding="utf-8"))
    summary = json.loads(paths.run_summary.read_text(encoding="utf-8"))
    assert payload["mode"] == "scenario-stratified"
    assert summary["split_mode"] == "scenario-stratified"


def test_pipeline_declares_production_prediction_cache_stage(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    assert "build_production_prediction_cache" in STAGES
    assert (
        STAGES.index("build_production_manifest")
        < STAGES.index("build_production_prediction_cache")
        < STAGES.index("build_production_resolver_metadata")
        < STAGES.index("candidate_search")
    )
    assert (
        STAGES.index("hard_alignment_validation")
        < STAGES.index("build_gt_hard_resolver_metadata")
        < STAGES.index("freeze_resolver_metadata")
    )

    contract = _contract_for("build_production_prediction_cache", args, paths)

    assert str(paths.production_manifest) in contract.required_files
    assert str(paths.production_cache) in contract.outputs
    assert str(paths.production_cache_sentinel) in contract.outputs


def test_production_resolver_metadata_contract_uses_image_cache_and_static_weights(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    contract = _contract_for("build_production_resolver_metadata", args, paths)

    assert str(paths.production_manifest) in contract.required_files
    assert str(paths.production_cache) in contract.required_files
    assert str(paths.production_cache_sentinel) in contract.required_files
    assert str(paths.run_static_weights) in contract.required_files
    assert str(paths.production_resolver_metadata) in contract.outputs
    assert str(paths.production_resolver_metadata_sentinel) in contract.outputs


def test_production_prediction_cache_command_uses_production_manifest_and_cache(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, production_cache_prediction_arg=["--extra-cache-option"])
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    command = _command_production_prediction_cache(args, paths)

    assert "cache_predictions.py" in command[1]
    assert command[command.index("--manifest") + 1] == str(paths.production_manifest)
    assert command[command.index("--cache-dir") + 1] == str(paths.production_cache)
    assert command[command.index("--models") + 1] == args.models
    assert "--run-models" in command
    assert "--extra-cache-option" in command


def test_production_resolver_metadata_command_writes_image_aware_sidecar(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, production_resolver_metadata_arg=["--extra-metadata-option"])
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    command = _command_production_resolver_metadata(args, paths)

    assert "build_production_resolver_metadata.py" in command[1]
    assert command[command.index("--manifest") + 1] == str(paths.production_manifest)
    assert command[command.index("--cache-dir") + 1] == str(paths.production_cache)
    assert command[command.index("--weights") + 1] == str(paths.run_static_weights)
    assert command[command.index("--candidates") + 1] == args.candidates
    assert command[command.index("--output") + 1] == str(paths.production_resolver_metadata)
    assert "--extra-metadata-option" in command


def test_prediction_cache_mode_fixtures_does_not_add_run_models(tmp_path: Path) -> None:
    args = _args(tmp_path, prediction_cache_mode="fixtures")
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    assert "--run-models" not in _command_production_prediction_cache(args, paths)
    assert "--run-models" not in _command_hard_source_prediction_cache(args, paths)


def test_candidate_search_contract_requires_built_production_prediction_cache(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    contract = _contract_for("candidate_search", args, paths)

    assert str(paths.production_cache) in contract.required_files
    assert str(paths.production_cache_sentinel) in contract.required_files
    assert str(paths.production_resolver_metadata) in contract.required_files
    assert str(paths.production_resolver_metadata_sentinel) in contract.required_files


def test_candidate_search_command_uses_full_manifest_with_splits(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    paths.run_report_manifest.parent.mkdir(parents=True)
    paths.run_report_manifest.write_text("{}", encoding="utf-8")

    command = _command_candidate_search(args, paths)

    assert command[command.index("--manifest") + 1] == str(paths.run_manifest)
    assert command[command.index("--splits") + 1] == str(paths.run_splits)
    assert command[command.index("--production-resolver-metadata") + 1] == str(
        paths.production_resolver_metadata
    )


@pytest.mark.parametrize(
    "promoted_scorer_version",
    ("continuous_regret_v1_1", "learned_quality_v2"),
)
def test_candidate_search_command_requires_effective_ensemble(
    tmp_path: Path,
    promoted_scorer_version: str,
) -> None:
    """Every promoted scorer version this pipeline supports installs a
    learned_quality_* runtime policy whose ranker has to choose between
    multiple fusion candidates built from the promoted setup. The pipeline
    therefore always passes ``--require-effective-ensemble`` to the search
    so a single-model or collapsed ensemble cannot reach disk as
    ``best_setup.json`` and leave the ranker with nothing to score."""
    args = _args(tmp_path, promoted_scorer_version=promoted_scorer_version)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    command = _command_candidate_search(args, paths)

    assert "--require-effective-ensemble" in command


def test_gt_hard_resolver_metadata_contract_runs_after_hard_validation(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    contract = _contract_for("build_gt_hard_resolver_metadata", args, paths)

    assert str(paths.hard_manifest) in contract.required_files
    assert str(paths.run_cache) in contract.required_files
    assert str(paths.best_weights) in contract.required_files
    assert str(paths.frozen_gt_metadata) in contract.outputs


def test_gt_hard_resolver_metadata_command_writes_frozen_sidecar(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, gt_hard_resolver_metadata_arg=["--extra-runtime-option"])
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    command = _command_gt_hard_resolver_metadata(args, paths)

    assert "build_gt_hard_resolver_metadata.py" in command[1]
    assert command[command.index("--manifest") + 1] == str(paths.hard_manifest)
    assert command[command.index("--cache-dir") + 1] == str(paths.run_cache)
    assert command[command.index("--weights") + 1] == str(paths.best_weights)
    assert command[command.index("--candidates") + 1] == args.candidates
    assert command[command.index("--output") + 1] == str(paths.frozen_gt_metadata)
    assert "--allow-image-backfill" in command
    assert "--extra-runtime-option" in command


def test_gt_hard_resolver_metadata_command_references_existing_cli(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    command = _command_gt_hard_resolver_metadata(args, paths)
    script = Path(command[1])

    assert script.name == "build_gt_hard_resolver_metadata.py"
    assert script.is_file()


def test_gt_hard_resolver_metadata_generation_ignores_stale_manifest_runtime_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.landmarks import build_gt_hard_resolver_metadata as metadata_cli

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "gt_hard",
                "samples": [
                    {
                        "sample_id": "s1",
                        "image": "s1.png",
                        "landmarks": "s1.npy",
                        "metadata": {
                            "face_index": 0,
                            "landmark_ensemble": {
                                "runtime_bucket": "stale_profile",
                                "runtime_bucket_source": "stale_manifest",
                                "selected_candidate": "stale_candidate",
                            },
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sample = metadata_cli.LandmarkSample(
        sample_id="s1",
        image=str(tmp_path / "s1.png"),
        landmarks=str(tmp_path / "s1.npy"),
        dataset="gt_hard",
        condition="profile",
        metadata={
            "face_index": 0,
            "landmark_ensemble": {
                "runtime_bucket": "stale_profile",
                "runtime_bucket_source": "stale_manifest",
                "selected_candidate": "stale_candidate",
            },
        },
    )
    observed: dict[str, object] = {}

    def fake_build_sample_context(sample_arg: object, **_: object) -> SimpleNamespace:
        observed["metadata"] = dict(sample_arg.metadata)  # type: ignore[attr-defined]
        return SimpleNamespace(
            runtime_bucket="generated_frontal",
            runtime_bucket_source="derived_no_image_evidence",
            current_policy_choice="hrnet",
            risk_route="low_risk",
            candidate_yaw_disagreement=None,
            max_disagreement_px=0.0,
            roll_estimate=0.0,
            yaw_estimate=0.0,
            model_predictions_available={"hrnet": True},
        )

    monkeypatch.setattr(metadata_cli, "load_manifest", lambda _: [sample])
    monkeypatch.setattr(metadata_cli, "load_weights", lambda _: {"hrnet": [1.0]})
    monkeypatch.setattr(metadata_cli, "DiskPredictionCache", lambda _: object())
    monkeypatch.setattr(metadata_cli, "build_sample_context", fake_build_sample_context)

    output = tmp_path / "resolver_metadata.jsonl"
    rows = metadata_cli.build_gt_hard_resolver_metadata(
        manifest=manifest,
        cache_dir=tmp_path / "cache",
        weights=tmp_path / "weights.json",
        candidates=("hrnet",),
        output=output,
    )

    assert "landmark_ensemble" not in observed["metadata"]  # type: ignore[operator]
    assert rows[0]["landmark_ensemble"]["runtime_bucket"] == "generated_frontal"
    assert rows[0]["landmark_ensemble"]["runtime_bucket_source"] == "derived_no_image_evidence"
    assert rows[0]["landmark_ensemble"]["selected_candidate"] == "hrnet"

    output_text = output.read_text(encoding="utf-8")
    assert "stale_profile" not in output_text
    assert "stale_manifest" not in output_text
    assert "stale_candidate" not in output_text


def test_stage_contract_declares_required_files(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    contract = _contract_for("continuous_scorer_training", args, paths)
    v2_contract = _contract_for("v2_scorer_training", args, paths)
    eval_contract = _contract_for("scorer_evaluation", args, paths)

    assert str(paths.hard_manifest) in contract.required_files
    assert str(paths.hard_source_cache_sentinel) in contract.required_files
    assert str(paths.production_cache_sentinel) in contract.required_files
    assert str(paths.frozen_gt_metadata) in contract.required_files
    assert str(paths.canonical_continuous_scorer_artifact) in contract.outputs
    assert str(paths.scorer_rows_csv) in contract.outputs
    assert str(paths.scorer_dataset_manifest) in contract.outputs
    assert str(paths.scorer_suite_metrics) in contract.outputs
    assert str(paths.scorer_training_sentinel) in contract.outputs
    assert str(paths.continuous_scorer_eval_rows) not in contract.outputs
    assert str(paths.hard_manifest) in v2_contract.required_files
    assert str(paths.hard_source_cache_sentinel) in v2_contract.required_files
    assert str(paths.production_cache_sentinel) in v2_contract.required_files
    assert str(paths.frozen_gt_metadata) in v2_contract.required_files
    assert str(paths.canonical_v2_scorer_artifact) in v2_contract.outputs
    assert str(paths.scorer_rows_csv) in v2_contract.outputs
    assert str(paths.scorer_dataset_manifest) in v2_contract.outputs
    assert str(paths.scorer_suite_metrics) in v2_contract.outputs
    assert str(paths.scorer_training_sentinel) in v2_contract.outputs
    assert str(paths.v2_scorer_training_sentinel) not in v2_contract.outputs
    assert str(paths.canonical_continuous_scorer_artifact) in eval_contract.required_files
    assert str(paths.canonical_binary_scorer_artifact) in eval_contract.required_files
    assert str(paths.canonical_v2_scorer_artifact) in eval_contract.required_files
    assert str(paths.scorer_rows_csv) in eval_contract.required_files
    assert str(paths.scorer_dataset_manifest) in eval_contract.required_files
    assert str(paths.continuous_scorer_eval_rows) not in eval_contract.required_files
    assert str(paths.hard_manifest) not in eval_contract.required_files
    assert str(paths.run_cache) not in eval_contract.required_files
    assert str(paths.hard_source_cache_sentinel) not in eval_contract.required_files
    assert str(paths.production_manifest) not in eval_contract.required_files
    assert str(paths.production_cache) not in eval_contract.required_files
    assert str(paths.production_cache_sentinel) not in eval_contract.required_files
    assert str(paths.best_weights) not in eval_contract.required_files
    assert str(paths.frozen_gt_metadata) not in eval_contract.required_files


def test_scorer_training_resume_requires_matching_sentinel_for_stage_aliases(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    required_and_outputs = (
        paths.hard_manifest,
        paths.production_manifest,
        paths.hard_source_cache_sentinel,
        paths.production_cache_sentinel,
        paths.best_weights,
        paths.frozen_gt_metadata,
        paths.canonical_binary_scorer_artifact,
        paths.canonical_continuous_scorer_artifact,
        paths.canonical_v2_scorer_artifact,
        paths.scorer_rows_csv,
        paths.scorer_dataset_manifest,
        paths.scorer_suite_metrics,
    )
    for path in required_and_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    assert not _stage_complete("scorer_training", args, paths)
    assert not _stage_complete("v2_scorer_training", args, paths)

    _write_scorer_training_sentinel(args, paths)

    assert _stage_complete("scorer_training", args, paths)
    assert _stage_complete("binary_scorer_training", args, paths)
    assert _stage_complete("continuous_scorer_training", args, paths)
    assert _stage_complete("v2_scorer_training", args, paths)

    args.v2_iterations += 1
    assert not _stage_complete("scorer_training", args, paths)
    assert not _stage_complete("v2_scorer_training", args, paths)


def test_scorer_train_and_eval_commands_allow_image_backfill_by_default(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    train_command = _command_scorer_training(
        args, paths, output_dir=paths.continuous_scorer_train_dir, target=args.scorer_target
    )
    v2_train_command = _command_v2_scorer_training(args, paths)
    eval_command = _command_scorer_eval(args, paths)

    assert "--allow-image-backfill" in train_command
    assert "--allow-image-backfill" in v2_train_command
    assert "--allow-image-backfill" in eval_command
    assert v2_train_command[v2_train_command.index("--training-mode") + 1] == "learned_quality_v2"
    assert v2_train_command[v2_train_command.index("--target") + 1] == "selection_cost"
    assert v2_train_command[v2_train_command.index("--eval-fraction") + 1] == str(
        args.v2_eval_fraction
    )
    assert v2_train_command[v2_train_command.index("--learning-rate") + 1] == str(
        args.v2_learning_rate
    )
    assert v2_train_command[v2_train_command.index("--iterations") + 1] == str(args.v2_iterations)
    assert v2_train_command[v2_train_command.index("--num-leaves") + 1] == str(args.v2_num_leaves)
    assert v2_train_command[v2_train_command.index("--output-dir") + 1] == str(
        paths.v2_scorer_train_dir
    )
    assert eval_command[eval_command.index("--v2-scorer") + 1] == str(
        paths.canonical_v2_scorer_artifact
    )


def test_scorer_train_and_eval_commands_allow_image_backfill_can_be_disabled(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, allow_image_backfill=False)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    train_command = _command_scorer_training(
        args, paths, output_dir=paths.continuous_scorer_train_dir, target=args.scorer_target
    )
    v2_train_command = _command_v2_scorer_training(args, paths)
    eval_command = _command_scorer_eval(args, paths)

    assert "--allow-image-backfill" not in train_command
    assert "--allow-image-backfill" not in v2_train_command
    assert "--allow-image-backfill" not in eval_command


def test_stage_output_validation_reports_missing_outputs(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    with pytest.raises(PipelineContractError, match="candidate_search.*best_setup.json"):
        _validate_stage_outputs("candidate_search", paths)


def test_freeze_metadata_requires_explicit_source_when_generation_is_disabled(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, generate_gt_hard_resolver_metadata=False)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)

    with pytest.raises(FileNotFoundError, match="will not derive runtime resolver metadata"):
        _freeze_metadata(args, paths)


def test_freeze_metadata_does_not_derive_from_plain_hard_manifest(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
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

    with pytest.raises(FileNotFoundError, match="will not derive runtime resolver metadata"):
        _freeze_metadata(args, paths)


def test_freeze_metadata_copies_explicit_sidecar(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
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
    sidecar = tmp_path / "explicit_resolver_metadata.jsonl"
    sidecar.write_text(
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
    args.gt_hard_resolver_metadata = sidecar

    notes = _freeze_metadata(args, paths)

    assert "copied and validated frozen metadata" in notes[0]
    rows = [
        json.loads(line)
        for line in paths.frozen_gt_metadata.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "face_index": 0,
            "landmark_ensemble": {"runtime_bucket": "frontal"},
            "sample_id": "s1",
        }
    ]


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


def test_resume_with_explicit_metadata_sidecar_does_not_skip_freeze_stage(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        resume=True,
        start_at="freeze_resolver_metadata",
        stop_after="freeze_resolver_metadata",
    )
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
                "landmark_ensemble": {"runtime_bucket": "old"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replacement = tmp_path / "replacement_resolver_metadata.jsonl"
    replacement.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "face_index": 0,
                "landmark_ensemble": {"runtime_bucket": "new"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args.gt_hard_resolver_metadata = replacement
    args.overwrite_frozen_metadata = True

    summary = run_pipeline(args)

    assert summary["stages"][0]["status"] == "ok"
    assert "copied and validated frozen metadata" in summary["stages"][0]["notes"][0]
    assert "new" in paths.frozen_gt_metadata.read_text(encoding="utf-8")


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
    assert preview["updates"]["resolver_policy"] == "learned_quality_v1_1"
    assert "production_weights_path" not in preview["updates"]
    assert "coordinate_space" not in preview["updates"]
    # Path-bearing keys are no longer in the patch.
    assert "setup_path" not in preview["updates"]
    assert "weights_path" not in preview["updates"]
    assert "resolver_scorer_path" not in preview["updates"]
    assert "setup_mode" not in preview["updates"]
    assert patch.startswith("[align.ensemble]\n")
    assert "weights_path = " not in patch
    assert "resolver_scorer_path = " not in patch

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
    parser.optionxform = str  # type: ignore[assignment, method-assign]
    parser.read(args.config_path, encoding="utf-8")

    assert parser.get("global", "aligner_min_scale") == "0.03"
    assert parser.get("align.fan", "batch_size") == "16"
    assert parser.get("detect.scrfd", "model") == "10g"
    assert parser.get("align.ensemble", "stale_experimental") == "true"
    assert parser.get("align.ensemble", "use_alignment_resolver") == "True"
    assert parser.get("align.ensemble", "resolver_policy") == "learned_quality_v1_1"
    # Path-bearing keys are not written into extract.ini anymore; the runtime
    # resolves them from the installed production bundle.
    assert not parser.has_option("align.ensemble", "resolver_scorer_path")
    assert not parser.has_option("align.ensemble", "weights_path")
    assert not parser.has_option("align.ensemble", "setup_path")
    assert not parser.has_option("align.ensemble", "setup_mode")

    config_text = args.config_path.read_text(encoding="utf-8")
    assert "# global comment survives" in config_text
    assert "# align ensemble intro comment survives" in config_text
    assert "# align ensemble key comment survives" in config_text
    assert "; align ensemble footer comment survives" in config_text
    assert "# align fan comment survives" in config_text


def test_config_write_appends_missing_align_ensemble_section(tmp_path: Path) -> None:
    args = _args(tmp_path, write_config=True)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)
    args.config_path.write_text(
        "\n".join(
            [
                "# config header comment survives",
                "[global]",
                "aligner_min_scale = 0.03",
                "",
                "[detect.scrfd]",
                "model = 10g",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _apply_config(args, paths)

    config_text = args.config_path.read_text(encoding="utf-8")
    assert "# config header comment survives" in config_text
    assert "[global]" in config_text
    assert "[detect.scrfd]" in config_text
    assert "[align.ensemble]" in config_text
    assert "resolver_policy = learned_quality_v1_1" in config_text

    parser = configparser.ConfigParser()
    parser.optionxform = str  # type: ignore[assignment, method-assign]
    parser.read(args.config_path, encoding="utf-8")
    assert parser.get("global", "aligner_min_scale") == "0.03"
    assert parser.get("detect.scrfd", "model") == "10g"
    assert parser.get("align.ensemble", "resolver_policy") == "learned_quality_v1_1"
    # Path-bearing keys are no longer written into extract.ini.
    assert not parser.has_option("align.ensemble", "weights_path")


def test_config_write_preserves_inline_comments_on_managed_keys(tmp_path: Path) -> None:
    args = _args(tmp_path, write_config=True)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)
    args.config_path.write_text(
        "\n".join(
            [
                "[align.ensemble]",
                "resolver_policy = stale_policy # keep this comment",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _apply_config(args, paths)

    config_text = args.config_path.read_text(encoding="utf-8")
    assert "resolver_policy = learned_quality_v1_1 # keep this comment" in config_text


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


def test_artifact_export_records_promoted_scorer_provenance(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)
    source_scorer_payload = {
        "artifact_schema_version": 1,
        "model_type": "linear_regression",
        "target": "selection_cost",
        "score_semantics": "predicted_cost",
        "higher_is_better": False,
        "features": ["candidate_name=hrnet"],
        "coefficients": [1.0],
        "intercept": 0.0,
        "version": "continuous_regret_v1_1",
    }
    paths.canonical_continuous_scorer_artifact.write_text(
        json.dumps(source_scorer_payload) + "\n",
        encoding="utf-8",
    )

    manifest = _export_artifacts(args, paths)

    assert manifest["active_policy"] == "learned_quality_v1_1"
    assert manifest["best_setup"] == str(paths.best_setup)
    assert manifest["best_weights"] == str(paths.best_weights)
    assert manifest["runtime_resolver_scorer_source"] == str(
        paths.canonical_continuous_scorer_artifact
    )
    assert manifest["runtime_resolver_scorer_version"] == "continuous_regret_v1_1"
    assert manifest["promotion"]["active_policy"] == "learned_quality_v1_1"
    assert manifest["promotion"]["source"] == str(paths.canonical_continuous_scorer_artifact)
    assert manifest["promotion"]["runtime_export"] == "production_bundle"
    assert manifest["promotion"]["immutable_scorer_artifact"] is True
    assert paths.promotion_manifest.is_file()

    assert not paths.exported_scorer_artifact.exists()
    assert not paths.exported_scorer_rows.exists()
    assert not paths.exported_scorer_dataset_manifest.exists()
    assert (
        json.loads(paths.canonical_continuous_scorer_artifact.read_text(encoding="utf-8"))
        == source_scorer_payload
    )


def test_v2_promotion_check_requires_v2_policy_metrics(tmp_path: Path) -> None:
    args = _args(tmp_path, promoted_scorer_version="learned_quality_v2")
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)

    with pytest.raises(RuntimeError, match="learned_quality_v2 promotion check failed"):
        _promotion_check(paths, promotion_scope=args.promotion_scope, args=args)


def test_artifact_export_blocks_v2_when_v2_metrics_regress(tmp_path: Path) -> None:
    args = _args(tmp_path, promoted_scorer_version="learned_quality_v2")
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)
    _write_v2_promotion_report(paths, v2_mean_nme=0.020, static_mean_nme=0.016)
    paths.canonical_v2_scorer_artifact.write_text(
        json.dumps(
            {
                "artifact_schema_version": 2,
                "model_type": "lightgbm_lambdarank",
                "target": "selection_cost",
                "score_semantics": "predicted_cost",
                "higher_is_better": False,
                "features": ["candidate_name=hrnet"],
                "model_data": "fake-model",
                "version": "learned_quality_v2",
                "scorer_version": "learned_quality_v2",
                "runtime_policy": "learned_quality_v2",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="learned_quality_v2 promotion check failed"):
        _export_artifacts(args, paths)

    assert not paths.exported_scorer_artifact.exists()


def test_artifact_export_can_promote_v2_scorer(tmp_path: Path) -> None:
    args = _args(tmp_path, promoted_scorer_version="learned_quality_v2")
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)
    _write_v2_promotion_report(paths)
    v2_payload = {
        "artifact_schema_version": 2,
        "model_type": "lightgbm_lambdarank",
        "target": "selection_cost",
        "score_semantics": "predicted_cost",
        "higher_is_better": False,
        "features": ["candidate_name=hrnet"],
        "model_data": "fake-model",
        "version": "learned_quality_v2",
        "scorer_version": "learned_quality_v2",
        "runtime_policy": "learned_quality_v2",
    }
    paths.canonical_v2_scorer_artifact.write_text(
        json.dumps(v2_payload) + "\n",
        encoding="utf-8",
    )

    manifest = _export_artifacts(args, paths)

    assert manifest["active_policy"] == "learned_quality_v2"
    assert manifest["runtime_resolver_scorer_source"] == str(paths.canonical_v2_scorer_artifact)
    assert manifest["runtime_resolver_scorer_version"] == "learned_quality_v2"
    assert manifest["promotion"]["active_policy"] == "learned_quality_v2"
    assert manifest["promotion"]["source"] == str(paths.canonical_v2_scorer_artifact)
    assert manifest["promotion"]["model_type"] == "lightgbm_lambdarank"
    assert manifest["promotion"]["runtime_export"] == "production_bundle"
    assert not paths.exported_scorer_artifact.exists()


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
    assert "resolver_policy = learned_quality_v1_1" in config_text
    assert "# config comment survives" in config_text


# ---------------------------------------------------------------------------
# Production bundle install (Phase 2)
# ---------------------------------------------------------------------------


def test_apply_config_installs_production_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--write-config`` installs a bundle so runtime can resolve artifacts.

    Validates the Phase 2 contract: the pipeline copies the promoted setup,
    weights, and every present scorer artifact into the bundle directory and
    writes a manifest whose ``active_policy`` matches the runtime policy
    derived from the promoted scorer version.
    """
    from lib.landmarks.ensemble import production_artifacts as pa

    args = _args(tmp_path, write_config=True)
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)
    args.config_path.write_text("[align.ensemble]\n", encoding="utf-8")

    bundle_root = tmp_path / "fs_cache" / "current"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle_root))

    notes = _apply_config(args, paths)

    assert any("installed production landmark-ensemble bundle" in note for note in notes)
    loaded = pa.load_production_bundle()
    # continuous_regret_v1_1 is the default promoted_scorer_version in _args.
    assert loaded.active_policy == "learned_quality_v1_1"
    assert loaded.setup_path.is_file()
    # All three scorer sources existed via _touch_pipeline_outputs, so all
    # three policy slots should be populated.
    assert set(loaded.scorers) == {
        "learned_quality_v1",
        "learned_quality_v1_1",
        "learned_quality_v2",
    }
    assert loaded.scorer_path_for("learned_quality_v1_1").is_file()  # type: ignore[union-attr]
    assert loaded.scorer_path_for(pa.ROLL_AWARE_VETO_POLICY) is None


def test_apply_config_without_write_skips_bundle_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preview-only invocations must not touch the production bundle dir."""
    from lib.landmarks.ensemble import production_artifacts as pa

    args = _args(tmp_path)  # write_config defaults to False
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    _touch_pipeline_outputs(paths)

    bundle_root = tmp_path / "fs_cache" / "current"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle_root))

    _apply_config(args, paths)

    assert not bundle_root.exists()
