#!/usr/bin/env python3
"""Run the landmark resolver promotion pipeline end to end."""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import shutil
import subprocess
import sys
import time
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.pipeline_conventions import (
    RESOLVER_METADATA_JSONL,
    SCORER_POLICY_REPORT_JSON,
    SOURCE_GT_HARD,
    SOURCE_PRODUCTION_VALIDATED,
    write_json,
)
from tools.landmarks.pipeline_progress import PipelineProgress, configure_logging

logger = logging.getLogger("run_landmark_resolver_pipeline")

STAGES: tuple[str, ...] = (
    "static_pipeline",
    "candidate_search",
    "hard_alignment_validation",
    "freeze_resolver_metadata",
    "scorer_training",
    "scorer_evaluation",
    "production_promotion_check",
    "artifact_export",
    "config_update",
)
DEFAULT_MODELS = "hrnet,spiga,orformer"
DEFAULT_CANDIDATES = "hrnet,spiga,orformer,plain_average,static_weighted,static_weighted_downweight,static_weighted_hard_drop,weighted_median"
DEFAULT_CONFIG_SECTION = "align.ensemble"
CONFIG_PREVIEW_FILENAME = "config_update_preview.json"
CONFIG_PATCH_FILENAME = "config_update_patch.ini"
PROGRESS_LOG_FILENAME = "pipeline_progress.jsonl"


class PipelineContractError(RuntimeError):
    """A pipeline stage did not satisfy its declared contract."""


@dataclass(frozen=True)
class PipelinePaths:
    run_root: Path
    production_root: Path
    output_root: Path
    run_manifest: Path = field(init=False)
    run_report_manifest: Path = field(init=False)
    run_cache: Path = field(init=False)
    run_splits: Path = field(init=False)
    run_static_weights: Path = field(init=False)
    run_summary: Path = field(init=False)
    production_manifest: Path = field(init=False)
    production_cache: Path = field(init=False)
    candidate_dir: Path = field(init=False)
    best_setup: Path = field(init=False)
    best_weights: Path = field(init=False)
    hard_dir: Path = field(init=False)
    hard_manifest: Path = field(init=False)
    frozen_gt_metadata: Path = field(init=False)
    scorer_train_dir: Path = field(init=False)
    scorer_artifact: Path = field(init=False)
    scorer_eval_dir: Path = field(init=False)
    scorer_report: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    progress_log: Path = field(init=False)
    summary_json: Path = field(init=False)
    summary_md: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_manifest", self.run_root / "dataset" / "manifest.json")
        object.__setattr__(
            self, "run_report_manifest", self.run_root / "dataset" / "report_manifest.json"
        )
        object.__setattr__(self, "run_cache", self.run_root / "cache")
        object.__setattr__(self, "run_splits", self.run_root / "splits" / "splits.json")
        object.__setattr__(
            self, "run_static_weights", self.run_root / "weights" / "static_landmark_weights.json"
        )
        object.__setattr__(self, "run_summary", self.run_root / "run_summary.json")
        object.__setattr__(
            self,
            "production_manifest",
            _first_existing(
                self.production_root / "manifest.json",
                self.production_root / "dataset" / "manifest.json",
            ),
        )
        object.__setattr__(
            self,
            "production_cache",
            _first_existing(
                self.production_root / "cache", self.production_root / "prediction_cache"
            ),
        )
        object.__setattr__(self, "candidate_dir", self.output_root / "candidate_search")
        object.__setattr__(self, "best_setup", self.candidate_dir / "best_setup.json")
        object.__setattr__(self, "best_weights", self.candidate_dir / "best_weights.json")
        object.__setattr__(self, "hard_dir", self.output_root / "gt_hard_validation")
        object.__setattr__(self, "hard_manifest", self.hard_dir / "manifest.json")
        object.__setattr__(self, "frozen_gt_metadata", self.output_root / RESOLVER_METADATA_JSONL)
        object.__setattr__(self, "scorer_train_dir", self.output_root / "scorer_training")
        object.__setattr__(
            self, "scorer_artifact", self.scorer_train_dir / "runtime_resolver_scorer.json"
        )
        object.__setattr__(self, "scorer_eval_dir", self.output_root / "scorer_evaluation")
        object.__setattr__(self, "scorer_report", self.scorer_eval_dir / SCORER_POLICY_REPORT_JSON)
        object.__setattr__(self, "artifacts_dir", self.output_root / "artifacts")
        object.__setattr__(self, "progress_log", self.output_root / PROGRESS_LOG_FILENAME)
        object.__setattr__(self, "summary_json", self.output_root / "pipeline_summary.json")
        object.__setattr__(self, "summary_md", self.output_root / "pipeline_summary.md")


@dataclass(frozen=True)
class StageContract:
    name: str
    inputs: list[str]
    outputs: list[str]
    cache_behavior: str
    required_files: list[str]
    success_criteria: list[str]

    def to_json(self) -> dict[str, T.Any]:
        return dict(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            cache_behavior=self.cache_behavior,
            required_files=self.required_files,
            success_criteria=self.success_criteria,
        )


@dataclass
class StageResult:
    name: str
    status: str
    duration_seconds: float = 0.0
    command: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    validated_outputs: list[str] = field(default_factory=list)
    contract: dict[str, T.Any] = field(default_factory=dict)
    error: str = ""
    notes: list[str] = field(default_factory=list)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _jsonable(value: T.Any) -> T.Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _read_json(path: Path) -> dict[str, T.Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _append_extra(argv: list[str], extra: T.Sequence[str]) -> list[str]:
    argv.extend(extra)
    return argv


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required {label}: {path}")


def _stage_slice(start_at: str | None, stop_after: str | None) -> tuple[str, ...]:
    start = STAGES.index(start_at) if start_at else 0
    stop = STAGES.index(stop_after) if stop_after else len(STAGES) - 1
    if start > stop:
        raise ValueError(f"--start-at {start_at!r} occurs after --stop-after {stop_after!r}")
    return STAGES[start : stop + 1]


def _script(path: str) -> str:
    return str(ROOT / "tools" / "landmarks" / path)


def _command_candidate_search(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    manifest = (
        paths.run_report_manifest if paths.run_report_manifest.exists() else paths.run_manifest
    )
    argv = [
        args.python_executable,
        _script("search_ensemble_setup.py"),
        "--manifest",
        str(manifest),
        "--cache-dir",
        str(paths.run_cache),
        "--splits",
        str(paths.run_splits),
        "--models",
        args.models,
        "--output-dir",
        str(paths.candidate_dir),
        "--include-geometry-metrics",
        "--production-manifest",
        str(paths.production_manifest),
        "--production-cache-dir",
        str(paths.production_cache),
        "--production-gate-output",
        str(paths.candidate_dir / "production_gate"),
    ]
    return _append_extra(argv, args.candidate_search_arg)


def _command_hard_validation(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    source_manifest = args.hard_source_manifest or paths.run_manifest
    weights = paths.best_weights if paths.best_weights.exists() else paths.run_static_weights
    return _append_extra(
        [
            args.python_executable,
            _script("build_hard_alignment_validation.py"),
            "--manifest",
            str(source_manifest),
            "--output-dir",
            str(paths.hard_dir),
            "--cache-dir",
            str(paths.run_cache),
            "--models",
            args.models,
            "--weights",
            str(weights),
        ],
        args.hard_validation_arg,
    )


def _command_scorer_training(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    return _append_extra(
        [
            args.python_executable,
            _script("train_runtime_resolver_scorer.py"),
            "--gt-manifest",
            str(paths.hard_manifest),
            "--gt-cache-dir",
            str(paths.run_cache),
            "--production-manifest",
            str(paths.production_manifest),
            "--production-cache-dir",
            str(paths.production_cache),
            "--weights",
            str(paths.best_weights),
            "--candidates",
            args.candidates,
            "--output-dir",
            str(paths.scorer_train_dir),
            "--gt-hard-resolver-metadata",
            str(paths.frozen_gt_metadata),
        ],
        args.scorer_train_arg,
    )


def _command_scorer_eval(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    return _append_extra(
        [
            args.python_executable,
            _script("evaluate_runtime_resolver_scorer.py"),
            "--gt-manifest",
            str(paths.hard_manifest),
            "--gt-cache-dir",
            str(paths.run_cache),
            "--production-manifest",
            str(paths.production_manifest),
            "--production-cache-dir",
            str(paths.production_cache),
            "--weights",
            str(paths.best_weights),
            "--scorer",
            str(paths.scorer_artifact),
            "--candidates",
            args.candidates,
            "--output-dir",
            str(paths.scorer_eval_dir),
            "--promotion-scope",
            args.promotion_scope,
            "--gt-hard-resolver-metadata",
            str(paths.frozen_gt_metadata),
        ],
        args.scorer_eval_arg,
    )


def _outputs_for(stage: str, paths: PipelinePaths) -> list[Path]:
    return {
        "static_pipeline": [
            paths.run_summary,
            paths.run_manifest,
            paths.run_cache,
            paths.run_splits,
            paths.run_static_weights,
        ],
        "candidate_search": [paths.best_setup, paths.best_weights],
        "hard_alignment_validation": [paths.hard_manifest],
        "freeze_resolver_metadata": [paths.frozen_gt_metadata],
        "scorer_training": [paths.scorer_artifact],
        "scorer_evaluation": [paths.scorer_report],
        "production_promotion_check": [paths.scorer_report],
        "artifact_export": [paths.artifacts_dir / "artifacts_manifest.json"],
        "config_update": [
            paths.output_root / CONFIG_PREVIEW_FILENAME,
            paths.output_root / CONFIG_PATCH_FILENAME,
        ],
    }[stage]


def _required_inputs_for(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> list[Path]:
    if stage == "static_pipeline":
        return [
            paths.run_summary,
            paths.run_manifest,
            paths.run_cache,
            paths.run_splits,
            paths.run_static_weights,
        ]
    return {
        "candidate_search": [
            paths.run_cache,
            paths.run_splits,
            paths.production_manifest,
            paths.production_cache,
        ],
        "hard_alignment_validation": [
            args.hard_source_manifest or paths.run_manifest,
            paths.run_cache,
            paths.best_weights if paths.best_weights.exists() else paths.run_static_weights,
        ],
        "freeze_resolver_metadata": []
        if paths.frozen_gt_metadata.exists()
        else ([args.gt_hard_resolver_metadata] if args.gt_hard_resolver_metadata else []),
        "scorer_training": [
            paths.hard_manifest,
            paths.run_cache,
            paths.production_manifest,
            paths.production_cache,
            paths.best_weights,
            paths.frozen_gt_metadata,
        ],
        "scorer_evaluation": [
            paths.hard_manifest,
            paths.run_cache,
            paths.production_manifest,
            paths.production_cache,
            paths.best_weights,
            paths.scorer_artifact,
            paths.frozen_gt_metadata,
        ],
        "production_promotion_check": [paths.scorer_report],
        "artifact_export": [
            paths.best_setup,
            paths.best_weights,
            paths.scorer_artifact,
            paths.scorer_report,
            paths.frozen_gt_metadata,
        ],
        "config_update": [Path(args.config_path)] if args.write_config else [],
    }[stage]


def _contract_for(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> StageContract:
    cache_behavior = {
        "static_pipeline": "reuse existing run-root artifacts; generation moved to active dataset/cache/search tools",
        "candidate_search": "writes candidate_search artifacts; --resume skips when best_setup.json and best_weights.json exist",
        "hard_alignment_validation": "writes gt_hard_validation artifacts; --resume skips when manifest.json exists",
        "freeze_resolver_metadata": "copies caller-supplied GT-hard sidecar once; never silently regenerates; --resume reuses existing frozen copy",
        "scorer_training": "writes scorer_training artifacts; --resume skips when runtime_resolver_scorer.json exists",
        "scorer_evaluation": "writes scorer_evaluation reports; --resume skips when scorer_policy_report.json exists",
        "production_promotion_check": "reads scorer report only; no recompute; requires promotion_status/status pass",
        "artifact_export": "copies final artifacts into artifacts/; --resume skips when artifacts_manifest.json exists",
        "config_update": "writes deterministic config_update_preview.json and config_update_patch.ini; writes config only with --write-config after promotion passes",
    }[stage]
    success = {
        "static_pipeline": ["run summary, manifest, cache, splits, and static weights exist"],
        "candidate_search": ["best_setup.json and best_weights.json exist"],
        "hard_alignment_validation": ["GT-hard manifest exists"],
        "freeze_resolver_metadata": [
            "frozen resolver_metadata.jsonl exists and was not regenerated implicitly"
        ],
        "scorer_training": ["runtime_resolver_scorer.json exists"],
        "scorer_evaluation": ["scorer_policy_report.json exists"],
        "production_promotion_check": ["scorer report promotion_status/status is pass"],
        "artifact_export": ["artifacts_manifest.json exists"],
        "config_update": [
            "config_update_preview.json exists",
            "config_update_patch.ini exists",
            "referenced artifacts exist",
            "when --write-config is set, target config file is updated",
        ],
    }[stage]
    required = [str(path) for path in _required_inputs_for(stage, args, paths) if path is not None]
    return StageContract(
        stage,
        required,
        [str(path) for path in _outputs_for(stage, paths)],
        cache_behavior,
        required,
        success,
    )


def _stage_complete(stage: str, paths: PipelinePaths) -> bool:
    return all(path.exists() for path in _outputs_for(stage, paths))


def _run_command(command: list[str]) -> None:
    logger.info("Running command: %s", " ".join(command))
    logger.debug("Pipeline child command argv=%r", command)
    if logger.isEnabledFor(5):
        logger.log(5, "Pipeline child command cwd=%s argv=%s", ROOT, json.dumps(command))
    subprocess.run(command, check=True, cwd=str(ROOT))


def _freeze_metadata(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    if paths.frozen_gt_metadata.exists() and args.resume:
        return [f"reused existing frozen metadata: {paths.frozen_gt_metadata}"]
    if paths.frozen_gt_metadata.exists() and not args.overwrite_frozen_metadata:
        return [
            f"frozen metadata already exists and was left unchanged: {paths.frozen_gt_metadata}"
        ]
    if args.gt_hard_resolver_metadata is None:
        raise FileNotFoundError(
            "missing required GT-hard resolver sidecar; pass --gt-hard-resolver-metadata. The pipeline will not silently regenerate GT-hard metadata."
        )
    source = Path(args.gt_hard_resolver_metadata)
    _require(source, "GT-hard resolver sidecar")
    paths.frozen_gt_metadata.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, paths.frozen_gt_metadata)
    return [f"copied frozen metadata from {source}"]


def _promotion_check(paths: PipelinePaths) -> list[str]:
    _require(paths.scorer_report, "scorer evaluation report")
    report = _read_json(paths.scorer_report)
    status = str(report.get("promotion_status") or report.get("status") or "")
    failed = report.get("failed_gates") or []
    if status != "pass":
        raise RuntimeError(
            f"production promotion check failed: status={status!r}, failed_gates={failed}"
        )
    return ["promotion_status=pass"]


def _validate_required_files(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> None:
    for required in _required_inputs_for(stage, args, paths):
        if required is not None:
            logger.debug("Validating required file for stage=%s path=%s", stage, required)
            _require(required, f"{stage} input")


def _validate_stage_outputs(stage: str, paths: PipelinePaths) -> list[str]:
    missing = [path for path in _outputs_for(stage, paths) if not path.exists()]
    if missing:
        details = ", ".join(str(path) for path in missing)
        raise PipelineContractError(
            f"stage {stage!r} did not produce required output(s): {details}"
        )
    if stage == "production_promotion_check":
        report = _read_json(paths.scorer_report)
        status = str(report.get("promotion_status") or report.get("status") or "")
        if status != "pass":
            failed = report.get("failed_gates") or []
            raise PipelineContractError(
                f"stage 'production_promotion_check' expected promotion_status=pass but got {status!r}; failed_gates={failed}"
            )
    validated = [str(path) for path in _outputs_for(stage, paths)]
    logger.debug("Validated outputs for stage=%s outputs=%s", stage, validated)
    return validated


def _copy_if_exists(source: Path, dest_dir: Path, *, name: str | None = None) -> str:
    if not source.exists():
        return ""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (name or source.name)
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copyfile(source, target)
    logger.debug("Exported artifact source=%s target=%s", source, target)
    return str(target)


def _export_artifacts(paths: PipelinePaths) -> dict[str, str]:
    exported = {
        "best_setup": _copy_if_exists(paths.best_setup, paths.artifacts_dir),
        "best_weights": _copy_if_exists(paths.best_weights, paths.artifacts_dir),
        "runtime_resolver_scorer": _copy_if_exists(paths.scorer_artifact, paths.artifacts_dir),
        "scorer_policy_report": _copy_if_exists(paths.scorer_report, paths.artifacts_dir),
        "gt_hard_resolver_metadata": _copy_if_exists(
            paths.frozen_gt_metadata, paths.artifacts_dir
        ),
    }
    manifest = {key: value for key, value in exported.items() if value}
    write_json(paths.artifacts_dir / "artifacts_manifest.json", manifest)
    return manifest


def _config_updates(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, str]:
    models = ",".join(item.strip() for item in str(args.models).split(",") if item.strip())
    return {
        "coordinate_space": "frame",
        "crop_scale": "1.6",
        "crop_to_frame": "true",
        "debug_metadata": "true",
        "debug_metadata_format": "jsonl",
        "enabled": "true",
        "fallback_model": "orformer",
        "fallback_strategy": "plain_average",
        "hard_case_strategy": "static_weighted_downweight",
        "hard_disagreement_px": "12.0",
        "hard_roll_degrees": "30.0",
        "min_models": "1",
        "models": models or DEFAULT_MODELS,
        "normalized_output": "true",
        "outlier_threshold": "3.5",
        "production_artifact_root": str(paths.artifacts_dir),
        "production_scorer_path": str(paths.scorer_artifact),
        "production_setup_path": str(paths.best_setup),
        "production_weights_path": str(paths.best_weights),
        "reject_outliers": "false",
        "resolver_policy": "learned_quality_v1",
        "resolver_scorer_path": str(paths.scorer_artifact),
        "roll_veto_degrees": "15.0",
        "secondary_hard_case_strategy": "static_weighted_hard_drop",
        "setup_mode": "strict",
        "setup_path": str(paths.best_setup),
        "strategy": "static_weighted_downweight",
        "strict": "true",
        "use_alignment_resolver": "true",
        "weights_path": str(paths.best_weights),
    }


def _config_patch_text(section: str, updates: T.Mapping[str, str]) -> str:
    lines = [f"[{section}]"]
    lines.extend(f"{key} = {updates[key]}" for key in sorted(updates))
    return "\n".join(lines) + "\n"


def _validate_config_artifacts(paths: PipelinePaths) -> None:
    for path, label in (
        (paths.best_setup, "promoted setup artifact"),
        (paths.best_weights, "promoted weights artifact"),
        (paths.scorer_artifact, "runtime resolver scorer artifact"),
        (paths.scorer_report, "scorer evaluation report"),
    ):
        _require(path, label)


def _write_config_patch_files(
    args: argparse.Namespace, paths: PipelinePaths, updates: T.Mapping[str, str]
) -> str:
    patch = _config_patch_text(args.config_section, updates)
    write_json(
        paths.output_root / CONFIG_PREVIEW_FILENAME,
        {
            "config_path": str(args.config_path),
            "section": args.config_section,
            "write_config": bool(args.write_config),
            "updates": dict(updates),
            "patch": patch,
        },
    )
    patch_path = paths.output_root / CONFIG_PATCH_FILENAME
    patch_path.write_text(patch, encoding="utf-8")
    if getattr(args, "print_config_patch", False):
        print(patch, end="")
    logger.debug(
        "Wrote config patch files preview=%s patch=%s",
        paths.output_root / CONFIG_PREVIEW_FILENAME,
        patch_path,
    )
    return patch


def _apply_config(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    updates = _config_updates(args, paths)
    _validate_config_artifacts(paths)
    _write_config_patch_files(args, paths, updates)
    if not args.write_config:
        return ["wrote config update preview only"]
    _promotion_check(paths)
    config_path = Path(args.config_path)
    _require(config_path, "config file for --write-config")
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(config_path, encoding="utf-8")
    if parser.has_section(args.config_section):
        parser.remove_section(args.config_section)
    parser.add_section(args.config_section)
    for key in sorted(updates):
        parser.set(args.config_section, key, updates[key])
    with config_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return [f"updated {config_path} [{args.config_section}] with finalized ensemble settings"]


def _execute_stage(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> StageResult:
    contract = _contract_for(stage, args, paths)
    outputs = contract.outputs
    if args.resume and _stage_complete(stage, paths):
        validated = _validate_stage_outputs(stage, paths)
        return StageResult(
            stage,
            "skipped",
            outputs=outputs,
            validated_outputs=validated,
            contract=contract.to_json(),
            notes=["resume: outputs already exist"],
        )
    command: list[str] = []
    started = time.time()
    try:
        logger.debug(
            "Starting stage=%s contract=%s", stage, json.dumps(contract.to_json(), sort_keys=True)
        )
        if logger.isEnabledFor(5):
            logger.log(5, "stage_contract=%s", json.dumps(contract.to_json(), sort_keys=True))
        if not args.dry_run:
            _validate_required_files(stage, args, paths)
        if stage == "static_pipeline":
            if args.dry_run:
                status = "planned" if not paths.run_summary.exists() else "ok"
                return StageResult(
                    stage,
                    status,
                    command=[],
                    outputs=outputs,
                    contract=contract.to_json(),
                )
        elif stage == "candidate_search":
            command = _command_candidate_search(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
        elif stage == "hard_alignment_validation":
            command = _command_hard_validation(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
        elif stage == "freeze_resolver_metadata":
            if args.dry_run:
                return StageResult(
                    stage,
                    "planned",
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=["would copy or reuse frozen GT-hard resolver metadata"],
                )
            notes = _freeze_metadata(args, paths)
            return StageResult(
                stage,
                "ok",
                outputs=outputs,
                validated_outputs=_validate_stage_outputs(stage, paths),
                contract=contract.to_json(),
                notes=notes,
            )
        elif stage == "scorer_training":
            command = _command_scorer_training(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
        elif stage == "scorer_evaluation":
            command = _command_scorer_eval(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
        elif stage == "production_promotion_check":
            if args.dry_run:
                return StageResult(
                    stage,
                    "planned",
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=["would require promotion_status=pass"],
                )
            notes = _promotion_check(paths)
            return StageResult(
                stage,
                "ok",
                outputs=outputs,
                validated_outputs=_validate_stage_outputs(stage, paths),
                contract=contract.to_json(),
                notes=notes,
            )
        elif stage == "artifact_export":
            if args.dry_run:
                return StageResult(
                    stage,
                    "planned",
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=["would copy promoted artifacts"],
                )
            exported = _export_artifacts(paths)
            return StageResult(
                stage,
                "ok",
                outputs=outputs,
                validated_outputs=_validate_stage_outputs(stage, paths),
                contract=contract.to_json(),
                notes=[f"exported {len(exported)} artifact(s)"],
            )
        elif stage == "config_update":
            if args.dry_run:
                return StageResult(
                    stage,
                    "planned",
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=["would write config preview/update"],
                )
            notes = _apply_config(args, paths)
            return StageResult(
                stage,
                "ok",
                outputs=outputs,
                validated_outputs=_validate_stage_outputs(stage, paths),
                contract=contract.to_json(),
                notes=notes,
            )
        else:
            raise AssertionError(stage)
        validated = _validate_stage_outputs(stage, paths)
    except Exception as err:
        logger.exception("Stage failed: %s", stage)
        return StageResult(
            stage,
            "failed",
            command=command,
            outputs=outputs,
            contract=contract.to_json(),
            error=f"{type(err).__name__}: {err}",
            duration_seconds=round(time.time() - started, 3),
        )
    result = StageResult(
        stage,
        "ok",
        command=command,
        outputs=outputs,
        validated_outputs=validated,
        contract=contract.to_json(),
        duration_seconds=round(time.time() - started, 3),
    )
    logger.debug(
        "Finished stage=%s result=%s",
        stage,
        json.dumps(_jsonable(result.__dict__), sort_keys=True),
    )
    return result


def _fallback_counts(report: T.Mapping[str, T.Any]) -> dict[str, int]:
    return {
        "fallback_count": int(report.get("fallback_count", 0) or 0),
        "safe_fallback_count": int(report.get("safe_fallback_count", 0) or 0),
        "hard_slice_fallback_count": int(report.get("hard_slice_fallback_count", 0) or 0),
        "consensus_collapse_fallback_count": int(
            report.get("consensus_collapse_fallback_count", 0) or 0
        ),
    }


def _summary_payload(
    args: argparse.Namespace, paths: PipelinePaths, results: T.Sequence[StageResult]
) -> dict[str, T.Any]:
    report = _read_json(paths.scorer_report)
    updates = _config_updates(args, paths)
    promotion_status = str(report.get("promotion_status") or report.get("status") or "")
    return {
        "status": "fail"
        if any(row.status == "failed" for row in results)
        else ("planned" if args.dry_run else "pass"),
        "promotion_scope": args.promotion_scope,
        "promotion_status": promotion_status,
        "failed_gates": report.get("failed_gates", []),
        "selected_production_policy": updates["resolver_policy"],
        "fallback_counts": _fallback_counts(report),
        "best_weights_path": str(paths.best_weights),
        "scorer_path": str(paths.scorer_artifact),
        "eval_report_path": str(paths.scorer_report),
        "config_fields_changed": sorted(updates),
        "config_patch_path": str(paths.output_root / CONFIG_PATCH_FILENAME),
        "progress_log_path": str(paths.progress_log),
        "run_root": str(paths.run_root),
        "production_root": str(paths.production_root),
        "output_root": str(paths.output_root),
        "sources": {
            SOURCE_GT_HARD: {
                "manifest": str(paths.hard_manifest),
                "cache": str(paths.run_cache),
                "resolver_metadata": str(paths.frozen_gt_metadata),
            },
            SOURCE_PRODUCTION_VALIDATED: {
                "manifest": str(paths.production_manifest),
                "cache": str(paths.production_cache),
            },
        },
        "artifacts": {
            "best_setup": str(paths.best_setup),
            "best_weights": str(paths.best_weights),
            "runtime_resolver_scorer": str(paths.scorer_artifact),
            "scorer_policy_report": str(paths.scorer_report),
        },
        "scorer_report_status": promotion_status,
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "stages": [_jsonable(row.__dict__) for row in results],
        "config_update": {
            "config_path": str(args.config_path),
            "section": args.config_section,
            "write_config": bool(args.write_config),
            "updates": updates,
        },
    }


def _write_summary_md(path: Path, summary: dict[str, T.Any]) -> None:
    lines = [
        "# Landmark resolver pipeline summary",
        "",
        f"Status: **{summary['status']}**",
        f"Promotion status: `{summary.get('promotion_status', '')}`",
        f"Promotion scope: `{summary['promotion_scope']}`",
        f"Selected production policy: `{summary['selected_production_policy']}`",
        f"Best weights: `{summary['best_weights_path']}`",
        f"Scorer: `{summary['scorer_path']}`",
        f"Eval report: `{summary['eval_report_path']}`",
        f"Failed gates: `{summary['failed_gates']}`",
        f"Fallback counts: `{summary['fallback_counts']}`",
        f"Config patch: `{summary['config_patch_path']}`",
        f"Progress log: `{summary['progress_log_path']}`",
        "",
        "## Stages",
        "",
        "| Stage | Status | Duration | Outputs | Notes |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for stage in summary["stages"]:
        notes = "; ".join(stage.get("notes") or [])
        if stage.get("error"):
            notes = stage["error"]
        outputs = "<br>".join(stage.get("validated_outputs") or stage.get("outputs") or [])
        lines.append(
            f"| `{stage['name']}` | {stage['status']} | {stage.get('duration_seconds', 0)}s | {outputs} | {notes} |"
        )
    lines.extend(["", "## Config fields changed", ""])
    lines.extend(f"- `{field_name}`" for field_name in summary["config_fields_changed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _progress_enabled(args: argparse.Namespace) -> bool | None:
    if getattr(args, "no_progress", False):
        return False
    if getattr(args, "progress", False):
        return True
    return None


def run_pipeline(args: argparse.Namespace) -> dict[str, T.Any]:
    paths = PipelinePaths(args.run_root, args.production_root, args.output_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    selected = _stage_slice(args.start_at, args.stop_after)
    progress = PipelineProgress(
        total=len(selected),
        output_path=paths.progress_log,
        enabled=_progress_enabled(args),
        logger=logger,
    )
    results: list[StageResult] = []
    for index, stage in enumerate(selected, start=1):
        progress.start(stage, index=index, message="stage starting")
        logger.info("Starting pipeline stage %s/%s: %s", index, len(selected), stage)
        result = _execute_stage(stage, args, paths)
        results.append(result)
        progress.finish(
            stage,
            index=index,
            status=result.status,
            message=result.error or "; ".join(result.notes),
        )
        if result.status == "failed":
            break
    summary = _summary_payload(args, paths, results)
    write_json(paths.summary_json, summary)
    _write_summary_md(paths.summary_md, summary)
    failed = next((row for row in results if row.status == "failed"), None)
    if failed is not None:
        raise RuntimeError(f"pipeline failed at {failed.name}: {failed.error}")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--promotion-scope", choices=("universal", "production"), default="production"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=STAGES)
    parser.add_argument("--start-at", choices=STAGES)
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--print-config-patch", action="store_true")
    parser.add_argument(
        "--progress", action="store_true", help="Force terminal progress bar output."
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bar output while keeping JSONL progress logs.",
    )
    parser.add_argument("--config-path", type=Path, default=Path("config/extract.ini"))
    parser.add_argument("--config-section", default=DEFAULT_CONFIG_SECTION)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--gt-hard-resolver-metadata", type=Path)
    parser.add_argument("--overwrite-frozen-metadata", action="store_true")
    parser.add_argument("--hard-source-manifest", type=Path)
    parser.add_argument("--candidate-search-arg", action="append", default=[])
    parser.add_argument("--hard-validation-arg", action="append", default=[])
    parser.add_argument("--scorer-train-arg", action="append", default=[])
    parser.add_argument("--scorer-eval-arg", action="append", default=[])
    parser.add_argument(
        "--log-level", default="INFO", help="TRACE, DEBUG, INFO, WARNING, or ERROR"
    )
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    summary = run_pipeline(args)
    logger.info("Pipeline summary: %s", summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
