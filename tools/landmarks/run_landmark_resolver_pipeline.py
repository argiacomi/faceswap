#!/usr/bin/env python3
"""Run the landmark resolver promotion pipeline end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
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

from lib.landmarks.datasets.manifest_io import filter_canonical_68_samples, load_manifest
from lib.landmarks.ensemble.production_artifacts import (
    LEARNED_POLICIES,
    install_production_bundle,
)
from lib.landmarks.ensemble.scorer_dataset import (
    SCORER_DATASET_DIR,
    SCORER_DATASET_MANIFEST_JSON,
    SCORER_ROWS_CSV,
)
from lib.landmarks.ensemble.scorer_target_config import (
    SCORER_TARGETS,
    TARGET_SELECTION_COST,
)
from lib.landmarks.ensemble.scorer_training import SCORER_SUITE_METRICS_JSON
from lib.landmarks.ensemble.static_weight_fit import compute_static_weights
from lib.landmarks.ensemble.weights import save_weights
from lib.landmarks.evaluation.splits import (
    RANDOM,
    SCENARIO_STRATIFIED,
    SplitRatios,
    save_split_file,
    split_manifest_samples,
    split_summary_counts,
    write_split_manifest,
)
from lib.landmarks.pipeline_conventions import (
    REPORT_MANIFEST_FILENAME,
    RESOLVER_METADATA_JSONL,
    SCORER_POLICY_REPORT_JSON,
    SOURCE_GT_HARD,
    SOURCE_PRODUCTION_VALIDATED,
    load_resolver_metadata_sidecar,
    validate_resolver_metadata_for_manifest,
    write_json,
)
from lib.landmarks.search.gt_runtime_bucket_metrics import (
    aggregate_runtime_bucket_metrics,
    load_candidate_table_csv,
    write_runtime_bucket_csv,
    write_runtime_bucket_json,
)
from tools.landmarks.pipeline_progress import PipelineProgress, configure_logging

logger = logging.getLogger("run_landmark_resolver_pipeline")

STAGES: tuple[str, ...] = (
    "build_dataset_manifest",
    "build_hard_source_manifest",
    "build_prediction_cache",
    "build_hard_source_prediction_cache",
    "build_splits",
    "fit_static_weights",
    "build_production_manifest",
    "build_production_prediction_cache",
    "build_production_resolver_metadata",
    "candidate_search",
    "hard_alignment_validation",
    "build_gt_hard_resolver_metadata",
    "freeze_resolver_metadata",
    "scorer_training",
    "scorer_evaluation",
    "production_promotion_check",
    "artifact_export",
    "config_update",
)
DEFAULT_MODELS = "fan,hrnet,spiga,orformer"
DEFAULT_CANDIDATES = "fan,hrnet,spiga,orformer,plain_average,static_weighted,static_weighted_downweight,static_weighted_hard_drop,weighted_median"
DEFAULT_HARD_SOURCE_DATASET = "aflw2000-3d"
DEFAULT_PREDICTION_CACHE_DEVICE = "gpu"
DEFAULT_PREDICTION_CACHE_COMPILE = "default"
BASE_DATASET_MANIFEST_SENTINEL_FILENAME = ".base_dataset_manifest_complete.json"
BASE_CACHE_SENTINEL_FILENAME = ".base_prediction_cache_complete.json"
HARD_SOURCE_CACHE_SENTINEL_FILENAME = ".hard_source_prediction_cache_complete.json"
PRODUCTION_CACHE_SENTINEL_FILENAME = ".production_prediction_cache_complete.json"
PRODUCTION_RESOLVER_METADATA_SENTINEL_FILENAME = ".production_resolver_metadata_complete.json"
V2_SCORER_TRAINING_SENTINEL_FILENAME = ".v2_scorer_training_complete.json"
SCORER_TRAINING_SENTINEL_FILENAME = ".scorer_training_complete.json"
DEFAULT_CONFIG_SECTION = "align.ensemble"
CONFIG_PREVIEW_FILENAME = "config_update_preview.json"
CONFIG_PATCH_FILENAME = "config_update_patch.ini"
SCORER_VERSION_CONTINUOUS_REGRET = "continuous_regret_v1_1"
SCORER_VERSION_BINARY = "learned_quality_v1"
SCORER_POLICY_CONTINUOUS_REGRET = "learned_quality_v1_1"
SCORER_VERSION_LEARNED_QUALITY_V2 = "learned_quality_v2"
PROMOTED_POLICIES = (
    SCORER_VERSION_BINARY,
    SCORER_POLICY_CONTINUOUS_REGRET,
    SCORER_VERSION_LEARNED_QUALITY_V2,
)
STAGE_ALIASES = {
    "binary_scorer_training": "scorer_training",
    "continuous_scorer_training": "scorer_training",
    "v2_scorer_training": "scorer_training",
}

PROGRESS_LOG_FILENAME = "pipeline_progress.jsonl"
VALID_ALIGN_ENSEMBLE_CONFIG_KEYS = frozenset(
    {
        "batch_size",
        "models",
        "crop_scale",
        "strategy",
        "reject_outliers",
        "outlier_threshold",
        "min_models",
        "resolver_policy",
        "use_alignment_resolver",
        "hard_case_strategy",
        "secondary_hard_case_strategy",
        "hard_disagreement_px",
        "fallback_model",
        "fallback_strategy",
        "strict",
        "roll_veto_degrees",
        "hard_roll_degrees",
    }
)


class PipelineContractError(RuntimeError):
    """A pipeline stage did not satisfy its declared contract."""


@dataclass(frozen=True)
class PipelinePaths:
    run_root: Path
    production_root: Path
    output_root: Path
    stable_root: Path = field(init=False)
    run_dataset_dir: Path = field(init=False)
    run_manifest: Path = field(init=False)
    run_manifest_sentinel: Path = field(init=False)
    run_fit_manifest: Path = field(init=False)
    run_select_manifest: Path = field(init=False)
    run_report_manifest: Path = field(init=False)
    run_cache: Path = field(init=False)
    run_cache_sentinel: Path = field(init=False)
    hard_source_dataset_dir: Path = field(init=False)
    hard_source_manifest: Path = field(init=False)
    hard_source_cache_sentinel: Path = field(init=False)
    run_splits: Path = field(init=False)
    run_static_weights: Path = field(init=False)
    run_summary: Path = field(init=False)
    production_manifest: Path = field(init=False)
    production_cache: Path = field(init=False)
    production_cache_sentinel: Path = field(init=False)
    production_resolver_metadata: Path = field(init=False)
    production_resolver_metadata_sentinel: Path = field(init=False)
    candidate_dir: Path = field(init=False)
    best_setup: Path = field(init=False)
    best_weights: Path = field(init=False)
    gt_runtime_bucket_candidate_table: Path = field(init=False)
    gt_runtime_bucket_metrics_json: Path = field(init=False)
    gt_runtime_bucket_metrics_csv: Path = field(init=False)
    hard_dir: Path = field(init=False)
    hard_manifest: Path = field(init=False)
    frozen_gt_metadata: Path = field(init=False)
    scorer_train_dir: Path = field(init=False)
    binary_scorer_train_dir: Path = field(init=False)
    binary_scorer_artifact: Path = field(init=False)
    continuous_scorer_train_dir: Path = field(init=False)
    continuous_scorer_eval_rows: Path = field(init=False)
    scorer_artifact: Path = field(init=False)
    v2_scorer_train_dir: Path = field(init=False)
    v2_scorer_artifact: Path = field(init=False)
    v2_scorer_training_sentinel: Path = field(init=False)
    scorer_dataset_dir: Path = field(init=False)
    scorer_rows_csv: Path = field(init=False)
    scorer_dataset_manifest: Path = field(init=False)
    canonical_scorers_dir: Path = field(init=False)
    canonical_binary_scorer_artifact: Path = field(init=False)
    canonical_continuous_scorer_artifact: Path = field(init=False)
    canonical_v2_scorer_artifact: Path = field(init=False)
    scorer_suite_metrics: Path = field(init=False)
    scorer_training_sentinel: Path = field(init=False)
    scorer_eval_dir: Path = field(init=False)
    scorer_report: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    promotion_manifest: Path = field(init=False)
    exported_candidate_dir: Path = field(init=False)
    exported_scorer_dir: Path = field(init=False)
    exported_best_setup: Path = field(init=False)
    exported_best_weights: Path = field(init=False)
    exported_scorer_artifact: Path = field(init=False)
    exported_scorer_dataset_dir: Path = field(init=False)
    exported_scorer_rows: Path = field(init=False)
    exported_scorer_dataset_manifest: Path = field(init=False)
    progress_log: Path = field(init=False)
    summary_json: Path = field(init=False)
    summary_md: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stable_root", self.output_root / "current")
        object.__setattr__(self, "run_dataset_dir", self.run_root / "dataset")
        object.__setattr__(self, "run_manifest", self.run_dataset_dir / "manifest.json")
        object.__setattr__(
            self,
            "run_manifest_sentinel",
            self.run_dataset_dir / BASE_DATASET_MANIFEST_SENTINEL_FILENAME,
        )
        object.__setattr__(self, "run_fit_manifest", self.run_dataset_dir / "fit_manifest.json")
        object.__setattr__(
            self, "run_select_manifest", self.run_dataset_dir / "select_manifest.json"
        )
        object.__setattr__(
            self, "run_report_manifest", self.run_dataset_dir / REPORT_MANIFEST_FILENAME
        )
        object.__setattr__(self, "run_cache", self.run_root / "cache")
        object.__setattr__(
            self, "run_cache_sentinel", self.run_cache / BASE_CACHE_SENTINEL_FILENAME
        )
        object.__setattr__(self, "hard_source_dataset_dir", self.run_root / "hard_source")
        object.__setattr__(
            self, "hard_source_manifest", self.hard_source_dataset_dir / "manifest.json"
        )
        object.__setattr__(
            self,
            "hard_source_cache_sentinel",
            self.run_cache / HARD_SOURCE_CACHE_SENTINEL_FILENAME,
        )
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
        object.__setattr__(
            self,
            "production_cache_sentinel",
            self.production_cache / PRODUCTION_CACHE_SENTINEL_FILENAME,
        )
        object.__setattr__(
            self,
            "production_resolver_metadata",
            self.production_root / RESOLVER_METADATA_JSONL,
        )
        object.__setattr__(
            self,
            "production_resolver_metadata_sentinel",
            self.production_root / PRODUCTION_RESOLVER_METADATA_SENTINEL_FILENAME,
        )
        object.__setattr__(self, "candidate_dir", self.output_root / "candidate_search")
        object.__setattr__(self, "best_setup", self.candidate_dir / "best_setup.json")
        object.__setattr__(self, "best_weights", self.candidate_dir / "best_weights.json")
        object.__setattr__(
            self,
            "gt_runtime_bucket_candidate_table",
            self.candidate_dir / "gt_runtime_bucket_candidate_table.csv",
        )
        object.__setattr__(
            self,
            "gt_runtime_bucket_metrics_json",
            self.candidate_dir / "gt_runtime_bucket_metrics.json",
        )
        object.__setattr__(
            self,
            "gt_runtime_bucket_metrics_csv",
            self.candidate_dir / "gt_runtime_bucket_metrics.csv",
        )
        object.__setattr__(self, "hard_dir", self.output_root / "gt_hard_validation")
        object.__setattr__(self, "hard_manifest", self.hard_dir / "manifest.json")
        object.__setattr__(self, "frozen_gt_metadata", self.output_root / RESOLVER_METADATA_JSONL)
        object.__setattr__(self, "scorer_train_dir", self.output_root / "scorer_training")
        object.__setattr__(self, "binary_scorer_train_dir", self.scorer_train_dir / "v1_binary")
        object.__setattr__(
            self,
            "binary_scorer_artifact",
            self.binary_scorer_train_dir / "runtime_resolver_scorer.json",
        )
        object.__setattr__(
            self,
            "continuous_scorer_train_dir",
            self.scorer_train_dir / "v1_1_selection_cost",
        )
        object.__setattr__(
            self,
            "continuous_scorer_eval_rows",
            self.continuous_scorer_train_dir / "runtime_resolver_scorer_eval_rows.csv",
        )
        object.__setattr__(
            self,
            "scorer_artifact",
            self.continuous_scorer_train_dir / "runtime_resolver_scorer.json",
        )
        object.__setattr__(
            self,
            "v2_scorer_train_dir",
            self.scorer_train_dir / "v2_lambdarank",
        )
        object.__setattr__(
            self,
            "v2_scorer_artifact",
            self.v2_scorer_train_dir / "runtime_resolver_scorer_v2.json",
        )
        object.__setattr__(
            self,
            "v2_scorer_training_sentinel",
            self.v2_scorer_train_dir / V2_SCORER_TRAINING_SENTINEL_FILENAME,
        )
        # Issue #206 canonical scorer-suite outputs. Keep the legacy path
        # attributes pointed at the suite subdirectories so existing evaluator,
        # promotion, and export code can continue to use them while the pipeline
        # runs only one scorer-training stage.
        object.__setattr__(self, "binary_scorer_train_dir", self.scorer_train_dir / "v1_binary")
        object.__setattr__(
            self,
            "binary_scorer_artifact",
            self.binary_scorer_train_dir / "runtime_resolver_scorer.json",
        )
        object.__setattr__(
            self,
            "continuous_scorer_train_dir",
            self.scorer_train_dir / "v1_1_selection_cost",
        )
        object.__setattr__(
            self,
            "continuous_scorer_eval_rows",
            self.continuous_scorer_train_dir / "runtime_resolver_scorer_eval_rows.csv",
        )
        object.__setattr__(
            self,
            "scorer_artifact",
            self.continuous_scorer_train_dir / "runtime_resolver_scorer.json",
        )
        object.__setattr__(self, "v2_scorer_train_dir", self.scorer_train_dir / "v2_lambdarank")
        object.__setattr__(
            self,
            "v2_scorer_artifact",
            self.v2_scorer_train_dir / "runtime_resolver_scorer_v2.json",
        )
        object.__setattr__(self, "scorer_dataset_dir", self.scorer_train_dir / SCORER_DATASET_DIR)
        object.__setattr__(self, "scorer_rows_csv", self.scorer_dataset_dir / SCORER_ROWS_CSV)
        object.__setattr__(
            self,
            "scorer_dataset_manifest",
            self.scorer_dataset_dir / SCORER_DATASET_MANIFEST_JSON,
        )
        object.__setattr__(self, "canonical_scorers_dir", self.scorer_train_dir / "scorers")
        object.__setattr__(
            self,
            "canonical_binary_scorer_artifact",
            self.canonical_scorers_dir / "learned_quality_v1.json",
        )
        object.__setattr__(
            self,
            "canonical_continuous_scorer_artifact",
            self.canonical_scorers_dir / "learned_quality_v1_1.json",
        )
        object.__setattr__(
            self,
            "canonical_v2_scorer_artifact",
            self.canonical_scorers_dir / "learned_quality_v2.json",
        )
        object.__setattr__(
            self,
            "scorer_suite_metrics",
            self.canonical_scorers_dir / SCORER_SUITE_METRICS_JSON,
        )
        object.__setattr__(
            self,
            "scorer_training_sentinel",
            self.scorer_train_dir / SCORER_TRAINING_SENTINEL_FILENAME,
        )
        object.__setattr__(self, "scorer_eval_dir", self.output_root / "scorer_evaluation")
        object.__setattr__(self, "scorer_report", self.scorer_eval_dir / SCORER_POLICY_REPORT_JSON)
        object.__setattr__(self, "artifacts_dir", self.output_root / "promotion")
        object.__setattr__(
            self, "promotion_manifest", self.artifacts_dir / "promotion_manifest.json"
        )
        object.__setattr__(self, "exported_candidate_dir", self.stable_root / "candidate_search")
        object.__setattr__(self, "exported_scorer_dir", self.stable_root / "scorer_training")
        object.__setattr__(
            self, "exported_best_setup", self.exported_candidate_dir / "best_setup.json"
        )
        object.__setattr__(
            self, "exported_best_weights", self.exported_candidate_dir / "best_weights.json"
        )
        object.__setattr__(
            self,
            "exported_scorer_artifact",
            self.exported_scorer_dir / "runtime_resolver_scorer.json",
        )
        object.__setattr__(
            self,
            "exported_scorer_dataset_dir",
            self.exported_scorer_dir / SCORER_DATASET_DIR,
        )
        object.__setattr__(
            self,
            "exported_scorer_rows",
            self.exported_scorer_dataset_dir / SCORER_ROWS_CSV,
        )
        object.__setattr__(
            self,
            "exported_scorer_dataset_manifest",
            self.exported_scorer_dataset_dir / SCORER_DATASET_MANIFEST_JSON,
        )
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


def _canonical_stage_name(stage: str | None) -> str | None:
    if stage is None:
        return None
    return STAGE_ALIASES.get(stage, stage)


def _stage_slice(start_at: str | None, stop_after: str | None) -> tuple[str, ...]:
    start_at = _canonical_stage_name(start_at)
    stop_after = _canonical_stage_name(stop_after)
    start = STAGES.index(start_at) if start_at else 0
    stop = STAGES.index(stop_after) if stop_after else len(STAGES) - 1
    if start > stop:
        raise ValueError(f"--start-at {start_at!r} occurs after --stop-after {stop_after!r}")
    return STAGES[start : stop + 1]


def _script(path: str) -> str:
    return str(ROOT / "tools" / "landmarks" / path)


def _split_csv_values(value: T.Any) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_items = value if isinstance(value, list | tuple) else [value]
    items: list[str] = []
    for raw in raw_items:
        items.extend(part.strip() for part in str(raw).split(",") if part.strip())
    return tuple(items)


def _base_datasets(args: argparse.Namespace) -> tuple[str, ...]:
    return _split_csv_values(getattr(args, "dataset", None)) or ("wflw",)


def _validate_dataset_source_args(args: argparse.Namespace) -> None:
    datasets = _base_datasets(args)
    if (
        getattr(args, "source_dir", None) is not None
        and getattr(args, "source_zip", None) is not None
    ):
        raise ValueError("Pass only one of --source-dir or --source-zip for a single base dataset")
    if getattr(args, "source_dir", None) is not None and len(datasets) != 1:
        raise ValueError("--source-dir is only valid when exactly one --dataset is selected")
    if getattr(args, "source_zip", None) is not None and len(datasets) != 1:
        raise ValueError("--source-zip is only valid when exactly one --dataset is selected")
    source_dirs = set(_dataset_source_map(args))
    source_zips = set(_dataset_source_zip_map(args))
    overlap = sorted(source_dirs & source_zips)
    if overlap:
        raise ValueError(
            "Datasets cannot have both --dataset-source and --dataset-source-zip: "
            + ", ".join(overlap)
        )
    if (
        getattr(args, "hard_source_dir", None) is not None
        and getattr(args, "hard_source_zip", None) is not None
    ):
        raise ValueError("Pass only one of --hard-source-dir or --hard-source-zip")


def _dataset_source_map(args: argparse.Namespace) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for spec in getattr(args, "dataset_source", []) or []:
        if "=" not in str(spec):
            raise ValueError(f"--dataset-source must use dataset=source_dir format, got {spec!r}")
        name, source = str(spec).split("=", 1)
        name = name.strip()
        source = source.strip()
        if not name or not source:
            raise ValueError(f"--dataset-source must use dataset=source_dir format, got {spec!r}")
        mapping[name] = Path(source)
    source_dir = getattr(args, "source_dir", None)
    datasets = _base_datasets(args)
    if source_dir is not None and len(datasets) == 1:
        mapping.setdefault(datasets[0], Path(source_dir))
    return mapping


def _dataset_source_zip_map(args: argparse.Namespace) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for spec in getattr(args, "dataset_source_zip", []) or []:
        if "=" not in str(spec):
            raise ValueError(
                f"--dataset-source-zip must use dataset=source_zip format, got {spec!r}"
            )
        name, source = str(spec).split("=", 1)
        name = name.strip()
        source = source.strip()
        if not name or not source:
            raise ValueError(
                f"--dataset-source-zip must use dataset=source_zip format, got {spec!r}"
            )
        mapping[name] = Path(source)
    source_zip = getattr(args, "source_zip", None)
    datasets = _base_datasets(args)
    if source_zip is not None and len(datasets) == 1:
        mapping.setdefault(datasets[0], Path(source_zip))
    return mapping


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_hard_source_manifest(args: argparse.Namespace, paths: PipelinePaths) -> Path:
    return (
        Path(args.hard_source_manifest)
        if args.hard_source_manifest
        else paths.hard_source_manifest
    )


def _prediction_cache_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "prediction_cache_mode", "run-models"))


def _prediction_cache_device(args: argparse.Namespace) -> str:
    return str(getattr(args, "prediction_cache_device", DEFAULT_PREDICTION_CACHE_DEVICE))


def _prediction_cache_compile(args: argparse.Namespace) -> str:
    return str(getattr(args, "prediction_cache_compile", DEFAULT_PREDICTION_CACHE_COMPILE))


def _append_prediction_cache_mode(
    args: argparse.Namespace, argv: list[str], extra: T.Sequence[str]
) -> list[str]:
    if _prediction_cache_mode(args) == "run-models":
        argv.append("--run-models")
        argv.extend(["--device", _prediction_cache_device(args)])
        argv.extend(["--compile", _prediction_cache_compile(args)])
    return _append_extra(argv, extra)


def _dataset_build_command(
    args: argparse.Namespace,
    paths: PipelinePaths,
    *,
    dataset: str,
    output_dir: Path,
    manifest_mode: str,
    source_dir: Path | None = None,
    source_zip: Path | None = None,
    extra: T.Sequence[str] = (),
) -> list[str]:
    argv = [
        args.python_executable,
        _script("build_quality_dataset.py"),
        "--dataset",
        dataset,
        "--output-dir",
        str(output_dir),
        "--manifest-mode",
        manifest_mode,
    ]
    if source_dir is not None and source_zip is not None:
        raise ValueError(f"dataset {dataset!r} received both source_dir and source_zip")
    if source_dir is not None:
        argv.extend(["--source-dir", str(source_dir)])
    if source_zip is not None:
        argv.extend(["--source-zip", str(source_zip)])
    return _append_extra(argv, extra)


def _commands_dataset_build(args: argparse.Namespace, paths: PipelinePaths) -> list[list[str]]:
    source_map = _dataset_source_map(args)
    source_zip_map = _dataset_source_zip_map(args)
    commands: list[list[str]] = []
    for index, dataset in enumerate(_base_datasets(args)):
        commands.append(
            _dataset_build_command(
                args,
                paths,
                dataset=dataset,
                output_dir=paths.run_dataset_dir,
                manifest_mode="replace" if index == 0 else "merge",
                source_dir=source_map.get(dataset),
                source_zip=source_zip_map.get(dataset),
                extra=getattr(args, "dataset_build_arg", []),
            )
        )
    return commands


def _command_dataset_build(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    return _commands_dataset_build(args, paths)[0]


def _command_hard_source_manifest(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    return _dataset_build_command(
        args,
        paths,
        dataset=getattr(args, "hard_source_dataset", DEFAULT_HARD_SOURCE_DATASET),
        output_dir=paths.hard_source_dataset_dir,
        manifest_mode="replace",
        source_dir=getattr(args, "hard_source_dir", None),
        source_zip=getattr(args, "hard_source_zip", None),
        extra=getattr(args, "hard_source_dataset_build_arg", []),
    )


def _command_prediction_cache(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    return _append_prediction_cache_mode(
        args,
        [
            args.python_executable,
            _script("cache_predictions.py"),
            "--manifest",
            str(paths.run_manifest),
            "--cache-dir",
            str(paths.run_cache),
            "--models",
            args.models,
        ],
        args.cache_prediction_arg,
    )


def _command_hard_source_prediction_cache(
    args: argparse.Namespace, paths: PipelinePaths
) -> list[str]:
    return _append_prediction_cache_mode(
        args,
        [
            args.python_executable,
            _script("cache_predictions.py"),
            "--manifest",
            str(_effective_hard_source_manifest(args, paths)),
            "--cache-dir",
            str(paths.run_cache),
            "--models",
            args.models,
        ],
        getattr(args, "hard_source_cache_prediction_arg", []),
    )


def _write_dataset_manifest_sentinel(args: argparse.Namespace, paths: PipelinePaths) -> None:
    source_map = _dataset_source_map(args)
    source_zip_map = _dataset_source_zip_map(args)
    write_json(
        paths.run_manifest_sentinel,
        {
            "manifest": str(paths.run_manifest),
            "manifest_sha256": _sha256_file(paths.run_manifest),
            "datasets": list(_base_datasets(args)),
            "dataset_sources": {
                dataset: str(source_map[dataset])
                for dataset in sorted(source_map)
                if dataset in _base_datasets(args)
            },
            "dataset_source_zips": {
                dataset: str(source_zip_map[dataset])
                for dataset in sorted(source_zip_map)
                if dataset in _base_datasets(args)
            },
            "dataset_build_args": list(getattr(args, "dataset_build_arg", [])),
        },
    )


def _write_cache_sentinel(
    path: Path,
    *,
    manifest: Path,
    models: str,
    mode: str,
    device: str,
    compile_mode: str,
    extra_args: T.Sequence[str],
) -> None:
    write_json(
        path,
        {
            "manifest": str(manifest),
            "manifest_sha256": _sha256_file(manifest),
            "models": models,
            "mode": mode,
            "device": device,
            "compile_mode": compile_mode,
            "extra_args": list(extra_args),
        },
    )


def _write_production_resolver_metadata_sentinel(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> None:
    write_json(
        paths.production_resolver_metadata_sentinel,
        {
            "manifest": str(paths.production_manifest),
            "manifest_sha256": _sha256_file(paths.production_manifest),
            "cache_sentinel": str(paths.production_cache_sentinel),
            "cache_sentinel_sha256": _sha256_file(paths.production_cache_sentinel),
            "weights": str(paths.run_static_weights),
            "weights_sha256": _sha256_file(paths.run_static_weights),
            "candidates": str(args.candidates),
            "allow_image_backfill": True,
            "source": SOURCE_PRODUCTION_VALIDATED,
            "extra_args": list(getattr(args, "production_resolver_metadata_arg", [])),
        },
    )


def _v2_scorer_training_sentinel_payload(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> dict[str, T.Any]:
    return {
        "hard_manifest": str(paths.hard_manifest),
        "hard_manifest_sha256": _sha256_file(paths.hard_manifest),
        "production_manifest": str(paths.production_manifest),
        "production_manifest_sha256": _sha256_file(paths.production_manifest),
        "hard_source_cache_sentinel": str(paths.hard_source_cache_sentinel),
        "hard_source_cache_sentinel_sha256": _sha256_file(paths.hard_source_cache_sentinel),
        "production_cache_sentinel": str(paths.production_cache_sentinel),
        "production_cache_sentinel_sha256": _sha256_file(paths.production_cache_sentinel),
        "best_weights": str(paths.best_weights),
        "best_weights_sha256": _sha256_file(paths.best_weights),
        "frozen_gt_metadata": str(paths.frozen_gt_metadata),
        "frozen_gt_metadata_sha256": _sha256_file(paths.frozen_gt_metadata),
        "candidates": str(args.candidates),
        "split_seed": int(args.split_seed),
        "eval_fraction": float(args.v2_eval_fraction),
        "learning_rate": float(args.v2_learning_rate),
        "iterations": int(args.v2_iterations),
        "num_leaves": int(args.v2_num_leaves),
        "scorer_train_arg": list(getattr(args, "scorer_train_arg", [])),
        "allow_image_backfill": bool(args.allow_image_backfill),
        "training_mode": SCORER_VERSION_LEARNED_QUALITY_V2,
    }


def _write_v2_scorer_training_sentinel(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> None:
    write_json(
        paths.v2_scorer_training_sentinel,
        _v2_scorer_training_sentinel_payload(args, paths),
    )


def _command_production_manifest(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    argv = [
        args.python_executable,
        _script("build_faceswap_validated_manifest.py"),
        "--images",
        str(args.production_images),
        "--alignments",
        str(args.production_alignments),
        "--output-dir",
        str(paths.production_root),
    ]
    return _append_extra(argv, args.production_manifest_arg)


def _command_production_prediction_cache(
    args: argparse.Namespace, paths: PipelinePaths
) -> list[str]:
    return _append_prediction_cache_mode(
        args,
        [
            args.python_executable,
            _script("cache_predictions.py"),
            "--manifest",
            str(paths.production_manifest),
            "--cache-dir",
            str(paths.production_cache),
            "--models",
            args.models,
        ],
        args.production_cache_prediction_arg,
    )


def _command_production_resolver_metadata(
    args: argparse.Namespace, paths: PipelinePaths
) -> list[str]:
    return _append_extra(
        [
            args.python_executable,
            _script("build_production_resolver_metadata.py"),
            "--manifest",
            str(paths.production_manifest),
            "--cache-dir",
            str(paths.production_cache),
            "--weights",
            str(paths.run_static_weights),
            "--candidates",
            args.candidates,
            "--output",
            str(paths.production_resolver_metadata),
        ],
        getattr(args, "production_resolver_metadata_arg", []),
    )


def _command_candidate_search(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    argv = [
        args.python_executable,
        _script("search_ensemble_setup.py"),
        "--manifest",
        str(paths.run_manifest),
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
        "--production-resolver-metadata",
        str(paths.production_resolver_metadata),
        "--production-gate-output",
        str(paths.candidate_dir / "production_gate"),
    ]
    # Every promoted scorer version this pipeline supports
    # (``continuous_regret_v1_1`` → runtime policy ``learned_quality_v1``,
    # ``learned_quality_v2`` → ``learned_quality_v2``) installs a learned
    # quality runtime policy whose ranker has to choose between *multiple*
    # fusion candidates that ``runtime_resolver.build_candidates`` derives
    # from the promoted setup. A single-model or collapsed setup
    # degenerates to one usable candidate and makes the trained ranker
    # meaningless, so the candidate search must promote a real ensemble.
    # ``--require-effective-ensemble`` activates the effective-ensemble
    # gate (which also blocks dominant-model collapse) without enabling
    # ``--allow-single-model-promotion``.
    argv.append("--require-effective-ensemble")
    return _append_extra(argv, args.candidate_search_arg)


def _command_gt_runtime_bucket_candidate_table(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> list[str]:
    """Build full-GT candidate diagnostics for production/runtime bucket reports.

    This intentionally uses the full candidate-search GT manifest/cache, not
    the GT-hard scorer manifest. The emitted table is then aggregated into
    gt_runtime_bucket_metrics.json/.csv.
    """
    argv = [
        args.python_executable,
        _script("export_resolver_candidate_table.py"),
        "--manifest",
        str(paths.run_manifest),
        "--cache-dir",
        str(paths.run_cache),
        "--weights",
        str(paths.best_weights),
        "--candidates",
        args.candidates,
        "--output-csv",
        str(paths.gt_runtime_bucket_candidate_table),
    ]
    if args.allow_image_backfill:
        argv.append("--allow-image-backfill")
    return argv


def _command_hard_validation(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    source_manifest = _effective_hard_source_manifest(args, paths)
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


def _command_gt_hard_resolver_metadata(
    args: argparse.Namespace, paths: PipelinePaths
) -> list[str]:
    argv = [
        args.python_executable,
        _script("build_gt_hard_resolver_metadata.py"),
        "--manifest",
        str(paths.hard_manifest),
        "--cache-dir",
        str(paths.run_cache),
        "--weights",
        str(paths.best_weights),
        "--candidates",
        args.candidates,
        "--output",
        str(paths.frozen_gt_metadata),
    ]
    if args.allow_image_backfill:
        argv.append("--allow-image-backfill")
    return _append_extra(argv, args.gt_hard_resolver_metadata_arg)


def _scorer_training_sentinel_payload(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> dict[str, T.Any]:
    payload = dict(_v2_scorer_training_sentinel_payload(args, paths))
    payload.update(
        {
            "training_mode": "scorer_suite",
            "failure_threshold": float(getattr(args, "failure_threshold", 0.08)),
            "high_gap_threshold": float(getattr(args, "high_gap_threshold", 0.01)),
        }
    )
    return payload


def _sentinel_matches(path: Path, expected: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected  # type: ignore[no-any-return]


def _scorer_training_sentinel_matches(args: argparse.Namespace, paths: PipelinePaths) -> bool:
    return _sentinel_matches(
        paths.scorer_training_sentinel,
        _scorer_training_sentinel_payload(args, paths),
    )


def _write_scorer_training_sentinel(args: argparse.Namespace, paths: PipelinePaths) -> None:
    write_json(paths.scorer_training_sentinel, _scorer_training_sentinel_payload(args, paths))


def _command_scorer_training(
    args: argparse.Namespace,
    paths: PipelinePaths,
    *,
    output_dir: Path,
    target: str,
) -> list[str]:
    argv = [
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
        str(output_dir),
        "--target",
        target,
    ]
    if args.allow_image_backfill:
        argv.append("--allow-image-backfill")
    argv.extend(["--gt-hard-resolver-metadata", str(paths.frozen_gt_metadata)])
    return _append_extra(argv, args.scorer_train_arg)


def _command_v2_scorer_training(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    argv = [
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
        str(paths.v2_scorer_train_dir),
        "--training-mode",
        "learned_quality_v2",
        "--target",
        TARGET_SELECTION_COST,
        "--split-seed",
        str(args.split_seed),
        "--eval-fraction",
        str(args.v2_eval_fraction),
        "--learning-rate",
        str(args.v2_learning_rate),
        "--iterations",
        str(args.v2_iterations),
        "--num-leaves",
        str(args.v2_num_leaves),
    ]
    if args.allow_image_backfill:
        argv.append("--allow-image-backfill")
    argv.extend(["--gt-hard-resolver-metadata", str(paths.frozen_gt_metadata)])
    return _append_extra(argv, args.scorer_train_arg)


def _command_scorer_suite_training(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    argv = [
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
        "--training-mode",
        "scorer_suite",
        "--target",
        TARGET_SELECTION_COST,
        "--split-seed",
        str(args.split_seed),
        "--eval-fraction",
        str(args.v2_eval_fraction),
        "--learning-rate",
        str(args.v2_learning_rate),
        "--iterations",
        str(args.v2_iterations),
        "--num-leaves",
        str(args.v2_num_leaves),
    ]
    if args.allow_image_backfill:
        argv.append("--allow-image-backfill")
    argv.extend(["--gt-hard-resolver-metadata", str(paths.frozen_gt_metadata)])
    return _append_extra(argv, args.scorer_train_arg)


def _command_scorer_eval(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    argv = [
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
        str(paths.canonical_continuous_scorer_artifact),
        "--binary-scorer",
        str(paths.canonical_binary_scorer_artifact),
        "--v2-scorer",
        str(paths.canonical_v2_scorer_artifact),
        "--scorer-rows",
        str(paths.scorer_rows_csv),
        "--candidates",
        args.candidates,
        "--output-dir",
        str(paths.scorer_eval_dir),
        "--promotion-scope",
        args.promotion_scope,
        "--promotion-policy",
        _promoted_runtime_policy(args),
        "--installed-scorer-dir",
        str(
            getattr(
                args, "installed_scorer_dir", Path(".fs_cache/landmark_ensemble/current/scorers")
            )
        ),
    ]
    if args.allow_image_backfill:
        argv.append("--allow-image-backfill")
    argv.extend(["--gt-hard-resolver-metadata", str(paths.frozen_gt_metadata)])
    return _append_extra(argv, args.scorer_eval_arg)


def _outputs_for(stage: str, paths: PipelinePaths) -> list[Path]:
    stage = _canonical_stage_name(stage) or stage
    return {
        "build_dataset_manifest": [paths.run_manifest, paths.run_manifest_sentinel],
        "build_hard_source_manifest": [paths.hard_source_manifest],
        "build_prediction_cache": [paths.run_cache, paths.run_cache_sentinel],
        "build_hard_source_prediction_cache": [paths.run_cache, paths.hard_source_cache_sentinel],
        "build_splits": [
            paths.run_splits,
            paths.run_fit_manifest,
            paths.run_select_manifest,
            paths.run_report_manifest,
            paths.run_summary,
        ],
        "fit_static_weights": [paths.run_static_weights],
        "build_production_manifest": [
            paths.production_manifest,
            paths.production_root / RESOLVER_METADATA_JSONL,
            paths.production_root / "audit.json",
        ],
        "build_production_prediction_cache": [
            paths.production_cache,
            paths.production_cache_sentinel,
        ],
        "build_production_resolver_metadata": [
            paths.production_resolver_metadata,
            paths.production_resolver_metadata_sentinel,
        ],
        "candidate_search": [
            paths.best_setup,
            paths.best_weights,
            paths.gt_runtime_bucket_candidate_table,
            paths.gt_runtime_bucket_metrics_json,
            paths.gt_runtime_bucket_metrics_csv,
        ],
        "hard_alignment_validation": [paths.hard_manifest],
        "build_gt_hard_resolver_metadata": [paths.frozen_gt_metadata],
        "freeze_resolver_metadata": [paths.frozen_gt_metadata],
        "scorer_training": [
            paths.canonical_binary_scorer_artifact,
            paths.canonical_continuous_scorer_artifact,
            paths.canonical_v2_scorer_artifact,
            paths.scorer_rows_csv,
            paths.scorer_dataset_manifest,
            paths.scorer_suite_metrics,
            paths.scorer_training_sentinel,
        ],
        "scorer_evaluation": [paths.scorer_report],
        "production_promotion_check": [paths.scorer_report],
        "artifact_export": [paths.promotion_manifest],
        "config_update": [
            paths.output_root / CONFIG_PREVIEW_FILENAME,
            paths.output_root / CONFIG_PATCH_FILENAME,
        ],
    }[stage]


def _required_inputs_for(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> list[Path]:
    stage = _canonical_stage_name(stage) or stage
    return {
        "build_dataset_manifest": [],
        "build_hard_source_manifest": [],
        "build_prediction_cache": [paths.run_manifest],
        "build_hard_source_prediction_cache": [_effective_hard_source_manifest(args, paths)],
        "build_splits": [paths.run_manifest],
        "fit_static_weights": [
            paths.run_fit_manifest if paths.run_fit_manifest.exists() else paths.run_manifest,
            paths.run_cache,
            paths.run_cache_sentinel,
        ],
        "build_production_manifest": []
        if paths.production_manifest.exists()
        else [
            args.production_images,
            args.production_alignments,
        ],
        "build_production_prediction_cache": [paths.production_manifest],
        "build_production_resolver_metadata": [
            paths.production_manifest,
            paths.production_cache,
            paths.production_cache_sentinel,
            paths.run_static_weights,
        ],
        "candidate_search": [
            paths.run_manifest,
            paths.run_cache,
            paths.run_cache_sentinel,
            paths.run_splits,
            paths.production_manifest,
            paths.production_cache,
            paths.production_cache_sentinel,
            paths.production_resolver_metadata,
            paths.production_resolver_metadata_sentinel,
        ],
        "hard_alignment_validation": [
            _effective_hard_source_manifest(args, paths),
            paths.run_cache,
            paths.hard_source_cache_sentinel,
            paths.best_weights if paths.best_weights.exists() else paths.run_static_weights,
        ],
        "build_gt_hard_resolver_metadata": []
        if paths.frozen_gt_metadata.exists()
        else (
            [args.gt_hard_resolver_metadata]
            if args.gt_hard_resolver_metadata
            else (
                [
                    paths.hard_manifest,
                    paths.run_cache,
                    paths.hard_source_cache_sentinel,
                    paths.best_weights,
                ]
                if args.generate_gt_hard_resolver_metadata
                else []
            )
        ),
        "freeze_resolver_metadata": []
        if paths.frozen_gt_metadata.exists()
        else (
            [args.gt_hard_resolver_metadata]
            if args.gt_hard_resolver_metadata
            else [paths.frozen_gt_metadata]
        ),
        "scorer_training": [
            paths.hard_manifest,
            paths.run_cache,
            paths.hard_source_cache_sentinel,
            paths.production_manifest,
            paths.production_cache,
            paths.production_cache_sentinel,
            paths.best_weights,
            paths.frozen_gt_metadata,
        ],
        "scorer_evaluation": [
            paths.canonical_continuous_scorer_artifact,
            paths.canonical_binary_scorer_artifact,
            paths.canonical_v2_scorer_artifact,
            paths.scorer_rows_csv,
            paths.scorer_dataset_manifest,
        ],
        "production_promotion_check": [paths.scorer_report],
        "artifact_export": [
            paths.best_setup,
            paths.best_weights,
            _promoted_scorer_source(args, paths),
            paths.scorer_report,
            paths.frozen_gt_metadata,
        ],
        "config_update": [Path(args.config_path)] if args.write_config else [],
    }[stage]


def _contract_for(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> StageContract:
    stage = _canonical_stage_name(stage) or stage
    cache_behavior = {
        "build_dataset_manifest": "writes merged run-root base GT manifest; first dataset uses --manifest-mode replace and later datasets use --manifest-mode merge; completion sentinel is written only after all dataset builds succeed",
        "build_hard_source_manifest": "writes separate AFLW2000-3D hard-source manifest under run-root; --resume skips when manifest exists",
        "build_prediction_cache": "writes base GT predictions into the shared run-root prediction cache and records a base-cache sentinel",
        "build_hard_source_prediction_cache": "writes hard-source predictions into the same run-root prediction cache and records a hard-source-cache sentinel",
        "build_splits": "writes fit/select/report splits and split manifests; --resume skips when all split artifacts exist",
        "fit_static_weights": "fits initial static weights from the fit split; --resume skips when weights exist",
        "build_production_manifest": "writes production_validated manifest, resolver metadata, and audit from Faceswap alignments; --resume skips when outputs exist",
        "build_production_prediction_cache": "writes production_validated prediction cache from the production manifest and records a production-cache sentinel",
        "build_production_resolver_metadata": "runs the runtime resolver over production_validated with image access and writes image-aware resolver metadata plus a completion sentinel",
        "candidate_search": "writes candidate_search artifacts; --resume skips when best_setup.json and best_weights.json exist",
        "hard_alignment_validation": "writes gt_hard_validation artifacts; --resume skips when manifest.json exists",
        "build_gt_hard_resolver_metadata": "runs the GT-hard resolver metadata CLI on the GT-hard manifest and writes the frozen resolver metadata sidecar; explicit sidecars are copied into the same frozen path",
        "freeze_resolver_metadata": "validates the caller-supplied or freshly generated frozen GT-hard resolver metadata sidecar; never derives runtime metadata from a plain manifest",
        "scorer_training": "loads scorer contexts once, writes canonical scorer rows, trains v1/v1.1/v2 scorer artifacts, and records an input/hyperparameter sentinel; --resume skips only when all canonical artifacts and the sentinel match",
        "scorer_evaluation": "writes scorer_evaluation reports; --resume skips when scorer_policy_report.json exists",
        "production_promotion_check": "reads scorer report only; no recompute; requires promotion_status/status pass",
        "artifact_export": "writes promotion_manifest.json with immutable runtime artifact provenance; --resume skips when the manifest matches the promoted scorer source",
        "config_update": "writes deterministic config_update_preview.json and config_update_patch.ini; writes only promoted align.ensemble keys with --write-config after promotion passes",
    }[stage]
    success = {
        "scorer_training": [
            "v1 binary runtime_resolver_scorer.json exists",
            "v1.1 runtime_resolver_scorer.json and held-out eval rows exist",
            "learned_quality_v2 runtime_resolver_scorer_v2.json exists",
            "canonical scorer rows and manifest exist",
            "scorer training sentinel matches requested inputs and hyperparameters",
        ],
        "build_dataset_manifest": ["merged base GT run manifest and completion sentinel exist"],
        "build_hard_source_manifest": ["hard-source manifest exists"],
        "build_prediction_cache": ["base GT prediction cache sentinel exists"],
        "build_hard_source_prediction_cache": ["hard-source prediction cache sentinel exists"],
        "build_splits": ["splits.json and fit/select/report manifests exist"],
        "fit_static_weights": ["static landmark weights exist"],
        "build_production_manifest": [
            "production manifest, resolver_metadata.jsonl, and audit.json exist"
        ],
        "build_production_prediction_cache": ["production prediction cache sentinel exists"],
        "build_production_resolver_metadata": [
            "image-aware production resolver_metadata.jsonl and completion sentinel exist"
        ],
        "candidate_search": ["best_setup.json and best_weights.json exist"],
        "hard_alignment_validation": ["GT-hard manifest exists"],
        "build_gt_hard_resolver_metadata": [
            "frozen resolver_metadata.jsonl exists and validates against the hard manifest"
        ],
        "freeze_resolver_metadata": [
            "frozen resolver_metadata.jsonl exists and validates against the hard manifest"
        ],
        "binary_scorer_training": ["v1 binary runtime_resolver_scorer.json exists"],
        "continuous_scorer_training": [
            "v1.1 runtime_resolver_scorer.json and held-out eval rows exist"
        ],
        "v2_scorer_training": [
            "learned_quality_v2 runtime_resolver_scorer_v2.json exists",
            "v2 scorer training sentinel matches requested inputs and hyperparameters",
        ],
        "scorer_evaluation": ["scorer_policy_report.json exists"],
        "production_promotion_check": ["scorer report promotion_status/status is pass"],
        "artifact_export": ["promotion_manifest.json exists"],
        "config_update": [
            "config_update_preview.json exists",
            "config_update_patch.ini exists",
            "referenced artifacts exist",
            "when --write-config is set, target config file is updated without deleting unmanaged sections/options",
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


def _cache_sentinel_matches(
    path: Path,
    *,
    manifest: Path,
    models: str,
    mode: str,
    device: str,
    compile_mode: str,
    extra_args: T.Sequence[str],
) -> bool:
    payload = _read_json(path)
    if not payload:
        return False
    expected_hash = _sha256_file(manifest) if manifest.is_file() else ""
    return (
        payload.get("manifest") == str(manifest)
        and payload.get("manifest_sha256") == expected_hash
        and payload.get("models") == models
        and payload.get("mode") == mode
        and payload.get("device") == device
        and payload.get("compile_mode") == compile_mode
        and payload.get("extra_args") == list(extra_args)
    )


def _production_resolver_metadata_sentinel_matches(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> bool:
    payload = _read_json(paths.production_resolver_metadata_sentinel)
    if (
        not payload
        or not paths.production_manifest.is_file()
        or not paths.production_cache_sentinel.is_file()
        or not paths.run_static_weights.is_file()
    ):
        return False
    return (
        payload.get("manifest") == str(paths.production_manifest)
        and payload.get("manifest_sha256") == _sha256_file(paths.production_manifest)
        and payload.get("cache_sentinel") == str(paths.production_cache_sentinel)
        and payload.get("cache_sentinel_sha256") == _sha256_file(paths.production_cache_sentinel)
        and payload.get("weights") == str(paths.run_static_weights)
        and payload.get("weights_sha256") == _sha256_file(paths.run_static_weights)
        and payload.get("candidates") == str(args.candidates)
        and payload.get("allow_image_backfill") is True
        and payload.get("source") == SOURCE_PRODUCTION_VALIDATED
        and payload.get("extra_args")
        == list(getattr(args, "production_resolver_metadata_arg", []))
    )


def _dataset_manifest_sentinel_matches(args: argparse.Namespace, paths: PipelinePaths) -> bool:
    payload = _read_json(paths.run_manifest_sentinel)
    if not payload or not paths.run_manifest.is_file():
        return False
    source_map = _dataset_source_map(args)
    expected_sources = {
        dataset: str(source_map[dataset])
        for dataset in sorted(source_map)
        if dataset in _base_datasets(args)
    }
    return (
        payload.get("manifest") == str(paths.run_manifest)
        and payload.get("manifest_sha256") == _sha256_file(paths.run_manifest)
        and payload.get("datasets") == list(_base_datasets(args))
        and payload.get("dataset_sources") == expected_sources
        and payload.get("dataset_build_args") == list(getattr(args, "dataset_build_arg", []))
    )


def _v2_scorer_training_sentinel_matches(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> bool:
    if (
        not paths.v2_scorer_training_sentinel.is_file()
        or not paths.hard_manifest.is_file()
        or not paths.production_manifest.is_file()
        or not paths.hard_source_cache_sentinel.is_file()
        or not paths.production_cache_sentinel.is_file()
        or not paths.best_weights.is_file()
        or not paths.frozen_gt_metadata.is_file()
    ):
        return False
    return _read_json(paths.v2_scorer_training_sentinel) == _v2_scorer_training_sentinel_payload(
        args, paths
    )


def _artifact_export_matches(args: argparse.Namespace, paths: PipelinePaths) -> bool:
    manifest = _read_json(paths.promotion_manifest)
    scorer_source = _promoted_scorer_source(args, paths)
    scorer_payload = _read_json(scorer_source)
    expected_version = str(
        scorer_payload.get("scorer_version")
        or scorer_payload.get("version")
        or args.promoted_scorer_version
    )
    return (
        manifest.get("active_policy") == _promoted_runtime_policy(args)
        and manifest.get("runtime_resolver_scorer_source") == str(scorer_source)
        and manifest.get("runtime_resolver_scorer_source_sha256") == _sha256_file(scorer_source)
        and str(manifest.get("runtime_resolver_scorer_version") or "") == expected_version
    )


def _selected_candidate_from_best_setup_payload(
    payload: T.Mapping[str, T.Any],
    candidate_names: T.AbstractSet[str],
) -> str | None:
    """Extract the selected candidate name from ``best_setup.json``.

    The canonical writer stores setup fields at top level (for example
    ``strategy`` and ``weight_generator.name``), while older tests and
    ad-hoc artifacts used a nested ``candidate`` object. Prefer values
    that exactly match the candidate table so the report's
    ``selected_candidate_*`` columns are populated only with a real row.
    """

    ordered: list[str] = []

    def _add(value: T.Any) -> None:
        if isinstance(value, str) and value and value not in ordered:
            ordered.append(value)

    candidate_payload = payload.get("candidate")
    if isinstance(candidate_payload, str):
        _add(candidate_payload)
    elif isinstance(candidate_payload, T.Mapping):
        _add(candidate_payload.get("name"))
        _add(candidate_payload.get("strategy"))
        nested_weight_generator = candidate_payload.get("weight_generator")
        if isinstance(nested_weight_generator, T.Mapping):
            _add(nested_weight_generator.get("name"))
        else:
            _add(nested_weight_generator)

    _add(payload.get("selected_candidate"))
    _add(payload.get("candidate_name"))
    _add(payload.get("strategy"))
    weight_generator = payload.get("weight_generator")
    if isinstance(weight_generator, T.Mapping):
        _add(weight_generator.get("name"))
    else:
        _add(weight_generator)

    for candidate in ordered:
        if candidate in candidate_names:
            return candidate
    return ordered[0] if ordered else None


def _emit_gt_runtime_bucket_artifacts(
    args_or_paths: argparse.Namespace | PipelinePaths,
    paths: PipelinePaths | None = None,
) -> None:
    """Aggregate full-GT candidate diagnostics into runtime-bucket metrics.

    Issue #205/#206: this report must be based on the full GT candidate-search
    manifest/cache, not the GT-hard scorer rows. The candidate table is produced
    by ``export_resolver_candidate_table.py`` during ``candidate_search`` and is
    then aggregated here into the canonical candidate-search diagnostics:

    * ``gt_runtime_bucket_candidate_table.csv``
    * ``gt_runtime_bucket_metrics.json``
    * ``gt_runtime_bucket_metrics.csv``
    """
    if paths is None:
        args = argparse.Namespace(models=DEFAULT_MODELS)
        paths = T.cast(PipelinePaths, args_or_paths)
    else:
        args = T.cast(argparse.Namespace, args_or_paths)

    candidate_table = Path(
        getattr(
            paths,
            "gt_runtime_bucket_candidate_table",
            Path(paths.candidate_dir) / "gt_runtime_bucket_candidate_table.csv",
        )
    )

    # Backward-compatible fallback for older tests/callers that construct a
    # minimal paths object with only binary_scorer_train_dir. The production
    # pipeline writes the full-GT table above during candidate_search.
    if not candidate_table.is_file() and hasattr(paths, "binary_scorer_train_dir"):
        legacy_candidate_table = Path(paths.binary_scorer_train_dir) / "candidate_table.csv"
        if legacy_candidate_table.is_file():
            candidate_table = legacy_candidate_table

    if not candidate_table.is_file():
        logger.info("gt_runtime_bucket aggregation skipped: %s not found", candidate_table)
        return

    best_setup_payload: dict[str, T.Any] = {}
    if paths.best_setup.is_file():
        try:
            payload = json.loads(paths.best_setup.read_text(encoding="utf-8"))
            best_setup_payload = payload if isinstance(payload, dict) else {}
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "gt_runtime_bucket aggregation could not read selected candidate from %s: %s",
                paths.best_setup,
                err,
            )

    try:
        rows = load_candidate_table_csv(candidate_table)
        if not rows:
            logger.info(
                "gt_runtime_bucket aggregation skipped: %s has no rows",
                candidate_table,
            )
            return
        candidate_names = {
            str(row.get("candidate")) for row in rows if row.get("candidate") is not None
        }
        selected_candidate = _selected_candidate_from_best_setup_payload(
            best_setup_payload,
            candidate_names,
        )
        metrics = aggregate_runtime_bucket_metrics(
            rows,
            selected_candidate=selected_candidate,
            single_model_candidates=_split_csv_values(args.models),
        )
        json_path = write_runtime_bucket_json(
            metrics,
            Path(
                getattr(
                    paths,
                    "gt_runtime_bucket_metrics_json",
                    Path(paths.candidate_dir) / "gt_runtime_bucket_metrics.json",
                )
            ),
        )
        csv_path = write_runtime_bucket_csv(
            metrics,
            Path(
                getattr(
                    paths,
                    "gt_runtime_bucket_metrics_csv",
                    Path(paths.candidate_dir) / "gt_runtime_bucket_metrics.csv",
                )
            ),
        )
        logger.info(
            "Wrote full-GT gt_runtime_bucket_metrics across %d bucket(s): %s, %s",
            len(metrics),
            json_path,
            csv_path,
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("gt_runtime_bucket aggregation skipped: %s", err)


def _stage_complete(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> bool:
    stage = _canonical_stage_name(stage) or stage
    if stage == "build_hard_source_manifest" and args.hard_source_manifest is not None:
        return True
    if not all(path.exists() for path in _outputs_for(stage, paths)):
        return False
    if stage == "build_dataset_manifest":
        return _dataset_manifest_sentinel_matches(args, paths)
    if stage == "build_prediction_cache":
        return _cache_sentinel_matches(
            paths.run_cache_sentinel,
            manifest=paths.run_manifest,
            models=args.models,
            mode=_prediction_cache_mode(args),
            device=_prediction_cache_device(args),
            compile_mode=_prediction_cache_compile(args),
            extra_args=getattr(args, "cache_prediction_arg", []),
        )
    if stage == "build_hard_source_prediction_cache":
        return _cache_sentinel_matches(
            paths.hard_source_cache_sentinel,
            manifest=_effective_hard_source_manifest(args, paths),
            models=args.models,
            mode=_prediction_cache_mode(args),
            device=_prediction_cache_device(args),
            compile_mode=_prediction_cache_compile(args),
            extra_args=getattr(args, "hard_source_cache_prediction_arg", []),
        )
    if stage == "build_production_prediction_cache":
        return _cache_sentinel_matches(
            paths.production_cache_sentinel,
            manifest=paths.production_manifest,
            models=args.models,
            mode=_prediction_cache_mode(args),
            device=_prediction_cache_device(args),
            compile_mode=_prediction_cache_compile(args),
            extra_args=getattr(args, "production_cache_prediction_arg", []),
        )
    if stage == "build_production_resolver_metadata":
        return _production_resolver_metadata_sentinel_matches(args, paths)
    if stage == "scorer_training":
        return _scorer_training_sentinel_matches(args, paths)
    if stage == "artifact_export":
        return _artifact_export_matches(args, paths)
    return True


def _should_skip_completed_stage(stage: str, args: argparse.Namespace) -> bool:
    if _stage_forced(stage, args):
        return False
    if stage == "config_update" and args.write_config:
        return False
    return not (
        stage in {"build_gt_hard_resolver_metadata", "freeze_resolver_metadata"}
        and (args.overwrite_frozen_metadata or args.gt_hard_resolver_metadata is not None)
    )


def _stage_forced(stage: str, args: argparse.Namespace) -> bool:
    overwrite_from = _canonical_stage_name(getattr(args, "overwrite_from", None))
    if overwrite_from is not None and STAGES.index(stage) >= STAGES.index(overwrite_from):
        return True
    force_downstream_of = _canonical_stage_name(getattr(args, "force_downstream_of", None))
    return force_downstream_of is not None and STAGES.index(stage) > STAGES.index(
        force_downstream_of
    )


def _run_command(command: list[str]) -> None:
    logger.info("Running command: %s", " ".join(command))
    logger.debug("Pipeline child command argv=%r", command)
    if logger.isEnabledFor(5):
        logger.log(5, "Pipeline child command cwd=%s argv=%s", ROOT, json.dumps(command))
    subprocess.run(command, check=True, cwd=str(ROOT))


def _load_manifest_payload(path: Path) -> dict[str, T.Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def _normalized_split_mode(value: str) -> str:
    return SCENARIO_STRATIFIED if value == "scenario_stratified" else value


def _build_splits(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    base_manifest = _load_manifest_payload(paths.run_manifest)
    samples = base_manifest.get("samples", base_manifest.get("scenarios", []))
    if not isinstance(samples, list):
        raise ValueError(f"manifest samples must be a list: {paths.run_manifest}")

    if all(
        isinstance(sample, dict) and (sample.get("landmarks") or sample.get("ground_truth"))
        for sample in samples
    ):
        canonical_ids = {
            sample.sample_id
            for sample in filter_canonical_68_samples(
                load_manifest(paths.run_manifest), context="pipeline split construction"
            )
        }
        if len(canonical_ids) != len(samples):
            samples = [
                sample
                for sample in samples
                if isinstance(sample, dict)
                and str(sample.get("sample_id") or sample.get("id") or sample.get("name"))
                in canonical_ids
            ]
            base_manifest = dict(base_manifest)
            base_manifest["samples"] = samples

    ratios = SplitRatios(args.split_fit, args.split_select, args.split_report)
    normalized_mode = _normalized_split_mode(args.split_mode)
    assignment, diagnostics = split_manifest_samples(
        [sample for sample in samples if isinstance(sample, dict)],
        mode=normalized_mode,
        ratios=ratios,
        seed=args.split_seed,
    )
    save_split_file(
        paths.run_splits,
        assignment,
        mode=normalized_mode,
        ratios=ratios,
        seed=args.split_seed,
        diagnostics=diagnostics,
    )
    write_split_manifest(paths.run_fit_manifest, base_manifest, assignment.fit)
    write_split_manifest(paths.run_select_manifest, base_manifest, assignment.select)
    write_split_manifest(paths.run_report_manifest, base_manifest, assignment.report)
    write_json(
        paths.run_summary,
        {
            "run_root": str(paths.run_root),
            "manifest": str(paths.run_manifest),
            "cache": str(paths.run_cache),
            "splits": str(paths.run_splits),
            "split_mode": normalized_mode,
            "split_seed": args.split_seed,
            "split_counts": split_summary_counts(assignment, samples),
        },
    )
    return [f"wrote split assignment with {assignment.sample_count} sample(s)"]


def _fit_static_weights(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    manifest = paths.run_fit_manifest if paths.run_fit_manifest.exists() else paths.run_manifest
    models = tuple(item.strip() for item in str(args.models).split(",") if item.strip())
    weights, mean_errors = compute_static_weights(manifest, paths.run_cache, models=models)
    save_weights(paths.run_static_weights, weights)
    write_json(
        paths.run_static_weights.with_name("static_landmark_weight_errors.json"),
        {"manifest": str(manifest), "models": list(models), "mean_errors": mean_errors},
    )
    return [f"fit static weights from {manifest}"]


def _validate_frozen_metadata(paths: PipelinePaths) -> None:
    metadata = load_resolver_metadata_sidecar(paths.frozen_gt_metadata)
    validate_resolver_metadata_for_manifest(
        paths.hard_manifest,
        metadata,
        source=SOURCE_GT_HARD,
        require_complete=True,
    )


def _copy_gt_hard_resolver_metadata(source: Path, paths: PipelinePaths) -> None:
    _require(source, "GT-hard resolver sidecar")
    paths.frozen_gt_metadata.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, paths.frozen_gt_metadata)
    _validate_frozen_metadata(paths)


def _freeze_metadata(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    if args.gt_hard_resolver_metadata is not None:
        source = Path(args.gt_hard_resolver_metadata)
        _copy_gt_hard_resolver_metadata(source, paths)
        return [f"copied and validated frozen metadata from {source}"]
    if paths.frozen_gt_metadata.exists() and args.resume:
        _validate_frozen_metadata(paths)
        return [f"reused existing frozen metadata: {paths.frozen_gt_metadata}"]
    if paths.frozen_gt_metadata.exists() and not args.overwrite_frozen_metadata:
        _validate_frozen_metadata(paths)
        return [
            f"frozen metadata already exists and was left unchanged: {paths.frozen_gt_metadata}"
        ]
    if paths.frozen_gt_metadata.exists() and args.gt_hard_resolver_metadata is None:
        _validate_frozen_metadata(paths)
        return [f"validated freshly generated frozen metadata: {paths.frozen_gt_metadata}"]
    raise FileNotFoundError(
        "missing generated GT-hard resolver metadata sidecar; run the "
        "build_gt_hard_resolver_metadata stage or pass --gt-hard-resolver-metadata. "
        "The pipeline will not derive runtime resolver metadata from a plain hard manifest."
    )


def _metric_value(metrics: T.Mapping[str, T.Any], key: str, *, policy_name: str) -> float:
    if key not in metrics:
        raise RuntimeError(f"{policy_name} promotion check failed: missing metric {key!r}")
    try:
        return float(metrics[key])
    except (TypeError, ValueError) as err:
        raise RuntimeError(
            f"{policy_name} promotion check failed: invalid metric {key!r}: {metrics[key]!r}"
        ) from err


def _selected_policy_metrics(report: T.Mapping[str, T.Any], policy_name: str) -> dict[str, T.Any]:
    metrics_by_policy = report.get("production_only_policy_metrics")
    if not isinstance(metrics_by_policy, dict):
        raise RuntimeError(
            f"{policy_name} promotion check failed: missing production_only_policy_metrics"
        )
    metrics = metrics_by_policy.get(policy_name)
    if not isinstance(metrics, dict):
        raise RuntimeError(
            f"{policy_name} promotion check failed: missing production metrics for {policy_name!r}"
        )
    return metrics


def _static_downweight_metrics(report: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    metrics_by_policy = report.get("production_only_policy_metrics")
    if isinstance(metrics_by_policy, dict):
        metrics = metrics_by_policy.get("static_weighted_downweight")
        if isinstance(metrics, dict):
            return metrics
    for key in ("static_downweight", "best_static_ensemble", "current_promoted_setup"):
        metrics = report.get(key)
        if isinstance(metrics, dict):
            return metrics
    raise RuntimeError(
        "learned_quality_v2 promotion check failed: missing static_weighted_downweight "
        "baseline metrics"
    )


def _validate_v2_promotion_gate(report: T.Mapping[str, T.Any]) -> list[str]:
    """Validate legacy learned_quality_v2 promotion fields when no installed baseline exists."""
    policy_name = SCORER_VERSION_LEARNED_QUALITY_V2
    status_keys = (
        "learned_quality_v2_gate_status",
        "learned_quality_v2_production_gate_status",
        "learned_quality_v2_promotion_status",
        "v2_gate_status",
        "v2_production_gate_status",
        "v2_promotion_status",
    )
    failed_keys = (
        "learned_quality_v2_failed_gates",
        "learned_quality_v2_production_failed_gates",
        "v2_failed_gates",
        "v2_production_failed_gates",
    )
    for key in status_keys:
        status = report.get(key)
        if status is not None and str(status) != "pass":
            raise RuntimeError(f"{policy_name} promotion check failed: {key}={status!r}")
    for key in failed_keys:
        failed = report.get(key) or []
        if failed:
            raise RuntimeError(f"{policy_name} promotion check failed: {key}={failed}")

    metrics = _selected_policy_metrics(report, policy_name)
    baseline = _static_downweight_metrics(report)
    gate_config = report.get("gate_config")
    if not isinstance(gate_config, dict):
        gate_config = {}

    failure_epsilon = float(gate_config.get("failure_rate_epsilon", 0.0) or 0.0)
    mean_epsilon = float(gate_config.get("mean_epsilon_nme", 0.001) or 0.001)
    p90_epsilon = float(gate_config.get("p90_epsilon_nme", 0.003) or 0.003)

    policy_failure = _metric_value(metrics, "failure_rate", policy_name=policy_name)
    baseline_failure = _metric_value(baseline, "failure_rate", policy_name=policy_name)
    if policy_failure > baseline_failure + failure_epsilon:
        raise RuntimeError(
            f"{policy_name} promotion check failed: failure_rate={policy_failure} "
            f"exceeds static_weighted_downweight failure_rate={baseline_failure} "
            f"+ epsilon={failure_epsilon}"
        )

    policy_mean = _metric_value(metrics, "mean_nme", policy_name=policy_name)
    baseline_mean = _metric_value(baseline, "mean_nme", policy_name=policy_name)
    if policy_mean > baseline_mean + mean_epsilon:
        raise RuntimeError(
            f"{policy_name} promotion check failed: mean_nme={policy_mean} "
            f"exceeds static_weighted_downweight mean_nme={baseline_mean} "
            f"+ epsilon={mean_epsilon}"
        )

    policy_p90 = _metric_value(metrics, "p90_nme", policy_name=policy_name)
    baseline_p90 = _metric_value(baseline, "p90_nme", policy_name=policy_name)
    if policy_p90 > baseline_p90 + p90_epsilon:
        raise RuntimeError(
            f"{policy_name} promotion check failed: p90_nme={policy_p90} "
            f"exceeds static_weighted_downweight p90_nme={baseline_p90} "
            f"+ epsilon={p90_epsilon}"
        )

    return [
        f"{policy_name}=pass",
        f"{policy_name}_failure_rate={policy_failure}",
        f"{policy_name}_mean_nme={policy_mean}",
        f"{policy_name}_p90_nme={policy_p90}",
    ]


def _installed_baseline_gate(
    report: T.Mapping[str, T.Any],
    *,
    policy_name: str,
    require_installed_baseline: bool,
) -> list[str]:
    if not require_installed_baseline:
        return [
            "installed_baseline_promotion=skipped_by_request",
            f"selected_policy={policy_name}",
        ]

    gates = report.get("installed_baseline_promotion")
    if not isinstance(gates, dict):
        if require_installed_baseline:
            raise RuntimeError("promotion check failed: missing installed_baseline_promotion")
        return ["installed_baseline_promotion=skipped_missing_report_payload"]

    gate = gates.get(policy_name)
    if not isinstance(gate, dict):
        if require_installed_baseline:
            raise RuntimeError(
                f"promotion check failed: missing installed baseline gate for {policy_name!r}"
            )
        return [f"installed_baseline_promotion=skipped_missing_policy:{policy_name}"]

    status = str(gate.get("status") or "")
    if status == "skipped_no_installed_baseline" and not require_installed_baseline:
        return [f"installed_baseline_promotion={status}", f"selected_policy={policy_name}"]
    if status != "pass":
        raise RuntimeError(
            "promotion check failed against installed current scorer: "
            f"selected_policy={policy_name!r}, status={status!r}, "
            f"failed_gates={gate.get('failed_gates', [])}, "
            f"selected_metrics={gate.get('selected_metrics', {})}, "
            f"installed_metrics={gate.get('installed_metrics', {})}"
        )
    return [
        "installed_baseline_promotion=pass",
        f"selected_policy={policy_name}",
        f"installed_policy={gate.get('installed_policy', '')}",
    ]


def _promotion_check(
    paths: PipelinePaths,
    *,
    promotion_scope: str = "production",
    args: argparse.Namespace | None = None,
) -> list[str]:
    """
    Validate promotion using installed-baseline reports, with legacy fallback.

    Reports containing installed_baseline_promotion use the installed-current
    scorer gate. Reports without that payload are legacy reports and must still
    honor promotion_status/status plus failed_gates.

    Important compatibility rule: args=None does not skip validation. Some unit
    tests and older callers invoke _promotion_check(paths) directly and still
    expect failed legacy reports to raise.
    """
    _require(paths.scorer_report, "scorer evaluation report")
    report = _read_json(paths.scorer_report)

    if args is not None:
        policy_name = _promoted_runtime_policy(args)
    else:
        policy_name = str(
            report.get("promotion_policy")
            or report.get("promoted_policy")
            or report.get("primary_scorer_policy")
            or report.get("promoted_scorer_label")
            or SCORER_POLICY_CONTINUOUS_REGRET
        )

    gates = report.get("installed_baseline_promotion")

    if isinstance(gates, dict):
        return _installed_baseline_gate(
            report,
            policy_name=policy_name,
            require_installed_baseline=bool(getattr(args, "require_installed_baseline", True)),
        )

    raw_status = report.get("promotion_status", report.get("status"))
    failed = report.get("failed_gates") or []
    status = str(raw_status) if raw_status is not None else ("fail" if failed else "pass")
    if status != "pass" or failed:
        raise RuntimeError(
            f"production promotion check failed: status={status!r}, failed_gates={failed}"
        )

    production_failed = report.get("production_failed_gates") or []
    raw_production_status = report.get("production_gate_status")
    production_status = (
        str(raw_production_status)
        if raw_production_status is not None
        else ("fail" if production_failed else "pass")
    )
    if production_status != "pass" or production_failed:
        raise RuntimeError(
            "production promotion check failed: "
            f"production_gate_status={production_status!r}, "
            f"production_failed_gates={production_failed}"
        )

    promoted_notes: list[str] = []
    if policy_name == SCORER_VERSION_LEARNED_QUALITY_V2:
        promoted_notes = _validate_v2_promotion_gate(report)

    if promotion_scope == "universal":
        gt_status = str(report.get("gt_hard_gate_status") or "")
        gt_failed = report.get("gt_hard_failed_gates") or []
        if gt_status not in {"pass", ""} or gt_failed:
            raise RuntimeError(
                "GT-hard diagnostic gate failed: "
                f"gt_hard_gate_status={gt_status!r}, gt_hard_failed_gates={gt_failed}"
            )

    return ["status=pass", "production_gate_status=pass", *promoted_notes]


def _validate_required_files(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> None:
    for required in _required_inputs_for(stage, args, paths):
        if required is not None:
            logger.debug("Validating required file for stage=%s path=%s", stage, required)
            _require(required, f"{stage} input")


def _validate_stage_outputs(
    stage: str, paths: PipelinePaths, args: argparse.Namespace | None = None
) -> list[str]:
    missing = [path for path in _outputs_for(stage, paths) if not path.exists()]
    if missing:
        details = ", ".join(str(path) for path in missing)
        raise PipelineContractError(
            f"stage {stage!r} did not produce required output(s): {details}"
        )
    if (
        args is not None
        and stage == "build_dataset_manifest"
        and not _dataset_manifest_sentinel_matches(args, paths)
    ):
        raise PipelineContractError(
            "stage 'build_dataset_manifest' completion sentinel does not match "
            "the requested dataset build inputs"
        )
    if (
        args is not None
        and stage == "build_prediction_cache"
        and not _cache_sentinel_matches(
            paths.run_cache_sentinel,
            manifest=paths.run_manifest,
            models=args.models,
            mode=_prediction_cache_mode(args),
            device=_prediction_cache_device(args),
            compile_mode=_prediction_cache_compile(args),
            extra_args=getattr(args, "cache_prediction_arg", []),
        )
    ):
        raise PipelineContractError(
            "stage 'build_prediction_cache' sentinel does not match the requested cache inputs"
        )
    if (
        args is not None
        and stage == "build_hard_source_prediction_cache"
        and not _cache_sentinel_matches(
            paths.hard_source_cache_sentinel,
            manifest=_effective_hard_source_manifest(args, paths),
            models=args.models,
            mode=_prediction_cache_mode(args),
            device=_prediction_cache_device(args),
            compile_mode=_prediction_cache_compile(args),
            extra_args=getattr(args, "hard_source_cache_prediction_arg", []),
        )
    ):
        raise PipelineContractError(
            "stage 'build_hard_source_prediction_cache' sentinel does not match "
            "the requested cache inputs"
        )
    if (
        args is not None
        and stage == "build_production_prediction_cache"
        and not _cache_sentinel_matches(
            paths.production_cache_sentinel,
            manifest=paths.production_manifest,
            models=args.models,
            mode=_prediction_cache_mode(args),
            device=_prediction_cache_device(args),
            compile_mode=_prediction_cache_compile(args),
            extra_args=getattr(args, "production_cache_prediction_arg", []),
        )
    ):
        raise PipelineContractError(
            "stage 'build_production_prediction_cache' sentinel does not match "
            "the requested cache inputs"
        )
    if (
        args is not None
        and stage == "build_production_resolver_metadata"
        and not _production_resolver_metadata_sentinel_matches(args, paths)
    ):
        raise PipelineContractError(
            "stage 'build_production_resolver_metadata' sentinel does not match "
            "the requested production metadata inputs"
        )
    if (
        args is not None
        and stage == "scorer_training"
        and not _scorer_training_sentinel_matches(args, paths)
    ):
        raise PipelineContractError(
            "stage 'scorer_training' sentinel does not match the requested "
            "training inputs or hyperparameters"
        )
    if args is not None and stage == "artifact_export":
        _promotion_check(paths, promotion_scope=args.promotion_scope, args=args)
        if not _artifact_export_matches(args, paths):
            raise PipelineContractError(
                "stage 'artifact_export' did not export the selected promoted scorer version"
            )
    if args is not None and stage == "config_update" and args.write_config:
        _promotion_check(paths, promotion_scope=args.promotion_scope, args=args)
    if stage == "build_production_resolver_metadata":
        metadata = load_resolver_metadata_sidecar(paths.production_resolver_metadata)
        validate_resolver_metadata_for_manifest(
            paths.production_manifest,
            metadata,
            source=SOURCE_PRODUCTION_VALIDATED,
            require_complete=True,
        )
    if stage in {"build_gt_hard_resolver_metadata", "freeze_resolver_metadata"}:
        _validate_frozen_metadata(paths)
    if stage == "production_promotion_check" and args is not None:
        _promotion_check(paths, promotion_scope=args.promotion_scope, args=args)
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


def _copy_scorer_with_promotion_metadata(source: Path, target: Path, *, promoted_from: str) -> str:
    """Copy the scorer artifact without mutating its JSON payload.

    Promotion provenance belongs in artifacts_manifest.json / the production
    bundle manifest. The trained scorer artifact should remain immutable so its
    hash and contents match the training output.
    """
    if not source.exists():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PipelineContractError(f"scorer artifact must be a JSON object: {source}")
    shutil.copyfile(source, target)
    logger.debug(
        "Exported immutable scorer artifact source=%s target=%s promoted_from=%s",
        source,
        target,
        promoted_from,
    )
    return str(target)


def _promoted_scorer_source(args: argparse.Namespace, paths: PipelinePaths) -> Path:
    policy = _promoted_runtime_policy(args)
    if policy == SCORER_VERSION_BINARY:
        return paths.canonical_binary_scorer_artifact
    if policy == SCORER_VERSION_LEARNED_QUALITY_V2:
        return paths.canonical_v2_scorer_artifact
    return paths.canonical_continuous_scorer_artifact


def _promoted_runtime_policy(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "promoted_policy", "") or "")
    if explicit:
        return explicit
    if args.promoted_scorer_version == SCORER_VERSION_LEARNED_QUALITY_V2:
        return SCORER_VERSION_LEARNED_QUALITY_V2
    return SCORER_POLICY_CONTINUOUS_REGRET


def _promoted_scorer_target(args: argparse.Namespace, paths: PipelinePaths) -> str:
    payload = _read_json(_promoted_scorer_source(args, paths))
    return str(payload.get("target") or TARGET_SELECTION_COST)


def _export_artifacts(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, T.Any]:
    """Write the promotion manifest without copying runtime artifacts.

    The production bundle is the runtime export. The run directory keeps
    immutable training/eval artifacts in place; this manifest records provenance
    and hashes for promotion/config stages without maintaining a redundant
    output_root/current tree.
    """
    _promotion_check(paths, promotion_scope=args.promotion_scope, args=args)
    scorer_source = _promoted_scorer_source(args, paths)
    scorer_payload = _read_json(scorer_source)
    active_policy = _promoted_runtime_policy(args)
    manifest: dict[str, T.Any] = {
        "active_policy": active_policy,
        "best_setup": str(paths.best_setup),
        "best_setup_sha256": _sha256_file(paths.best_setup),
        "best_weights": str(paths.best_weights),
        "best_weights_sha256": _sha256_file(paths.best_weights),
        "runtime_resolver_scorer_source": str(scorer_source),
        "runtime_resolver_scorer_source_sha256": _sha256_file(scorer_source),
        "runtime_resolver_scorer_version": str(
            scorer_payload.get("scorer_version")
            or scorer_payload.get("version")
            or args.promoted_scorer_version
        ),
        "promotion": {
            "active_policy": active_policy,
            "source": str(scorer_source),
            "source_sha256": _sha256_file(scorer_source),
            "target": str(scorer_payload.get("target") or ""),
            "model_type": str(scorer_payload.get("model_type") or ""),
            "immutable_scorer_artifact": True,
            "runtime_export": "production_bundle",
        },
        "diagnostics": {
            "scorer_policy_report": str(paths.scorer_report),
            "scorer_policy_report_sha256": _sha256_file(paths.scorer_report),
            "scorer_dataset_manifest": str(paths.scorer_dataset_manifest)
            if paths.scorer_dataset_manifest.exists()
            else "",
            "scorer_rows": str(paths.scorer_rows_csv) if paths.scorer_rows_csv.exists() else "",
            "gt_hard_resolver_metadata": str(paths.frozen_gt_metadata),
            "gt_hard_resolver_metadata_sha256": _sha256_file(paths.frozen_gt_metadata),
        },
    }
    paths.promotion_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(paths.promotion_manifest, manifest)

    # #206: artifact_export no longer maintains output_root/current copies.
    # Remove stale copies from older runs so the default contract is unambiguous.
    for stale in (
        paths.exported_scorer_artifact,
        paths.exported_scorer_rows,
        paths.exported_scorer_dataset_manifest,
        paths.exported_best_setup,
        paths.exported_best_weights,
        paths.stable_root / "artifacts" / "artifacts_manifest.json",
    ):
        stale.unlink(missing_ok=True)
    return manifest


def _config_updates(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, str]:
    models = ", ".join(item.strip() for item in str(args.models).split(",") if item.strip())
    updates = {
        "batch_size": "16",
        "models": models or DEFAULT_MODELS,
        "crop_scale": "1.6",
        "strategy": "static_weighted_downweight",
        "reject_outliers": "False",
        "outlier_threshold": "3.5",
        "min_models": "1",
        "resolver_policy": _promoted_runtime_policy(args),
        "use_alignment_resolver": "True",
        "hard_case_strategy": "static_weighted_downweight",
        "secondary_hard_case_strategy": "static_weighted_hard_drop",
        "hard_disagreement_px": "12.0",
        "fallback_model": "orformer",
        "fallback_strategy": "plain_average",
        "strict": "True",
        "roll_veto_degrees": "15.0",
        "hard_roll_degrees": "30.0",
    }
    unsupported = sorted(set(updates).difference(VALID_ALIGN_ENSEMBLE_CONFIG_KEYS))
    if unsupported:
        raise PipelineContractError(
            f"unsupported {DEFAULT_CONFIG_SECTION} config update key(s): {unsupported}"
        )
    return updates


def _config_patch_text(section: str, updates: T.Mapping[str, str]) -> str:
    lines = [f"[{section}]"]
    lines.extend(f"{key} = {updates[key]}" for key in sorted(updates))
    return "\n".join(lines) + "\n"


def _validate_config_artifacts(args: argparse.Namespace, paths: PipelinePaths) -> None:
    for path, label in (
        (paths.best_setup, "promoted setup artifact"),
        (paths.best_weights, "promoted weights artifact"),
        (_promoted_scorer_source(args, paths), "promoted runtime resolver scorer artifact"),
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


def _line_ending_for(text: str) -> str:
    return "\r\n" if text.count("\r\n") > text.count("\n") - text.count("\r\n") else "\n"


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def _section_name(line: str) -> str | None:
    body, _newline = _split_line_ending(line)
    match = re.match(r"^\s*\[([^\]]+)\]\s*(?:[;#].*)?$", body)
    return match.group(1).strip() if match else None


def _split_inline_comment(value: str) -> tuple[str, str]:
    for index, char in enumerate(value):
        if char in {"#", ";"} and (index == 0 or value[index - 1].isspace()):
            return value[:index], value[index:]
    return value, ""


def _patch_option_line(line: str, updates: T.Mapping[str, str]) -> tuple[str, str | None]:
    body, newline = _split_line_ending(line)
    stripped = body.lstrip()
    if not stripped or stripped[0] in {"#", ";", "["}:
        return line, None
    match = re.match(
        r"^(?P<indent>\s*)(?P<key>[^:=\s][^:=]*?)(?P<spacing>\s*(?:=|:)\s*)(?P<value>.*)$",
        body,
    )
    if not match:
        return line, None
    key = match.group("key").strip()
    if key not in updates:
        return line, None
    value_prefix, comment = _split_inline_comment(match.group("value"))
    trailing_space = re.search(r"\s*$", value_prefix)
    spacer = trailing_space.group(0) if trailing_space else ""
    patched = (
        f"{match.group('indent')}{match.group('key')}{match.group('spacing')}"
        f"{updates[key]}{spacer}{comment}{newline}"
    )
    return patched, key


def _patch_config_section_text(text: str, section: str, updates: T.Mapping[str, str]) -> str:
    newline = _line_ending_for(text)
    lines = text.splitlines(keepends=True)
    section_start: int | None = None
    for index, line in enumerate(lines):
        if _section_name(line) == section:
            section_start = index
            break

    new_update_lines = [f"{key} = {updates[key]}{newline}" for key in sorted(updates)]
    if section_start is None:
        prefix = text
        if prefix and not prefix.endswith(("\n", "\r")):
            prefix += newline
        if prefix and not prefix.endswith(f"{newline}{newline}"):
            prefix += newline
        return prefix + f"[{section}]{newline}" + "".join(new_update_lines)

    section_end = len(lines)
    for index in range(section_start + 1, len(lines)):
        if _section_name(lines[index]) is not None:
            section_end = index
            break

    seen: set[str] = set()
    for index in range(section_start + 1, section_end):
        patched, key = _patch_option_line(lines[index], updates)
        lines[index] = patched
        if key is not None:
            seen.add(key)

    missing = [key for key in sorted(updates) if key not in seen]
    if missing:
        insert_at = section_end
        while insert_at > section_start + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        if insert_at > 0 and not lines[insert_at - 1].endswith(("\n", "\r")):
            lines[insert_at - 1] += newline
        lines[insert_at:insert_at] = [f"{key} = {updates[key]}{newline}" for key in missing]

    return "".join(lines)


def _write_config_updates_in_place(
    config_path: Path, section: str, updates: T.Mapping[str, str]
) -> None:
    original = config_path.read_text(encoding="utf-8")
    patched = _patch_config_section_text(original, section, updates)
    config_path.write_text(patched, encoding="utf-8")


def _install_production_bundle_artifacts(args: argparse.Namespace, paths: PipelinePaths) -> Path:
    """Install the runtime artifact bundle that the extract plugin reads.

    Replaces the legacy contract where extract.ini carried setup_path /
    weights_path / resolver_scorer_path. The bundle puts those files in a
    known location (``.fs_cache/landmark_ensemble/current/`` by default,
    overridable via FACESWAP_LANDMARK_ENSEMBLE_ARTIFACTS) and a manifest
    maps resolver_policy → scorer file so changing the policy in config
    automatically selects the matching scorer.
    """
    scorer_sources = {
        "learned_quality_v1": paths.canonical_binary_scorer_artifact,
        "learned_quality_v1_1": paths.canonical_continuous_scorer_artifact,
        "learned_quality_v2": paths.canonical_v2_scorer_artifact,
    }
    return install_production_bundle(
        setup_src=paths.best_setup,
        weights_src=paths.best_weights,
        scorer_sources={
            policy: source
            for policy, source in scorer_sources.items()
            if policy in LEARNED_POLICIES and source.is_file()
        },
        active_policy=_promoted_runtime_policy(args),
        source_output_root=paths.output_root,
        created_by="run_landmark_resolver_pipeline.py",
    )


def _apply_config(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    updates = _config_updates(args, paths)
    _validate_config_artifacts(args, paths)
    _write_config_patch_files(args, paths, updates)
    if not args.write_config:
        return ["wrote config update preview only"]

    if not _artifact_export_matches(args, paths):
        _export_artifacts(args, paths)
    _promotion_check(paths, promotion_scope=args.promotion_scope, args=args)
    config_path = Path(args.config_path)
    _require(config_path, "config file for --write-config")
    _write_config_updates_in_place(config_path, args.config_section, updates)
    manifest_file = _install_production_bundle_artifacts(args, paths)

    return [
        f"updated {config_path} [{args.config_section}] with finalized ensemble settings "
        "and preserved unmanaged config entries, comments, and formatting",
        f"installed production landmark-ensemble bundle manifest at {manifest_file}",
    ]


def _execute_stage(stage: str, args: argparse.Namespace, paths: PipelinePaths) -> StageResult:
    contract = _contract_for(stage, args, paths)
    outputs = contract.outputs
    if (
        args.resume
        and _stage_complete(stage, args, paths)
        and _should_skip_completed_stage(stage, args)
    ):
        validated = _validate_stage_outputs(stage, paths, args)
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
        if stage == "build_dataset_manifest":
            commands = _commands_dataset_build(args, paths)
            command = commands[0]
            if args.dry_run:
                return StageResult(
                    stage,
                    "planned",
                    command=command,
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=[f"would run {len(commands)} dataset build command(s)"],
                )
            for command in commands:
                _run_command(command)
            _write_dataset_manifest_sentinel(args, paths)
        elif stage == "build_hard_source_manifest":
            if args.hard_source_manifest is not None:
                return StageResult(
                    stage,
                    "skipped",
                    outputs=outputs,
                    validated_outputs=[str(_effective_hard_source_manifest(args, paths))],
                    contract=contract.to_json(),
                    notes=["using caller-supplied hard-source manifest"],
                )
            command = _command_hard_source_manifest(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
        elif stage == "build_prediction_cache":
            command = _command_prediction_cache(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
            _write_cache_sentinel(
                paths.run_cache_sentinel,
                manifest=paths.run_manifest,
                models=args.models,
                mode=_prediction_cache_mode(args),
                device=_prediction_cache_device(args),
                compile_mode=_prediction_cache_compile(args),
                extra_args=getattr(args, "cache_prediction_arg", []),
            )
        elif stage == "build_hard_source_prediction_cache":
            command = _command_hard_source_prediction_cache(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
            _write_cache_sentinel(
                paths.hard_source_cache_sentinel,
                manifest=_effective_hard_source_manifest(args, paths),
                models=args.models,
                mode=_prediction_cache_mode(args),
                device=_prediction_cache_device(args),
                compile_mode=_prediction_cache_compile(args),
                extra_args=getattr(args, "hard_source_cache_prediction_arg", []),
            )
        elif stage == "build_splits":
            if args.dry_run:
                return StageResult(
                    stage,
                    "planned",
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=["would write split assignment and split manifests"],
                )
            notes = _build_splits(args, paths)
            return StageResult(
                stage,
                "ok",
                outputs=outputs,
                validated_outputs=_validate_stage_outputs(stage, paths, args),
                contract=contract.to_json(),
                notes=notes,
            )
        elif stage == "fit_static_weights":
            if args.dry_run:
                return StageResult(
                    stage,
                    "planned",
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=["would fit static landmark weights from fit split"],
                )
            notes = _fit_static_weights(args, paths)
            return StageResult(
                stage,
                "ok",
                outputs=outputs,
                validated_outputs=_validate_stage_outputs(stage, paths, args),
                contract=contract.to_json(),
                notes=notes,
            )
        elif stage == "build_production_manifest":
            if (
                paths.production_manifest.exists()
                and (paths.production_root / RESOLVER_METADATA_JSONL).exists()
                and (paths.production_root / "audit.json").exists()
                and not _stage_forced(stage, args)
            ):
                validated = _validate_stage_outputs(stage, paths, args)
                return StageResult(
                    stage,
                    "ok",
                    outputs=outputs,
                    validated_outputs=validated,
                    contract=contract.to_json(),
                    notes=["reused existing production manifest artifacts"],
                )
            command = _command_production_manifest(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            if args.production_images is None or args.production_alignments is None:
                raise FileNotFoundError(
                    "missing production manifest inputs; pass --production-images and "
                    "--production-alignments or provide existing production manifest artifacts"
                )
            _run_command(command)
        elif stage == "build_production_prediction_cache":
            command = _command_production_prediction_cache(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
            _write_cache_sentinel(
                paths.production_cache_sentinel,
                manifest=paths.production_manifest,
                models=args.models,
                mode=_prediction_cache_mode(args),
                device=_prediction_cache_device(args),
                compile_mode=_prediction_cache_compile(args),
                extra_args=getattr(args, "production_cache_prediction_arg", []),
            )
        elif stage == "build_production_resolver_metadata":
            command = _command_production_resolver_metadata(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
            _write_production_resolver_metadata_sentinel(args, paths)
        elif stage == "candidate_search":
            command = _command_candidate_search(args, paths)
            if args.dry_run:
                diagnostic_command = _command_gt_runtime_bucket_candidate_table(args, paths)
                return StageResult(
                    stage,
                    "planned",
                    command=command,
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=[
                        "would run candidate search",
                        "would export full-GT runtime-bucket candidate diagnostics",
                        "diagnostic command: " + " ".join(diagnostic_command),
                    ],
                )
            _run_command(command)
            diagnostic_command = _command_gt_runtime_bucket_candidate_table(args, paths)
            _run_command(diagnostic_command)
            _emit_gt_runtime_bucket_artifacts(args, paths)
        elif stage == "hard_alignment_validation":
            command = _command_hard_validation(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
        elif stage == "build_gt_hard_resolver_metadata":
            if args.gt_hard_resolver_metadata is not None:
                if args.dry_run:
                    return StageResult(
                        stage,
                        "planned",
                        outputs=outputs,
                        contract=contract.to_json(),
                        notes=["would copy caller-supplied GT-hard resolver metadata sidecar"],
                    )
                _copy_gt_hard_resolver_metadata(Path(args.gt_hard_resolver_metadata), paths)
                return StageResult(
                    stage,
                    "ok",
                    outputs=outputs,
                    validated_outputs=_validate_stage_outputs(stage, paths, args),
                    contract=contract.to_json(),
                    notes=[
                        f"copied GT-hard resolver metadata from {args.gt_hard_resolver_metadata}"
                    ],
                )
            if not args.generate_gt_hard_resolver_metadata:
                raise FileNotFoundError(
                    "missing required GT-hard resolver sidecar; pass --gt-hard-resolver-metadata "
                    "or enable --generate-gt-hard-resolver-metadata"
                )
            command = _command_gt_hard_resolver_metadata(args, paths)
            if args.dry_run:
                return StageResult(
                    stage,
                    "planned",
                    command=command,
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=["would run runtime resolver to write GT-hard metadata sidecar"],
                )
            _run_command(command)
        elif stage == "freeze_resolver_metadata":
            if args.dry_run:
                return StageResult(
                    stage,
                    "planned",
                    outputs=outputs,
                    contract=contract.to_json(),
                    notes=["would validate, copy, or reuse frozen GT-hard resolver metadata"],
                )
            notes = _freeze_metadata(args, paths)
            return StageResult(
                stage,
                "ok",
                outputs=outputs,
                validated_outputs=_validate_stage_outputs(stage, paths, args),
                contract=contract.to_json(),
                notes=notes,
            )
        elif stage == "scorer_training":
            command = _command_scorer_suite_training(args, paths)
            if args.dry_run:
                return StageResult(
                    stage, "planned", command=command, outputs=outputs, contract=contract.to_json()
                )
            _run_command(command)
            _write_scorer_training_sentinel(args, paths)
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
            notes = _promotion_check(paths, promotion_scope=args.promotion_scope, args=args)
            return StageResult(
                stage,
                "ok",
                outputs=outputs,
                validated_outputs=_validate_stage_outputs(stage, paths, args),
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
            exported = _export_artifacts(args, paths)
            return StageResult(
                stage,
                "ok",
                outputs=outputs,
                validated_outputs=_validate_stage_outputs(stage, paths, args),
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
                validated_outputs=_validate_stage_outputs(stage, paths, args),
                contract=contract.to_json(),
                notes=notes,
            )
        else:
            raise AssertionError(stage)
        validated = _validate_stage_outputs(stage, paths, args)
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
    updates = _config_updates(args, paths)
    report = _read_json(paths.scorer_report)
    promotion_status = str(report.get("promotion_status") or report.get("status") or "")
    promoted_scorer_version = str(
        _read_json(_promoted_scorer_source(args, paths)).get("scorer_version")
        or _read_json(_promoted_scorer_source(args, paths)).get("version")
        or args.promoted_scorer_version
    )
    promoted_scorer_target = _promoted_scorer_target(args, paths)
    return {
        "status": "fail"
        if any(row.status == "failed" for row in results)
        else ("planned" if args.dry_run else "pass"),
        "promotion_scope": args.promotion_scope,
        "promotion_status": promotion_status,
        "failed_gates": report.get("failed_gates", []),
        "selected_runtime_policy": updates["resolver_policy"],
        "selected_production_policy": updates["resolver_policy"],
        "promoted_scorer_version": promoted_scorer_version,
        "promoted_scorer_target": promoted_scorer_target,
        "fallback_counts": _fallback_counts(report),
        "best_weights_path": str(paths.exported_best_weights),
        "scorer_path": str(paths.exported_scorer_artifact),
        "binary_scorer_path": str(paths.binary_scorer_artifact),
        "eval_report_path": str(paths.scorer_report),
        "config_fields_changed": sorted(updates),
        "config_patch_path": str(paths.output_root / CONFIG_PATCH_FILENAME),
        "progress_log_path": str(paths.progress_log),
        "run_root": str(paths.run_root),
        "production_root": str(paths.production_root),
        "output_root": str(paths.output_root),
        "sources": {
            "base_gt": {
                "manifest": str(paths.run_manifest),
                "cache": str(paths.run_cache),
                "datasets": list(_base_datasets(args)),
            },
            "hard_source": {
                "manifest": str(_effective_hard_source_manifest(args, paths)),
                "cache": str(paths.run_cache),
                "dataset": getattr(args, "hard_source_dataset", DEFAULT_HARD_SOURCE_DATASET),
            },
            SOURCE_GT_HARD: {
                "manifest": str(paths.hard_manifest),
                "cache": str(paths.run_cache),
                "resolver_metadata": str(paths.frozen_gt_metadata),
            },
            SOURCE_PRODUCTION_VALIDATED: {
                "manifest": str(paths.production_manifest),
                "cache": str(paths.production_cache),
                "resolver_metadata": str(paths.production_resolver_metadata),
            },
        },
        "artifacts": {
            "best_setup": str(paths.exported_best_setup),
            "best_weights": str(paths.exported_best_weights),
            "runtime_resolver_scorer": str(paths.exported_scorer_artifact),
            "scorer_policy_report": str(paths.scorer_report),
        },
        "scorer_report_status": promotion_status,
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "overwrite_from": str(getattr(args, "overwrite_from", "") or ""),
        "force_downstream_of": str(getattr(args, "force_downstream_of", "") or ""),
        "prediction_cache_mode": _prediction_cache_mode(args),
        "prediction_cache_device": _prediction_cache_device(args),
        "prediction_cache_compile": _prediction_cache_compile(args),
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
        f"Selected runtime policy: `{summary['selected_runtime_policy']}`",
        f"Promoted scorer version: `{summary['promoted_scorer_version']}`",
        f"Promoted scorer target: `{summary['promoted_scorer_target']}`",
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
    if getattr(args, "write_summary_md", False):
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
    parser.add_argument(
        "--overwrite-from",
        choices=tuple(STAGES) + tuple(STAGE_ALIASES),
        help=(
            "With --resume, rerun this stage and every later selected stage even when "
            "their outputs already exist."
        ),
    )
    parser.add_argument(
        "--force-downstream-of",
        choices=tuple(STAGES) + tuple(STAGE_ALIASES),
        help=(
            "With --resume, rerun every selected stage after this stage even when "
            "their outputs already exist."
        ),
    )
    parser.add_argument("--stop-after", choices=STAGES)
    parser.add_argument("--start-at", choices=STAGES)
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--print-config-patch", action="store_true")
    parser.add_argument(
        "--write-summary-md",
        action="store_true",
        help="Write the optional human-readable pipeline_summary.md alongside pipeline_summary.json.",
    )
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
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Base GT dataset to include in the search pool. Repeat the flag or pass a "
            "comma-separated list. The first dataset is written with manifest-mode=replace; "
            "later datasets use manifest-mode=merge. Defaults to wflw."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Source directory for a single --dataset base GT build.",
    )
    parser.add_argument(
        "--dataset-source",
        action="append",
        default=[],
        help="Per-base-dataset source mapping in dataset=source_dir form.",
    )
    parser.add_argument(
        "--source-zip",
        type=Path,
        help="Source archive for a single --dataset base GT build.",
    )
    parser.add_argument(
        "--dataset-source-zip",
        action="append",
        default=[],
        help="Per-base-dataset archive mapping in dataset=source_zip form.",
    )
    parser.add_argument(
        "--hard-source-dataset",
        default=DEFAULT_HARD_SOURCE_DATASET,
        help="Dataset used to build the separate pose-driven GT-hard source manifest.",
    )
    parser.add_argument(
        "--hard-source-dir",
        type=Path,
        help="Source directory for --hard-source-dataset, normally AFLW2000-3D.",
    )
    parser.add_argument(
        "--hard-source-zip",
        type=Path,
        help="Source archive for --hard-source-dataset.",
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--production-images", type=Path)
    parser.add_argument("--production-alignments", type=Path)
    parser.add_argument("--gt-hard-resolver-metadata", type=Path)
    parser.add_argument("--overwrite-frozen-metadata", action="store_true")
    parser.add_argument(
        "--generate-gt-hard-resolver-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the runtime resolver on the GT-hard manifest to build the frozen "
            "resolver metadata sidecar when --gt-hard-resolver-metadata is not supplied. "
            "Enabled by default; use --no-generate-gt-hard-resolver-metadata to require "
            "an explicit sidecar."
        ),
    )
    parser.add_argument("--hard-source-manifest", type=Path)
    parser.add_argument(
        "--split-mode",
        choices=(SCENARIO_STRATIFIED, "scenario_stratified", RANDOM),
        default=SCENARIO_STRATIFIED,
    )
    parser.add_argument("--split-fit", type=float, default=0.60)
    parser.add_argument("--split-select", type=float, default=0.20)
    parser.add_argument("--split-report", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument(
        "--scorer-target",
        choices=SCORER_TARGETS,
        default=TARGET_SELECTION_COST,
    )
    parser.add_argument(
        "--promoted-scorer-version",
        choices=(SCORER_VERSION_CONTINUOUS_REGRET, SCORER_VERSION_LEARNED_QUALITY_V2),
        default=SCORER_VERSION_CONTINUOUS_REGRET,
        help=(
            "Scorer artifact to export into the stable runtime resolver scorer path. "
            "The config resolver_policy is learned_quality_v2 only when this is "
            "learned_quality_v2."
        ),
    )
    parser.add_argument(
        "--promoted-policy",
        choices=PROMOTED_POLICIES,
        default="",
        help=(
            "Runtime resolver policy to export/configure. Overrides "
            "--promoted-scorer-version. Allows explicitly choosing any trained "
            "learned policy artifact."
        ),
    )
    parser.add_argument(
        "--installed-scorer-dir",
        type=Path,
        default=Path(".fs_cache/landmark_ensemble/current/scorers"),
        help="Directory containing currently installed production scorer artifacts.",
    )
    parser.add_argument(
        "--require-installed-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the selected policy to beat the installed current scorer baseline. "
            "Use --no-require-installed-baseline only for first-time bootstrap runs."
        ),
    )
    parser.add_argument("--v2-eval-fraction", type=float, default=0.20)
    parser.add_argument("--v2-learning-rate", type=float, default=0.05)
    parser.add_argument("--v2-iterations", type=int, default=150)
    parser.add_argument("--v2-num-leaves", type=int, default=31)
    parser.add_argument(
        "--prediction-cache-mode",
        choices=("run-models", "fixtures"),
        default="run-models",
        help=(
            "How cache_predictions.py should populate caches. Defaults to run-models "
            "for end-to-end runs; fixtures preserves precomputed prediction import behavior."
        ),
    )
    parser.add_argument(
        "--prediction-cache-device",
        default=DEFAULT_PREDICTION_CACHE_DEVICE,
        help=(
            "Torch device passed to cache_predictions.py for run-model cache stages. "
            "The default 'gpu' requires CUDA/ROCm or MPS and will not silently use CPU. "
            "Use 'cpu' only for explicit CPU/debug runs."
        ),
    )
    parser.add_argument(
        "--prediction-cache-compile",
        default=DEFAULT_PREDICTION_CACHE_COMPILE,
        choices=(
            "off",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        help="Torch compile mode passed to cache_predictions.py for run-model cache stages.",
    )
    parser.add_argument(
        "--allow-image-backfill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow scorer train/eval CLIs to backfill image paths from manifests "
            "when metadata is incomplete. Enabled by default for the blessed pipeline; "
            "use --no-allow-image-backfill to opt out."
        ),
    )
    parser.add_argument("--dataset-build-arg", action="append", default=[])
    parser.add_argument("--hard-source-dataset-build-arg", action="append", default=[])
    parser.add_argument("--cache-prediction-arg", action="append", default=[])
    parser.add_argument("--hard-source-cache-prediction-arg", action="append", default=[])
    parser.add_argument("--production-manifest-arg", action="append", default=[])
    parser.add_argument("--production-cache-prediction-arg", action="append", default=[])
    parser.add_argument("--production-resolver-metadata-arg", action="append", default=[])
    parser.add_argument("--candidate-search-arg", action="append", default=[])
    parser.add_argument("--hard-validation-arg", action="append", default=[])
    parser.add_argument("--gt-hard-resolver-metadata-arg", action="append", default=[])
    parser.add_argument("--scorer-train-arg", action="append", default=[])
    parser.add_argument("--scorer-eval-arg", action="append", default=[])
    parser.add_argument(
        "--log-level", default="INFO", help="TRACE, DEBUG, INFO, WARNING, or ERROR"
    )
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_dataset_source_args(args)
    configure_logging(args.log_level)
    summary = run_pipeline(args)
    logger.info("Pipeline summary: %s", summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
