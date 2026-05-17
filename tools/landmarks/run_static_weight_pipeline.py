#!/usr/bin/env python3
"""Run the landmark static-weight validation pipeline end to end."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.landmarks.datasets.sources import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = DEFAULT_CACHE_DIR / "runs" / "static_weight_validation"
_EXCLUSIVE_SOURCE_FLAGS = (
    "--wflw-annotations",
    "--wflw-download-official",
    "--cofw-json",
    "--source-zip",
    "--download-url",
)


@dataclass(frozen=True)
class PipelinePaths:
    """Structured output paths for one pipeline run."""

    root: Path
    dataset: Path = field(init=False)
    manifest: Path = field(init=False)
    cache: Path = field(init=False)
    baseline_metrics: Path = field(init=False)
    weights: Path = field(init=False)
    weight_file: Path = field(init=False)
    weight_report: Path = field(init=False)
    weighted_metrics: Path = field(init=False)
    debug: Path = field(init=False)
    summary: Path = field(init=False)
    detector_bbox_report: Path = field(init=False)
    splits: Path = field(init=False)
    splits_file: Path = field(init=False)
    fit_manifest: Path = field(init=False)
    select_manifest: Path = field(init=False)
    report_manifest: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", self.root / "dataset")
        object.__setattr__(self, "manifest", self.root / "dataset" / "manifest.json")
        object.__setattr__(self, "cache", self.root / "cache")
        object.__setattr__(self, "baseline_metrics", self.root / "baseline_metrics")
        object.__setattr__(self, "weights", self.root / "weights")
        object.__setattr__(
            self, "weight_file", self.root / "weights" / "static_landmark_weights.json"
        )
        object.__setattr__(
            self, "weight_report", self.root / "weights" / "static_weight_report.json"
        )
        object.__setattr__(self, "weighted_metrics", self.root / "weighted_metrics")
        object.__setattr__(self, "debug", self.root / "debug")
        object.__setattr__(self, "summary", self.root / "run_summary.json")
        object.__setattr__(
            self,
            "detector_bbox_report",
            self.root / "dataset" / "detector_bboxes.json",
        )
        # The filtered manifests must live next to the source manifest so that
        # their relative ``image``/``landmarks`` paths still resolve.
        object.__setattr__(self, "splits", self.root / "splits")
        object.__setattr__(self, "splits_file", self.root / "splits" / "splits.json")
        object.__setattr__(self, "fit_manifest", self.root / "dataset" / "fit_manifest.json")
        object.__setattr__(self, "select_manifest", self.root / "dataset" / "select_manifest.json")
        object.__setattr__(self, "report_manifest", self.root / "dataset" / "report_manifest.json")


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _append_if(argv: list[str], flag: str, value: str) -> None:
    if value:
        argv.extend([flag, value])


def _append_source_dir_if_no_competing_source(argv: list[str], value: str) -> None:
    """Append source-dir only when no explicit source/download mode is active."""
    if value and not any(flag in argv for flag in _EXCLUSIVE_SOURCE_FLAGS):
        argv.extend(["--source-dir", value])


def _default_dataset_source_dir(dataset: str) -> str:
    """Return a conventional local extracted source path when it exists."""
    candidate = DEFAULT_CACHE_DIR / dataset / "extracted"
    return str(candidate) if candidate.is_dir() else ""


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _json_ready(value: T.Any) -> T.Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _ensure_dirs(paths: PipelinePaths) -> None:
    for path in (
        paths.root,
        paths.dataset,
        paths.cache,
        paths.baseline_metrics,
        paths.weights,
        paths.weighted_metrics,
        paths.debug,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _initial_summary(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, T.Any]:
    return {
        "args": _json_ready(vars(args)),
        "git_commit": _git_commit(),
        "output_root": str(paths.root),
        "stages": [],
        "dataset_counts": {},
        "cache_counts": {},
        "generated_weight_path": "",
        "baseline_best_model": "",
        "baseline_best_single_metrics": {},
        "baseline_best_variant": "",
        "baseline_best_variant_metrics": {},
        "weighted_best_variant": "",
        "weighted_best_variant_metrics": {},
        "ensemble_deltas_vs_best_single": {},
        "threshold_failed": False,
        "ensemble_improved_over_best_single": None,
        "bbox_source": args.bbox_source,
    }


def _write_summary(paths: PipelinePaths, summary: dict[str, T.Any]) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _format_duration(seconds: float) -> str:
    """Return a compact human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.0f}s"


def _progress(message: str) -> None:
    """Print a human-visible pipeline progress message."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _stage(summary: dict[str, T.Any], name: str, fn: T.Callable[[], T.Any]) -> T.Any:
    started = time.time()
    record: dict[str, T.Any] = {"name": name, "status": "running", "error": ""}
    summary["stages"].append(record)
    _progress(f"START {name}")
    try:
        result = fn()
    except Exception as err:
        duration = round(time.time() - started, 3)
        record.update(
            {
                "status": "failed",
                "error": f"{type(err).__name__}: {err}",
                "duration_seconds": duration,
            }
        )
        _progress(f"FAIL  {name} after {_format_duration(duration)}: {type(err).__name__}: {err}")
        raise
    duration = round(time.time() - started, 3)
    record.update({"status": "ok", "duration_seconds": duration})
    _progress(f"OK    {name} in {_format_duration(duration)}")
    return result


def _prediction_roots(values: T.Sequence[str]) -> list[str]:
    roots = []
    for value in values:
        if "=" not in value:
            raise ValueError("--prediction-root must be provided as model=path")
        model, root = value.split("=", 1)
        if not model.strip() or not root.strip():
            raise ValueError("--prediction-root must be provided as model=path")
        roots.append(f"{model.strip().lower()}={Path(root).expanduser()}")
    return roots


def _dataset_build_args(
    args: argparse.Namespace, dataset: str, paths: PipelinePaths, *, first: bool
) -> list[str]:
    argv = [
        "--dataset",
        dataset,
        "--output-dir",
        str(paths.dataset),
        "--manifest-mode",
        "replace" if first else "merge",
        "--log-level",
        args.log_level,
    ]
    if args.write_overlays:
        argv.append("--write-overlays")
    if args.samples_per_scenario is not None:
        argv.extend(["--samples-per-scenario", str(args.samples_per_scenario)])
    _append_if(argv, "--scenarios", args.scenarios)
    _append_if(argv, "--cache-dir", args.dataset_cache_dir)
    if args.allow_overlap:
        argv.append("--allow-overlap")
    if args.no_download:
        argv.append("--no-download")
    if args.force_download:
        argv.append("--force-download")

    if dataset == "wflw":
        _append_if(argv, "--wflw-annotations", args.wflw_annotations)
        _append_if(argv, "--image-root", args.wflw_image_root)
        _append_if(argv, "--source-zip", args.wflw_source_zip)
        _append_if(argv, "--download-url", args.wflw_download_url)
        if args.wflw_download_official:
            argv.append("--wflw-download-official")
        _append_source_dir_if_no_competing_source(argv, args.wflw_source_dir)
    elif dataset == "cofw":
        _append_if(argv, "--cofw-json", args.cofw_json)
        _append_if(argv, "--image-root", args.cofw_image_root)
        _append_if(argv, "--source-zip", args.cofw_source_zip)
        _append_if(argv, "--download-url", args.cofw_download_url)
        _append_source_dir_if_no_competing_source(argv, args.cofw_source_dir)
    elif dataset == "300w":
        _append_if(argv, "--source-zip", args.w300_source_zip)
        _append_if(argv, "--download-url", args.w300_download_url)
        _append_source_dir_if_no_competing_source(argv, args.w300_source_dir)
    elif dataset == "directory":
        if not args.directory_source_dir:
            raise ValueError(
                "--directory-source-dir is required when --datasets includes directory"
            )
        argv.extend(["--source-dir", args.directory_source_dir])
        if args.recursive:
            argv.append("--recursive")
    elif dataset == "merl-rav":
        _append_if(argv, "--source-zip", args.merl_rav_source_zip)
        _append_if(argv, "--download-url", args.merl_rav_download_url)
        _append_source_dir_if_no_competing_source(argv, args.merl_rav_source_dir)
        _append_if(argv, "--aflw-source-zip", args.merl_rav_aflw_source_zip)
        _append_if(argv, "--aflw-download-url", args.merl_rav_aflw_download_url)
        if (
            args.merl_rav_aflw_source_dir
            and "--aflw-source-zip" not in argv
            and "--aflw-download-url" not in argv
        ):
            argv.extend(["--aflw-source-dir", args.merl_rav_aflw_source_dir])
    elif dataset == "aflw2000-3d":
        _append_if(argv, "--source-zip", args.aflw2000_3d_source_zip)
        _append_if(argv, "--download-url", args.aflw2000_3d_download_url)
        _append_source_dir_if_no_competing_source(argv, args.aflw2000_3d_source_dir)
    return argv


def _build_datasets(
    args: argparse.Namespace, paths: PipelinePaths, summary: dict[str, T.Any]
) -> None:
    from tools.landmarks.build_quality_dataset import main as build_quality_dataset_main

    successful = 0
    datasets = _parse_csv(args.datasets)
    _progress(f"Building datasets: {', '.join(datasets)}")
    for dataset in datasets:
        first = successful == 0

        def _run_one(dataset: str = dataset, first: bool = first) -> None:
            build_quality_dataset_main(_dataset_build_args(args, dataset, paths, first=first))

        try:
            _stage(summary, f"dataset:{dataset}", _run_one)
            successful += 1
        except Exception:
            if not args.continue_on_error:
                raise
            logger.exception("Dataset build failed for %s; continuing", dataset)
    if successful == 0:
        raise RuntimeError("no dataset manifests were built successfully")
    _require_manifest_samples(paths.manifest)


def _apply_detector_bboxes(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, T.Any]:
    """Enrich the manifest with detector-derived face bboxes."""
    if args.bbox_source != "faceswap-detector":
        return {}

    from tools.landmarks.apply_detector_bboxes import (
        apply_detector_bboxes,
        build_parser,
    )

    detector_argv = [
        "--manifest",
        str(paths.manifest),
        "--detector",
        args.detector,
        "--selection",
        args.detector_selection,
        "--min-iou",
        str(args.detector_min_iou),
        "--on-missing",
        args.detector_on_missing,
        "--output-json",
        str(paths.detector_bbox_report),
        "--log-level",
        args.log_level,
    ]

    detector_args = build_parser().parse_args(detector_argv)
    return apply_detector_bboxes(detector_args)


def _require_manifest_samples(manifest_path: Path) -> None:
    from lib.landmarks.evaluation.harness import load_manifest

    sample_count = sum(1 for _sample in load_manifest(manifest_path))
    if sample_count:
        return
    raise ValueError(
        f"manifest contains no validation samples: {manifest_path}. "
        "For --datasets directory, ensure --directory-source-dir contains matching "
        "*.npy landmarks and same-stem image files; pass --recursive for nested fixtures."
    )


def _prediction_mode(args: argparse.Namespace) -> str:
    if args.prediction_mode != "auto":
        return args.prediction_mode
    return "import" if args.prediction_root else "run"


def _cache_predictions(args: argparse.Namespace, paths: PipelinePaths) -> None:
    from tools.landmarks.cache_predictions import main as cache_predictions_main

    mode = _prediction_mode(args)
    argv = [
        "--manifest",
        str(paths.manifest),
        "--models",
        args.models,
        "--cache-dir",
        str(paths.cache),
    ]
    if args.refresh_predictions:
        argv.append("--refresh")
    if mode == "import":
        argv.extend(["--checkpoint", args.checkpoint_tag])
        roots = _prediction_roots(args.prediction_root)
        if not roots:
            raise ValueError("prediction import mode requires at least one --prediction-root")
        for root in roots:
            argv.extend(["--prediction-root", root])
    else:
        argv.extend(
            [
                "--checkpoint-tag",
                args.checkpoint_tag,
                "--run-models",
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--gt-roi-scale",
                str(args.gt_roi_scale),
            ]
        )
        if args.no_gt_roi:
            argv.append("--no-gt-roi")
    cache_predictions_main(argv)


def _create_splits(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, T.Any]:
    """Compute or load the fit/select/report split and write per-split manifests."""
    from lib.landmarks.evaluation.splits import (
        SplitRatios,
        load_split_file,
        save_split_file,
        scenario_bucket,
        split_assignment_hash,
        split_manifest_samples,
        split_summary_counts,
        write_split_manifest,
    )

    base_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    samples = base_manifest.get("samples", base_manifest.get("scenarios", []))
    if not samples:
        raise ValueError(f"manifest at {paths.manifest} contains no samples")
    ratios = SplitRatios(args.fit_ratio, args.select_ratio, args.report_ratio)

    diagnostics: list[T.Any] = []
    if args.split_mode == "file":
        if not args.split_file:
            raise ValueError("--split-mode=file requires --split-file")
        assignment = load_split_file(args.split_file)
        # Verify every referenced ID exists in the manifest.
        manifest_ids = {
            str(sample.get("sample_id") or sample.get("id") or sample.get("name"))
            for sample in samples
        }
        referenced = set(assignment.fit) | set(assignment.select) | set(assignment.report)
        missing = sorted(referenced - manifest_ids)
        if missing:
            raise ValueError(f"--split-file references samples not in the manifest: {missing}")
    else:
        assignment, diagnostics = split_manifest_samples(
            samples, mode=args.split_mode, ratios=ratios, seed=args.split_seed
        )

    splits_path = Path(args.split_output) if args.split_output else paths.splits_file
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    save_split_file(
        splits_path,
        assignment,
        mode=args.split_mode,
        ratios=ratios,
        seed=args.split_seed,
        diagnostics=diagnostics,
    )

    write_split_manifest(paths.fit_manifest, base_manifest, assignment.fit)
    if assignment.select:
        write_split_manifest(paths.select_manifest, base_manifest, assignment.select)
    if assignment.report:
        write_split_manifest(paths.report_manifest, base_manifest, assignment.report)

    counts = split_summary_counts(assignment, samples)
    return {
        "mode": args.split_mode,
        "ratios": ratios.as_dict(),
        "seed": int(args.split_seed),
        "assignment_hash": split_assignment_hash(assignment),
        "splits_file": str(splits_path),
        "fit_manifest": str(paths.fit_manifest),
        "select_manifest": str(paths.select_manifest) if assignment.select else "",
        "report_manifest": str(paths.report_manifest) if assignment.report else "",
        "counts": counts,
        "diagnostics": [diagnostic.to_payload() for diagnostic in diagnostics],
        "scenario_buckets": sorted({scenario_bucket(sample) for sample in samples}),
    }


def _fit_manifest_path(args: argparse.Namespace, paths: PipelinePaths) -> Path:
    """Return the manifest path that weight derivation should consume.

    When splits are disabled the pipeline falls back to the full manifest.
    """
    if args.split_mode == "none":
        return paths.manifest
    if not paths.fit_manifest.is_file():
        raise FileNotFoundError(
            f"fit-split manifest not found at {paths.fit_manifest}; the create_splits "
            "stage must run before compute_static_weights"
        )
    return paths.fit_manifest


def _compute_weights(args: argparse.Namespace, paths: PipelinePaths) -> None:
    from lib.landmarks.ensemble.weights import save_weights
    from tools.landmarks.compute_static_weights import compute_static_weights

    weights, mean_errors = compute_static_weights(
        _fit_manifest_path(args, paths), paths.cache, _parse_csv(args.models)
    )
    save_weights(paths.weight_file, weights)
    dominant = {}
    if weights:
        import numpy as np

        model_names = list(weights)
        dominant_indices = np.asarray([weights[model] for model in model_names]).argmax(axis=0)
        dominant = {
            str(index): model_names[int(model_index)]
            for index, model_index in enumerate(dominant_indices)
        }
    paths.weight_report.write_text(
        json.dumps(
            {
                "models": list(weights),
                "mean_errors": mean_errors,
                "dominant_model_by_landmark": dominant,
                "weights": weights,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_harness(
    args: argparse.Namespace, paths: PipelinePaths, *, weighted: bool
) -> dict[str, T.Any]:
    from lib.landmarks.evaluation.harness import run_quality_harness

    return run_quality_harness(
        paths.manifest,
        paths.cache,
        models=_parse_csv(args.models),
        variants=_parse_csv(args.weighted_variants if weighted else args.baseline_variants),
        weights_path=paths.weight_file if weighted else None,
        output_dir=paths.weighted_metrics if weighted else paths.baseline_metrics,
        failure_threshold=args.failure_threshold,
        outlier_threshold=args.outlier_threshold,
    )


def _run_failure_viewer(args: argparse.Namespace, paths: PipelinePaths) -> None:
    from tools.landmarks.failure_viewer import main as failure_viewer_main

    failure_viewer_main(
        [
            "--metrics",
            str(paths.weighted_metrics / "metrics.json"),
            "--manifest",
            str(paths.manifest),
            "--cache-dir",
            str(paths.cache),
            "--output-dir",
            str(paths.debug),
            "--models",
            args.models,
            "--weights",
            str(paths.weight_file),
            "--limit",
            str(args.failure_viewer_limit),
            "--outlier-threshold",
            str(args.outlier_threshold),
        ]
    )


def _dataset_counts(manifest_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not manifest_path.is_file():
        return counts
    from lib.landmarks.evaluation.harness import load_manifest

    for sample in load_manifest(manifest_path):
        key = sample.dataset or "unspecified"
        counts[key] = counts.get(key, 0) + 1
    counts["total"] = sum(counts.values())
    return counts


def _cache_counts(
    manifest_path: Path, cache_dir: Path, models: T.Sequence[str]
) -> dict[str, T.Any]:
    if not manifest_path.is_file():
        return {"samples": 0, "predictions": 0, "models": {}}
    from lib.landmarks.cache.prediction_cache import DiskPredictionCache
    from lib.landmarks.evaluation.harness import load_manifest

    cache = DiskPredictionCache(cache_dir)
    model_counts = {model: 0 for model in models}
    sample_count = prediction_count = 0
    for sample in load_manifest(manifest_path):
        sample_count += 1
        available = set(cache.available_models(sample.sample_id))
        for model in models:
            if model in available:
                model_counts[model] += 1
                prediction_count += 1
    return {"samples": sample_count, "predictions": prediction_count, "models": model_counts}


def _compact_best(result: dict[str, T.Any] | None) -> dict[str, T.Any]:
    """Return the harness best-variant block if present."""
    if not result:
        return {}
    best = result.get("best_variant", {})
    return best if isinstance(best, dict) else {}


def _update_summary_outputs(
    summary: dict[str, T.Any],
    paths: PipelinePaths,
    args: argparse.Namespace,
    *,
    baseline: dict[str, T.Any] | None,
    weighted: dict[str, T.Any] | None,
) -> None:
    models = _parse_csv(args.models)
    summary["dataset_counts"] = _dataset_counts(paths.manifest)
    summary["cache_counts"] = _cache_counts(paths.manifest, paths.cache, models)
    if paths.detector_bbox_report.is_file():
        try:
            summary["detector_bboxes"] = json.loads(
                paths.detector_bbox_report.read_text(encoding="utf-8")
            )
        except Exception:
            summary["detector_bboxes"] = {"path": str(paths.detector_bbox_report)}
    if paths.weight_file.is_file():
        summary["generated_weight_path"] = str(paths.weight_file)
    baseline_best = _compact_best(baseline)
    if baseline_best:
        summary["baseline_best_model"] = str(baseline_best.get("best_single_model", ""))
        summary["baseline_best_single_metrics"] = baseline_best.get("best_single", {})
        summary["baseline_best_variant"] = str(baseline_best.get("best_variant", ""))
        summary["baseline_best_variant_metrics"] = baseline_best.get("best_variant_metrics", {})
    weighted_best = _compact_best(weighted)
    if weighted_best:
        summary["weighted_best_variant"] = str(
            weighted_best.get("best_variant") or weighted_best.get("label") or ""
        )
        summary["weighted_best_variant_metrics"] = weighted_best.get(
            "best_variant_metrics"
        ) or weighted_best.get("metrics", {})
        summary["ensemble_deltas_vs_best_single"] = weighted_best.get(
            "ensemble_deltas_vs_best_single", {}
        )
        summary["threshold_failed"] = bool(weighted_best.get("threshold_failed"))
        deltas = summary["ensemble_deltas_vs_best_single"]
        summary["ensemble_improved_over_best_single"] = (
            any(float(delta) < 0 for delta in deltas.values()) if deltas else None
        )
    elif baseline_best:
        summary["threshold_failed"] = bool(baseline_best.get("threshold_failed"))


def _metric_percent(metrics: T.Mapping[str, T.Any], key: str) -> str:
    """Return a percentage metric string for NME/failure rate values."""
    value = (
        metrics.get("overall_nme", metrics.get(key, None)) if key == "nme" else metrics.get(key)
    )
    if value is None or value == "":
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _metric_float(metrics: T.Mapping[str, T.Any], key: str) -> str:
    value = metrics.get(key)
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.4f}"


def _format_metrics(label: str, metrics: T.Mapping[str, T.Any]) -> str:
    if not metrics:
        return f"{label}: n/a"
    return (
        f"{label}: NME {_metric_percent(metrics, 'nme')}, "
        f"FR {_metric_percent(metrics, 'failure_rate')}, "
        f"AUC {_metric_float(metrics, 'auc')}"
    )


def _format_dataset_counts(counts: T.Mapping[str, T.Any]) -> str:
    if not counts:
        return "none"
    total = counts.get("total", 0)
    parts = [f"{key}={value}" for key, value in sorted(counts.items()) if key != "total"]
    if parts:
        return f"total={total} ({', '.join(parts)})"
    return f"total={total}"


def _format_cache_counts(counts: T.Mapping[str, T.Any]) -> str:
    if not counts:
        return "none"
    model_counts = counts.get("models", {}) if isinstance(counts.get("models"), dict) else {}
    models = ", ".join(f"{model}={count}" for model, count in sorted(model_counts.items()))
    return f"samples={counts.get('samples', 0)}, predictions={counts.get('predictions', 0)}" + (
        f" ({models})" if models else ""
    )


def _best_delta(summary: T.Mapping[str, T.Any], variant: str) -> str:
    deltas = summary.get("ensemble_deltas_vs_best_single", {})
    if not isinstance(deltas, dict) or variant not in deltas:
        return "n/a"
    delta = float(deltas[variant])
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.3f} pp NME"


def _format_run_report(summary: dict[str, T.Any], *, exit_code: int) -> str:
    """Return a compact human-readable final pipeline report."""
    status = "FAILED" if exit_code else "COMPLETE"
    weighted_variant = str(summary.get("weighted_best_variant") or "")
    lines = [
        "",
        f"Pipeline {status}",
        f"Output root: {summary.get('output_root', '')}",
        f"Run summary: {Path(str(summary.get('output_root', ''))) / 'run_summary.json'}",
        f"Datasets: {_format_dataset_counts(summary.get('dataset_counts', {}))}",
        f"Prediction cache: {_format_cache_counts(summary.get('cache_counts', {}))}",
        f"BBox source: {summary.get('bbox_source', 'manifest')}",
    ]
    failed_error = summary.get("failed_stage_error")
    if failed_error:
        lines.append(f"Failure: {failed_error}")
    lines.extend(
        [
            _format_metrics(
                f"Best single ({summary.get('baseline_best_model') or 'n/a'})",
                summary.get("baseline_best_single_metrics", {}),
            ),
            _format_metrics(
                f"Best weighted ({weighted_variant or 'n/a'})",
                summary.get("weighted_best_variant_metrics", {}),
            ),
        ]
    )
    if weighted_variant:
        lines.append(f"Weighted delta vs best single: {_best_delta(summary, weighted_variant)}")
    improved = summary.get("ensemble_improved_over_best_single")
    if improved is not None:
        lines.append(
            "Finding: ensemble improved over best single"
            if improved
            else "Finding: ensemble did not beat best single"
        )
    if summary.get("threshold_failed"):
        lines.append("Threshold: failed at least one configured failure threshold")
    if summary.get("generated_weight_path"):
        lines.append(f"Weights: {summary['generated_weight_path']}")
    lines.append(
        f"Metrics: {summary.get('output_root', '')}/weighted_metrics/best_variant_summary.json"
    )
    return "\n".join(lines)


def _dry_run(args: argparse.Namespace, paths: PipelinePaths, summary: dict[str, T.Any]) -> int:
    _ensure_dirs(paths)
    planned = []
    if not args.skip_dataset_build:
        planned.append("dataset:auto-build-or-reuse")
        planned.extend(f"dataset:{dataset}" for dataset in _parse_csv(args.datasets))
    else:
        planned.append("dataset:reuse-existing")
    if args.bbox_source == "faceswap-detector":
        planned.append(f"detector_bboxes:{args.detector}:{args.detector_selection}")
    planned.append(
        f"predictions:{_prediction_mode(args)}"
        if args.run_predictions and not args.skip_predictions
        else "predictions:skipped"
    )
    if not args.skip_baseline:
        planned.append("baseline_harness")
    if args.split_mode != "none":
        planned.append(f"create_splits:{args.split_mode}")
    planned.extend(["compute_static_weights", "weighted_harness"])
    if not args.skip_failure_viewer:
        planned.append("failure_viewer")
    summary["dry_run"] = True
    summary["planned_stages"] = planned
    _write_summary(paths, summary)
    print(f"Dry run only. Planned stages written to: {paths.summary}")
    print("Planned stages:")
    for index, stage in enumerate(planned, start=1):
        print(f"  {index}. {stage}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="wflw,cofw")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--models", default="hrnet,spiga,orformer")
    parser.add_argument(
        "--build-datasets",
        action="store_true",
        help="Force dataset rebuild even when manifest exists.",
    )
    parser.add_argument("--run-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-dataset-build", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-failure-viewer", action="store_true")
    parser.add_argument("--prediction-mode", choices=("auto", "import", "run"), default="auto")
    parser.add_argument("--prediction-root", action="append", default=[])
    parser.add_argument("--checkpoint-tag", default="validation")
    parser.add_argument("--refresh-predictions", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--bbox-source",
        choices=("manifest", "faceswap-detector"),
        default="manifest",
        help=(
            "ROI source for prediction crops. 'manifest' uses manifest face_bbox / "
            "dataset defaults. 'faceswap-detector' enriches the manifest with "
            "detector-derived face_bbox values before prediction."
        ),
    )
    parser.add_argument("--detector", default="scrfd")
    parser.add_argument(
        "--detector-selection",
        choices=("gt-iou", "confidence", "largest"),
        default="gt-iou",
        help=(
            "How to choose a detector box. Use gt-iou for labeled validation "
            "sets so the selected detection matches the GT face."
        ),
    )
    parser.add_argument("--detector-min-iou", type=float, default=0.25)
    parser.add_argument(
        "--detector-on-missing",
        choices=("error", "skip", "gt"),
        default="error",
        help=(
            "What to do when no detector box can be matched. 'error' is strict, "
            "'skip' keeps the sample without changing face_bbox, and 'gt' falls "
            "back to a GT-landmark bbox."
        ),
    )
    parser.add_argument("--no-gt-roi", action="store_true")
    parser.add_argument("--gt-roi-scale", type=float, default=1.0)
    parser.add_argument("--write-overlays", action="store_true")
    parser.add_argument("--failure-threshold", type=float, default=0.08)
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    parser.add_argument("--failure-viewer-limit", type=int, default=32)
    parser.add_argument("--baseline-variants", default="plain_average")
    parser.add_argument(
        "--weighted-variants", default="static_weighted,static_weighted_outliers,weighted_median"
    )
    parser.add_argument(
        "--split-mode",
        choices=("scenario-stratified", "random", "file", "none"),
        default="scenario-stratified",
        help=(
            "How to derive the fit/select/report splits. 'scenario-stratified' is the "
            "default and preserves dataset:condition bucket coverage; 'random' splits "
            "the manifest as a single pool; 'file' loads a precomputed splits.json; "
            "'none' disables splits and weights are fit on the full manifest."
        ),
    )
    parser.add_argument("--fit-ratio", type=float, default=0.6)
    parser.add_argument("--select-ratio", type=float, default=0.2)
    parser.add_argument("--report-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1337)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split-file",
        default="",
        help=(
            "Path to an existing splits.json to consume. Required when "
            "--split-mode=file. Mutually exclusive with --split-output."
        ),
    )
    split_group.add_argument(
        "--split-output",
        default="",
        help=(
            "Path to write the generated splits.json. Defaults to "
            "<output-root>/splits/splits.json. Mutually exclusive with --split-file."
        ),
    )
    parser.add_argument("--samples-per-scenario", type=int, default=None)
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--dataset-cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--wflw-annotations", default="")
    parser.add_argument("--wflw-image-root", default="")
    parser.add_argument("--wflw-source-dir", default=_default_dataset_source_dir("wflw"))
    parser.add_argument("--wflw-source-zip", default="")
    parser.add_argument("--wflw-download-url", default="")
    parser.add_argument("--wflw-download-official", action="store_true")
    parser.add_argument("--cofw-json", default="")
    parser.add_argument("--cofw-image-root", default="")
    parser.add_argument("--cofw-source-dir", default=_default_dataset_source_dir("cofw"))
    parser.add_argument("--cofw-source-zip", default="")
    parser.add_argument("--cofw-download-url", default="")
    parser.add_argument(
        "--300w-source-dir", dest="w300_source_dir", default=_default_dataset_source_dir("300w")
    )
    parser.add_argument("--300w-source-zip", dest="w300_source_zip", default="")
    parser.add_argument("--300w-download-url", dest="w300_download_url", default="")
    parser.add_argument("--merl-rav-source-dir", default=_default_dataset_source_dir("merl-rav"))
    parser.add_argument("--merl-rav-source-zip", default="")
    parser.add_argument("--merl-rav-download-url", default="")
    parser.add_argument(
        "--merl-rav-aflw-source-dir",
        default=_default_dataset_source_dir("aflw"),
        help="Explicit AFLW image directory for MERL-RAV label matching.",
    )
    parser.add_argument(
        "--merl-rav-aflw-source-zip",
        default="",
        help="Explicit AFLW image archive for MERL-RAV label matching.",
    )
    parser.add_argument(
        "--merl-rav-aflw-download-url",
        default="",
        help="Override AFLW image archive URL for MERL-RAV label matching.",
    )
    parser.add_argument(
        "--aflw2000-3d-source-dir", default=_default_dataset_source_dir("aflw2000-3d")
    )
    parser.add_argument("--aflw2000-3d-source-zip", default="")
    parser.add_argument("--aflw2000-3d-download-url", default="")
    parser.add_argument("--directory-source-dir", default=_default_dataset_source_dir("directory"))
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def _validate_split_args(args: argparse.Namespace) -> None:
    """Reject incompatible split-mode / split-file / split-output combinations."""
    if args.split_mode == "none":
        if args.split_file:
            raise SystemExit("--split-file is not allowed when --split-mode=none")
        if args.split_output:
            raise SystemExit("--split-output is not allowed when --split-mode=none")
        return
    if args.split_mode == "file":
        if not args.split_file:
            raise SystemExit("--split-mode=file requires --split-file")
        return
    # scenario-stratified or random
    if args.split_file:
        raise SystemExit(
            "--split-file is only valid with --split-mode=file; got "
            f"--split-mode={args.split_mode}"
        )
    ratios = (args.fit_ratio, args.select_ratio, args.report_ratio)
    if any(value <= 0 for value in ratios):
        raise SystemExit("--fit-ratio, --select-ratio, and --report-ratio must all be > 0")
    if not 0.99 < sum(ratios) < 1.01:
        raise SystemExit(
            f"--fit-ratio + --select-ratio + --report-ratio must sum to 1.0, got {sum(ratios):.3f}"
        )


def _log_pipeline_failure(args: argparse.Namespace, err: Exception) -> None:
    if args.log_level == "DEBUG":
        logger.exception("Pipeline failed")
        return
    logger.error("Pipeline failed: %s: %s", type(err).__name__, err)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s"
    )
    paths = PipelinePaths(Path(args.output_root).expanduser())
    summary = _initial_summary(args, paths)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.failure_viewer_limit <= 0:
        raise SystemExit("--failure-viewer-limit must be greater than zero")
    if args.gt_roi_scale <= 0:
        raise SystemExit("--gt-roi-scale must be greater than zero")
    if args.detector_min_iou < 0.0 or args.detector_min_iou > 1.0:
        raise SystemExit("--detector-min-iou must be between 0 and 1")
    _validate_split_args(args)
    if args.dry_run:
        return _dry_run(args, paths, summary)

    baseline: dict[str, T.Any] | None = None
    weighted: dict[str, T.Any] | None = None
    exit_code = 0
    _ensure_dirs(paths)
    _progress(
        f"Pipeline start: datasets={args.datasets}, models={args.models}, output_root={paths.root}"
    )
    try:
        if not args.skip_dataset_build and (args.build_datasets or not paths.manifest.is_file()):
            _stage(summary, "build_datasets", lambda: _build_datasets(args, paths, summary))
        elif not paths.manifest.is_file():
            raise FileNotFoundError(
                f"manifest not found at {paths.manifest}. "
                "Remove --skip-dataset-build, pass --build-datasets, or create it first."
            )
        else:
            _require_manifest_samples(paths.manifest)
            _progress(f"Using existing manifest: {paths.manifest}")
        if args.bbox_source == "faceswap-detector":
            detector_summary = _stage(
                summary, "detector_bboxes", lambda: _apply_detector_bboxes(args, paths)
            )
            summary["detector_bboxes"] = detector_summary
        if args.run_predictions and not args.skip_predictions:
            _stage(summary, "cache_predictions", lambda: _cache_predictions(args, paths))
        else:
            _progress("Skipping prediction caching")
        if not args.skip_baseline:
            baseline = _stage(
                summary, "baseline_harness", lambda: _run_harness(args, paths, weighted=False)
            )
        else:
            _progress("Skipping baseline harness")
        if args.split_mode != "none":
            split_summary = _stage(summary, "create_splits", lambda: _create_splits(args, paths))
            summary["splits"] = split_summary
        _stage(summary, "compute_static_weights", lambda: _compute_weights(args, paths))
        weighted = _stage(
            summary, "weighted_harness", lambda: _run_harness(args, paths, weighted=True)
        )
        if not args.skip_failure_viewer:
            _stage(summary, "failure_viewer", lambda: _run_failure_viewer(args, paths))
        else:
            _progress("Skipping failure viewer")
    except Exception as err:
        exit_code = 1
        summary["failed_stage_error"] = f"{type(err).__name__}: {err}"
        _log_pipeline_failure(args, err)
    finally:
        _update_summary_outputs(summary, paths, args, baseline=baseline, weighted=weighted)
        _write_summary(paths, summary)

    report = _format_run_report(summary, exit_code=exit_code)
    print(report, file=sys.stderr if exit_code else sys.stdout)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
