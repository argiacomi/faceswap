#!/usr/bin/env python3
"""Search landmark ensemble setups and emit promoted artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import typing as T
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.landmarks.ensemble.promoted_setup import (
    PROMOTION_REPORT_FILENAME,
    SETUP_FILENAME,
    WEIGHTS_FILENAME,
    write_best_setup,
    write_best_weights,
)
from lib.landmarks.eval.candidate_search import (
    DEFAULT_OBJECTIVE,
    DEFAULT_REGRESSION_EPSILON_NME,
    Candidate,
    CandidateResult,
    enumerate_candidates,
    evaluate_candidate,
    load_split_samples,
    run_candidate_search,
)
from lib.landmarks.eval.geometry_metrics import (
    GEOMETRY_OBJECTIVE,
    GeometryAggregate,
)
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.eval.profile_metrics import (
    DEFAULT_NORMALIZER,
    DEFAULT_PCK_THRESHOLDS,
    DEFAULT_PRIORITY_FAILURE_REGIONS,
    DEFAULT_REGION_FAILURE_THRESHOLD,
    NORMALIZERS,
    ProfileAggregate,
    aggregate_profile_samples,
    evaluate_profile_sample,
)
from lib.landmarks.eval.promotion_gates import (
    DEFAULT_REPORT_IMPROVEMENT_TOLERANCE,
    GateApplication,
    GateConfig,
    GeometryScore,
    ProfileScore,
    apply_gates,
    no_promotion_payload,
)
from lib.landmarks.eval.splits import SplitAssignment, load_split_file, split_assignment_hash


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.0f}s"


def _progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _stage(name: str, fn: T.Callable[[], T.Any]) -> T.Any:
    started = time.time()
    _progress(f"START {name}")
    try:
        result = fn()
    except Exception as err:
        _progress(
            f"FAIL  {name} after {_format_duration(time.time() - started)}: "
            f"{type(err).__name__}: {err}"
        )
        raise
    _progress(f"OK    {name} in {_format_duration(time.time() - started)}")
    return result


def _show_progress(args: argparse.Namespace) -> bool:
    return not args.no_progress and sys.stderr.isatty()


def _candidate_progress(
    candidates: T.Sequence[Candidate], *, enabled: bool
) -> T.Iterable[Candidate]:
    return tqdm(
        candidates,
        total=len(candidates),
        desc="Evaluate candidates",
        unit="candidate",
        disable=not enabled,
    )


def _geometry_candidate_progress(
    results: T.Sequence[CandidateResult], *, enabled: bool
) -> T.Iterable[CandidateResult]:
    return tqdm(
        results,
        total=len(results),
        desc="Evaluate geometry candidates",
        unit="candidate",
        disable=not enabled,
    )


def _load_inputs(args: argparse.Namespace) -> tuple[SplitAssignment, str, DiskPredictionCache]:
    assignment = load_split_file(args.splits)
    return assignment, split_assignment_hash(assignment), DiskPredictionCache(args.cache_dir)


def _load_samples(
    args: argparse.Namespace, assignment: SplitAssignment
) -> tuple[list[T.Any], list[T.Any], list[T.Any]]:
    fit_samples = load_split_samples(args.manifest, assignment, "fit")
    select_samples = load_split_samples(args.manifest, assignment, "select")
    report_samples = load_split_samples(args.manifest, assignment, "report")
    _progress(
        f"Loaded samples: fit={len(fit_samples)}, select={len(select_samples)}, "
        f"report={len(report_samples)}"
    )
    return fit_samples, select_samples, report_samples


# The candidate-aware fusion helper now lives in
# lib.landmarks.search.fusion_variants. Re-exported here so the profile-
# aggregate path below keeps the legacy name; new code should import from
# :mod:`lib.landmarks.search.fusion_variants` directly.
from lib.landmarks.search.fusion_variants import fuse_candidate as _fuse_for_profile


def _candidate_profile_aggregate(
    result: CandidateResult,
    *,
    samples: T.Sequence[T.Any],
    cache: DiskPredictionCache,
    normalizer: str,
    region_failure_threshold: float,
    pck_thresholds: T.Sequence[float],
    priority_failure_regions: T.Sequence[str],
) -> ProfileAggregate:
    import numpy as np

    per_sample: list[T.Any] = []
    for sample in samples:
        bbox = sample.face_bbox
        try:
            truth = np.load(sample.landmarks).astype("float32")
        except OSError:
            continue
        if bbox is None:
            left, top = np.min(truth, axis=0)
            right, bottom = np.max(truth, axis=0)
            bbox = (float(left), float(top), float(right), float(bottom))
        cached_points = [
            cache.read(sample.sample_id, m).landmarks for m in result.candidate.models
        ]
        fused = _fuse_for_profile(result.candidate, cached_points, weights=result.weights)
        per_sample.append(
            evaluate_profile_sample(
                fused,
                truth,
                sample_id=sample.sample_id,
                face_bbox=bbox,
                visibility=sample.visibility,
                normalizer_method=normalizer,
                region_failure_threshold=region_failure_threshold,
                priority_failure_regions=priority_failure_regions,
                pck_thresholds=pck_thresholds,
            )
        )
    return aggregate_profile_samples(
        result.candidate_id,
        per_sample,
        priority_failure_regions=priority_failure_regions,
        pck_thresholds=pck_thresholds,
    )


def _gate_config_from_args(args: argparse.Namespace) -> GateConfig:
    return GateConfig(
        require_report_improvement=args.require_report_improvement,
        report_improvement_tolerance=args.report_improvement_tolerance,
        max_overall_regression_nme=args.max_overall_regression_nme,
        max_bucket_regression_rate=args.max_bucket_regression_rate,
        require_profile_improvement=args.require_profile_improvement,
        max_profile_region_failure_rate=args.max_profile_region_failure_rate,
        require_effective_ensemble=args.require_effective_ensemble,
        effective_models_floor=args.effective_models_floor,
        allow_single_model_baselines=args.allow_single_model_baselines,
        require_geometry_improvement=args.require_geometry_improvement,
        max_catastrophic_geometry_failure_rate=args.max_catastrophic_geometry_failure_rate,
        max_p95_transform_error=args.max_p95_transform_error,
        max_p95_crop_center_error=args.max_p95_crop_center_error,
        max_p95_roll_error=args.max_p95_roll_error,
        min_hull_iou=args.min_hull_iou,
        max_hard_slice_regression_rate=args.max_hard_slice_regression_rate,
        allow_nme_only_promotion=args.allow_nme_only_promotion,
    )


def _gates_need_profile(config: GateConfig) -> bool:
    return bool(
        config.require_profile_improvement or config.max_profile_region_failure_rate is not None
    )


def _gates_need_geometry(config: GateConfig, args: argparse.Namespace) -> bool:
    return bool(config.requires_geometry() or args.include_geometry_metrics)


def _candidate_models(results: T.Sequence[CandidateResult]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(model for result in results for model in result.candidate.models))


# Geometry candidate-evaluation primitives live in lib.landmarks.search now.
# These names are re-exported under the legacy ``_`` prefix so the rest of
# the CLI doesn't need updating; new call sites should import directly from
# :mod:`lib.landmarks.search.geometry_candidate_eval`.
from lib.landmarks.search.geometry_candidate_eval import (
    build_geometry_context as _build_geometry_context,
)
from lib.landmarks.search.geometry_candidate_eval import (
    evaluate_candidate_geometry as _evaluate_candidate_geometry,
)
from lib.landmarks.search.geometry_candidate_eval import (
    geometry_score_from_aggregate as _geometry_score_from_aggregate,
)


def _enumerate_search_candidates(args: argparse.Namespace) -> list[Candidate]:
    include_baselines = (
        args.include_single_model_baselines
        or args.require_effective_ensemble
        or args.require_report_improvement
        or args.max_overall_regression_nme is not None
        or args.require_profile_improvement
    )
    candidates = enumerate_candidates(
        models=_parse_csv(args.models),
        model_subset_presets=_parse_csv(args.model_subsets),
        weight_generators=_parse_csv(args.weight_generators),
        strategies=_parse_csv(args.strategies),
        outlier_thresholds=_parse_csv_floats(args.outlier_thresholds),
        bbox_source=args.bbox_source,
        crop_scale=args.crop_scale,
        include_single_model_baselines=include_baselines,
    )
    if not candidates:
        raise SystemExit("no candidates were enumerated from the requested dimensions")
    _progress(
        f"Enumerated {len(candidates)} candidates from models={args.models}, "
        f"subsets={args.model_subsets}, generators={args.weight_generators}, "
        f"strategies={args.strategies}"
    )
    return candidates


def _geometry_per_candidate_payload(
    *,
    geometry_aggregates: T.Mapping[str, GeometryAggregate],
    geometry_scores: T.Mapping[str, GeometryScore],
    baseline_per_bucket: T.Mapping[str, float] | None,
    global_baseline_score: float | None,
) -> dict[str, dict[str, T.Any]]:
    """Build the per-candidate geometry section persisted in candidate_results.json.

    Surfaces every per-bucket diagnostic the search produced (overall score,
    catastrophic rate, hull IoU mean/p05, crop-center P95, transform P95)
    alongside the worst-bucket regression summary so downstream consumers
    don't have to recompute anything to understand *which* slice drove a
    candidate's ``max_bucket_regression_score``.
    """
    payload: dict[str, dict[str, T.Any]] = {}
    for candidate_id, aggregate in geometry_aggregates.items():
        score = geometry_scores.get(candidate_id)
        payload[candidate_id] = {
            "overall_score": aggregate.overall_score,
            "catastrophic_failure_rate": aggregate.catastrophic_failure_rate,
            "p95_translation_normalized": aggregate.p95_translation_normalized,
            "p95_roi_center_normalized": aggregate.p95_roi_center_normalized,
            "p95_roll_degrees_delta": aggregate.p95_roll_degrees_delta,
            "mean_hull_iou": aggregate.mean_hull_iou,
            "p05_hull_iou": aggregate.p05_hull_iou,
            "max_bucket_regression_score": (
                float(score.max_bucket_regression_score) if score is not None else 0.0
            ),
            "worst_bucket": score.worst_bucket if score is not None else "",
            "worst_bucket_score": (float(score.worst_bucket_score) if score is not None else 0.0),
            "worst_bucket_baseline_score": (
                float(score.worst_bucket_baseline_score) if score is not None else 0.0
            ),
            "per_bucket": {
                bucket: dict(values) for bucket, values in aggregate.per_bucket.items()
            },
        }
    if payload:
        payload["__baseline__"] = {
            "global_baseline_score": (
                float(global_baseline_score) if global_baseline_score is not None else None
            ),
            "per_bucket_baseline_score": (
                {bucket: float(value) for bucket, value in baseline_per_bucket.items()}
                if baseline_per_bucket is not None
                else None
            ),
        }
    return payload


def _write_candidate_results(
    output_dir: Path,
    results: T.Sequence[CandidateResult],
    *,
    objective: str,
    regression_epsilon_nme: float,
    geometry_aggregates: T.Mapping[str, GeometryAggregate] | None = None,
    geometry_scores: T.Mapping[str, GeometryScore] | None = None,
    baseline_per_bucket: T.Mapping[str, float] | None = None,
    global_baseline_score: float | None = None,
) -> tuple[Path, Path]:
    csv_path = output_dir / "candidate_results.csv"
    json_path = output_dir / "candidate_results.json"
    geometry_payload = (
        _geometry_per_candidate_payload(
            geometry_aggregates=geometry_aggregates,
            geometry_scores=geometry_scores or {},
            baseline_per_bucket=baseline_per_bucket,
            global_baseline_score=global_baseline_score,
        )
        if geometry_aggregates
        else {}
    )
    fieldnames = [
        "rank",
        "candidate_id",
        "score",
        "objective",
        "models",
        "weight_generator",
        "strategy",
        "outlier_threshold",
        "overall_nme",
        "failure_rate",
        "auc",
        "regression_rate_vs_best_single",
        "bucket_regression_rate_vs_best_single",
        "best_single_model",
        "weights_hash",
    ]
    if geometry_payload:
        fieldnames.extend(
            [
                "geometry_overall_score",
                "geometry_catastrophic_failure_rate",
                "geometry_p95_translation_normalized",
                "geometry_p95_roi_center_normalized",
                "geometry_mean_hull_iou",
                "geometry_p05_hull_iou",
                "geometry_max_bucket_regression_score",
                "geometry_worst_bucket",
                "geometry_worst_bucket_score",
                "geometry_worst_bucket_baseline_score",
            ]
        )
    rows = []
    for rank, result in enumerate(results, start=1):
        row = {
            "rank": rank,
            "candidate_id": result.candidate_id,
            "score": result.score,
            "objective": result.objective,
            "models": "|".join(result.candidate.models),
            "weight_generator": result.candidate.weight_generator,
            "strategy": result.candidate.strategy,
            "outlier_threshold": ""
            if result.candidate.outlier_threshold is None
            else result.candidate.outlier_threshold,
            "overall_nme": result.metrics.overall_nme,
            "failure_rate": result.metrics.failure_rate,
            "auc": result.metrics.auc,
            "regression_rate_vs_best_single": result.metrics.regression_rate_vs_best_single,
            "bucket_regression_rate_vs_best_single": result.metrics.bucket_regression_rate_vs_best_single,
            "best_single_model": result.metrics.best_single_model,
            "weights_hash": result.weights_hash,
        }
        geom = geometry_payload.get(result.candidate_id) if geometry_payload else None
        if geom is not None:
            row.update(
                {
                    "geometry_overall_score": geom["overall_score"],
                    "geometry_catastrophic_failure_rate": geom["catastrophic_failure_rate"],
                    "geometry_p95_translation_normalized": geom["p95_translation_normalized"],
                    "geometry_p95_roi_center_normalized": geom["p95_roi_center_normalized"],
                    "geometry_mean_hull_iou": geom["mean_hull_iou"],
                    "geometry_p05_hull_iou": geom["p05_hull_iou"],
                    "geometry_max_bucket_regression_score": geom["max_bucket_regression_score"],
                    "geometry_worst_bucket": geom["worst_bucket"],
                    "geometry_worst_bucket_score": geom["worst_bucket_score"],
                    "geometry_worst_bucket_baseline_score": geom["worst_bucket_baseline_score"],
                }
            )
        rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_doc: dict[str, T.Any] = {
        "objective": objective,
        "regression_epsilon_nme": regression_epsilon_nme,
        "candidates": [result.to_payload() for result in results],
    }
    if geometry_payload:
        json_doc["geometry_per_candidate"] = geometry_payload
    json_path.write_text(
        json.dumps(json_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def _write_promotion_report(
    output_dir: Path,
    *,
    winner: CandidateResult,
    results: T.Sequence[CandidateResult],
    objective: str,
    regression_epsilon_nme: float,
    report_metrics: T.Mapping[str, T.Any],
    gate_application: GateApplication | None = None,
    profile_aggregate: ProfileAggregate | None = None,
) -> Path:
    path = output_dir / PROMOTION_REPORT_FILENAME
    lines = [
        "# Promotion Report",
        "",
        f"- objective: `{objective}`",
        f"- regression_epsilon_nme: `{regression_epsilon_nme}`",
        f"- evaluated_candidates: `{len(results)}`",
        "",
        "## Winner",
        "",
        f"- candidate_id: `{winner.candidate_id}`",
        f"- models: `{', '.join(winner.candidate.models)}`",
        f"- weight_generator: `{winner.candidate.weight_generator}`",
        f"- strategy: `{winner.candidate.strategy}`",
        f"- outlier_threshold: `{winner.candidate.outlier_threshold}`",
        f"- selection_score: `{winner.score:.6f}`",
        f"- selection_nme: `{winner.metrics.overall_nme:.6f}`",
        f"- selection_failure_rate: `{winner.metrics.failure_rate:.6f}`",
        f"- selection_regression_rate: `{winner.metrics.regression_rate_vs_best_single:.6f}`",
        f"- bucket_regression_rate: `{winner.metrics.bucket_regression_rate_vs_best_single:.6f}`",
        "",
        "## Held-out report metrics",
        "",
        f"- report_nme: `{report_metrics.get('overall_nme', 0.0):.6f}`",
        f"- report_failure_rate: `{report_metrics.get('failure_rate', 0.0):.6f}`",
        f"- report_regression_rate: `{report_metrics.get('regression_rate_vs_best_single', 0.0):.6f}`",
        f"- report_bucket_regression_rate: `{report_metrics.get('bucket_regression_rate_vs_best_single', 0.0):.6f}`",
    ]
    if winner.effective_ensemble is not None:
        diag = winner.effective_ensemble
        lines.extend(
            [
                "",
                "## Effective ensemble diagnostics",
                "",
                f"- mean_effective_models: `{diag.mean_effective_models:.3f}` (floor `{diag.effective_models_floor:.3f}`)",
                f"- collapsed: `{diag.collapsed}`",
                f"- weighted_median_collapsed: `{diag.weighted_median_collapsed}`",
                "- landmark share by model: "
                + ", ".join(
                    f"{model}={share:.2f}"
                    for model, share in sorted(
                        diag.landmark_share_by_model.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ),
            ]
        )
        if winner.is_single_model_baseline:
            lines.append(
                "- note: this winner is a single-model baseline; promotion of single-model setups was explicitly allowed."
            )
    if gate_application is not None:
        lines.extend(
            [
                "",
                "## Promotion gates",
                "",
                f"- gates_passed: `{gate_application.passed_count}`",
                f"- gates_failed: `{gate_application.failed_count}`",
            ]
        )
        if gate_application.promoted_outcome is not None:
            lines.append("- selected candidate cleared every active gate.")
        if profile_aggregate is not None:
            lines.extend(
                [
                    "",
                    "## Profile metrics (held-out report split)",
                    "",
                    f"- profile_overall_score: `{profile_aggregate.overall_score:.6f}`",
                    f"- profile_region_failure_rate: `{profile_aggregate.region_failure_rate:.6f}`",
                    f"- profile_p90_visible_error: `{profile_aggregate.p90_visible_error:.6f}`",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_promoted_artifacts(
    output_dir: Path,
    *,
    winner: CandidateResult,
    results: T.Sequence[CandidateResult],
    json_path: Path,
    report_metrics: T.Any,
    fit_samples: T.Sequence[T.Any],
    sah: str,
    args: argparse.Namespace,
    gate_application: GateApplication | None = None,
    profile_aggregate: ProfileAggregate | None = None,
) -> tuple[Path, Path, Path]:
    weights_path = output_dir / WEIGHTS_FILENAME
    setup_path = output_dir / SETUP_FILENAME
    write_best_weights(weights_path, winner.weights, models=winner.candidate.models)
    write_best_setup(
        setup_path,
        candidate_id=winner.candidate_id,
        models=winner.candidate.models,
        strategy=winner.candidate.strategy,
        outlier_threshold=winner.candidate.outlier_threshold,
        weight_generator_name=winner.candidate.weight_generator,
        weight_generator_params=winner.candidate.generator_params_dict(),
        crop_scale=winner.candidate.crop_scale,
        bbox_source=winner.candidate.bbox_source,
        regression_epsilon_nme=args.regression_epsilon_nme,
        reproducibility={
            "split_assignment_hash": sah,
            "candidate_search_seed": int(args.seed),
            "objective": args.objective,
        },
        fit={
            "sample_count": len(fit_samples),
            "datasets": sorted({sample.dataset for sample in fit_samples if sample.dataset}),
            "scenario_buckets": sorted(
                {
                    f"{sample.dataset or 'unspecified'}:{sample.condition or 'unspecified'}"
                    for sample in fit_samples
                }
            ),
        },
        selection_metrics=winner.metrics.to_payload(),
        report_metrics=report_metrics.to_payload(),
        evaluation_log_path=str(json_path.name),
        weights_path=WEIGHTS_FILENAME,
    )
    report_path = _write_promotion_report(
        output_dir,
        winner=winner,
        results=results,
        objective=args.objective,
        regression_epsilon_nme=args.regression_epsilon_nme,
        report_metrics=report_metrics.to_payload(),
        gate_application=gate_application,
        profile_aggregate=profile_aggregate,
    )
    return setup_path, weights_path, report_path


def _write_no_promotion(
    output_dir: Path,
    application: GateApplication,
    *,
    args: argparse.Namespace,
    results: T.Sequence[CandidateResult],
    geometry_aggregates: T.Mapping[str, GeometryAggregate] | None = None,
    geometry_scores: T.Mapping[str, GeometryScore] | None = None,
    baseline_per_bucket: T.Mapping[str, float] | None = None,
    global_baseline_score: float | None = None,
) -> Path:
    payload = no_promotion_payload(application)
    payload["objective"] = args.objective
    payload["evaluated_candidates"] = len(results)
    if geometry_aggregates:
        payload["geometry_per_candidate"] = _geometry_per_candidate_payload(
            geometry_aggregates=geometry_aggregates,
            geometry_scores=geometry_scores or {},
            baseline_per_bucket=baseline_per_bucket,
            global_baseline_score=global_baseline_score,
        )
    payload["gate_config"] = {
        "require_report_improvement": args.require_report_improvement,
        "report_improvement_tolerance": args.report_improvement_tolerance,
        "max_overall_regression_nme": args.max_overall_regression_nme,
        "max_bucket_regression_rate": args.max_bucket_regression_rate,
        "require_profile_improvement": args.require_profile_improvement,
        "max_profile_region_failure_rate": args.max_profile_region_failure_rate,
        "require_effective_ensemble": args.require_effective_ensemble,
        "effective_models_floor": args.effective_models_floor,
        "allow_single_model_baselines": args.allow_single_model_baselines,
        "require_geometry_improvement": args.require_geometry_improvement,
        "max_catastrophic_geometry_failure_rate": args.max_catastrophic_geometry_failure_rate,
        "max_p95_transform_error": args.max_p95_transform_error,
        "max_p95_crop_center_error": args.max_p95_crop_center_error,
        "max_p95_roll_error": args.max_p95_roll_error,
        "min_hull_iou": args.min_hull_iou,
        "max_hard_slice_regression_rate": args.max_hard_slice_regression_rate,
        "allow_nme_only_promotion": args.allow_nme_only_promotion,
    }
    path = output_dir / "no_promotion.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--models", default="hrnet,spiga,orformer")
    parser.add_argument("--model-subsets", default="all")
    parser.add_argument(
        "--weight-generators", default="equal,inverse_mean_error,regularized_inverse_error"
    )
    parser.add_argument(
        "--strategies", default="static_weighted,static_weighted_downweight,weighted_median"
    )
    parser.add_argument("--outlier-thresholds", default="2.5,3.5,4.5")
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument(
        "--regression-epsilon-nme", type=float, default=DEFAULT_REGRESSION_EPSILON_NME
    )
    parser.add_argument("--bbox-source", default="manifest")
    parser.add_argument("--crop-scale", type=float, default=1.6)
    parser.add_argument("--failure-threshold", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--include-single-model-baselines", action="store_true")
    parser.add_argument("--allow-single-model-baselines", action="store_true")
    parser.add_argument("--require-report-improvement", action="store_true")
    parser.add_argument(
        "--report-improvement-tolerance", type=float, default=DEFAULT_REPORT_IMPROVEMENT_TOLERANCE
    )
    parser.add_argument("--max-overall-regression-nme", type=float, default=None)
    parser.add_argument("--max-bucket-regression-rate", type=float, default=None)
    parser.add_argument("--require-profile-improvement", action="store_true")
    parser.add_argument("--max-profile-region-failure-rate", type=float, default=None)
    parser.add_argument("--require-effective-ensemble", action="store_true")
    parser.add_argument("--effective-models-floor", type=float, default=1.5)
    parser.add_argument("--profile-normalizer", choices=NORMALIZERS, default=DEFAULT_NORMALIZER)
    parser.add_argument(
        "--profile-region-failure-threshold", type=float, default=DEFAULT_REGION_FAILURE_THRESHOLD
    )
    parser.add_argument(
        "--profile-pck-thresholds", default=",".join(f"{t:.2f}" for t in DEFAULT_PCK_THRESHOLDS)
    )
    parser.add_argument(
        "--profile-priority-regions", default=",".join(DEFAULT_PRIORITY_FAILURE_REGIONS)
    )
    parser.add_argument("--include-geometry-metrics", action="store_true")
    parser.add_argument("--geometry-aligned-size", type=int, default=512)
    parser.add_argument("--geometry-region-failure-threshold", type=float, default=0.05)
    parser.add_argument("--require-geometry-improvement", action="store_true")
    parser.add_argument("--max-catastrophic-geometry-failure-rate", type=float, default=None)
    parser.add_argument("--max-p95-transform-error", type=float, default=None)
    parser.add_argument("--max-p95-crop-center-error", type=float, default=None)
    parser.add_argument("--max-p95-roll-error", type=float, default=None)
    parser.add_argument("--min-hull-iou", type=float, default=None)
    parser.add_argument("--max-hard-slice-regression-rate", type=float, default=None)
    parser.add_argument("--allow-nme-only-promotion", action="store_true")
    args = parser.parse_args(argv)

    if (
        any(
            getattr(args, name, None) is not None
            for name in (
                "max_catastrophic_geometry_failure_rate",
                "max_p95_transform_error",
                "max_p95_crop_center_error",
                "max_p95_roll_error",
                "min_hull_iou",
                "max_hard_slice_regression_rate",
            )
        )
        or args.require_geometry_improvement
    ):
        args.include_geometry_metrics = True
    if args.objective == GEOMETRY_OBJECTIVE and not args.allow_nme_only_promotion:
        args.include_geometry_metrics = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment, sah, cache = _stage("load_splits", lambda: _load_inputs(args))
    fit_samples, select_samples, report_samples = _stage(
        "load_samples", lambda: _load_samples(args, assignment)
    )
    candidates = _stage("enumerate_candidates", lambda: _enumerate_search_candidates(args))
    results = _stage(
        "candidate_search",
        lambda: run_candidate_search(
            candidates,
            fit_samples=fit_samples,
            select_samples=select_samples,
            cache=cache,
            split_assignment_hash=sah,
            objective=args.objective,
            regression_epsilon_nme=args.regression_epsilon_nme,
            failure_threshold=args.failure_threshold,
            progress=lambda values: _candidate_progress(values, enabled=_show_progress(args)),
        ),
    )

    gate_config = _gate_config_from_args(args)
    geometry_scores: dict[str, GeometryScore] = {}
    geometry_aggregates: dict[str, GeometryAggregate] = {}
    geometry_baseline_per_bucket: dict[str, float] = {}
    geometry_global_baseline_score: float | None = None
    if _gates_need_geometry(gate_config, args):

        def _geometry_eval() -> None:
            nonlocal geometry_global_baseline_score
            context = _build_geometry_context(
                select_samples,
                cache=cache,
                models=_candidate_models(results),
                aligned_size=args.geometry_aligned_size,
            )
            for result in _geometry_candidate_progress(results, enabled=_show_progress(args)):
                aggregate = _evaluate_candidate_geometry(
                    result,
                    context=context,
                    aligned_size=args.geometry_aligned_size,
                    region_failure_threshold=args.geometry_region_failure_threshold,
                )
                geometry_aggregates[result.candidate_id] = aggregate
            baseline_score: float | None = None
            for result in results:
                if not result.is_single_model_baseline:
                    continue
                aggregate = geometry_aggregates[result.candidate_id]
                baseline_score = (
                    aggregate.overall_score
                    if baseline_score is None
                    else min(baseline_score, aggregate.overall_score)
                )
                for bucket, values in aggregate.per_bucket.items():
                    bucket_score = float(values.get("overall_score", 0.0))
                    current = geometry_baseline_per_bucket.get(bucket)
                    if current is None or bucket_score < current:
                        geometry_baseline_per_bucket[bucket] = bucket_score
            geometry_global_baseline_score = baseline_score
            for candidate_id, aggregate in geometry_aggregates.items():
                geometry_scores[candidate_id] = _geometry_score_from_aggregate(
                    aggregate,
                    baseline_score=baseline_score,
                    baseline_per_bucket=geometry_baseline_per_bucket or None,
                )

        _stage("geometry_evaluate_candidates", _geometry_eval)

    profile_scores: dict[str, ProfileScore] = {}
    profile_aggregates: dict[str, ProfileAggregate] = {}
    if _gates_need_profile(gate_config):
        pck_thresholds = _parse_csv_floats(args.profile_pck_thresholds) or DEFAULT_PCK_THRESHOLDS
        priority_regions = (
            _parse_csv(args.profile_priority_regions) or DEFAULT_PRIORITY_FAILURE_REGIONS
        )

        def _profile_eval() -> None:
            for result in results:
                aggregate = _candidate_profile_aggregate(
                    result,
                    samples=report_samples,
                    cache=cache,
                    normalizer=args.profile_normalizer,
                    region_failure_threshold=args.profile_region_failure_threshold,
                    pck_thresholds=pck_thresholds,
                    priority_failure_regions=priority_regions,
                )
                profile_aggregates[result.candidate_id] = aggregate
                profile_scores[result.candidate_id] = ProfileScore(
                    overall_score=aggregate.overall_score,
                    region_failure_rate=aggregate.region_failure_rate,
                )

        _stage("profile_evaluate_candidates", _profile_eval)

    if args.objective == GEOMETRY_OBJECTIVE:
        if geometry_scores:
            _progress(
                f"Re-ranking {len(results)} candidates by alignment_geometry_v1 score before promotion"
            )
            results = sorted(
                results,
                key=lambda r: (
                    geometry_scores[r.candidate_id].overall_score,
                    r.metrics.overall_nme,
                ),
            )
        elif not args.allow_nme_only_promotion:
            raise SystemExit(
                "--objective alignment_geometry_v1 selected but geometry metrics were not computed."
            )
        else:
            _progress(
                "alignment_geometry_v1 objective set but --allow-nme-only-promotion is on; falling back to NME ranking."
            )

    csv_path, json_path = _stage(
        "write_candidate_results",
        lambda: _write_candidate_results(
            output_dir,
            results,
            objective=args.objective,
            regression_epsilon_nme=args.regression_epsilon_nme,
            geometry_aggregates=geometry_aggregates or None,
            geometry_scores=geometry_scores or None,
            baseline_per_bucket=geometry_baseline_per_bucket or None,
            global_baseline_score=geometry_global_baseline_score,
        ),
    )

    if gate_config.is_active():
        gate_application = _stage(
            "apply_promotion_gates",
            lambda: apply_gates(
                results,
                gate_config,
                profile_scores=profile_scores or None,
                geometry_scores=geometry_scores or None,
            ),
        )
        winner = gate_application.promoted
        if winner is None:
            no_promotion_path = _stage(
                "write_no_promotion",
                lambda: _write_no_promotion(
                    output_dir,
                    gate_application,
                    args=args,
                    results=results,
                    geometry_aggregates=geometry_aggregates or None,
                    geometry_scores=geometry_scores or None,
                    baseline_per_bucket=geometry_baseline_per_bucket or None,
                    global_baseline_score=geometry_global_baseline_score,
                ),
            )
            print(f"No candidate passed the configured promotion gates; see {no_promotion_path}")
            return 1
    else:
        gate_application = None
        winner = results[0] if results else None
        if winner is None:
            raise SystemExit("no candidates were evaluated")

    _fit_result, report_metrics = _stage(
        "report_evaluate_winner",
        lambda: evaluate_candidate(
            winner.candidate,
            fit_samples=fit_samples,
            select_samples=report_samples,
            cache=cache,
            failure_threshold=args.failure_threshold,
            regression_epsilon_nme=args.regression_epsilon_nme,
        ),
    )
    promoted_profile_aggregate = (
        profile_aggregates.get(winner.candidate_id) if profile_aggregates else None
    )
    setup_path, weights_path, _report_path = _stage(
        "write_promoted_artifacts",
        lambda: _write_promoted_artifacts(
            output_dir,
            winner=winner,
            results=results,
            json_path=json_path,
            report_metrics=report_metrics,
            fit_samples=fit_samples,
            sah=sah,
            args=args,
            gate_application=gate_application,
            profile_aggregate=promoted_profile_aggregate,
        ),
    )
    print(
        f"Promoted candidate {winner.candidate_id} (strategy={winner.candidate.strategy}, score={winner.score:.6f})"
    )
    print(f"  setup:   {setup_path}")
    print(f"  weights: {weights_path}")
    print(f"  csv:     {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
