#!/usr/bin/env python3
"""Search the landmark ensemble setup space and emit promoted artifacts (#69).

Reads a manifest, a populated prediction cache, and a fit/select/report split
assignment (#67), enumerates ensemble candidates over the requested model
subsets / weight generators / strategies / outlier thresholds, scores each on
the select split, and writes ``best_setup.json``, ``best_weights.json``,
``candidate_results.{csv,json}``, and ``promotion_report.md``.

The search is cache-only: it never invokes landmark adapters. Re-prediction
belongs in the cache-building stage.
"""

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
from lib.landmarks.ensemble.strategies import (
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
)
from lib.landmarks.ensemble.weights import weights_matrix_for_models
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
    aggregate_geometry_samples,
    evaluate_geometry_sample,
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
from lib.landmarks.fusion import (
    normalize_weight_matrix,
    plain_average,
    static_weighted,
)
from lib.landmarks.rejection import weighted_median


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


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
    """Print a human-visible progress message."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _stage(name: str, fn: T.Callable[[], T.Any]) -> T.Any:
    """Run one visible CLI stage with pipeline-style timing output."""
    started = time.time()
    _progress(f"START {name}")
    try:
        result = fn()
    except Exception as err:
        duration = round(time.time() - started, 3)
        _progress(f"FAIL  {name} after {_format_duration(duration)}: {type(err).__name__}: {err}")
        raise
    duration = round(time.time() - started, 3)
    _progress(f"OK    {name} in {_format_duration(duration)}")
    return result


def _show_progress(args: argparse.Namespace) -> bool:
    """Return whether tqdm progress bars should be shown."""
    return not args.no_progress and sys.stderr.isatty()


def _candidate_progress(
    candidates: T.Sequence[Candidate], *, enabled: bool
) -> T.Iterable[Candidate]:
    """Wrap candidates in a tqdm bar when interactive progress is enabled."""
    return tqdm(
        candidates,
        total=len(candidates),
        desc="Evaluate candidates",
        unit="candidate",
        disable=not enabled,
    )


def _load_inputs(args: argparse.Namespace) -> tuple[SplitAssignment, str, DiskPredictionCache]:
    """Load split metadata and prediction cache handles."""
    assignment = load_split_file(args.splits)
    sah = split_assignment_hash(assignment)
    cache = DiskPredictionCache(args.cache_dir)
    return assignment, sah, cache


def _load_samples(
    args: argparse.Namespace, assignment: SplitAssignment
) -> tuple[list[T.Any], list[T.Any], list[T.Any]]:
    """Load fit/select/report sample lists from the manifest and split file."""
    fit_samples = load_split_samples(args.manifest, assignment, "fit")
    select_samples = load_split_samples(args.manifest, assignment, "select")
    report_samples = load_split_samples(args.manifest, assignment, "report")
    _progress(
        "Loaded samples: "
        f"fit={len(fit_samples)}, select={len(select_samples)}, report={len(report_samples)}"
    )
    return fit_samples, select_samples, report_samples


def _fuse_for_profile(
    candidate: Candidate,
    cached_points: T.Sequence[T.Any],
    *,
    weights: dict[str, list[float]],
) -> T.Any:
    """Fuse one face's cached predictions using a candidate's strategy + weights."""
    import numpy as np

    from lib.landmarks.schema import LandmarkPrediction

    predictions = [
        LandmarkPrediction(np.asarray(points, dtype="float32"), source=model)
        for points, model in zip(cached_points, candidate.models, strict=True)
    ]
    method = strategy_outlier_method(candidate.strategy)
    threshold = candidate.outlier_threshold if strategy_uses_threshold(candidate.strategy) else 3.5

    if not strategy_requires_weights(candidate.strategy):
        return plain_average(
            predictions, outlier_method=method, outlier_threshold=threshold
        ).points
    matrix = weights_matrix_for_models(weights, candidate.models)
    if candidate.strategy == "weighted_median":
        stack = np.stack([prediction.canonical_68().points for prediction in predictions], axis=0)
        normalized = normalize_weight_matrix(
            matrix, model_count=stack.shape[0], landmark_count=stack.shape[1]
        )
        return weighted_median(stack, normalized)
    return static_weighted(
        predictions,
        matrix,
        outlier_method=method,
        outlier_threshold=threshold,
    ).points


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
    """Compute the report-split ProfileAggregate for one CandidateResult.

    Cache-only: reuses the same DiskPredictionCache and never invokes adapters.
    Skips samples without a usable face bbox (those samples are also skipped
    by the AFLW CLI tool, so the behavior matches end-to-end).
    """
    import numpy as np

    per_sample: list[T.Any] = []
    for sample in samples:
        bbox = sample.face_bbox
        if bbox is None:
            try:
                truth = np.load(sample.landmarks).astype("float32")
            except OSError:
                continue
            left, top = np.min(truth, axis=0)
            right, bottom = np.max(truth, axis=0)
            bbox = (float(left), float(top), float(right), float(bottom))
        truth = np.load(sample.landmarks).astype("float32")
        cached_points = [
            cache.read(sample.sample_id, model).landmarks for model in result.candidate.models
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
    """Translate CLI flags into a :class:`GateConfig`."""
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
    """Return True when we should pre-compute geometry metrics for every candidate."""
    return bool(config.requires_geometry() or args.include_geometry_metrics)


def _candidate_geometry_aggregate(
    result: CandidateResult,
    *,
    samples: T.Sequence[T.Any],
    cache: DiskPredictionCache,
    aligned_size: int,
    region_failure_threshold: float,
    truth_summaries: dict[str, T.Any] | None = None,
    truth_landmarks: dict[str, T.Any] | None = None,
) -> GeometryAggregate:
    """Compute the report-split GeometryAggregate for one CandidateResult.

    Cache-only: reuses cached predictions and fuses with the candidate's
    strategy + weights. The GT side runs Faceswap's own ``AlignedFace`` so the
    measurements are exactly what extract would see at runtime.

    Callers that evaluate many candidates against the same report split can
    pass ``truth_summaries`` / ``truth_landmarks`` keyed by ``sample_id`` to
    avoid re-reading the GT npy and re-running ``AlignedFace`` once per
    candidate. Absent caches fall back to per-call IO.
    """
    import numpy as np

    truth_summaries = truth_summaries or {}
    truth_landmarks_cache = truth_landmarks or {}
    per_sample: list[T.Any] = []
    for sample in samples:
        bbox = sample.face_bbox
        truth = truth_landmarks_cache.get(sample.sample_id)
        if truth is None:
            try:
                truth = np.load(sample.landmarks).astype("float32")
            except OSError:
                continue
        if bbox is None:
            left, top = np.min(truth, axis=0)
            right, bottom = np.max(truth, axis=0)
            bbox = (float(left), float(top), float(right), float(bottom))
        cached_points = [
            cache.read(sample.sample_id, model).landmarks for model in result.candidate.models
        ]
        fused = _fuse_for_profile(result.candidate, cached_points, weights=result.weights)
        per_sample.append(
            evaluate_geometry_sample(
                fused,
                truth,
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                condition=sample.condition,
                bbox=bbox,
                visibility=sample.visibility,
                aligned_size=aligned_size,
                region_failure_threshold=region_failure_threshold,
                truth_summary=truth_summaries.get(sample.sample_id),
            )
        )
    return aggregate_geometry_samples(result.candidate_id, per_sample)


def _geometry_score_from_aggregate(
    aggregate: GeometryAggregate, baseline_score: float | None
) -> GeometryScore:
    """Pack a GeometryAggregate into the per-candidate score the gates consume."""
    if aggregate.per_bucket:
        bucket_scores = [
            float(values.get("overall_score", 0.0)) for values in aggregate.per_bucket.values()
        ]
        max_bucket = max(bucket_scores) if bucket_scores else 0.0
    else:
        max_bucket = 0.0
    max_bucket_regression = (
        max(0.0, max_bucket - baseline_score) if baseline_score is not None else 0.0
    )
    return GeometryScore(
        overall_score=aggregate.overall_score,
        catastrophic_failure_rate=aggregate.catastrophic_failure_rate,
        p95_translation_normalized=aggregate.p95_translation_normalized,
        p95_roi_center_normalized=aggregate.p95_roi_center_normalized,
        p95_roll_degrees=aggregate.p95_roll_degrees_delta,
        mean_hull_iou=aggregate.mean_hull_iou,
        p05_hull_iou=aggregate.p05_hull_iou,
        max_bucket_regression_score=max_bucket_regression,
    )


def _enumerate_search_candidates(args: argparse.Namespace) -> list[Candidate]:
    """Enumerate candidate setups and print a compact search-space summary."""
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
        f"Enumerated {len(candidates)} candidates "
        f"from models={args.models}, subsets={args.model_subsets}, "
        f"generators={args.weight_generators}, strategies={args.strategies}"
    )
    return candidates


def _write_candidate_results(
    output_dir: Path,
    results: T.Sequence[CandidateResult],
    *,
    objective: str,
    regression_epsilon_nme: float,
) -> tuple[Path, Path]:
    """Persist the full candidate evaluation log as CSV + JSON."""
    csv_path = output_dir / "candidate_results.csv"
    json_path = output_dir / "candidate_results.json"
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
    rows: list[dict[str, T.Any]] = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "candidate_id": result.candidate_id,
                "score": result.score,
                "objective": result.objective,
                "models": "|".join(result.candidate.models),
                "weight_generator": result.candidate.weight_generator,
                "strategy": result.candidate.strategy,
                "outlier_threshold": (
                    ""
                    if result.candidate.outlier_threshold is None
                    else result.candidate.outlier_threshold
                ),
                "overall_nme": result.metrics.overall_nme,
                "failure_rate": result.metrics.failure_rate,
                "auc": result.metrics.auc,
                "regression_rate_vs_best_single": result.metrics.regression_rate_vs_best_single,
                "bucket_regression_rate_vs_best_single": result.metrics.bucket_regression_rate_vs_best_single,
                "best_single_model": result.metrics.best_single_model,
                "weights_hash": result.weights_hash,
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_payload = {
        "objective": objective,
        "regression_epsilon_nme": regression_epsilon_nme,
        "candidates": [result.to_payload() for result in results],
    }
    json_path.write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    """Write a short human-readable Markdown summary of the promotion decision."""
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
        f"- report_bucket_regression_rate: "
        f"`{report_metrics.get('bucket_regression_rate_vs_best_single', 0.0):.6f}`",
    ]
    diagnostics = winner.effective_ensemble
    if diagnostics is not None:
        lines.extend(
            [
                "",
                "## Effective ensemble diagnostics",
                "",
                f"- mean_effective_models: `{diagnostics.mean_effective_models:.3f}` "
                f"(floor `{diagnostics.effective_models_floor:.3f}`)",
                f"- collapsed: `{diagnostics.collapsed}`",
                f"- weighted_median_collapsed: `{diagnostics.weighted_median_collapsed}`",
                "- landmark share by model: "
                + ", ".join(
                    f"{model}={share:.2f}"
                    for model, share in sorted(
                        diagnostics.landmark_share_by_model.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ),
            ]
        )
        if winner.is_single_model_baseline:
            lines.append(
                "- note: this winner is a single-model baseline; promotion of "
                "single-model setups was explicitly allowed."
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
                    f"- profile_region_failure_rate: "
                    f"`{profile_aggregate.region_failure_rate:.6f}`",
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
    """Write best setup, best weights, and the human-readable promotion report."""
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
) -> Path:
    """Write ``no_promotion.json`` when no candidate satisfies the configured gates."""
    payload = no_promotion_payload(application)
    payload["objective"] = args.objective
    payload["evaluated_candidates"] = len(results)
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
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--splits", required=True, help="Path to splits.json (#67).")
    parser.add_argument("--models", default="hrnet,spiga,orformer")
    parser.add_argument(
        "--model-subsets",
        default="all",
        help="Comma-separated subset presets: 'all', 'pairs', and/or 'triples'.",
    )
    parser.add_argument(
        "--weight-generators",
        default="equal,inverse_mean_error,regularized_inverse_error",
    )
    parser.add_argument(
        "--strategies",
        default="static_weighted,static_weighted_downweight,weighted_median",
    )
    parser.add_argument("--outlier-thresholds", default="2.5,3.5,4.5")
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    parser.add_argument(
        "--regression-epsilon-nme",
        type=float,
        default=DEFAULT_REGRESSION_EPSILON_NME,
    )
    parser.add_argument("--bbox-source", default="manifest")
    parser.add_argument("--crop-scale", type=float, default=1.6)
    parser.add_argument("--failure-threshold", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars while keeping stage timing output.",
    )
    parser.add_argument(
        "--include-single-model-baselines",
        action="store_true",
        help=(
            "Add one plain_average candidate per model so promotion can compare the "
            "ensemble against the obvious single-model alternative. Auto-enabled when "
            "any gate that needs a baseline is configured."
        ),
    )
    parser.add_argument(
        "--allow-single-model-baselines",
        action="store_true",
        help="Allow a single-model baseline candidate to be promoted (off by default).",
    )
    parser.add_argument("--require-report-improvement", action="store_true")
    parser.add_argument(
        "--report-improvement-tolerance",
        type=float,
        default=DEFAULT_REPORT_IMPROVEMENT_TOLERANCE,
        help="Tolerance (in NME units) when comparing candidate report NME to baseline.",
    )
    parser.add_argument("--max-overall-regression-nme", type=float, default=None)
    parser.add_argument("--max-bucket-regression-rate", type=float, default=None)
    parser.add_argument("--require-profile-improvement", action="store_true")
    parser.add_argument("--max-profile-region-failure-rate", type=float, default=None)
    parser.add_argument("--require-effective-ensemble", action="store_true")
    parser.add_argument("--effective-models-floor", type=float, default=1.5)
    parser.add_argument(
        "--profile-normalizer",
        choices=NORMALIZERS,
        default=DEFAULT_NORMALIZER,
    )
    parser.add_argument(
        "--profile-region-failure-threshold",
        type=float,
        default=DEFAULT_REGION_FAILURE_THRESHOLD,
    )
    parser.add_argument(
        "--profile-pck-thresholds",
        default=",".join(f"{t:.2f}" for t in DEFAULT_PCK_THRESHOLDS),
    )
    parser.add_argument(
        "--profile-priority-regions",
        default=",".join(DEFAULT_PRIORITY_FAILURE_REGIONS),
    )
    parser.add_argument(
        "--include-geometry-metrics",
        action="store_true",
        help=(
            "Compute GT-derived alignment-geometry metrics (#76) for every candidate on the "
            "report split. Auto-enabled when any geometry gate is configured."
        ),
    )
    parser.add_argument(
        "--geometry-aligned-size",
        type=int,
        default=512,
        help="Pixel size used when running AlignedFace for geometry evaluation.",
    )
    parser.add_argument(
        "--geometry-region-failure-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument("--require-geometry-improvement", action="store_true")
    parser.add_argument("--max-catastrophic-geometry-failure-rate", type=float, default=None)
    parser.add_argument("--max-p95-transform-error", type=float, default=None)
    parser.add_argument("--max-p95-crop-center-error", type=float, default=None)
    parser.add_argument("--max-p95-roll-error", type=float, default=None)
    parser.add_argument("--min-hull-iou", type=float, default=None)
    parser.add_argument("--max-hard-slice-regression-rate", type=float, default=None)
    parser.add_argument(
        "--allow-nme-only-promotion",
        action="store_true",
        help=(
            "Allow promotion based on NME-shaped objectives when geometry metrics are absent. "
            "Off by default once any geometry gate is configured."
        ),
    )
    args = parser.parse_args(argv)
    # Resolve dependent defaults: any geometry gate implies geometry computation
    # and disables the NME-only promotion escape unless the user explicitly opts in.
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
        # Geometry objective implies geometry-side enforcement.
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
    if _gates_need_geometry(gate_config, args):

        def _geometry_eval() -> None:
            # Precompute the GT-side AlignedFace summary + raw landmarks for
            # every report-split sample so every candidate evaluated below
            # reuses them. Without this we re-read the GT npy and rebuild
            # AlignedFace once per (sample, candidate) pair.
            import numpy as np

            from lib.landmarks.eval.geometry_signals import alignment_summary

            truth_landmarks_cache: dict[str, T.Any] = {}
            truth_summaries: dict[str, T.Any] = {}
            for sample in report_samples:
                try:
                    truth = np.load(sample.landmarks).astype("float32")
                except OSError:
                    continue
                truth_landmarks_cache[sample.sample_id] = truth
                truth_summaries[sample.sample_id] = alignment_summary(
                    truth, size=args.geometry_aligned_size
                )

            interim: dict[str, GeometryAggregate] = {}
            for result in results:
                aggregate = _candidate_geometry_aggregate(
                    result,
                    samples=report_samples,
                    cache=cache,
                    aligned_size=args.geometry_aligned_size,
                    region_failure_threshold=args.geometry_region_failure_threshold,
                    truth_summaries=truth_summaries,
                    truth_landmarks=truth_landmarks_cache,
                )
                interim[result.candidate_id] = aggregate
                geometry_aggregates[result.candidate_id] = aggregate
            # Resolve baseline (best single model) to populate hard-slice regression scores.
            baseline_score: float | None = None
            for result in results:
                if not result.is_single_model_baseline:
                    continue
                aggregate = interim[result.candidate_id]
                if baseline_score is None or aggregate.overall_score < baseline_score:
                    baseline_score = aggregate.overall_score
            for candidate_id, aggregate in interim.items():
                geometry_scores[candidate_id] = _geometry_score_from_aggregate(
                    aggregate, baseline_score
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

    # When the operator explicitly asks for alignment_geometry_v1 the
    # ranking must come from geometry scores, not the NME-shaped objective
    # that run_candidate_search returns sorted by. Re-rank before writing
    # the candidate log so the CSV / JSON / promotion path all see the
    # geometry-driven order.
    if args.objective == GEOMETRY_OBJECTIVE:
        if geometry_scores:
            _progress(
                f"Re-ranking {len(results)} candidates by alignment_geometry_v1 "
                "score before promotion"
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
                "--objective alignment_geometry_v1 selected but geometry metrics were not "
                "computed. Pass --include-geometry-metrics (or any geometry gate) to enable "
                "geometry-based ranking, or set --allow-nme-only-promotion explicitly to "
                "keep the legacy NME ranking."
            )
        else:
            _progress(
                "alignment_geometry_v1 objective set but --allow-nme-only-promotion is on; "
                "falling back to NME ranking."
            )

    csv_path, json_path = _stage(
        "write_candidate_results",
        lambda: _write_candidate_results(
            output_dir,
            results,
            objective=args.objective,
            regression_epsilon_nme=args.regression_epsilon_nme,
        ),
    )

    gate_application: GateApplication | None = None
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
                ),
            )
            print(f"No candidate passed the configured promotion gates; see {no_promotion_path}")
            return 1
    else:
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
        f"Promoted candidate {winner.candidate_id} "
        f"(strategy={winner.candidate.strategy}, score={winner.score:.6f})"
    )
    print(f"  setup:   {setup_path}")
    print(f"  weights: {weights_path}")
    print(f"  csv:     {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
