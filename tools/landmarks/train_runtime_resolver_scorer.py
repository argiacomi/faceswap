#!/usr/bin/env python3
"""Train a candidate-quality scorer for the landmark runtime resolver."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import typing as T
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.ensemble.runtime_resolver_scorer import (
    RuntimeResolverScorer,
    feature_matrix,
    sigmoid,
    write_runtime_resolver_scorer,
)
from lib.landmarks.ensemble.weights import load_weights
from tools.landmarks.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_HIGH_GAP_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    DEFAULT_SCORER_CANDIDATE_CSV,
    CandidateQualityRow,
    candidate_table_rows,
    load_contexts,
    parse_candidates,
    rows_for_context,
    write_candidate_table_csv,
)

logger = logging.getLogger("train_runtime_resolver_scorer")

SCORER_ARTIFACT = "runtime_resolver_scorer.json"
TRAINING_ROWS_CSV = "runtime_resolver_scorer_training_rows.csv"
EVAL_ROWS_CSV = "runtime_resolver_scorer_eval_rows.csv"
TRAINING_CANDIDATE_TABLE_CSV = "candidate_table.csv"
TRAINING_METRICS_JSON = "runtime_resolver_scorer_training_metrics.json"
SOURCE_GT_HARD = "gt_hard"
SOURCE_PRODUCTION_VALIDATED = "production_validated"
TaggedRow = tuple[CandidateQualityRow, str]


def _collect_rows(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
    weights_path: Path,
    candidates: T.Sequence[str],
    failure_threshold: float,
    high_gap_threshold: float,
    outlier_threshold: float,
    allow_image_backfill: bool,
) -> tuple[list[TaggedRow], list[dict[str, T.Any]]]:
    specs = [
        (SOURCE_GT_HARD, gt_manifest, gt_cache_dir),
        (SOURCE_PRODUCTION_VALIDATED, production_manifest, production_cache_dir),
    ]
    rows: list[TaggedRow] = []
    candidate_rows: list[dict[str, T.Any]] = []
    for label, manifest_path, cache_dir in specs:
        if manifest_path is None and cache_dir is None:
            continue
        if manifest_path is None or cache_dir is None:
            raise ValueError(f"{label} manifest/cache inputs must be supplied together")
        logger.info("Loading %s scorer rows from %s", label, manifest_path)
        contexts = load_contexts(
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            weights_path=weights_path,
            candidates=candidates,
            failure_threshold=failure_threshold,
            outlier_threshold=outlier_threshold,
            allow_image_backfill=allow_image_backfill,
        )
        candidate_rows.extend(candidate_table_rows(contexts))
        for context in contexts:
            rows.extend(
                (row, label)
                for row in rows_for_context(context, high_gap_threshold=high_gap_threshold)
            )
    if not rows:
        raise ValueError("no scorer training rows were loaded")
    return rows, candidate_rows


def _split_rows(
    rows: T.Sequence[TaggedRow],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[TaggedRow], list[TaggedRow]]:
    """Split rows by sample while stratifying within source/dataset/condition groups."""
    if eval_fraction < 0.0 or eval_fraction >= 1.0:
        raise ValueError("--eval-fraction must be >= 0.0 and < 1.0")
    groups: dict[tuple[str, str, str], set[str]] = {}
    for row, source in rows:
        groups.setdefault((source, row.dataset, row.condition), set()).add(row.sample_id)

    eval_samples: set[tuple[str, str]] = set()
    for (source, dataset, condition), sample_ids in groups.items():
        ordered = sorted(
            sample_ids,
            key=lambda sample_id: hashlib.sha256(
                "|".join((str(seed), source, dataset, condition, sample_id)).encode("utf-8")
            ).hexdigest(),
        )
        if eval_fraction <= 0.0 or len(ordered) <= 1:
            continue
        eval_count = max(1, int(round(len(ordered) * eval_fraction)))
        eval_count = min(eval_count, len(ordered) - 1)
        eval_samples.update((source, sample_id) for sample_id in ordered[:eval_count])

    train_rows: list[TaggedRow] = []
    eval_rows: list[TaggedRow] = []
    for tagged in rows:
        row, source = tagged
        if (source, row.sample_id) in eval_samples:
            eval_rows.append(tagged)
        else:
            train_rows.append(tagged)
    if not train_rows:
        raise ValueError("scorer split produced no training rows")
    return train_rows, eval_rows


def _untag(rows: T.Sequence[TaggedRow]) -> list[CandidateQualityRow]:
    return [row for row, _source in rows]


def _source_rows(rows: T.Sequence[TaggedRow], source: str) -> list[CandidateQualityRow]:
    return [row for row, row_source in rows if row_source == source]


def _write_tagged_rows_csv(rows: T.Sequence[TaggedRow], path: Path) -> Path:
    """Write scorer rows with an explicit source column for held-out split reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = sorted({name for row, _source in rows for name in row.feature_values})
    base_fieldnames = [
        "sample_id",
        "dataset",
        "condition",
        "candidate_name",
        "candidate_nme",
        "failure_label",
        "is_oracle",
        "was_selected_by_current_policy",
        "gap_vs_oracle",
        "runtime_bucket",
        "runtime_bucket_source",
        "risk_route",
        "geometry_veto_reasons",
        "selected_by_current_policy",
        "selected_candidate_missing_from_eval",
        "oracle",
        "features_json",
    ]
    fieldnames = ["source", *base_fieldnames, *feature_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, source in rows:
            writer.writerow(
                {
                    "source": source,
                    **row.to_csv_row(),
                    **{name: row.feature_values.get(name, 0.0) for name in feature_names},
                }
            )
    return path


def _feature_order(rows: T.Sequence[CandidateQualityRow]) -> tuple[str, ...]:
    names: set[str] = set()
    for row in rows:
        names.update(row.feature_values)
    preferred = [
        "candidate_is_single_model",
        "candidate_is_fusion",
        "cloud_area_ratio",
        "hull_area_ratio",
        "points_outside_expanded_bbox_fraction",
        "eye_mouth_order_valid_after_deroll",
        "roi_center_consensus_distance",
        "landmark_consensus_distance",
        "roll_degrees",
        "yaw_degrees",
        "roll_delta_to_consensus",
        "yaw_delta_to_consensus",
        "candidate_yaw_disagreement",
        "max_disagreement_px",
        "has_geometry_veto",
    ]
    ordered = [name for name in preferred if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return tuple(ordered)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return float(np.log(clipped / (1.0 - clipped)))


def _fit_logistic(
    x_raw: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    learning_rate: float,
    iterations: int,
) -> tuple[np.ndarray, float]:
    if x_raw.shape[0] == 0:
        raise ValueError("cannot train scorer on an empty matrix")
    positive_rate = float(np.mean(y))
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return np.zeros(x_raw.shape[1], dtype="float64"), _logit(positive_rate)

    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x = (x_raw - mean) / std
    coef = np.zeros(x.shape[1], dtype="float64")
    intercept = _logit(positive_rate)
    n_rows = float(x.shape[0])
    for _ in range(iterations):
        linear = x @ coef + intercept
        pred = np.asarray([sigmoid(float(item)) for item in linear], dtype="float64")
        error = pred - y
        grad_coef = (x.T @ error) / n_rows + (l2 * coef)
        grad_intercept = float(np.mean(error))
        coef -= learning_rate * grad_coef
        intercept -= learning_rate * grad_intercept
    raw_coef = coef / std
    raw_intercept = float(intercept - np.sum((coef * mean) / std))
    return raw_coef, raw_intercept


def _metrics(
    scorer: RuntimeResolverScorer,
    rows: T.Sequence[CandidateQualityRow],
) -> dict[str, T.Any]:
    if not rows:
        return {
            "row_count": 0,
            "positive_count": 0,
            "positive_rate": 0.0,
            "accuracy_at_0_5": 0.0,
            "log_loss": 0.0,
        }
    labels = np.asarray([float(row.failure_label) for row in rows], dtype="float64")
    scores = np.asarray(
        [scorer.score_feature_map(row.feature_values) for row in rows],
        dtype="float64",
    )
    predicted = scores >= 0.5
    accuracy = float(np.mean(predicted == labels)) if labels.size else 0.0
    loss = -np.mean(
        labels * np.log(np.clip(scores, 1e-8, 1.0))
        + (1.0 - labels) * np.log(np.clip(1.0 - scores, 1e-8, 1.0))
    )
    return {
        "row_count": len(rows),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "accuracy_at_0_5": accuracy,
        "log_loss": float(loss) if labels.size else 0.0,
    }


def train_runtime_resolver_scorer(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
    weights_path: Path,
    candidates: T.Sequence[str],
    output_dir: Path,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    high_gap_threshold: float = DEFAULT_HIGH_GAP_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    l2: float = 0.001,
    learning_rate: float = 0.1,
    iterations: int = 1500,
    eval_fraction: float = 0.20,
    split_seed: int = 42,
    allow_image_backfill: bool = False,
) -> dict[str, T.Any]:
    """Train the scorer and write the portable artifact plus diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tagged_rows, candidate_rows = _collect_rows(
        gt_manifest=gt_manifest,
        gt_cache_dir=gt_cache_dir,
        production_manifest=production_manifest,
        production_cache_dir=production_cache_dir,
        weights_path=weights_path,
        candidates=candidates,
        failure_threshold=failure_threshold,
        high_gap_threshold=high_gap_threshold,
        outlier_threshold=outlier_threshold,
        allow_image_backfill=allow_image_backfill,
    )
    train_tagged_rows, eval_tagged_rows = _split_rows(
        tagged_rows,
        eval_fraction=eval_fraction,
        seed=split_seed,
    )
    rows = _untag(tagged_rows)
    train_rows = _untag(train_tagged_rows)
    eval_rows = _untag(eval_tagged_rows)
    features = _feature_order(train_rows)
    x = feature_matrix([row.feature_values for row in train_rows], features)
    y = np.asarray([float(row.failure_label) for row in train_rows], dtype="float64")
    coefficients, intercept = _fit_logistic(
        x,
        y,
        l2=l2,
        learning_rate=learning_rate,
        iterations=iterations,
    )
    scorer = RuntimeResolverScorer(
        features=features,
        coefficients=tuple(float(item) for item in coefficients),
        intercept=float(intercept),
        failure_threshold=failure_threshold,
        calibration={"type": "none", "params": {}},
    )
    scorer_path = write_runtime_resolver_scorer(scorer, output_dir / SCORER_ARTIFACT)
    rows_path = _write_tagged_rows_csv(train_tagged_rows, output_dir / TRAINING_ROWS_CSV)
    eval_rows_path = _write_tagged_rows_csv(eval_tagged_rows, output_dir / EVAL_ROWS_CSV)
    candidate_table_path = write_candidate_table_csv(
        candidate_rows,
        output_dir / TRAINING_CANDIDATE_TABLE_CSV,
    )
    metrics = _metrics(scorer, rows)
    train_metrics = _metrics(scorer, train_rows)
    eval_metrics = _metrics(scorer, eval_rows)
    production_eval_metrics = _metrics(
        scorer, _source_rows(eval_tagged_rows, SOURCE_PRODUCTION_VALIDATED)
    )
    gt_eval_metrics = _metrics(scorer, _source_rows(eval_tagged_rows, SOURCE_GT_HARD))
    metrics.update(
        {
            "artifact": str(scorer_path),
            "training_rows": str(rows_path),
            "eval_rows": str(eval_rows_path),
            "candidate_table": str(candidate_table_path),
            "candidate_count": len(candidates),
            "candidates": list(candidates),
            "feature_count": len(features),
            "failure_threshold": failure_threshold,
            "high_gap_threshold": high_gap_threshold,
            "l2": l2,
            "split_seed": split_seed,
            "eval_fraction": eval_fraction,
            "allow_image_backfill": allow_image_backfill,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "production_only_eval_metrics": production_eval_metrics,
            "gt_hard_only_eval_metrics": gt_eval_metrics,
        }
    )
    metrics_path = output_dir / TRAINING_METRICS_JSON
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-manifest", type=Path)
    parser.add_argument("--gt-cache-dir", type=Path)
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--production-cache-dir", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        default="",
        help=f"Comma-separated candidate list. Defaults to {DEFAULT_SCORER_CANDIDATE_CSV}.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    parser.add_argument("--high-gap-threshold", type=float, default=DEFAULT_HIGH_GAP_THRESHOLD)
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_THRESHOLD)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--allow-image-backfill",
        action="store_true",
        help="Compute image-aware runtime metadata for rows without stored metadata.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    weights = load_weights(args.weights)
    candidates = parse_candidates(args.candidates, weights)
    metrics = train_runtime_resolver_scorer(
        gt_manifest=args.gt_manifest,
        gt_cache_dir=args.gt_cache_dir,
        production_manifest=args.production_manifest,
        production_cache_dir=args.production_cache_dir,
        weights_path=args.weights,
        candidates=candidates,
        output_dir=args.output_dir,
        failure_threshold=args.failure_threshold,
        high_gap_threshold=args.high_gap_threshold,
        outlier_threshold=args.outlier_threshold,
        l2=args.l2,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        eval_fraction=args.eval_fraction,
        split_seed=args.split_seed,
        allow_image_backfill=args.allow_image_backfill,
    )
    logger.info("Wrote runtime resolver scorer to %s", metrics["artifact"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
