#!/usr/bin/env python3
"""Production-validation promotion gate for landmark ensemble policies.

Standalone validation CLI retained for repeatable gate checks. Candidate for a
thin facade over the unified resolver pipeline once promotion flows are fully
centralized.
"""

from __future__ import annotations

import csv
import json
import typing as T
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_RESOLVER_CANDIDATES,
    SampleCandidateContext,
    load_contexts,
)
from lib.landmarks.ensemble.weights import load_weights
from lib.landmarks.pipeline_conventions import (
    PRODUCTION_PER_BUCKET_CSV,
    PRODUCTION_POLICY_FAILURES_CSV,
    PRODUCTION_PROMOTION_REPORT_JSON,
    PRODUCTION_PROMOTION_REPORT_MD,
    PRODUCTION_WORST_SAMPLES_JSON,
    write_json,
)

DEFAULT_PRODUCTION_MEAN_EPSILON_NME: float = 0.001
DEFAULT_PRODUCTION_P90_EPSILON_NME: float = 0.003
DEFAULT_PRODUCTION_FAILURE_THRESHOLD: float = 0.08
DEFAULT_WORST_SAMPLE_COUNT: int = 25
DEFAULT_MIN_HARD_BUCKET_GATE_COUNT: int = 20
PRODUCTION_REPORT_JSON = PRODUCTION_PROMOTION_REPORT_JSON
PRODUCTION_REPORT_MD = PRODUCTION_PROMOTION_REPORT_MD


@dataclass(frozen=True)
class ProductionGateConfig:
    """Thresholds and policy for production promotion validation."""

    policy: str = "bucket_aware_veto"
    mean_epsilon_nme: float = DEFAULT_PRODUCTION_MEAN_EPSILON_NME
    p90_epsilon_nme: float = DEFAULT_PRODUCTION_P90_EPSILON_NME
    failure_threshold: float = DEFAULT_PRODUCTION_FAILURE_THRESHOLD
    failure_rate_epsilon: float = 0.0
    worst_sample_count: int = DEFAULT_WORST_SAMPLE_COUNT
    outlier_threshold: float = 3.5
    min_hard_bucket_gate_count: int = DEFAULT_MIN_HARD_BUCKET_GATE_COUNT


@dataclass(frozen=True)
class ProductionSampleEvaluation:
    """Per-sample production validation result."""

    sample_id: str
    condition: str
    chosen: str
    oracle: str
    nme_by_candidate: dict[str, float]
    failure_by_candidate: dict[str, bool]
    missing_runtime_metadata: bool
    runtime_bucket_source: str


def _raw_manifest_metadata(path: Path) -> dict[str, dict[str, T.Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    retval: dict[str, dict[str, T.Any]] = {}
    for entry in payload.get("samples", payload.get("scenarios", [])):
        sample_id = str(entry.get("sample_id") or entry.get("id") or entry.get("name"))
        metadata = entry.get("metadata", {})
        retval[sample_id] = metadata if isinstance(metadata, dict) else {}
    return retval


def _runtime_metadata(metadata: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    resolver = metadata.get("landmark_ensemble")
    return resolver if isinstance(resolver, dict) else {}


def _has_runtime_metadata(metadata: T.Mapping[str, T.Any]) -> bool:
    resolver = _runtime_metadata(metadata)
    return bool(
        resolver.get("runtime_bucket")
        and resolver.get("bucket")
        and resolver.get("selected_candidate")
    )


def _resolve_policy_choice(
    policy: str,
    *,
    metadata: T.Mapping[str, T.Any],
    context: SampleCandidateContext,
) -> str:
    if policy.startswith("candidate:"):
        chosen = policy.split(":", 1)[1]
        if chosen not in context.nme_by_candidate:
            raise ValueError(f"production policy requested unknown candidate {chosen!r}")
        return chosen
    if policy == "manifest_selected_candidate":
        chosen = str(_runtime_metadata(metadata).get("selected_candidate", ""))
        if chosen not in context.nme_by_candidate:
            raise ValueError(
                f"manifest selected_candidate {chosen!r} is not in evaluated candidates"
            )
        return chosen
    if policy not in {"bucket_aware_veto", "roll_aware_veto"}:
        raise ValueError(
            "unknown production policy "
            f"{policy!r}; use bucket_aware_veto, roll_aware_veto, "
            "manifest_selected_candidate, or candidate:<name>"
        )
    return context.current_policy_choice


def _summarize(values: T.Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return {"mean": 0.0, "p90": 0.0}
    return {"mean": float(arr.mean()), "p90": float(np.percentile(arr, 90))}


def _candidate_summary(
    evaluations: T.Sequence[ProductionSampleEvaluation],
    candidate: str,
) -> dict[str, float]:
    values = [row.nme_by_candidate[candidate] for row in evaluations]
    summary = _summarize(values)
    failures = [row.failure_by_candidate[candidate] for row in evaluations]
    summary["failure_rate"] = float(sum(failures) / len(failures)) if failures else 0.0
    return summary


def _chosen_summary(evaluations: T.Sequence[ProductionSampleEvaluation]) -> dict[str, float]:
    values = [row.nme_by_candidate[row.chosen] for row in evaluations]
    summary = _summarize(values)
    failures = [row.failure_by_candidate[row.chosen] for row in evaluations]
    summary["failure_rate"] = float(sum(failures) / len(failures)) if failures else 0.0
    return summary


def _best_single(
    evaluations: T.Sequence[ProductionSampleEvaluation],
    models: T.Sequence[str],
) -> tuple[str, dict[str, float]]:
    summaries = {model: _candidate_summary(evaluations, model) for model in models}
    best = min(
        models,
        key=lambda model: (
            summaries[model]["mean"],
            summaries[model]["p90"],
            summaries[model]["failure_rate"],
            model,
        ),
    )
    return best, summaries[best]


def _oracle_summary(evaluations: T.Sequence[ProductionSampleEvaluation]) -> dict[str, float]:
    values = [row.nme_by_candidate[row.oracle] for row in evaluations]
    summary = _summarize(values)
    failures = [row.failure_by_candidate[row.oracle] for row in evaluations]
    summary["failure_rate"] = float(sum(failures) / len(failures)) if failures else 0.0
    return summary


def _per_bucket(
    evaluations: T.Sequence[ProductionSampleEvaluation],
    *,
    models: T.Sequence[str],
    static_candidate: str,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[ProductionSampleEvaluation]] = defaultdict(list)
    for row in evaluations:
        buckets[row.condition or "unknown"].append(row)
    payload: dict[str, dict[str, float]] = {}
    for bucket, rows in sorted(buckets.items()):
        chosen = _chosen_summary(rows)
        best_single_name, best_single = _best_single(rows, models)
        static = _candidate_summary(rows, static_candidate)
        payload[bucket] = {
            "sample_count": float(len(rows)),
            "mean_nme": chosen["mean"],
            "p90_nme": chosen["p90"],
            "failure_rate": chosen["failure_rate"],
            "best_single_candidate": best_single_name,
            "best_single_mean_nme": best_single["mean"],
            "static_downweight_mean_nme": static["mean"],
            "regression_vs_best_single": chosen["mean"] - best_single["mean"],
            "regression_vs_static_downweight": chosen["mean"] - static["mean"],
            "failure_regression_vs_best_single": (
                chosen["failure_rate"] - best_single["failure_rate"]
            ),
            "failure_regression_vs_static_downweight": (
                chosen["failure_rate"] - static["failure_rate"]
            ),
        }
    return payload


def _hard_bucket(bucket: str) -> bool:
    return "profile" in bucket or "large_yaw" in bucket


def _gate_failures(
    report: dict[str, T.Any],
    *,
    config: ProductionGateConfig,
) -> list[str]:
    failures: list[str] = []
    warnings: list[str] = []
    chosen = report["chosen_policy"]
    best = report["best_single"]
    static = report["static_downweight"]
    if chosen["mean_nme"] > best["mean_nme"] + config.mean_epsilon_nme:
        failures.append("chosen_policy_mean_nme_regresses_vs_best_single")
    if chosen["p90_nme"] > best["p90_nme"] + config.p90_epsilon_nme:
        failures.append("chosen_policy_p90_nme_regresses_vs_best_single")
    if chosen["failure_rate"] > best["failure_rate"] + config.failure_rate_epsilon:
        failures.append("chosen_policy_failure_rate_regresses_vs_best_single")
    if chosen["mean_nme"] > static["mean_nme"] + config.mean_epsilon_nme:
        failures.append("chosen_policy_mean_nme_regresses_vs_static_downweight")
    if chosen["failure_rate"] > static["failure_rate"] + config.failure_rate_epsilon:
        failures.append("chosen_policy_failure_rate_regresses_vs_static_downweight")
    if report["missing_runtime_metadata_count"] > 0:
        failures.append("missing_production_runtime_metadata")
    if report["derived_no_image_runtime_metadata_count"] > 0:
        failures.append("production_runtime_bucket_source_derived_no_image_evidence")
    for bucket, metrics in report["per_bucket"].items():
        if not _hard_bucket(bucket):
            continue
        sample_count = int(metrics["sample_count"])
        if sample_count < config.min_hard_bucket_gate_count:
            warnings.append(
                "bucket_"
                f"{bucket}_sample_count_{sample_count}_below_gate_min_"
                f"{config.min_hard_bucket_gate_count}"
            )
            continue
        if metrics["regression_vs_best_single"] > config.mean_epsilon_nme:
            failures.append(f"bucket_{bucket}_mean_regresses_vs_best_single")
        if metrics["regression_vs_static_downweight"] > config.mean_epsilon_nme:
            failures.append(f"bucket_{bucket}_mean_regresses_vs_static_downweight")
        if metrics["failure_regression_vs_best_single"] > config.failure_rate_epsilon:
            failures.append(f"bucket_{bucket}_failure_regresses_vs_best_single")
        if metrics["failure_regression_vs_static_downweight"] > config.failure_rate_epsilon:
            failures.append(f"bucket_{bucket}_failure_regresses_vs_static_downweight")
    report["warnings"] = warnings
    return failures


def evaluate_production_gate(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    config: ProductionGateConfig,
) -> tuple[dict[str, T.Any], list[ProductionSampleEvaluation]]:
    """Evaluate a production manifest and return gate report plus per-sample rows."""
    raw_metadata = _raw_manifest_metadata(manifest_path)
    weights = load_weights(weights_path)
    models = tuple(weights)
    if not models:
        raise ValueError("production weights did not contain any model weights")
    requested_candidates = tuple(DEFAULT_RESOLVER_CANDIDATES)
    contexts = load_contexts(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        candidates=requested_candidates,
        failure_threshold=config.failure_threshold,
        outlier_threshold=config.outlier_threshold,
    )
    missing_current = [
        context.sample_id for context in contexts if context.selected_candidate_missing_from_eval
    ]
    if missing_current:
        raise ValueError(
            "current runtime policy selected candidates missing from production evaluation set for "
            f"{len(missing_current)} sample(s): {missing_current[:10]}"
        )
    evaluations: list[ProductionSampleEvaluation] = []
    for context in contexts:
        metadata = raw_metadata.get(context.sample_id, {})
        chosen = _resolve_policy_choice(
            config.policy,
            metadata=metadata,
            context=context,
        )
        evaluations.append(
            ProductionSampleEvaluation(
                sample_id=context.sample_id,
                condition=context.condition or "unknown",
                chosen=chosen,
                oracle=context.oracle,
                nme_by_candidate=context.nme_by_candidate,
                failure_by_candidate=context.failure_by_candidate,
                missing_runtime_metadata=not _has_runtime_metadata(metadata),
                runtime_bucket_source=context.runtime_bucket_source,
            )
        )
    if not evaluations:
        raise ValueError("production manifest produced zero evaluable samples")

    best_single_name, best_single = _best_single(evaluations, models)
    static_summaries = {
        candidate: _candidate_summary(evaluations, candidate)
        for candidate in ("static_weighted", "static_weighted_downweight")
    }
    best_static_name = min(
        static_summaries,
        key=lambda name: (
            static_summaries[name]["mean"],
            static_summaries[name]["p90"],
            static_summaries[name]["failure_rate"],
            name,
        ),
    )
    static = _candidate_summary(evaluations, "static_weighted_downweight")
    chosen = _chosen_summary(evaluations)
    oracle = _oracle_summary(evaluations)
    pick_counts = Counter(row.chosen for row in evaluations)
    oracle_matches = sum(1 for row in evaluations if row.chosen == row.oracle)
    gaps = [
        row.nme_by_candidate[row.chosen] - row.nme_by_candidate[row.oracle] for row in evaluations
    ]
    per_bucket = _per_bucket(
        evaluations,
        models=models,
        static_candidate="static_weighted_downweight",
    )
    report: dict[str, T.Any] = {
        "status": "pending",
        "production_sample_count": len(evaluations),
        "production_condition_counts": dict(Counter(row.condition for row in evaluations)),
        "missing_runtime_metadata_count": sum(
            1 for row in evaluations if row.missing_runtime_metadata
        ),
        "derived_no_image_runtime_metadata_count": sum(
            1 for row in evaluations if row.runtime_bucket_source == "derived_no_image_evidence"
        ),
        "best_single": {
            "candidate": best_single_name,
            "mean_nme": best_single["mean"],
            "p90_nme": best_single["p90"],
            "failure_rate": best_single["failure_rate"],
        },
        "best_single_candidate": best_single_name,
        "best_single_mean_nme": best_single["mean"],
        "best_single_p90_nme": best_single["p90"],
        "best_single_failure_rate": best_single["failure_rate"],
        "static_downweight": {
            "mean_nme": static["mean"],
            "p90_nme": static["p90"],
            "failure_rate": static["failure_rate"],
        },
        "best_static_ensemble": {
            "candidate": best_static_name,
            "mean_nme": static_summaries[best_static_name]["mean"],
            "p90_nme": static_summaries[best_static_name]["p90"],
            "failure_rate": static_summaries[best_static_name]["failure_rate"],
        },
        "current_promoted_setup": {
            "candidate": "static_weighted_downweight",
            "mean_nme": static["mean"],
            "p90_nme": static["p90"],
            "failure_rate": static["failure_rate"],
        },
        "static_downweight_mean_nme": static["mean"],
        "static_downweight_p90_nme": static["p90"],
        "static_downweight_failure_rate": static["failure_rate"],
        "chosen_policy": {
            "name": config.policy,
            "mean_nme": chosen["mean"],
            "p90_nme": chosen["p90"],
            "failure_rate": chosen["failure_rate"],
            "pick_counts": dict(pick_counts),
            "oracle_match_rate": oracle_matches / len(evaluations),
            "mean_gap_vs_oracle": float(np.mean(gaps)),
        },
        "chosen_policy_mean_nme": chosen["mean"],
        "chosen_policy_p90_nme": chosen["p90"],
        "chosen_policy_failure_rate": chosen["failure_rate"],
        "chosen_policy_pick_counts": dict(pick_counts),
        "chosen_policy_oracle_match_rate": oracle_matches / len(evaluations),
        "chosen_policy_mean_gap_vs_oracle": float(np.mean(gaps)),
        "oracle": oracle,
        "per_bucket": per_bucket,
        "per_bucket_mean_nme": {
            bucket: values["mean_nme"] for bucket, values in per_bucket.items()
        },
        "per_bucket_p90_nme": {bucket: values["p90_nme"] for bucket, values in per_bucket.items()},
        "per_bucket_failure_rate": {
            bucket: values["failure_rate"] for bucket, values in per_bucket.items()
        },
        "per_bucket_regression_vs_best_single": {
            bucket: values["regression_vs_best_single"] for bucket, values in per_bucket.items()
        },
        "per_bucket_regression_vs_static_downweight": {
            bucket: values["regression_vs_static_downweight"]
            for bucket, values in per_bucket.items()
        },
        "gate_config": {
            "mean_epsilon_nme": config.mean_epsilon_nme,
            "p90_epsilon_nme": config.p90_epsilon_nme,
            "failure_threshold": config.failure_threshold,
            "failure_rate_epsilon": config.failure_rate_epsilon,
            "min_hard_bucket_gate_count": config.min_hard_bucket_gate_count,
            "policy": config.policy,
        },
    }
    failures = _gate_failures(report, config=config)
    report["status"] = "pass" if not failures else "fail"
    report["failed_gates"] = failures
    return report, evaluations


def _write_json(report: dict[str, T.Any], output_dir: Path) -> Path:
    return write_json(output_dir / PRODUCTION_REPORT_JSON, report)


def _write_markdown(report: dict[str, T.Any], output_dir: Path) -> Path:
    path = output_dir / PRODUCTION_REPORT_MD
    lines = [
        "# Production Promotion Gate",
        "",
        f"Status: **{report['status']}**",
        f"Samples: {report['production_sample_count']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Best single | {report['best_single_candidate']} |",
        f"| Best single mean NME | {report['best_single_mean_nme']:.6f} |",
        f"| Static downweight mean NME | {report['static_downweight_mean_nme']:.6f} |",
        f"| Chosen policy mean NME | {report['chosen_policy_mean_nme']:.6f} |",
        f"| Chosen policy p90 NME | {report['chosen_policy_p90_nme']:.6f} |",
        f"| Chosen policy failure rate | {report['chosen_policy_failure_rate']:.6f} |",
        f"| Oracle match rate | {report['chosen_policy_oracle_match_rate']:.6f} |",
        "",
        "## Failed Gates",
        "",
    ]
    if report["failed_gates"]:
        lines.extend(f"- {gate}" for gate in report["failed_gates"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if report.get("warnings"):
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Pick Counts", ""])
    for candidate, count in sorted(report["chosen_policy_pick_counts"].items()):
        lines.append(f"- {candidate}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_bucket_csv(report: dict[str, T.Any], output_dir: Path) -> Path:
    path = output_dir / PRODUCTION_PER_BUCKET_CSV
    fieldnames = [
        "bucket",
        "sample_count",
        "mean_nme",
        "p90_nme",
        "failure_rate",
        "best_single_candidate",
        "best_single_mean_nme",
        "static_downweight_mean_nme",
        "regression_vs_best_single",
        "regression_vs_static_downweight",
        "failure_regression_vs_best_single",
        "failure_regression_vs_static_downweight",
        "gate_enforced",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bucket, values in report["per_bucket"].items():
            writer.writerow(
                {
                    "bucket": bucket,
                    **values,
                    "gate_enforced": int(
                        not _hard_bucket(bucket)
                        or int(values["sample_count"])
                        >= int(report["gate_config"]["min_hard_bucket_gate_count"])
                    ),
                }
            )
    return path


def _write_policy_failures(
    evaluations: T.Sequence[ProductionSampleEvaluation],
    output_dir: Path,
) -> Path:
    path = output_dir / PRODUCTION_POLICY_FAILURES_CSV
    fieldnames = [
        "sample_id",
        "condition",
        "chosen",
        "chosen_nme",
        "chosen_failure",
        "oracle",
        "oracle_nme",
        "gap_vs_oracle",
        "missing_runtime_metadata",
        "runtime_bucket_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in evaluations:
            chosen_nme = row.nme_by_candidate[row.chosen]
            oracle_nme = row.nme_by_candidate[row.oracle]
            if (
                not row.failure_by_candidate[row.chosen]
                and chosen_nme <= oracle_nme
                and not row.missing_runtime_metadata
                and row.runtime_bucket_source != "derived_no_image_evidence"
            ):
                continue
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "condition": row.condition,
                    "chosen": row.chosen,
                    "chosen_nme": chosen_nme,
                    "chosen_failure": int(row.failure_by_candidate[row.chosen]),
                    "oracle": row.oracle,
                    "oracle_nme": oracle_nme,
                    "gap_vs_oracle": chosen_nme - oracle_nme,
                    "missing_runtime_metadata": int(row.missing_runtime_metadata),
                    "runtime_bucket_source": row.runtime_bucket_source,
                }
            )
    return path


def _write_worst_samples(
    evaluations: T.Sequence[ProductionSampleEvaluation],
    output_dir: Path,
    *,
    count: int,
) -> Path:
    path = output_dir / PRODUCTION_WORST_SAMPLES_JSON
    worst = sorted(
        evaluations,
        key=lambda row: row.nme_by_candidate[row.chosen] - row.nme_by_candidate[row.oracle],
        reverse=True,
    )[:count]
    payload = {
        "samples": [
            {
                "sample_id": row.sample_id,
                "condition": row.condition,
                "chosen": row.chosen,
                "chosen_nme": row.nme_by_candidate[row.chosen],
                "oracle": row.oracle,
                "oracle_nme": row.nme_by_candidate[row.oracle],
                "gap_vs_oracle": row.nme_by_candidate[row.chosen]
                - row.nme_by_candidate[row.oracle],
                "missing_runtime_metadata": row.missing_runtime_metadata,
                "runtime_bucket_source": row.runtime_bucket_source,
            }
            for row in worst
        ]
    }
    return write_json(path, payload)


def write_production_gate_artifacts(
    report: dict[str, T.Any],
    evaluations: T.Sequence[ProductionSampleEvaluation],
    output_dir: Path,
    *,
    worst_sample_count: int,
) -> dict[str, str]:
    """Write all production promotion gate artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": _write_json(report, output_dir),
        "markdown": _write_markdown(report, output_dir),
        "per_bucket_csv": _write_bucket_csv(report, output_dir),
        "policy_failures_csv": _write_policy_failures(evaluations, output_dir),
        "worst_samples_json": _write_worst_samples(
            evaluations,
            output_dir,
            count=worst_sample_count,
        ),
    }
    return {key: str(path) for key, path in paths.items()}


def run_production_promotion_gate(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    output_dir: Path,
    config: ProductionGateConfig | None = None,
) -> dict[str, T.Any]:
    """Evaluate production validation and write promotion-gate artifacts."""
    gate_config = config or ProductionGateConfig()
    report, evaluations = evaluate_production_gate(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        weights_path=weights_path,
        config=gate_config,
    )
    report["artifacts"] = write_production_gate_artifacts(
        report,
        evaluations,
        output_dir,
        worst_sample_count=gate_config.worst_sample_count,
    )
    _write_json(report, output_dir)
    return report


__all__ = [
    "DEFAULT_MIN_HARD_BUCKET_GATE_COUNT",
    "DEFAULT_PRODUCTION_FAILURE_THRESHOLD",
    "DEFAULT_PRODUCTION_MEAN_EPSILON_NME",
    "DEFAULT_PRODUCTION_P90_EPSILON_NME",
    "PRODUCTION_REPORT_JSON",
    "PRODUCTION_REPORT_MD",
    "ProductionGateConfig",
    "ProductionSampleEvaluation",
    "evaluate_production_gate",
    "run_production_promotion_gate",
    "write_production_gate_artifacts",
]
