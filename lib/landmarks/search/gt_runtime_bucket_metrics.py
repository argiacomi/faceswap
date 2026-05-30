#!/usr/bin/env python3
"""Production / runtime bucket aggregate metrics for the FULL GT corpus.

Issue #205. The candidate-search machinery already emits per-sample
candidate metrics grouped by ``dataset:scenario`` buckets and the scorer
emits GT-hard runtime-bucket slices. What was missing is a first-class
report for **all GT samples grouped by production/runtime buckets** so
the user can answer:

* Which candidate performs best on full GT data for ``frontal``,
  ``intermediate``, ``large_yaw_left``, ``profile_right``, etc.?
* Does the selected candidate setup regress in production-style buckets
  even if it looks good by dataset/scenario bucket?
* How should ``DEFAULT_PRIORITY`` / ``BUCKET_PRIORITIES`` be updated
  with GT evidence rather than only production-validation evidence?

This module is purely an aggregator: it consumes the canonical candidate
diagnostic table rows (see
:data:`lib.landmarks.ensemble.runtime_resolver_scorer_data.CANDIDATE_TABLE_COLUMNS`)
and emits one ``RuntimeBucketMetrics`` record per ``runtime_bucket``.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import typing as T
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Canonical single-model candidate names. The "best_single" winner per
# bucket is computed restricted to this set so the report can tell the
# user whether an ensemble actually beats the best individual model on
# that production slice.
SINGLE_MODEL_CANDIDATES: frozenset[str] = frozenset({"fan", "hrnet", "spiga", "orformer"})

# Stable column order for the CSV writer: one row per (bucket, candidate).
GT_RUNTIME_BUCKET_CSV_COLUMNS: tuple[str, ...] = (
    "runtime_bucket",
    "candidate",
    "sample_count",
    "mean_nme",
    "p90_nme",
    "failure_rate",
    "is_best_candidate",
    "is_best_single_candidate",
    "is_selected_candidate",
)

# Default tie-break order for ``best_candidate`` / ``best_single_candidate``.
_TIE_BREAK_FIELDS: tuple[str, ...] = ("mean_nme", "p90_nme", "failure_rate")


@dataclass(frozen=True)
class CandidateBucketMetrics:
    """Per-(bucket, candidate) aggregate."""

    candidate: str
    sample_count: int
    mean_nme: float
    p90_nme: float
    failure_rate: float

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "sample_count": int(self.sample_count),
            "mean_nme": float(self.mean_nme),
            "p90_nme": float(self.p90_nme),
            "failure_rate": float(self.failure_rate),
        }


@dataclass(frozen=True)
class RuntimeBucketMetrics:
    """Aggregate metrics for one production / runtime bucket."""

    runtime_bucket: str
    sample_count: int
    best_candidate: str | None
    best_candidate_mean_nme: float | None
    best_candidate_p90_nme: float | None
    best_candidate_failure_rate: float | None
    best_single_candidate: str | None
    best_single_mean_nme: float | None
    best_single_p90_nme: float | None
    best_single_failure_rate: float | None
    selected_candidate: str | None
    selected_candidate_mean_nme: float | None
    selected_candidate_p90_nme: float | None
    selected_candidate_failure_rate: float | None
    static_weighted_downweight_mean_nme: float | None
    weighted_median_mean_nme: float | None
    per_candidate: dict[str, CandidateBucketMetrics] = field(default_factory=dict)

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "sample_count": int(self.sample_count),
            "best_candidate": self.best_candidate,
            "best_candidate_mean_nme": _opt_float(self.best_candidate_mean_nme),
            "best_candidate_p90_nme": _opt_float(self.best_candidate_p90_nme),
            "best_candidate_failure_rate": _opt_float(self.best_candidate_failure_rate),
            "best_single_candidate": self.best_single_candidate,
            "best_single_mean_nme": _opt_float(self.best_single_mean_nme),
            "best_single_p90_nme": _opt_float(self.best_single_p90_nme),
            "best_single_failure_rate": _opt_float(self.best_single_failure_rate),
            "selected_candidate": self.selected_candidate,
            "selected_candidate_mean_nme": _opt_float(self.selected_candidate_mean_nme),
            "selected_candidate_p90_nme": _opt_float(self.selected_candidate_p90_nme),
            "selected_candidate_failure_rate": _opt_float(self.selected_candidate_failure_rate),
            "static_weighted_downweight_mean_nme": _opt_float(
                self.static_weighted_downweight_mean_nme
            ),
            "weighted_median_mean_nme": _opt_float(self.weighted_median_mean_nme),
            "per_candidate": {
                candidate: metrics.to_payload()
                for candidate, metrics in self.per_candidate.items()
            },
        }


def _opt_float(value: float | None) -> float | None:
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _aggregate_one_candidate(
    nmes: list[float],
    failures: list[bool],
) -> tuple[float, float, float]:
    """Return ``(mean_nme, p90_nme, failure_rate)`` for one candidate's rows."""
    if not nmes:
        return math.nan, math.nan, math.nan
    arr = np.asarray(nmes, dtype=np.float64)
    mean_nme = float(np.mean(arr))
    p90_nme = float(np.percentile(arr, 90))
    failure_rate = float(np.mean(failures)) if failures else 0.0
    return mean_nme, p90_nme, failure_rate


def _select_best(
    metrics_by_candidate: Mapping[str, CandidateBucketMetrics],
    *,
    restrict_to: Iterable[str] | None = None,
) -> str | None:
    """Pick the best candidate by ``(mean_nme, p90_nme, failure_rate, name)``.

    ``restrict_to`` filters the candidate set (e.g. to single-model names
    for the ``best_single_candidate`` selection). Returns ``None`` when no
    candidate has finite NME stats.
    """
    if restrict_to is not None:
        allowed = frozenset(restrict_to)
        items = [
            (name, metrics) for name, metrics in metrics_by_candidate.items() if name in allowed
        ]
    else:
        items = list(metrics_by_candidate.items())
    items = [(name, metrics) for name, metrics in items if math.isfinite(metrics.mean_nme)]
    if not items:
        return None
    items.sort(
        key=lambda kv: (
            kv[1].mean_nme,
            kv[1].p90_nme,
            kv[1].failure_rate,
            kv[0],
        )
    )
    return items[0][0]


def aggregate_runtime_bucket_metrics(
    rows: Sequence[Mapping[str, T.Any]],
    *,
    selected_candidate: str | None = None,
    single_model_candidates: Iterable[str] = SINGLE_MODEL_CANDIDATES,
) -> dict[str, RuntimeBucketMetrics]:
    """Aggregate candidate-table rows into per-runtime-bucket metrics.

    ``rows`` must carry the canonical candidate-table fields
    (``sample_id``, ``candidate``, ``nme``, ``failure``,
    ``runtime_bucket``); other columns are ignored. Returns a mapping
    keyed by runtime bucket label, ordered alphabetically with the
    ``unknown`` bucket sorting last so the JSON output is stable across
    runs.
    """
    grouped: dict[str, dict[str, tuple[list[float], list[bool], set[str]]]] = {}
    for row in rows:
        candidate = row.get("candidate")
        bucket = row.get("runtime_bucket") or "unknown"
        if candidate is None:
            continue
        try:
            nme_value = float(row.get("nme"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(nme_value):
            continue
        failure = bool(row.get("failure"))
        sample_id = str(row.get("sample_id") or "")

        bucket_entry = grouped.setdefault(bucket, {})
        nmes, failures, samples = bucket_entry.setdefault(str(candidate), ([], [], set()))
        nmes.append(nme_value)
        failures.append(failure)
        if sample_id:
            samples.add(sample_id)

    single_set = frozenset(single_model_candidates)
    result: dict[str, RuntimeBucketMetrics] = {}
    for bucket in sorted(grouped.keys(), key=lambda b: (b == "unknown", b)):
        per_candidate_aggregates: dict[str, CandidateBucketMetrics] = {}
        bucket_sample_ids: set[str] = set()
        for candidate, (nmes, failures, samples) in sorted(grouped[bucket].items()):
            mean_nme, p90_nme, failure_rate = _aggregate_one_candidate(nmes, failures)
            per_candidate_aggregates[candidate] = CandidateBucketMetrics(
                candidate=candidate,
                sample_count=len(nmes),
                mean_nme=mean_nme,
                p90_nme=p90_nme,
                failure_rate=failure_rate,
            )
            bucket_sample_ids.update(samples)

        best = _select_best(per_candidate_aggregates)
        best_single = _select_best(per_candidate_aggregates, restrict_to=single_set)
        sel = (
            selected_candidate
            if selected_candidate and selected_candidate in per_candidate_aggregates
            else None
        )

        def _opt(
            name: str | None,
            attr: str,
            _aggregates: dict[str, CandidateBucketMetrics] = per_candidate_aggregates,
        ) -> float | None:
            if name is None:
                return None
            return getattr(_aggregates[name], attr)

        bucket_sample_count = (
            len(bucket_sample_ids)
            if bucket_sample_ids
            else max(
                (m.sample_count for m in per_candidate_aggregates.values()),
                default=0,
            )
        )
        result[bucket] = RuntimeBucketMetrics(
            runtime_bucket=bucket,
            sample_count=bucket_sample_count,
            best_candidate=best,
            best_candidate_mean_nme=_opt(best, "mean_nme"),
            best_candidate_p90_nme=_opt(best, "p90_nme"),
            best_candidate_failure_rate=_opt(best, "failure_rate"),
            best_single_candidate=best_single,
            best_single_mean_nme=_opt(best_single, "mean_nme"),
            best_single_p90_nme=_opt(best_single, "p90_nme"),
            best_single_failure_rate=_opt(best_single, "failure_rate"),
            selected_candidate=sel,
            selected_candidate_mean_nme=_opt(sel, "mean_nme"),
            selected_candidate_p90_nme=_opt(sel, "p90_nme"),
            selected_candidate_failure_rate=_opt(sel, "failure_rate"),
            static_weighted_downweight_mean_nme=(
                per_candidate_aggregates["static_weighted_downweight"].mean_nme
                if "static_weighted_downweight" in per_candidate_aggregates
                else None
            ),
            weighted_median_mean_nme=(
                per_candidate_aggregates["weighted_median"].mean_nme
                if "weighted_median" in per_candidate_aggregates
                else None
            ),
            per_candidate=per_candidate_aggregates,
        )
    return result


def write_runtime_bucket_json(metrics: Mapping[str, RuntimeBucketMetrics], path: Path) -> Path:
    """Write the per-bucket aggregate payload as JSON to ``path``."""
    payload = {bucket: m.to_payload() for bucket, m in metrics.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_runtime_bucket_csv(metrics: Mapping[str, RuntimeBucketMetrics], path: Path) -> Path:
    """Write one row per ``(bucket, candidate)`` to ``path``.

    The CSV uses :data:`GT_RUNTIME_BUCKET_CSV_COLUMNS` so external tools
    can sort / filter by either dimension. ``is_best_candidate`` /
    ``is_best_single_candidate`` / ``is_selected_candidate`` are 1/0
    booleans so spreadsheets can pivot on them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GT_RUNTIME_BUCKET_CSV_COLUMNS))
        writer.writeheader()
        for bucket, bucket_metrics in metrics.items():
            for candidate, candidate_metrics in bucket_metrics.per_candidate.items():
                writer.writerow(
                    {
                        "runtime_bucket": bucket,
                        "candidate": candidate,
                        "sample_count": candidate_metrics.sample_count,
                        "mean_nme": _csv_float(candidate_metrics.mean_nme),
                        "p90_nme": _csv_float(candidate_metrics.p90_nme),
                        "failure_rate": _csv_float(candidate_metrics.failure_rate),
                        "is_best_candidate": int(candidate == bucket_metrics.best_candidate),
                        "is_best_single_candidate": int(
                            candidate == bucket_metrics.best_single_candidate
                        ),
                        "is_selected_candidate": int(
                            candidate == bucket_metrics.selected_candidate
                        ),
                    }
                )
    return path


def _csv_float(value: float | None) -> str:
    """Render an optional float for CSV — empty for NaN / None."""
    if value is None:
        return ""
    f = float(value)
    return "" if not math.isfinite(f) else f"{f:.6f}"


def load_candidate_table_csv(path: Path) -> list[dict[str, T.Any]]:
    """Load a ``candidate_table.csv`` file into the row-dict shape
    consumed by :func:`aggregate_runtime_bucket_metrics`."""
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, T.Any]] = []
        for raw in reader:
            row = dict(raw)
            # CSVs round-trip everything as strings — coerce the numeric
            # / boolean columns the aggregator needs.
            if "nme" in row and row["nme"] != "":
                with contextlib.suppress(ValueError):
                    row["nme"] = float(row["nme"])
            if "failure" in row:
                row["failure"] = row.get("failure") in {"1", "True", "true", "TRUE"}
            rows.append(row)
    return rows
