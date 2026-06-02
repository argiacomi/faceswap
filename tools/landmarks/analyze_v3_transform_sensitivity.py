#!/usr/bin/env python3
"""Analyze learned-quality v3 transform-cost sensitivity.

Sweep v3 transform-cost weights and near-tie thresholds over scorer-row CSVs.

Expected input: one row per sample/candidate. Required columns are sample_id and
candidate_name (or candidate). True weight sensitivity also needs component
columns. Supported aliases:

    center_delta_v3, transform_center_delta_v3, crop_center_delta_v3
    scale_delta_v3, transform_scale_delta_v3
    roll_delta_degrees_v3, transform_roll_delta_degrees_v3
    fit_delta_v3, transform_fit_delta_v3
    soft_structural_penalty_v3

If component columns are missing, this tool falls back to transform_cost_v3 and
only analyzes MIN_V3_ORACLE_GAP / policy-regret stability.

Outputs:
    v3_weight_sensitivity_summary.json
    v3_weight_sensitivity_by_bucket.csv
    v3_oracle_flip_examples.csv
    v3_gate_sensitivity.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import typing as T
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

BASELINE_WEIGHTS = {"center": 5.0, "scale": 1.0, "roll": 0.02, "fit": 1.0}

DEFAULT_SWEEP_VALUES = {
    "center": (2.5, 5.0, 7.5, 10.0),
    "scale": (0.5, 1.0, 2.0),
    "roll": (0.01, 0.02, 0.05),
    "fit": (0.5, 1.0, 2.0),
    "soft": (0.0, 0.025, 0.05, 0.10),
    "min_gap": (0.00005, 0.00010, 0.00020, 0.00050),
}

COMPONENT_COLUMN_ALIASES = {
    "center": ("center_delta_v3", "transform_center_delta_v3", "crop_center_delta_v3"),
    "scale": ("scale_delta_v3", "transform_scale_delta_v3"),
    "roll": ("roll_delta_degrees_v3", "transform_roll_delta_degrees_v3"),
    "fit": ("fit_delta_v3", "transform_fit_delta_v3"),
    "soft": ("soft_structural_penalty_v3",),
}


@dataclass(frozen=True)
class CandidateRow:
    sample_id: str
    face_index: str
    candidate_name: str
    source: str
    dataset: str
    condition: str
    runtime_bucket: str
    current_policy_choice: str
    baseline_policy_choice: str
    original_oracle: str
    rankable: bool
    hard_invalid: bool
    base_cost: float
    center_delta: float | None
    scale_delta: float | None
    roll_delta_degrees: float | None
    fit_delta: float | None
    soft_structural_penalty: float


@dataclass(frozen=True)
class SweepConfig:
    center: float
    scale: float
    roll: float
    fit: float
    soft: float
    min_gap: float

    @property
    def key(self) -> str:
        return (
            f"center={self.center:g}|scale={self.scale:g}|roll={self.roll:g}|"
            f"fit={self.fit:g}|soft={self.soft:g}|min_gap={self.min_gap:g}"
        )


@dataclass
class GroupSummary:
    key: str
    sample_id: str
    face_index: str
    source: str
    dataset: str
    condition: str
    runtime_bucket: str
    oracle_candidate: str
    oracle_cost: float
    oracle_gap: float
    near_tie_excluded: bool
    zero_valid: bool
    current_policy_choice: str
    current_policy_regret: float | None
    baseline_policy_choice: str
    baseline_policy_regret: float | None
    original_oracle: str


def parse_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        parsed = float(text)
    except ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "t", "yes", "y"}


def first_present(row: T.Mapping[str, str], names: T.Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value)
    return ""


def first_component(row: T.Mapping[str, str], component: str) -> float | None:
    for name in COMPONENT_COLUMN_ALIASES[component]:
        if name in row and str(row.get(name, "")).strip() != "":
            return parse_float(row.get(name), 0.0)
    return None


def load_rows(path: Path) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, raw in enumerate(reader, start=2):
            sample_id = first_present(raw, ("sample_id",))
            candidate_name = first_present(raw, ("candidate_name", "candidate"))
            if not sample_id or not candidate_name:
                raise ValueError(f"{path}:{line_number} missing sample_id/candidate_name")

            hard_invalid = parse_bool(raw.get("hard_invalid_v3"), False)
            if "rankable_v3" in raw:
                rankable = parse_bool(raw.get("rankable_v3"), False) and not hard_invalid
            else:
                rankable = not hard_invalid

            current_policy_choice = first_present(
                raw,
                ("selected_by_current_policy", "current_policy_choice", "selected_candidate"),
            )
            if not current_policy_choice and parse_bool(
                raw.get("was_selected_by_current_policy"), False
            ):
                current_policy_choice = candidate_name

            rows.append(
                CandidateRow(
                    sample_id=sample_id,
                    face_index=first_present(raw, ("face_index",)) or "0",
                    candidate_name=candidate_name,
                    source=first_present(raw, ("source",)),
                    dataset=first_present(raw, ("dataset",)),
                    condition=first_present(raw, ("condition",)),
                    runtime_bucket=first_present(raw, ("runtime_bucket",)),
                    current_policy_choice=current_policy_choice,
                    baseline_policy_choice=first_present(
                        raw,
                        (
                            "baseline_policy_choice",
                            "static_weighted_downweight_choice",
                            "static_weighted_choice",
                        ),
                    ),
                    original_oracle=first_present(
                        raw, ("transform_oracle_candidate_v3", "oracle")
                    ),
                    rankable=rankable,
                    hard_invalid=hard_invalid,
                    base_cost=parse_float(raw.get("transform_cost_v3"), 0.0),
                    center_delta=first_component(raw, "center"),
                    scale_delta=first_component(raw, "scale"),
                    roll_delta_degrees=first_component(raw, "roll"),
                    fit_delta=first_component(raw, "fit"),
                    soft_structural_penalty=first_component(raw, "soft") or 0.0,
                )
            )
    return rows


def has_component_columns(rows: T.Sequence[CandidateRow]) -> bool:
    return bool(rows) and all(
        row.center_delta is not None
        and row.scale_delta is not None
        and row.roll_delta_degrees is not None
        and row.fit_delta is not None
        for row in rows
    )


def cost_for(row: CandidateRow, config: SweepConfig, *, components_available: bool) -> float:
    if row.hard_invalid or not row.rankable:
        return 0.0
    if not components_available:
        return row.base_cost
    assert row.center_delta is not None
    assert row.scale_delta is not None
    assert row.roll_delta_degrees is not None
    assert row.fit_delta is not None
    soft_flag = 1.0 if row.soft_structural_penalty > 0.0 else 0.0
    return float(
        config.center * row.center_delta
        + config.scale * row.scale_delta
        + config.roll * row.roll_delta_degrees
        + config.fit * row.fit_delta
        + config.soft * soft_flag
    )


def group_key(row: CandidateRow) -> tuple[str, str]:
    return row.sample_id, row.face_index


def grouped_rows(rows: T.Sequence[CandidateRow]) -> dict[tuple[str, str], list[CandidateRow]]:
    groups: dict[tuple[str, str], list[CandidateRow]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return dict(groups)


def percentile(values: T.Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_groups(
    rows: T.Sequence[CandidateRow],
    config: SweepConfig,
    *,
    components_available: bool,
    baseline_policy: str,
) -> list[GroupSummary]:
    summaries: list[GroupSummary] = []
    for _key, group in grouped_rows(rows).items():
        valid = [row for row in group if row.rankable and not row.hard_invalid]
        exemplar = group[0]
        if not valid:
            summaries.append(
                GroupSummary(
                    key=config.key,
                    sample_id=exemplar.sample_id,
                    face_index=exemplar.face_index,
                    source=exemplar.source,
                    dataset=exemplar.dataset,
                    condition=exemplar.condition,
                    runtime_bucket=exemplar.runtime_bucket,
                    oracle_candidate="",
                    oracle_cost=0.0,
                    oracle_gap=0.0,
                    near_tie_excluded=False,
                    zero_valid=True,
                    current_policy_choice=exemplar.current_policy_choice,
                    current_policy_regret=None,
                    baseline_policy_choice=baseline_policy,
                    baseline_policy_regret=None,
                    original_oracle=exemplar.original_oracle,
                )
            )
            continue

        costs = {
            row.candidate_name: cost_for(row, config, components_available=components_available)
            for row in valid
        }
        ordered = sorted(costs.items(), key=lambda item: (item[1], item[0]))
        oracle_candidate, oracle_cost = ordered[0]
        oracle_gap = ordered[1][1] - ordered[0][1] if len(ordered) > 1 else 0.0
        near_tie = len(ordered) > 1 and oracle_gap < config.min_gap

        current_choice = exemplar.current_policy_choice
        current_regret = (
            max(costs[current_choice] - oracle_cost, 0.0)
            if current_choice in costs and not near_tie
            else None
        )
        baseline_choice = baseline_policy
        baseline_regret = (
            max(costs[baseline_choice] - oracle_cost, 0.0)
            if baseline_choice in costs and not near_tie
            else None
        )
        summaries.append(
            GroupSummary(
                key=config.key,
                sample_id=exemplar.sample_id,
                face_index=exemplar.face_index,
                source=exemplar.source,
                dataset=exemplar.dataset,
                condition=exemplar.condition,
                runtime_bucket=exemplar.runtime_bucket,
                oracle_candidate=oracle_candidate,
                oracle_cost=oracle_cost,
                oracle_gap=oracle_gap,
                near_tie_excluded=near_tie,
                zero_valid=False,
                current_policy_choice=current_choice,
                current_policy_regret=current_regret,
                baseline_policy_choice=baseline_choice,
                baseline_policy_regret=baseline_regret,
                original_oracle=exemplar.original_oracle,
            )
        )
    return summaries


def metric_summary(values: T.Sequence[float | None]) -> dict[str, float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"count": 0.0, "mean": 0.0, "p95": 0.0}
    return {
        "count": float(len(clean)),
        "mean": float(statistics.fmean(clean)),
        "p95": percentile(clean, 95),
    }


def aggregate_config(
    config: SweepConfig,
    summaries: T.Sequence[GroupSummary],
    baseline_summaries: T.Mapping[tuple[str, str], GroupSummary] | None,
) -> dict[str, T.Any]:
    total = len(summaries)
    zero_valid = sum(item.zero_valid for item in summaries)
    near_tie = sum(item.near_tie_excluded for item in summaries)
    eval_items = [
        item
        for item in summaries
        if not item.zero_valid and not item.near_tie_excluded and item.oracle_candidate
    ]

    oracle_flip_count = 0
    if baseline_summaries is not None:
        for item in eval_items:
            baseline = baseline_summaries.get((item.sample_id, item.face_index))
            if baseline and baseline.oracle_candidate != item.oracle_candidate:
                oracle_flip_count += 1

    current_regret = metric_summary([item.current_policy_regret for item in summaries])
    baseline_regret = metric_summary([item.baseline_policy_regret for item in summaries])
    return {
        **asdict(config),
        "config_key": config.key,
        "group_count": total,
        "eval_group_count": len(eval_items),
        "zero_valid_group_count": zero_valid,
        "near_tie_excluded_count": near_tie,
        "near_tie_excluded_rate": near_tie / total if total else 0.0,
        "oracle_flip_count_vs_baseline": oracle_flip_count,
        "oracle_flip_rate_vs_baseline": oracle_flip_count / len(eval_items) if eval_items else 0.0,
        "current_policy_regret": current_regret,
        "baseline_policy_regret": baseline_regret,
    }


def by_bucket_rows(
    config: SweepConfig,
    summaries: T.Sequence[GroupSummary],
    baseline_summaries: T.Mapping[tuple[str, str], GroupSummary] | None,
) -> list[dict[str, T.Any]]:
    buckets: dict[str, list[GroupSummary]] = defaultdict(list)
    for item in summaries:
        bucket = item.runtime_bucket or item.condition or item.dataset or "unknown"
        buckets[bucket].append(item)
    rows: list[dict[str, T.Any]] = []
    for bucket, items in sorted(buckets.items()):
        agg = aggregate_config(config, items, baseline_summaries)
        rows.append(
            {
                "config_key": config.key,
                "runtime_bucket": bucket,
                "group_count": agg["group_count"],
                "eval_group_count": agg["eval_group_count"],
                "zero_valid_group_count": agg["zero_valid_group_count"],
                "near_tie_excluded_count": agg["near_tie_excluded_count"],
                "oracle_flip_count_vs_baseline": agg["oracle_flip_count_vs_baseline"],
                "current_policy_mean_transform_regret_v3": agg["current_policy_regret"]["mean"],
                "current_policy_p95_transform_regret_v3": agg["current_policy_regret"]["p95"],
                "baseline_policy_mean_transform_regret_v3": agg["baseline_policy_regret"]["mean"],
                "baseline_policy_p95_transform_regret_v3": agg["baseline_policy_regret"]["p95"],
            }
        )
    return rows


def oracle_flip_examples(
    summaries: T.Sequence[GroupSummary],
    baseline_summaries: T.Mapping[tuple[str, str], GroupSummary],
    *,
    limit: int,
) -> list[dict[str, T.Any]]:
    rows: list[dict[str, T.Any]] = []
    for item in summaries:
        baseline = baseline_summaries.get((item.sample_id, item.face_index))
        if (
            not baseline
            or not item.oracle_candidate
            or item.oracle_candidate == baseline.oracle_candidate
        ):
            continue
        rows.append(
            {
                "config_key": item.key,
                "sample_id": item.sample_id,
                "face_index": item.face_index,
                "runtime_bucket": item.runtime_bucket,
                "baseline_oracle": baseline.oracle_candidate,
                "new_oracle": item.oracle_candidate,
                "baseline_oracle_gap": baseline.oracle_gap,
                "new_oracle_gap": item.oracle_gap,
                "current_policy_choice": item.current_policy_choice,
                "baseline_policy_choice": item.baseline_policy_choice,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def parse_csv_floats(raw: str | None, default: T.Sequence[float]) -> tuple[float, ...]:
    if not raw:
        return tuple(default)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("sweep list must contain at least one number")
    return tuple(values)


def build_configs(args: argparse.Namespace, *, components_available: bool) -> list[SweepConfig]:
    center_values = parse_csv_floats(args.center_weights, DEFAULT_SWEEP_VALUES["center"])
    scale_values = parse_csv_floats(args.scale_weights, DEFAULT_SWEEP_VALUES["scale"])
    roll_values = parse_csv_floats(args.roll_weights, DEFAULT_SWEEP_VALUES["roll"])
    fit_values = parse_csv_floats(args.fit_weights, DEFAULT_SWEEP_VALUES["fit"])
    soft_values = parse_csv_floats(args.soft_penalties, DEFAULT_SWEEP_VALUES["soft"])
    min_gap_values = parse_csv_floats(args.min_gaps, DEFAULT_SWEEP_VALUES["min_gap"])
    if not components_available:
        center_values = (BASELINE_WEIGHTS["center"],)
        scale_values = (BASELINE_WEIGHTS["scale"],)
        roll_values = (BASELINE_WEIGHTS["roll"],)
        fit_values = (BASELINE_WEIGHTS["fit"],)
        soft_values = (DEFAULT_SWEEP_VALUES["soft"][2],)
    return [
        SweepConfig(center=center, scale=scale, roll=roll, fit=fit, soft=soft, min_gap=min_gap)
        for center, scale, roll, fit, soft, min_gap in itertools.product(
            center_values, scale_values, roll_values, fit_values, soft_values, min_gap_values
        )
    ]


def write_csv(path: Path, rows: T.Sequence[T.Mapping[str, T.Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: T.Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: T.Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="Input scorer-row CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for reports.")
    parser.add_argument("--baseline-policy", default="static_weighted_downweight")
    parser.add_argument("--center-weights", default=None)
    parser.add_argument("--scale-weights", default=None)
    parser.add_argument("--roll-weights", default=None)
    parser.add_argument("--fit-weights", default=None)
    parser.add_argument("--soft-penalties", default=None)
    parser.add_argument("--min-gaps", default=None)
    parser.add_argument("--oracle-flip-example-limit", type=int, default=200)
    args = parser.parse_args(argv)

    rows = load_rows(args.rows)
    if not rows:
        raise SystemExit(f"no rows found in {args.rows}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    components_available = has_component_columns(rows)
    configs = build_configs(args, components_available=components_available)

    baseline_config = SweepConfig(
        center=BASELINE_WEIGHTS["center"],
        scale=BASELINE_WEIGHTS["scale"],
        roll=BASELINE_WEIGHTS["roll"],
        fit=BASELINE_WEIGHTS["fit"],
        soft=DEFAULT_SWEEP_VALUES["soft"][2],
        min_gap=DEFAULT_SWEEP_VALUES["min_gap"][1],
    )
    baseline_group_summaries = summarize_groups(
        rows,
        baseline_config,
        components_available=components_available,
        baseline_policy=args.baseline_policy,
    )
    baseline_by_group = {
        (item.sample_id, item.face_index): item for item in baseline_group_summaries
    }

    config_summaries: list[dict[str, T.Any]] = []
    bucket_rows: list[dict[str, T.Any]] = []
    flip_rows: list[dict[str, T.Any]] = []
    gate_sensitivity: list[dict[str, T.Any]] = []

    for config in configs:
        summaries = summarize_groups(
            rows,
            config,
            components_available=components_available,
            baseline_policy=args.baseline_policy,
        )
        aggregate = aggregate_config(config, summaries, baseline_by_group)
        config_summaries.append(aggregate)
        bucket_rows.extend(by_bucket_rows(config, summaries, baseline_by_group))
        if len(flip_rows) < args.oracle_flip_example_limit:
            remaining = args.oracle_flip_example_limit - len(flip_rows)
            flip_rows.extend(oracle_flip_examples(summaries, baseline_by_group, limit=remaining))
        gate_sensitivity.append(
            {
                "config_key": config.key,
                "min_gap": config.min_gap,
                "near_tie_excluded_count": aggregate["near_tie_excluded_count"],
                "near_tie_excluded_rate": aggregate["near_tie_excluded_rate"],
                "oracle_flip_rate_vs_baseline": aggregate["oracle_flip_rate_vs_baseline"],
                "current_policy_mean_transform_regret_v3": aggregate["current_policy_regret"][
                    "mean"
                ],
                "current_policy_p95_transform_regret_v3": aggregate["current_policy_regret"][
                    "p95"
                ],
                "baseline_policy_mean_transform_regret_v3": aggregate["baseline_policy_regret"][
                    "mean"
                ],
                "baseline_policy_p95_transform_regret_v3": aggregate["baseline_policy_regret"][
                    "p95"
                ],
            }
        )

    oracle_flip_rates = [float(item["oracle_flip_rate_vs_baseline"]) for item in config_summaries]
    near_tie_rates = [float(item["near_tie_excluded_rate"]) for item in config_summaries]
    summary_payload = {
        "input_rows": str(args.rows),
        "row_count": len(rows),
        "face_group_count": len(grouped_rows(rows)),
        "components_available": components_available,
        "component_mode": "weighted_components"
        if components_available
        else "existing_transform_cost_v3_only",
        "baseline_policy": args.baseline_policy,
        "baseline_config": asdict(baseline_config),
        "config_count": len(configs),
        "sweep_values": {
            "center": sorted({config.center for config in configs}),
            "scale": sorted({config.scale for config in configs}),
            "roll": sorted({config.roll for config in configs}),
            "fit": sorted({config.fit for config in configs}),
            "soft": sorted({config.soft for config in configs}),
            "min_gap": sorted({config.min_gap for config in configs}),
        },
        "stability": {
            "max_oracle_flip_rate_vs_baseline": max(oracle_flip_rates)
            if oracle_flip_rates
            else 0.0,
            "mean_oracle_flip_rate_vs_baseline": statistics.fmean(oracle_flip_rates)
            if oracle_flip_rates
            else 0.0,
            "max_near_tie_excluded_rate": max(near_tie_rates) if near_tie_rates else 0.0,
            "mean_near_tie_excluded_rate": statistics.fmean(near_tie_rates)
            if near_tie_rates
            else 0.0,
        },
        "configs": config_summaries,
        "notes": [
            "When components_available is false, only MIN_V3_ORACLE_GAP and policy regret are meaningful.",
            "Oracle flip rates compare each sweep config against the baseline v3 weight config.",
            "This report validates sensitivity/stability; promotion gates still require a real scorer run.",
        ],
    }

    write_json(args.output_dir / "v3_weight_sensitivity_summary.json", summary_payload)
    write_csv(args.output_dir / "v3_weight_sensitivity_by_bucket.csv", bucket_rows)
    write_csv(args.output_dir / "v3_oracle_flip_examples.csv", flip_rows)
    write_json(args.output_dir / "v3_gate_sensitivity.json", gate_sensitivity)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "row_count": len(rows),
                "config_count": len(configs),
                "components_available": components_available,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
