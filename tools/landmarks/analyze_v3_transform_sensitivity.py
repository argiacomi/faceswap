#!/usr/bin/env python3
"""Analyze learned-quality v3 transform-cost *oracle/cost* sensitivity.

This is an **oracle/cost sensitivity** report, not learned-scorer promotion
evidence. It sweeps the v3 transform-cost knobs over scorer-row CSVs and measures
how the within-face oracle (the lowest-cost candidate, which becomes the v3
ranking label) moves under perturbation. Promotion of a trained scorer still
requires the real scorer run and the learnability gates; nothing here speaks to
that.

v3 cost model (matches ``lib.landmarks.evaluation.transform_alignment_cost``):

    candidate_alignment_cost =
        w_corner * corner_delta_v3      # single grounded geometric primitive
        + w_fit * fit_delta_v3          # small adjunct
        + w_soft * soft_indicator       # soft structural suspect penalty

``corner_delta_v3`` is the RMS crop-corner displacement (output-frame units)
between candidate and GT transforms; it subsumes center/scale/roll, so there is
one geometric knob instead of three. ``center_delta_v3`` / ``scale_delta_v3`` /
``roll_delta_degrees_v3`` are read only as reported diagnostics.

Expected input: one row per sample/candidate. Required columns are ``sample_id``
and ``candidate_name`` (or ``candidate``). True weight sensitivity needs the
component columns ``corner_delta_v3`` and ``fit_delta_v3`` (and optionally
``soft_structural_penalty_v3``). If those are missing the tool falls back to
``transform_cost_v3`` and can only analyze ``min_gap`` near-tie behavior.

Outputs:
    v3_cost_sensitivity_summary.json     (envelope-scoped acceptance KPIs)
    v3_cost_sensitivity_by_bucket.csv    (per-bucket flips + hard-bucket gate)
    v3_oracle_flip_examples.csv
    v3_cost_sensitivity_gates.json       (per-config cost-sensitivity gate flips)
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

from tqdm import tqdm

# Baseline cost knobs mirror ``TransformCostWeightsV3`` defaults plus the default
# soft structural penalty. ``min_gap`` is the near-tie exclusion threshold.
BASELINE_WEIGHTS = {"corner": 1.0, "fit": 0.25, "soft": 0.05, "min_gap": 0.00010}

DEFAULT_SWEEP_VALUES = {
    "corner": (0.5, 1.0, 2.0),
    "fit": (0.1, 0.25, 0.5, 1.0),
    "soft": (0.0, 0.025, 0.05, 0.10),
    "min_gap": (0.00005, 0.00010, 0.00020, 0.00050),
}

# Configs within this normalized log-distance of the baseline weights count as a
# "reasonable" joint perturbation. Single-axis perturbations are always
# reasonable; everything else is the "stress/corner grid".
DEFAULT_JOINT_BALL_THRESHOLD = 0.75

# Cost-sensitivity acceptance gates (NOT learned-scorer promotion gates).
DEFAULT_FLIP_RATE_GATE = 0.10
DEFAULT_FLIP_COST_P95_GATE = 0.01
DEFAULT_NEAR_TIE_RATE_GATE = 0.20

REASONABLE_ENVELOPES = ("baseline", "single_axis", "joint_ball")
GAP_BUCKET_EDGES = (1e-4, 2e-4, 5e-4)

COMPONENT_COLUMN_ALIASES = {
    "corner": ("corner_delta_v3", "transform_corner_delta_v3"),
    "fit": ("fit_delta_v3", "transform_fit_delta_v3"),
    "soft": ("soft_structural_penalty_v3",),
    # Diagnostics only - reported elsewhere, never swept or summed into cost.
    "center": ("center_delta_v3", "transform_center_delta_v3"),
    "scale": ("scale_delta_v3", "transform_scale_delta_v3"),
    "roll": ("roll_delta_degrees_v3", "transform_roll_delta_degrees_v3"),
}


def progress_iter(
    values: T.Iterable[T.Any], *, total: int | None, desc: str, unit: str, enabled: bool
) -> T.Iterable[T.Any]:
    """Wrap an iterable in a tqdm progress bar when enabled."""
    return T.cast(
        T.Iterable[T.Any],
        tqdm(values, total=total, desc=desc, unit=unit, disable=not enabled),
    )


@dataclass(frozen=True)
class CandidateRow:
    """One candidate row parsed from a scorer-row CSV."""

    sample_id: str
    face_index: str
    candidate_name: str
    source: str
    dataset: str
    condition: str
    runtime_bucket: str
    current_policy_choice: str
    rankable: bool
    hard_invalid: bool
    base_cost: float
    corner_delta: float | None
    fit_delta: float | None
    soft_structural_penalty: float


@dataclass(frozen=True)
class SweepConfig:
    """One point in the (corner, fit, soft, min_gap) sweep grid."""

    corner: float
    fit: float
    soft: float
    min_gap: float

    @property
    def key(self) -> str:
        return (
            f"corner={self.corner:g}|fit={self.fit:g}|soft={self.soft:g}|min_gap={self.min_gap:g}"
        )


@dataclass(frozen=True)
class GroupCosts:
    """Per-face candidate costs and the resulting oracle for one config."""

    sample_id: str
    face_index: str
    runtime_bucket: str
    condition: str
    dataset: str
    costs: dict[str, float]
    oracle_candidate: str
    oracle_cost: float
    oracle_gap: float
    near_tie: bool
    zero_valid: bool
    current_policy_choice: str


@dataclass(frozen=True)
class FlipMetrics:
    """Oracle flip count/rate/cost for one config vs the baseline-weight oracle."""

    eval_group_count: int
    near_tie_excluded_count: int
    oracle_flip_count: int
    oracle_flip_rate: float
    oracle_flip_mean_delta_regret: float
    oracle_flip_p95_delta_regret: float
    flip_count_by_baseline_gap: dict[str, int]


@dataclass(frozen=True)
class GateResult:
    """Cost-sensitivity gate pass/fail for one scope (overall or one bucket)."""

    passed: bool
    failed_gates: tuple[str, ...]


def parse_float(value: object, default: float = 0.0) -> float:
    """Parse a CSV cell into a finite float, falling back to ``default``."""
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
    """Parse a CSV cell into a bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "t", "yes", "y"}


def first_present(row: T.Mapping[str, str], names: T.Sequence[str]) -> str:
    """Return the first non-empty value among ``names``."""
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value)
    return ""


def first_component(row: T.Mapping[str, str], component: str) -> float | None:
    """Return the first present component-delta value, or ``None`` if absent."""
    for name in COMPONENT_COLUMN_ALIASES[component]:
        if name in row and str(row.get(name, "")).strip() != "":
            return parse_float(row.get(name), 0.0)
    return None


def load_rows(path: Path) -> list[CandidateRow]:
    """Load and validate candidate rows from a scorer-row CSV."""
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
                raw, ("selected_by_current_policy", "current_policy_choice", "selected_candidate")
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
                    rankable=rankable,
                    hard_invalid=hard_invalid,
                    base_cost=parse_float(raw.get("transform_cost_v3"), 0.0),
                    corner_delta=first_component(raw, "corner"),
                    fit_delta=first_component(raw, "fit"),
                    soft_structural_penalty=first_component(raw, "soft") or 0.0,
                )
            )
    return rows


def has_component_columns(rows: T.Sequence[CandidateRow]) -> bool:
    """Return ``True`` if every row carries the corner + fit cost components."""
    return bool(rows) and all(
        row.corner_delta is not None and row.fit_delta is not None for row in rows
    )


def soft_penalty_active(rows: T.Sequence[CandidateRow]) -> bool:
    """Return ``True`` if any row carries a non-zero soft structural penalty."""
    return any(row.soft_structural_penalty > 0.0 for row in rows)


def cost_for(row: CandidateRow, config: SweepConfig, *, components_available: bool) -> float:
    """Return the candidate cost under ``config`` (0 for hard-invalid/unrankable)."""
    if row.hard_invalid or not row.rankable:
        return 0.0
    if not components_available:
        return row.base_cost
    assert row.corner_delta is not None
    assert row.fit_delta is not None
    soft_flag = 1.0 if row.soft_structural_penalty > 0.0 else 0.0
    return float(
        config.corner * row.corner_delta + config.fit * row.fit_delta + config.soft * soft_flag
    )


def group_key(row: CandidateRow) -> tuple[str, str]:
    """Return the per-face grouping key."""
    return row.sample_id, row.face_index


def grouped_rows(rows: T.Sequence[CandidateRow]) -> dict[tuple[str, str], list[CandidateRow]]:
    """Group candidate rows by face."""
    groups: dict[tuple[str, str], list[CandidateRow]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    return dict(groups)


def percentile(values: T.Sequence[float], pct: float) -> float:
    """Return the linear-interpolated percentile of ``values``."""
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


def group_costs(
    rows: T.Sequence[CandidateRow], config: SweepConfig, *, components_available: bool
) -> list[GroupCosts]:
    """Compute per-face candidate costs / oracle / near-tie flags for one config."""
    summaries: list[GroupCosts] = []
    for _key, group in grouped_rows(rows).items():
        exemplar = group[0]
        valid = [row for row in group if row.rankable and not row.hard_invalid]
        if not valid:
            summaries.append(
                GroupCosts(
                    sample_id=exemplar.sample_id,
                    face_index=exemplar.face_index,
                    runtime_bucket=exemplar.runtime_bucket,
                    condition=exemplar.condition,
                    dataset=exemplar.dataset,
                    costs={},
                    oracle_candidate="",
                    oracle_cost=0.0,
                    oracle_gap=0.0,
                    near_tie=False,
                    zero_valid=True,
                    current_policy_choice=exemplar.current_policy_choice,
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
        summaries.append(
            GroupCosts(
                sample_id=exemplar.sample_id,
                face_index=exemplar.face_index,
                runtime_bucket=exemplar.runtime_bucket,
                condition=exemplar.condition,
                dataset=exemplar.dataset,
                costs=costs,
                oracle_candidate=oracle_candidate,
                oracle_cost=oracle_cost,
                oracle_gap=oracle_gap,
                near_tie=len(ordered) > 1 and oracle_gap < config.min_gap,
                zero_valid=False,
                current_policy_choice=exemplar.current_policy_choice,
            )
        )
    return summaries


def _gap_bucket(gap: float) -> str:
    """Return the baseline-oracle-gap bucket label for a flip."""
    for edge in GAP_BUCKET_EDGES:
        if gap < edge:
            return f"lt_{edge:g}"
    return f"ge_{GAP_BUCKET_EDGES[-1]:g}"


def flip_metrics(
    summaries: T.Sequence[GroupCosts],
    baseline_by_group: T.Mapping[tuple[str, str], GroupCosts],
) -> FlipMetrics:
    """Compare a config's oracle to the baseline-weight oracle, costed at baseline.

    ``delta_regret`` is the *flip cost*: the extra baseline-weighted transform
    cost incurred by following this config's oracle instead of the baseline
    oracle. A relabel that picks a near-identical candidate has ~0 delta-regret;
    a materially different choice has a large one.
    """
    eval_count = 0
    near_tie_excluded = 0
    flip_count = 0
    deltas: list[float] = []
    by_gap = {f"lt_{edge:g}": 0 for edge in GAP_BUCKET_EDGES}
    by_gap[f"ge_{GAP_BUCKET_EDGES[-1]:g}"] = 0

    for item in summaries:
        if item.zero_valid or not item.oracle_candidate:
            continue
        if item.near_tie:
            near_tie_excluded += 1
            continue
        baseline = baseline_by_group.get((item.sample_id, item.face_index))
        if baseline is None or not baseline.oracle_candidate:
            continue
        eval_count += 1
        if item.oracle_candidate == baseline.oracle_candidate:
            continue
        flip_count += 1
        config_oracle_baseline_cost = baseline.costs.get(item.oracle_candidate)
        if config_oracle_baseline_cost is None:
            continue
        deltas.append(max(config_oracle_baseline_cost - baseline.oracle_cost, 0.0))
        by_gap[_gap_bucket(baseline.oracle_gap)] += 1

    return FlipMetrics(
        eval_group_count=eval_count,
        near_tie_excluded_count=near_tie_excluded,
        oracle_flip_count=flip_count,
        oracle_flip_rate=flip_count / eval_count if eval_count else 0.0,
        oracle_flip_mean_delta_regret=float(statistics.fmean(deltas)) if deltas else 0.0,
        oracle_flip_p95_delta_regret=percentile(deltas, 95),
        flip_count_by_baseline_gap=by_gap,
    )


def _config_distance(config: SweepConfig, baseline: SweepConfig) -> float:
    """Return normalized log-distance of ``config`` weights from ``baseline``."""
    distance = 0.0
    for axis in ("corner", "fit", "min_gap"):
        value = float(getattr(config, axis))
        base = float(getattr(baseline, axis))
        if value > 0.0 and base > 0.0:
            distance += math.log(value / base) ** 2
    soft_base = float(baseline.soft)
    if soft_base > 0.0:
        distance += ((config.soft - soft_base) / soft_base) ** 2
    else:
        distance += config.soft**2
    return math.sqrt(distance)


def envelope_for(
    config: SweepConfig, baseline: SweepConfig, *, joint_ball_threshold: float
) -> str:
    """Classify ``config`` relative to ``baseline`` for the reasonable envelope."""
    changed = [
        axis
        for axis in ("corner", "fit", "soft", "min_gap")
        if getattr(config, axis) != getattr(baseline, axis)
    ]
    if not changed:
        return "baseline"
    if len(changed) == 1:
        return "single_axis"
    if _config_distance(config, baseline) <= joint_ball_threshold:
        return "joint_ball"
    return "stress"


def evaluate_cost_sensitivity_gate(
    flips: FlipMetrics,
    *,
    flip_rate_gate: float,
    flip_cost_p95_gate: float,
    near_tie_rate_gate: float,
) -> GateResult:
    """Return the offline cost-sensitivity gate result for one flip summary.

    These are oracle/cost stability gates, deliberately distinct from the
    learned-scorer promotion gates. They answer "did reweighting move the oracle
    label too much / too expensively?", not "should this scorer ship?".
    """
    failed: list[str] = []
    if flips.oracle_flip_rate > flip_rate_gate:
        failed.append("oracle_flip_rate_above_gate")
    if flips.oracle_flip_p95_delta_regret > flip_cost_p95_gate:
        failed.append("oracle_flip_p95_delta_regret_above_gate")
    total_groups = flips.eval_group_count + flips.near_tie_excluded_count
    near_tie_rate = flips.near_tie_excluded_count / total_groups if total_groups else 0.0
    if near_tie_rate > near_tie_rate_gate:
        failed.append("near_tie_excluded_rate_above_gate")
    return GateResult(passed=not failed, failed_gates=tuple(failed))


def by_bucket_rows(
    config: SweepConfig,
    summaries: T.Sequence[GroupCosts],
    baseline_by_group: T.Mapping[tuple[str, str], GroupCosts],
    *,
    gate_kwargs: dict[str, float],
) -> list[dict[str, T.Any]]:
    """Return per-bucket flip metrics plus the per-bucket cost-sensitivity gate."""
    buckets: dict[str, list[GroupCosts]] = defaultdict(list)
    for item in summaries:
        bucket = item.runtime_bucket or item.condition or item.dataset or "unknown"
        buckets[bucket].append(item)
    rows: list[dict[str, T.Any]] = []
    for bucket, items in sorted(buckets.items()):
        flips = flip_metrics(items, baseline_by_group)
        gate = evaluate_cost_sensitivity_gate(flips, **gate_kwargs)
        rows.append(
            {
                "config_key": config.key,
                "runtime_bucket": bucket,
                "eval_group_count": flips.eval_group_count,
                "near_tie_excluded_count": flips.near_tie_excluded_count,
                "oracle_flip_count": flips.oracle_flip_count,
                "oracle_flip_rate": flips.oracle_flip_rate,
                "oracle_flip_mean_delta_regret": flips.oracle_flip_mean_delta_regret,
                "oracle_flip_p95_delta_regret": flips.oracle_flip_p95_delta_regret,
                "cost_sensitivity_gate_pass": int(gate.passed),
                "cost_sensitivity_failed_gates": "|".join(gate.failed_gates),
            }
        )
    return rows


def oracle_flip_examples(
    summaries: T.Sequence[GroupCosts],
    baseline_by_group: T.Mapping[tuple[str, str], GroupCosts],
    *,
    limit: int,
) -> list[dict[str, T.Any]]:
    """Return concrete oracle-flip examples (config oracle vs baseline oracle)."""
    rows: list[dict[str, T.Any]] = []
    for item in summaries:
        if item.zero_valid or item.near_tie or not item.oracle_candidate:
            continue
        baseline = baseline_by_group.get((item.sample_id, item.face_index))
        if not baseline or not baseline.oracle_candidate:
            continue
        if item.oracle_candidate == baseline.oracle_candidate:
            continue
        delta = max(
            baseline.costs.get(item.oracle_candidate, baseline.oracle_cost) - baseline.oracle_cost,
            0.0,
        )
        rows.append(
            {
                "sample_id": item.sample_id,
                "face_index": item.face_index,
                "runtime_bucket": item.runtime_bucket,
                "baseline_oracle": baseline.oracle_candidate,
                "new_oracle": item.oracle_candidate,
                "baseline_oracle_gap": baseline.oracle_gap,
                "flip_delta_regret": delta,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def parse_csv_floats(raw: str | None, default: T.Sequence[float]) -> tuple[float, ...]:
    """Parse a comma-separated float list, falling back to ``default``."""
    if not raw:
        return tuple(default)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("sweep list must contain at least one number")
    return tuple(values)


def build_configs(args: argparse.Namespace, *, components_available: bool) -> list[SweepConfig]:
    """Build the sweep grid. Collapses cost weights when components are absent."""
    corner_values = parse_csv_floats(args.corner_weights, DEFAULT_SWEEP_VALUES["corner"])
    fit_values = parse_csv_floats(args.fit_weights, DEFAULT_SWEEP_VALUES["fit"])
    soft_values = parse_csv_floats(args.soft_penalties, DEFAULT_SWEEP_VALUES["soft"])
    min_gap_values = parse_csv_floats(args.min_gaps, DEFAULT_SWEEP_VALUES["min_gap"])
    if not components_available:
        corner_values = (BASELINE_WEIGHTS["corner"],)
        fit_values = (BASELINE_WEIGHTS["fit"],)
        soft_values = (BASELINE_WEIGHTS["soft"],)
    return [
        SweepConfig(corner=corner, fit=fit, soft=soft, min_gap=min_gap)
        for corner, fit, soft, min_gap in itertools.product(
            corner_values, fit_values, soft_values, min_gap_values
        )
    ]


def baseline_config() -> SweepConfig:
    """Return the baseline-weight config used as the flip/envelope reference."""
    return SweepConfig(
        corner=BASELINE_WEIGHTS["corner"],
        fit=BASELINE_WEIGHTS["fit"],
        soft=BASELINE_WEIGHTS["soft"],
        min_gap=BASELINE_WEIGHTS["min_gap"],
    )


def write_csv(path: Path, rows: T.Sequence[T.Mapping[str, T.Any]]) -> None:
    """Write ``rows`` to ``path`` as CSV (empty file when no rows)."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: T.Any) -> None:
    """Write ``payload`` to ``path`` as pretty JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def envelope_stability(
    config_records: T.Sequence[dict[str, T.Any]], envelopes: T.Sequence[str]
) -> dict[str, T.Any]:
    """Aggregate flip stability across configs in the given envelope set."""
    selected = [rec for rec in config_records if rec["envelope"] in envelopes]
    flip_rates = [float(rec["oracle_flip_rate"]) for rec in selected]
    flip_costs = [float(rec["oracle_flip_p95_delta_regret"]) for rec in selected]
    stabilities = [1.0 - rate for rate in flip_rates]
    gate_pass = [bool(rec["cost_sensitivity_gate_pass"]) for rec in selected]
    return {
        "config_count": len(selected),
        "mean_oracle_winner_stability": float(statistics.fmean(stabilities))
        if stabilities
        else 1.0,
        "worst_oracle_winner_stability": min(stabilities) if stabilities else 1.0,
        "max_oracle_flip_rate": max(flip_rates) if flip_rates else 0.0,
        "max_oracle_flip_p95_delta_regret": max(flip_costs) if flip_costs else 0.0,
        "cost_sensitivity_gate_pass_rate": (
            float(statistics.fmean([1.0 if ok else 0.0 for ok in gate_pass])) if gate_pass else 1.0
        ),
    }


def main(argv: T.Sequence[str] | None = None) -> int:
    """Run the v3 oracle/cost sensitivity sweep and write the reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="Input scorer-row CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for reports.")
    parser.add_argument("--baseline-policy", default="static_weighted_downweight")
    parser.add_argument("--corner-weights", default=None)
    parser.add_argument("--fit-weights", default=None)
    parser.add_argument("--soft-penalties", default=None)
    parser.add_argument("--min-gaps", default=None)
    parser.add_argument("--joint-ball-threshold", type=float, default=DEFAULT_JOINT_BALL_THRESHOLD)
    parser.add_argument("--flip-rate-gate", type=float, default=DEFAULT_FLIP_RATE_GATE)
    parser.add_argument("--flip-cost-p95-gate", type=float, default=DEFAULT_FLIP_COST_P95_GATE)
    parser.add_argument("--near-tie-rate-gate", type=float, default=DEFAULT_NEAR_TIE_RATE_GATE)
    parser.add_argument("--oracle-flip-example-limit", type=int, default=200)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    rows = load_rows(args.rows)
    if not rows:
        raise SystemExit(f"no rows found in {args.rows}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    components_available = has_component_columns(rows)
    soft_active = soft_penalty_active(rows)
    progress_enabled = args.progress and not args.no_progress
    base_config = baseline_config()
    configs = build_configs(args, components_available=components_available)

    baseline_by_group = {
        (item.sample_id, item.face_index): item
        for item in group_costs(rows, base_config, components_available=components_available)
    }

    gate_kwargs = {
        "flip_rate_gate": args.flip_rate_gate,
        "flip_cost_p95_gate": args.flip_cost_p95_gate,
        "near_tie_rate_gate": args.near_tie_rate_gate,
    }

    config_records: list[dict[str, T.Any]] = []
    bucket_rows: list[dict[str, T.Any]] = []
    flip_rows: list[dict[str, T.Any]] = []
    gate_records: list[dict[str, T.Any]] = []

    for config in progress_iter(
        configs,
        total=len(configs),
        desc="Sweep transform configs",
        unit="config",
        enabled=progress_enabled,
    ):
        config = T.cast(SweepConfig, config)
        summaries = group_costs(rows, config, components_available=components_available)
        flips = flip_metrics(summaries, baseline_by_group)
        gate = evaluate_cost_sensitivity_gate(flips, **gate_kwargs)
        envelope = envelope_for(
            config, base_config, joint_ball_threshold=args.joint_ball_threshold
        )
        config_records.append(
            {
                **asdict(config),
                "config_key": config.key,
                "envelope": envelope,
                "eval_group_count": flips.eval_group_count,
                "near_tie_excluded_count": flips.near_tie_excluded_count,
                "oracle_flip_count": flips.oracle_flip_count,
                "oracle_flip_rate": flips.oracle_flip_rate,
                "oracle_flip_mean_delta_regret": flips.oracle_flip_mean_delta_regret,
                "oracle_flip_p95_delta_regret": flips.oracle_flip_p95_delta_regret,
                "flip_count_by_baseline_gap": flips.flip_count_by_baseline_gap,
                "cost_sensitivity_gate_pass": int(gate.passed),
                "cost_sensitivity_failed_gates": list(gate.failed_gates),
            }
        )
        bucket_rows.extend(
            by_bucket_rows(config, summaries, baseline_by_group, gate_kwargs=gate_kwargs)
        )
        if len(flip_rows) < args.oracle_flip_example_limit:
            remaining = args.oracle_flip_example_limit - len(flip_rows)
            flip_rows.extend(oracle_flip_examples(summaries, baseline_by_group, limit=remaining))
        gate_records.append(
            {
                "config_key": config.key,
                "envelope": envelope,
                "cost_sensitivity_gate_pass": int(gate.passed),
                "cost_sensitivity_failed_gates": list(gate.failed_gates),
                "oracle_flip_rate": flips.oracle_flip_rate,
                "oracle_flip_p95_delta_regret": flips.oracle_flip_p95_delta_regret,
                "near_tie_excluded_count": flips.near_tie_excluded_count,
            }
        )

    summary_payload = {
        "analysis_type": "oracle_cost_sensitivity",
        "promotion_evidence": False,
        "input_rows": str(args.rows),
        "row_count": len(rows),
        "face_group_count": len(grouped_rows(rows)),
        "components_available": components_available,
        "component_mode": "weighted_components"
        if components_available
        else "existing_transform_cost_v3_only",
        "soft_structural_penalty_active": soft_active,
        "baseline_policy": args.baseline_policy,
        "baseline_config": asdict(base_config),
        "joint_ball_threshold": args.joint_ball_threshold,
        "cost_sensitivity_gate_config": gate_kwargs,
        "config_count": len(configs),
        "envelope_config_counts": {
            envelope: sum(1 for rec in config_records if rec["envelope"] == envelope)
            for envelope in ("baseline", "single_axis", "joint_ball", "stress")
        },
        "reasonable_envelope": envelope_stability(config_records, REASONABLE_ENVELOPES),
        "single_axis": envelope_stability(config_records, ("single_axis",)),
        "joint_ball": envelope_stability(config_records, ("joint_ball",)),
        "stress_grid": envelope_stability(config_records, ("stress",)),
        "configs": config_records,
        "notes": [
            "analysis_type=oracle_cost_sensitivity: this is NOT learned-scorer promotion "
            "evidence. Promotion still requires the real scorer run and learnability gates.",
            "Acceptance KPIs use the reasonable_envelope (baseline + single_axis + joint_ball); "
            "stress_grid is the full Cartesian corner grid and is reported for context only.",
            "oracle_flip_*_delta_regret is flip COST in baseline-weighted transform-cost units, "
            "not flip count: near-tie relabels cost ~0, materially different choices cost more.",
            "cost_sensitivity_gate_* are oracle/cost stability gates, distinct from the v3 "
            "learnability promotion gates.",
            "soft_structural_penalty_active=false means no row carried a soft penalty, so the "
            "soft sweep is inert on this dataset.",
            "When components_available is false, only min_gap near-tie behavior is meaningful.",
        ],
    }

    write_json(args.output_dir / "v3_cost_sensitivity_summary.json", summary_payload)
    write_csv(args.output_dir / "v3_cost_sensitivity_by_bucket.csv", bucket_rows)
    write_csv(args.output_dir / "v3_oracle_flip_examples.csv", flip_rows)
    write_json(args.output_dir / "v3_cost_sensitivity_gates.json", gate_records)

    print(
        json.dumps(
            {
                "analysis_type": "oracle_cost_sensitivity",
                "output_dir": str(args.output_dir),
                "row_count": len(rows),
                "config_count": len(configs),
                "components_available": components_available,
                "soft_structural_penalty_active": soft_active,
                "reasonable_envelope_mean_oracle_winner_stability": summary_payload[
                    "reasonable_envelope"
                ]["mean_oracle_winner_stability"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
