#!/usr/bin/env python3
"""GT-geometry signal validation primitives (#80).

For each manifest sample we run every candidate (single models + ensemble
variants) and compare them against the GT-derived AlignedFace geometry. The
"oracle" per sample is the candidate with the lowest GT-geometry score; every
other candidate is, by construction, a worse choice for that sample.

This module exposes two complementary analyses:

* **Signal validation** — for each candidate signal (NME, transform error,
  hull IoU, crop center error, …) and each percentile threshold, compute
  precision / recall / AUC at predicting whether a candidate is "bad"
  (geometry score above the per-sample oracle by more than a margin). The
  question the report answers is: which signal would best identify bad
  candidates if we used it as a selector?

* **Selector ablations** — for each named selector strategy (lowest NME,
  lowest transform error, composite geometry score, oracle, …) count how
  often it picks the oracle's choice for each sample, broken down by
  scenario / hard-slice bucket. Answers: which selector matches the oracle
  most often on held-out data?

Both analyses are cache-only: they consume the same fused-prediction code
the search and geometry CLIs use.
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.landmarks.evaluation.geometry_metrics import GeometrySampleMetrics

DEFAULT_BAD_CANDIDATE_MARGIN: float = 0.05
"""How much worse than the per-sample oracle a candidate's geometry score must
be before we consider it a "bad" alternative."""


@dataclass(frozen=True)
class CandidateRecord:
    """A single (sample, candidate) row used by the validation analyses."""

    sample_id: str
    dataset: str
    condition: str
    hard_slice: str
    candidate_label: str
    is_baseline: bool
    geometry_score: float  # alignment_geometry_v1 (GT-derived)
    nme: float  # legacy point-error diagnostic
    transform_normalized: float  # matrix translation_normalized_distance
    crop_center_normalized: float  # ROI center / bbox-diagonal
    roll_degrees_delta: float
    hull_iou: float
    catastrophic: bool
    is_oracle: bool = False  # True when this is the per-sample best


@dataclass(frozen=True)
class SignalReport:
    """Precision/recall/AUC for one signal at a chosen threshold."""

    name: str
    direction: str  # "higher_is_worse" or "lower_is_worse"
    threshold: float
    precision: float
    recall: float
    auc: float

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "threshold": float(self.threshold),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "auc": float(self.auc),
        }


@dataclass(frozen=True)
class SelectorReport:
    """How often a selector strategy picks the oracle's choice."""

    name: str
    sample_count: int
    oracle_match_rate: float
    mean_score_gap_vs_oracle: float
    per_bucket: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "name": self.name,
            "sample_count": int(self.sample_count),
            "oracle_match_rate": float(self.oracle_match_rate),
            "mean_score_gap_vs_oracle": float(self.mean_score_gap_vs_oracle),
            "per_bucket": {bucket: dict(values) for bucket, values in self.per_bucket.items()},
        }


def tag_oracle(records: T.Sequence[CandidateRecord]) -> list[CandidateRecord]:
    """Mark the lowest-geometry-score candidate per sample as the oracle.

    Ties go to the earliest record (stable ordering); the goal is just to give
    each sample a single ground-truth best candidate for downstream comparison.
    """
    by_sample: dict[str, list[CandidateRecord]] = {}
    for record in records:
        by_sample.setdefault(record.sample_id, []).append(record)

    oracle_ids: dict[str, str] = {}
    for sample_id, items in by_sample.items():
        winner = min(items, key=lambda r: r.geometry_score)
        oracle_ids[sample_id] = winner.candidate_label

    tagged: list[CandidateRecord] = []
    for record in records:
        is_oracle = oracle_ids.get(record.sample_id) == record.candidate_label
        if is_oracle == record.is_oracle:
            tagged.append(record)
        else:
            tagged.append(_replace_oracle(record, is_oracle))
    return tagged


def _replace_oracle(record: CandidateRecord, is_oracle: bool) -> CandidateRecord:
    return CandidateRecord(
        sample_id=record.sample_id,
        dataset=record.dataset,
        condition=record.condition,
        hard_slice=record.hard_slice,
        candidate_label=record.candidate_label,
        is_baseline=record.is_baseline,
        geometry_score=record.geometry_score,
        nme=record.nme,
        transform_normalized=record.transform_normalized,
        crop_center_normalized=record.crop_center_normalized,
        roll_degrees_delta=record.roll_degrees_delta,
        hull_iou=record.hull_iou,
        catastrophic=record.catastrophic,
        is_oracle=is_oracle,
    )


def label_bad_candidates(
    records: T.Sequence[CandidateRecord],
    *,
    margin: float = DEFAULT_BAD_CANDIDATE_MARGIN,
) -> np.ndarray:
    """Return a boolean array marking candidates whose geometry score exceeds
    the per-sample oracle by more than ``margin``."""
    by_sample: dict[str, float] = {}
    for record in records:
        existing = by_sample.get(record.sample_id)
        if existing is None or record.geometry_score < existing:
            by_sample[record.sample_id] = record.geometry_score
    labels = np.zeros(len(records), dtype=bool)
    for idx, record in enumerate(records):
        labels[idx] = (record.geometry_score - by_sample[record.sample_id]) > margin
    return labels


def _values_for_signal(name: str, records: T.Sequence[CandidateRecord]) -> tuple[np.ndarray, str]:
    """Return per-record signal values + direction (higher_is_worse / lower_is_worse)."""
    if name == "nme":
        return np.array([r.nme for r in records]), "higher_is_worse"
    if name == "transform_normalized":
        return np.array([r.transform_normalized for r in records]), "higher_is_worse"
    if name == "crop_center_normalized":
        return np.array([r.crop_center_normalized for r in records]), "higher_is_worse"
    if name == "roll_degrees_delta":
        return np.array([r.roll_degrees_delta for r in records]), "higher_is_worse"
    if name == "hull_iou":
        return np.array([r.hull_iou for r in records]), "lower_is_worse"
    if name == "geometry_score":
        return np.array([r.geometry_score for r in records]), "higher_is_worse"
    raise KeyError(f"unknown signal {name!r}")


def _classify(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    """Return a boolean mask of candidates flagged as bad by this threshold.

    Comparisons are inclusive so a threshold landing exactly on the bad-cluster
    boundary still flags those candidates — important for bimodal signals.
    """
    if direction == "higher_is_worse":
        return values >= threshold
    return values <= threshold


def validate_signal(
    records: T.Sequence[CandidateRecord],
    *,
    signal: str,
    margin: float = DEFAULT_BAD_CANDIDATE_MARGIN,
    threshold_quantile: float = 0.75,
) -> SignalReport:
    """Score one signal at its ``threshold_quantile`` cut-point.

    AUC is approximated by sweeping thresholds across the values and averaging
    the true-positive / false-positive trade-off. Precision and recall are
    reported at the requested quantile so operators can read a single
    actionable number.
    """
    labels = label_bad_candidates(records, margin=margin)
    values, direction = _values_for_signal(signal, records)
    if len(values) == 0:
        return SignalReport(signal, direction, 0.0, 0.0, 0.0, 0.0)
    quantile = threshold_quantile if direction == "higher_is_worse" else 1.0 - threshold_quantile
    threshold = float(np.quantile(values, max(0.0, min(1.0, quantile))))
    predicted = _classify(values, threshold, direction)
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    auc = _approx_auc(values, labels, direction)
    return SignalReport(signal, direction, threshold, precision, recall, auc)


def _approx_auc(values: np.ndarray, labels: np.ndarray, direction: str) -> float:
    """Trapezoidal ROC-AUC over a 20-step threshold sweep."""
    if len(values) == 0 or not labels.any() or labels.all():
        return 0.0
    ordered = np.sort(np.unique(values))
    if len(ordered) <= 1:
        return 0.0
    thresholds = np.linspace(ordered[0], ordered[-1], 20)
    points: list[tuple[float, float]] = []
    for threshold in thresholds:
        predicted = _classify(values, threshold, direction)
        tp = int(np.sum(predicted & labels))
        fp = int(np.sum(predicted & ~labels))
        fn = int(np.sum(~predicted & labels))
        tn = int(np.sum(~predicted & ~labels))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        points.append((fpr, tpr))
    points.sort()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(np.trapezoid(ys, xs))


SELECTORS: tuple[tuple[str, str, str], ...] = (
    ("lowest_nme", "nme", "higher_is_worse"),
    ("lowest_transform_error", "transform_normalized", "higher_is_worse"),
    ("lowest_crop_center_error", "crop_center_normalized", "higher_is_worse"),
    ("lowest_roll_error", "roll_degrees_delta", "higher_is_worse"),
    ("highest_hull_iou", "hull_iou", "lower_is_worse"),
    ("composite_geometry", "geometry_score", "higher_is_worse"),
)


def selector_pick(
    records: T.Sequence[CandidateRecord], signal: str, direction: str
) -> CandidateRecord:
    """Return the best candidate per the requested signal."""
    values, _ = _values_for_signal(signal, records)
    idx = int(np.argmin(values)) if direction == "higher_is_worse" else int(np.argmax(values))
    return records[idx]


def evaluate_selector(
    records: T.Sequence[CandidateRecord], *, name: str, signal: str, direction: str
) -> SelectorReport:
    """Run one selector across all samples; report oracle-match rate."""
    by_sample: dict[str, list[CandidateRecord]] = {}
    for record in records:
        by_sample.setdefault(record.sample_id, []).append(record)
    matches: list[bool] = []
    gaps: list[float] = []
    per_bucket: dict[str, dict[str, list[float]]] = {}
    for _sample_id, items in by_sample.items():
        oracle = min(items, key=lambda r: r.geometry_score)
        chosen = selector_pick(items, signal=signal, direction=direction)
        match = chosen.candidate_label == oracle.candidate_label
        gap = chosen.geometry_score - oracle.geometry_score
        matches.append(match)
        gaps.append(gap)
        bucket = (
            chosen.hard_slice
            or f"{chosen.dataset or 'unspecified'}:{chosen.condition or 'unspecified'}"
        )
        per_bucket.setdefault(bucket, {"match": [], "gap": []})
        per_bucket[bucket]["match"].append(float(match))
        per_bucket[bucket]["gap"].append(float(gap))

    per_bucket_payload = {
        bucket: {
            "sample_count": float(len(values["match"])),
            "oracle_match_rate": float(np.mean(values["match"])),
            "mean_score_gap_vs_oracle": float(np.mean(values["gap"])),
        }
        for bucket, values in per_bucket.items()
    }
    return SelectorReport(
        name=name,
        sample_count=len(matches),
        oracle_match_rate=float(np.mean(matches)) if matches else 0.0,
        mean_score_gap_vs_oracle=float(np.mean(gaps)) if gaps else 0.0,
        per_bucket=per_bucket_payload,
    )


def evaluate_signals(
    records: T.Sequence[CandidateRecord],
    *,
    margin: float = DEFAULT_BAD_CANDIDATE_MARGIN,
    threshold_quantile: float = 0.75,
    signals: T.Sequence[str] = (
        "nme",
        "transform_normalized",
        "crop_center_normalized",
        "roll_degrees_delta",
        "hull_iou",
        "geometry_score",
    ),
) -> list[SignalReport]:
    """Validate every named signal against the bad-candidate labels."""
    return [
        validate_signal(
            records,
            signal=name,
            margin=margin,
            threshold_quantile=threshold_quantile,
        )
        for name in signals
    ]


def evaluate_selectors(
    records: T.Sequence[CandidateRecord],
    *,
    selectors: T.Sequence[tuple[str, str, str]] = SELECTORS,
) -> list[SelectorReport]:
    """Run every named selector across all samples."""
    return [
        evaluate_selector(records, name=name, signal=signal, direction=direction)
        for name, signal, direction in selectors
    ]


def candidate_record_from_geometry(
    metrics: GeometrySampleMetrics,
    *,
    candidate_label: str,
    nme: float,
    is_baseline: bool = False,
    hard_slice: str = "",
) -> CandidateRecord:
    """Convert a :class:`GeometrySampleMetrics` into a CandidateRecord."""
    return CandidateRecord(
        sample_id=metrics.sample_id,
        dataset=metrics.dataset,
        condition=metrics.condition,
        hard_slice=hard_slice or metrics.condition,
        candidate_label=candidate_label,
        is_baseline=is_baseline,
        geometry_score=metrics.overall_score,
        nme=float(nme),
        transform_normalized=metrics.matrix_delta.translation_normalized_distance,
        crop_center_normalized=metrics.roi_delta.center_normalized_distance,
        roll_degrees_delta=metrics.pose_delta.roll_delta_degrees,
        hull_iou=metrics.hull_iou,
        catastrophic=metrics.catastrophic_flags.any,
    )


__all__ = [
    "CandidateRecord",
    "DEFAULT_BAD_CANDIDATE_MARGIN",
    "SELECTORS",
    "SelectorReport",
    "SignalReport",
    "candidate_record_from_geometry",
    "evaluate_selector",
    "evaluate_selectors",
    "evaluate_signal",
    "evaluate_signals",
    "label_bad_candidates",
    "selector_pick",
    "tag_oracle",
    "validate_signal",
]


# Backwards-compatible alias mirroring the doc string "evaluate_signal" name.
evaluate_signal = validate_signal
