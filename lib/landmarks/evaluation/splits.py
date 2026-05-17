#!/usr/bin/env python3
"""Fit/select/report splits for the landmark static weight pipeline (#67).

The terminology is ``fit`` / ``select`` / ``report`` (not train / val / test)
because static weight derivation is closed-form and does not train model
parameters: ``fit`` samples produce weights, ``select`` samples compare
candidate setups, and ``report`` samples are reserved for held-out final
metrics.

Splits are scenario-stratified by default. Each manifest sample is bucketed by
``f"{dataset}:{condition}"`` and the splitter preserves bucket coverage across
fit/select/report when bucket size allows. Random split is provided for
debugging but must not be the pipeline default because dataset sizes are
imbalanced and the harness already evaluates by scenario bucket.
"""

from __future__ import annotations

import hashlib
import json
import typing as T
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLIT_NAMES: tuple[str, ...] = ("fit", "select", "report")
SPLIT_MODES: tuple[str, ...] = ("scenario-stratified", "random", "file", "none")
SCENARIO_STRATIFIED: str = "scenario-stratified"
RANDOM: str = "random"
FROM_FILE: str = "file"
NONE: str = "none"

SPLIT_FILE_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class SplitRatios:
    """Ratios that fit/select/report take from the full sample pool."""

    fit: float
    select: float
    report: float

    def __post_init__(self) -> None:
        total = self.fit + self.select + self.report
        if not 0.99 < total < 1.01:
            raise ValueError(f"split ratios must sum to 1.0, got {total!r}")
        if min(self.fit, self.select, self.report) <= 0:
            raise ValueError("every split ratio must be greater than zero")

    def as_dict(self) -> dict[str, float]:
        return {"fit": float(self.fit), "select": float(self.select), "report": float(self.report)}


@dataclass(frozen=True)
class SplitAssignment:
    """Three-way sample-id assignment for one pipeline run."""

    fit: tuple[str, ...]
    select: tuple[str, ...]
    report: tuple[str, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for name in SPLIT_NAMES:
            for sid in getattr(self, name):
                if sid in seen:
                    raise ValueError(f"sample {sid!r} appears in more than one split")
                seen.add(sid)

    @property
    def sample_count(self) -> int:
        return len(self.fit) + len(self.select) + len(self.report)

    def ids_for(self, name: str) -> tuple[str, ...]:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}; supported: {', '.join(SPLIT_NAMES)}")
        return T.cast(tuple[str, ...], getattr(self, name))

    def to_payload(self) -> dict[str, list[str]]:
        return {name: list(self.ids_for(name)) for name in SPLIT_NAMES}

    @classmethod
    def from_payload(cls, payload: T.Mapping[str, T.Any]) -> SplitAssignment:
        missing = [name for name in SPLIT_NAMES if name not in payload]
        if missing:
            raise ValueError(f"split payload missing required keys: {missing}")
        return cls(
            fit=tuple(str(item) for item in payload["fit"]),
            select=tuple(str(item) for item in payload["select"]),
            report=tuple(str(item) for item in payload["report"]),
        )


@dataclass(frozen=True)
class BucketDiagnostic:
    """Per-bucket split coverage diagnostic."""

    bucket: str
    fit: int
    select: int
    report: int
    total: int
    too_small: bool

    def to_payload(self) -> dict[str, T.Any]:
        return {
            "bucket": self.bucket,
            "fit": self.fit,
            "select": self.select,
            "report": self.report,
            "total": self.total,
            "too_small": self.too_small,
        }


def sample_id(sample: T.Mapping[str, T.Any]) -> str:
    """Return the canonical sample identifier from a manifest entry."""
    value = sample.get("sample_id") or sample.get("id") or sample.get("name")
    if not value:
        raise ValueError(f"manifest sample missing sample_id/id/name: {sample!r}")
    return str(value)


def scenario_bucket(sample: T.Mapping[str, T.Any]) -> str:
    """Return ``dataset:condition`` for a manifest sample (defaults to ``unspecified``)."""
    dataset = sample.get("dataset") or "unspecified"
    condition = sample.get("condition") or sample.get("scenario") or "unspecified"
    return f"{dataset}:{condition}"


def split_manifest_samples(
    samples: T.Sequence[T.Mapping[str, T.Any]],
    *,
    mode: str,
    ratios: SplitRatios,
    seed: int,
) -> tuple[SplitAssignment, list[BucketDiagnostic]]:
    """Compute a SplitAssignment for ``samples`` using ``mode``.

    ``mode`` must be ``scenario-stratified`` or ``random``. ``file`` and
    ``none`` are pipeline-level modes consumed elsewhere and are not valid
    inputs here.
    """
    if mode not in {SCENARIO_STRATIFIED, RANDOM}:
        raise ValueError(
            f"mode must be one of {SCENARIO_STRATIFIED!r} or {RANDOM!r}; got {mode!r}"
        )
    if not samples:
        raise ValueError("samples cannot be empty")
    sample_ids = [sample_id(sample) for sample in samples]
    duplicates = sorted({sid for sid in sample_ids if sample_ids.count(sid) > 1})
    if duplicates:
        raise ValueError(f"sample_ids must be unique; duplicates: {duplicates}")

    if mode == RANDOM:
        return _random_split(sample_ids, ratios=ratios, seed=seed)

    buckets: dict[str, list[str]] = {}
    for sample, sid in zip(samples, sample_ids, strict=True):
        buckets.setdefault(scenario_bucket(sample), []).append(sid)
    return _stratified_split(buckets, ratios=ratios, seed=seed)


def _shuffled(items: T.Sequence[str], *, rng: np.random.Generator) -> list[str]:
    """Sort ``items`` for determinism then shuffle in place with ``rng``."""
    ordered = sorted(items)
    rng.shuffle(ordered)
    return ordered


def _partition_counts(total: int, ratios: SplitRatios) -> tuple[int, int, int]:
    """Return (fit, select, report) counts for ``total`` samples honoring ratios.

    Each split receives at least one sample when ``total >= 3``. For ``total``
    of 1 or 2 the caller is expected to handle the bucket as ``too_small``.
    """
    if total < 3:
        raise ValueError(f"_partition_counts requires total >= 3, got {total}")
    n_fit = max(1, int(round(ratios.fit * total)))
    n_select = max(1, int(round(ratios.select * total)))
    n_report = total - n_fit - n_select
    # Re-balance if rounding squeezed report below one.
    while n_report < 1:
        if n_fit > n_select and n_fit > 1:
            n_fit -= 1
        elif n_select > 1:
            n_select -= 1
        else:
            n_fit -= 1
        n_report = total - n_fit - n_select
    return n_fit, n_select, n_report


def _random_split(
    sample_ids: T.Sequence[str],
    *,
    ratios: SplitRatios,
    seed: int,
) -> tuple[SplitAssignment, list[BucketDiagnostic]]:
    """Single-bucket random split treating all samples as one pool."""
    rng = np.random.default_rng(seed)
    shuffled = _shuffled(sample_ids, rng=rng)
    total = len(shuffled)
    if total < 3:
        # Treat as a too-small single bucket.
        fit, select, report = _allocate_too_small(shuffled)
        too_small = True
    else:
        n_fit, n_select, _ = _partition_counts(total, ratios)
        fit = shuffled[:n_fit]
        select = shuffled[n_fit : n_fit + n_select]
        report = shuffled[n_fit + n_select :]
        too_small = False
    diagnostic = BucketDiagnostic(
        bucket="__all__",
        fit=len(fit),
        select=len(select),
        report=len(report),
        total=total,
        too_small=too_small,
    )
    return (
        SplitAssignment(fit=tuple(fit), select=tuple(select), report=tuple(report)),
        [diagnostic],
    )


def _stratified_split(
    buckets: T.Mapping[str, T.Sequence[str]],
    *,
    ratios: SplitRatios,
    seed: int,
) -> tuple[SplitAssignment, list[BucketDiagnostic]]:
    """Stratified split that preserves bucket coverage when sample counts allow."""
    fit: list[str] = []
    select: list[str] = []
    report: list[str] = []
    diagnostics: list[BucketDiagnostic] = []
    base_rng = np.random.default_rng(seed)
    for bucket in sorted(buckets):
        members = list(buckets[bucket])
        bucket_seed = int(base_rng.integers(0, 2**63 - 1))
        rng = np.random.default_rng(bucket_seed)
        shuffled = _shuffled(members, rng=rng)
        total = len(shuffled)
        if total < 3:
            f_ids, s_ids, r_ids = _allocate_too_small(shuffled)
            too_small = True
        else:
            n_fit, n_select, _ = _partition_counts(total, ratios)
            f_ids = shuffled[:n_fit]
            s_ids = shuffled[n_fit : n_fit + n_select]
            r_ids = shuffled[n_fit + n_select :]
            too_small = False
        fit.extend(f_ids)
        select.extend(s_ids)
        report.extend(r_ids)
        diagnostics.append(
            BucketDiagnostic(
                bucket=bucket,
                fit=len(f_ids),
                select=len(s_ids),
                report=len(r_ids),
                total=total,
                too_small=too_small,
            )
        )
    return (
        SplitAssignment(fit=tuple(fit), select=tuple(select), report=tuple(report)),
        diagnostics,
    )


def _allocate_too_small(shuffled: T.Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    """Distribute 1 or 2 samples into fit (preferred) and select."""
    if len(shuffled) == 0:
        return [], [], []
    if len(shuffled) == 1:
        return list(shuffled), [], []
    return [shuffled[0]], [shuffled[1]], []


def split_assignment_hash(assignment: SplitAssignment) -> str:
    """Stable ``sha256:...`` hash over the sorted sample IDs per split."""
    payload = json.dumps(
        {name: sorted(assignment.ids_for(name)) for name in SPLIT_NAMES},
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_split_file(path: str | Path) -> SplitAssignment:
    """Load a SplitAssignment from a ``splits.json`` file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"split file must contain a JSON object, got {type(raw).__name__}")
    payload = raw.get("splits", raw)
    return SplitAssignment.from_payload(payload)


def save_split_file(
    path: str | Path,
    assignment: SplitAssignment,
    *,
    mode: str,
    ratios: SplitRatios,
    seed: int,
    diagnostics: T.Sequence[BucketDiagnostic],
) -> Path:
    """Write a versioned ``splits.json`` artifact alongside diagnostics."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SPLIT_FILE_SCHEMA_VERSION,
        "mode": mode,
        "ratios": ratios.as_dict(),
        "seed": int(seed),
        "assignment_hash": split_assignment_hash(assignment),
        "splits": assignment.to_payload(),
        "counts": {name: len(assignment.ids_for(name)) for name in SPLIT_NAMES},
        "diagnostics": [diagnostic.to_payload() for diagnostic in diagnostics],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def filter_manifest_samples(
    samples: T.Sequence[T.Mapping[str, T.Any]],
    sample_ids: T.Iterable[str],
) -> list[dict[str, T.Any]]:
    """Return manifest samples whose id is in ``sample_ids`` (preserves manifest order)."""
    wanted = set(sample_ids)
    by_id: dict[str, T.Mapping[str, T.Any]] = {}
    for sample in samples:
        sid = sample_id(sample)
        by_id[sid] = sample
    missing = sorted(wanted - by_id.keys())
    if missing:
        raise ValueError(f"split references missing samples: {missing}")
    return [dict(sample) for sid, sample in by_id.items() if sid in wanted]


def write_split_manifest(
    output_path: str | Path,
    base_manifest: T.Mapping[str, T.Any],
    sample_ids: T.Iterable[str],
) -> Path:
    """Write a filtered manifest containing only the named samples.

    Preserves any non-sample top-level metadata. The output uses the canonical
    ``samples`` key (legacy manifests stored them under ``scenarios``).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = base_manifest.get("samples", base_manifest.get("scenarios", []))
    filtered = filter_manifest_samples(samples, sample_ids)
    payload: dict[str, T.Any] = {
        key: value for key, value in base_manifest.items() if key not in {"samples", "scenarios"}
    }
    payload["samples"] = filtered
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def split_summary_counts(
    assignment: SplitAssignment,
    samples: T.Sequence[T.Mapping[str, T.Any]],
) -> dict[str, T.Any]:
    """Return per-split counts grouped by dataset, condition, and scenario bucket.

    Suitable for inclusion in ``run_summary.json`` under a ``splits`` block.
    """
    by_id = {sample_id(sample): sample for sample in samples}
    counts: dict[str, dict[str, dict[str, int]]] = {
        "by_dataset": {name: {} for name in SPLIT_NAMES},
        "by_condition": {name: {} for name in SPLIT_NAMES},
        "by_scenario_bucket": {name: {} for name in SPLIT_NAMES},
    }
    for split_name in SPLIT_NAMES:
        for sid in assignment.ids_for(split_name):
            sample = by_id.get(sid)
            if sample is None:
                raise ValueError(f"split sample {sid!r} not found in manifest")
            for key, value in (
                ("by_dataset", str(sample.get("dataset") or "unspecified")),
                (
                    "by_condition",
                    str(sample.get("condition") or sample.get("scenario") or "unspecified"),
                ),
                ("by_scenario_bucket", scenario_bucket(sample)),
            ):
                counts[key][split_name][value] = counts[key][split_name].get(value, 0) + 1
    counts["totals"] = {name: len(assignment.ids_for(name)) for name in SPLIT_NAMES}
    return counts


__all__ = [
    "BucketDiagnostic",
    "FROM_FILE",
    "NONE",
    "RANDOM",
    "SCENARIO_STRATIFIED",
    "SPLIT_FILE_SCHEMA_VERSION",
    "SPLIT_MODES",
    "SPLIT_NAMES",
    "SplitAssignment",
    "SplitRatios",
    "filter_manifest_samples",
    "load_split_file",
    "sample_id",
    "save_split_file",
    "scenario_bucket",
    "split_assignment_hash",
    "split_manifest_samples",
    "split_summary_counts",
    "write_split_manifest",
]
