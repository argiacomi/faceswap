#!/usr/bin/env python3
"""Merge dataset manifests into a prioritized profile/occlusion hard-negative manifest.

The landmark-resolver scorer needs more profile/occlusion hard negatives than a
naturally-sampled manifest provides. This tool reads any subset of the supported
landmark dataset manifests, classifies every sample into one of four buckets
(``profile_occlusion`` > ``profile`` > ``occlusion`` > ``anchor``) using
:mod:`lib.landmarks.datasets.hard_negative_mining`, dedupes by source key, then
quota-samples each bucket deterministically so future scorer runs are
reproducible.

Outputs (written to ``--output-dir``):

* ``manifest.json`` - merged, annotated samples (original fields preserved).
* ``hard_negative_mix.json`` - bucket counts, per-dataset counts, and weights.
* ``dataset_audit.json`` - per-dataset classification breakdown (``--write-audit``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import typing as T
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.landmarks.datasets.hard_negative_mining import (
    BUCKET_PRIORITY,
    BUCKET_WEIGHT,
    HardNegativeClass,
    annotate_sample,
    classify_hard_negative,
    source_key,
)

logger = logging.getLogger(__name__)

# Datasets that should fall back to a fixed bucket when label-based
# classification is inconclusive. Mirrors the priority table in the ticket:
# COFW is occlusion-only hard negatives, 300W is frontal anchors/controls.
DATASET_DEFAULT_BUCKET: dict[str, str] = {
    "cofw": "occlusion",
    "cofw68": "occlusion",
    "300w": "anchor",
    "w300": "anchor",
}

# Stable ordering for reports/quota iteration (highest priority first).
BUCKET_ORDER: tuple[str, ...] = ("profile_occlusion", "profile", "occlusion", "anchor")

# CLI manifest argument -> canonical dataset label.
MANIFEST_ARGS: tuple[tuple[str, str], ...] = (
    ("wflw_manifest", "wflw"),
    ("aflw2000_manifest", "aflw2000-3d"),
    ("merl_rav_manifest", "merl-rav"),
    ("cofw_manifest", "cofw"),
    ("menpo2d_manifest", "menpo2d"),
    ("multipie_manifest", "multipie"),
    ("w300_manifest", "300w"),
)


def _default_class(dataset: str) -> HardNegativeClass | None:
    """Return the configured default bucket classification for ``dataset``."""
    bucket = DATASET_DEFAULT_BUCKET.get(dataset.strip().lower())
    if bucket is None:
        return None
    return HardNegativeClass(
        bucket=bucket,
        priority=BUCKET_PRIORITY[bucket],
        weight=BUCKET_WEIGHT[bucket],
        reasons=(f"{dataset}_default",),
    )


def _load_samples(manifest_path: Path) -> list[dict[str, T.Any]]:
    """Return raw sample dicts from a manifest JSON file."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload.get("scenarios", []))
    if not isinstance(samples, list):
        raise ValueError(f"manifest {manifest_path} has no 'samples' list")
    return [dict(entry) for entry in samples if isinstance(entry, T.Mapping)]


def _stable_order(samples: T.Sequence[dict[str, T.Any]], *, seed: int) -> list[dict[str, T.Any]]:
    """Return ``samples`` ordered by a deterministic seeded hash."""

    def _hash(sample: dict[str, T.Any]) -> str:
        dataset, source_id = source_key(sample)
        return hashlib.sha256(f"{seed}|{dataset}|{source_id}".encode()).hexdigest()

    return sorted(samples, key=_hash)


def build_hard_negative_manifest(
    *,
    manifests: T.Mapping[str, Path],
    output_dir: Path,
    max_profile_occlusion: int | None = None,
    max_profile: int | None = None,
    max_occlusion: int | None = None,
    max_anchors: int | None = None,
    allow_overlap: bool = False,
    seed: int = 1337,
    write_audit: bool = False,
) -> dict[str, T.Any]:
    """Build a prioritized hard-negative manifest from dataset manifests.

    ``manifests`` maps canonical dataset labels to manifest JSON paths. Returns
    the in-memory mix report. Writes ``manifest.json`` and ``hard_negative_mix
    .json`` (and ``dataset_audit.json`` when ``write_audit`` is set) under
    ``output_dir``.
    """
    quotas: dict[str, int | None] = {
        "profile_occlusion": max_profile_occlusion,
        "profile": max_profile,
        "occlusion": max_occlusion,
        "anchor": max_anchors,
    }

    classified: dict[str, list[dict[str, T.Any]]] = {bucket: [] for bucket in BUCKET_ORDER}
    audit: dict[str, dict[str, int]] = {}
    seen_keys: set[tuple[str, str]] = set()

    for dataset, manifest_path in manifests.items():
        dataset_label = dataset.strip().lower()
        dataset_counts: dict[str, int] = dict.fromkeys((*BUCKET_ORDER, "skipped", "duplicate"), 0)
        for sample in _load_samples(manifest_path):
            sample.setdefault("dataset", dataset_label)
            classification = classify_hard_negative(sample) or _default_class(dataset_label)
            if classification is None:
                dataset_counts["skipped"] += 1
                continue
            key = source_key(sample)
            if not allow_overlap and key in seen_keys:
                dataset_counts["duplicate"] += 1
                continue
            seen_keys.add(key)
            annotated = annotate_sample(sample, classification)
            annotated.setdefault("dataset", dataset_label)
            classified[classification.bucket].append(annotated)
            dataset_counts[classification.bucket] += 1
        audit[dataset_label] = dataset_counts

    selected: list[dict[str, T.Any]] = []
    counts: dict[str, int] = {}
    by_dataset: dict[str, dict[str, int]] = {}
    for bucket in BUCKET_ORDER:
        ordered = _stable_order(classified[bucket], seed=seed)
        limit = quotas[bucket]
        if limit is not None:
            ordered = ordered[: max(limit, 0)]
        counts[bucket] = len(ordered)
        for sample in ordered:
            dataset_label = str(sample.get("dataset", "")).strip().lower() or "unknown"
            by_dataset.setdefault(dataset_label, {})
            by_dataset[dataset_label][bucket] = by_dataset[dataset_label].get(bucket, 0) + 1
        selected.extend(ordered)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {"samples": selected}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    mix_report = {
        "counts": counts,
        "by_dataset": by_dataset,
        "weights": dict(BUCKET_WEIGHT),
        "total": len(selected),
        "seed": seed,
        "allow_overlap": allow_overlap,
    }
    (output_dir / "hard_negative_mix.json").write_text(
        json.dumps(mix_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    if write_audit:
        (output_dir / "dataset_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )

    logger.info(
        "Wrote %d hard-negative samples to %s (%s)",
        len(selected),
        output_dir / "manifest.json",
        ", ".join(f"{bucket}={counts[bucket]}" for bucket in BUCKET_ORDER),
    )
    return mix_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wflw-manifest", type=Path)
    parser.add_argument("--aflw2000-manifest", type=Path)
    parser.add_argument("--merl-rav-manifest", type=Path)
    parser.add_argument("--cofw-manifest", type=Path)
    parser.add_argument("--menpo2d-manifest", type=Path)
    parser.add_argument("--multipie-manifest", type=Path)
    parser.add_argument("--w300-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-profile-occlusion", type=int, default=None)
    parser.add_argument("--max-profile", type=int, default=None)
    parser.add_argument("--max-occlusion", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--write-audit", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from lib.logger import configure_tool_logging

    configure_tool_logging(args.log_level)

    manifests: dict[str, Path] = {}
    for attr, dataset_label in MANIFEST_ARGS:
        path = getattr(args, attr)
        if path is not None:
            manifests[dataset_label] = path
    if not manifests:
        _parser().error("at least one dataset manifest is required")

    build_hard_negative_manifest(
        manifests=manifests,
        output_dir=args.output_dir,
        max_profile_occlusion=args.max_profile_occlusion,
        max_profile=args.max_profile,
        max_occlusion=args.max_occlusion,
        max_anchors=args.max_anchors,
        allow_overlap=args.allow_overlap,
        seed=args.seed,
        write_audit=args.write_audit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
