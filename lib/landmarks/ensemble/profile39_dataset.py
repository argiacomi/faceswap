#!/usr/bin/env python3
"""Parallel 39-point profile row loader for the profile specialist (#218/#219).

39-point profile GT is filtered out of the canonical-68 scorer path
(:func:`lib.landmarks.datasets.manifest_io.filter_canonical_68_samples`) because
it cannot be scored against 68-point output. This module is the *separate*
partial-schema path: it loads only the 39-point samples, builds the same 68-point
candidates from the prediction cache, and scores them with
:mod:`lib.landmarks.evaluation.profile39` (visible-side error, 39-point transform
cost / regret / oracle). The resulting rows feed ``learned_quality_v3_profile``
training and a profile-only report, and never enter the canonical-68 scorer
target.
"""

from __future__ import annotations

import logging
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.datasets.manifest_io import (
    LandmarkSample,
    bbox_from_truth_fallback,
    load_manifest,
    sample_is_canonical_68,
)
from lib.landmarks.ensemble.profile_features import profile_side_from_context
from lib.landmarks.ensemble.runtime_features import candidate_feature_map
from lib.landmarks.ensemble.runtime_resolver import build_candidates
from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_OUTLIER_THRESHOLD,
    _candidate_config,
    _candidate_extra_features,
    _metric_for_candidate,
    _read_predictions,
    _requested_candidate_sets,
    parse_candidates,
)
from lib.landmarks.ensemble.weights import load_optional_weight_blocks, load_weights
from lib.landmarks.evaluation.profile39 import (
    PROFILE39_POINT_COUNT,
    profile39_rows,
    profile39_side_from_sample,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile39Row:
    """One (sample, candidate) row scored against 39-point profile GT."""

    sample_id: str
    dataset: str
    side: str
    candidate_name: str
    profile39_transform_cost: float
    profile39_transform_regret: float
    profile39_visible_side_error: float
    profile39_oracle_candidate: str
    profile39_is_oracle: bool
    profile39_rankable: bool
    feature_values: dict[str, float] = field(default_factory=dict)


def _sample_label_mapping(sample: LandmarkSample) -> dict[str, T.Any]:
    return {
        "condition": sample.condition,
        "conditions": list(sample.conditions),
        "metadata": sample.metadata if isinstance(sample.metadata, dict) else {},
    }


def is_profile39_sample(sample: LandmarkSample) -> bool:
    """Return ``True`` when a sample carries 39-point profile GT (shape ``(39, 2)``)."""
    if sample_is_canonical_68(sample):
        return False
    try:
        truth = np.load(sample.landmarks)
    except OSError:
        return False
    return getattr(truth, "ndim", 0) == 2 and int(truth.shape[0]) == PROFILE39_POINT_COUNT


def profile39_samples(manifest_path: Path) -> list[LandmarkSample]:
    """Return only the 39-point profile samples from a manifest."""
    return [sample for sample in load_manifest(manifest_path) if is_profile39_sample(sample)]


def build_profile39_rows_for_sample(
    sample: LandmarkSample,
    *,
    cache: DiskPredictionCache,
    weights: T.Mapping[str, T.Sequence[float]],
    requested_candidates: T.Sequence[str],
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    bucket_weights: T.Mapping[str, T.Mapping[str, T.Sequence[float]]] | None = None,
    region_weights: T.Mapping[str, T.Mapping[str, float]] | None = None,
) -> list[Profile39Row]:
    """Return profile-39 rows for one sample, or ``[]`` when it should be skipped.

    Skips samples whose visible side cannot be resolved (conservative first
    implementation) and samples without cached predictions.
    """
    side = profile39_side_from_sample(_sample_label_mapping(sample))
    if side is None:
        side = profile_side_from_context(
            runtime_bucket=sample.condition, condition=sample.condition
        )
    if side not in {"left", "right"}:
        logger.debug("skipping profile39 sample %s: unresolved side", sample.sample_id)
        return []

    truth = np.load(sample.landmarks).astype("float32")
    if truth.shape != (PROFILE39_POINT_COUNT, 2):
        return []

    requested_adaptive, single_models = _requested_candidate_sets(requested_candidates)
    model_names = tuple(dict.fromkeys((*weights.keys(), *single_models)))
    predictions = _read_predictions(cache, sample, models=model_names)
    if not predictions:
        logger.debug("skipping profile39 sample %s: no cached predictions", sample.sample_id)
        return []

    config = _candidate_config(
        weights,
        outlier_threshold=outlier_threshold,
        bucket_weights=bucket_weights,
        region_weights=region_weights,
    )
    all_candidates = build_candidates(predictions, config)
    by_name = {candidate.name: candidate for candidate in all_candidates}
    seed_names = [name for name in requested_candidates if name not in requested_adaptive]
    if requested_adaptive:
        seed_names.extend(c.name for c in all_candidates if not c.is_fusion)
    candidates = tuple(by_name[name] for name in dict.fromkeys(seed_names) if name in by_name)
    if not candidates:
        return []

    rows = profile39_rows(candidates, truth, side=side, normalizer=sample.normalizer)
    if not rows:
        return []

    # Runtime feature payload, mirroring the canonical path so the specialist
    # sees the same features (including the profile/visible-side features).
    runtime_bucket = f"profile_{side}"
    reference_bbox = sample.face_bbox or bbox_from_truth_fallback(truth)
    metrics = {c.name: _metric_for_candidate(c, reference_bbox=reference_bbox) for c in candidates}
    extra = _candidate_extra_features(
        candidates,
        metrics,
        reference_bbox=reference_bbox,
        condition=runtime_bucket,
        runtime_bucket=runtime_bucket,
    )
    feature_maps = {
        c.name: candidate_feature_map(
            c,
            metrics[c.name],
            runtime_bucket=runtime_bucket,
            candidate_extra_features=extra,
        )
        for c in candidates
    }

    out: list[Profile39Row] = []
    for row in rows:
        name = str(row["candidate_name"])
        out.append(
            Profile39Row(
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                side=side,
                candidate_name=name,
                profile39_transform_cost=float(row["profile39_transform_cost"]),
                profile39_transform_regret=float(row["profile39_transform_regret"]),
                profile39_visible_side_error=float(row["profile39_visible_side_error"]),
                profile39_oracle_candidate=str(row["profile39_oracle_candidate"]),
                profile39_is_oracle=bool(row["profile39_is_oracle"]),
                profile39_rankable=bool(row["profile39_rankable"]),
                feature_values=feature_maps.get(name, {}),
            )
        )
    return out


def load_profile39_rows(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    candidates: T.Sequence[str] | None = None,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
) -> list[Profile39Row]:
    """Load profile-39 rows for every 39-point sample in a manifest/cache pair."""
    weights = load_weights(weights_path)
    bucket_weights, region_weights = load_optional_weight_blocks(weights_path)
    requested = tuple(candidates or parse_candidates(None, weights))
    cache = DiskPredictionCache(cache_dir)
    rows: list[Profile39Row] = []
    for sample in profile39_samples(manifest_path):
        try:
            rows.extend(
                build_profile39_rows_for_sample(
                    sample,
                    cache=cache,
                    weights=weights,
                    requested_candidates=requested,
                    outlier_threshold=outlier_threshold,
                    bucket_weights=bucket_weights,
                    region_weights=region_weights,
                )
            )
        except (FileNotFoundError, ValueError) as err:
            logger.warning("skipping profile39 sample %s: %s", sample.sample_id, err)
    return rows


def profile39_mix_report(rows: T.Sequence[Profile39Row]) -> dict[str, T.Any]:
    """Return separate profile-39 summary metrics (kept apart from canonical GT-hard)."""
    if not rows:
        return {
            "row_count": 0,
            "sample_count": 0,
            "side_counts": {"left": 0, "right": 0},
            "mean_visible_side_error": 0.0,
            "p95_visible_side_error": 0.0,
            "mean_transform_regret": 0.0,
            "p95_transform_regret": 0.0,
            "oracle_row_count": 0,
        }
    errors = np.asarray([row.profile39_visible_side_error for row in rows], dtype="float64")
    regrets = np.asarray([row.profile39_transform_regret for row in rows], dtype="float64")
    side_counts = {"left": 0, "right": 0}
    for row in rows:
        if row.side in side_counts:
            side_counts[row.side] += 1
    return {
        "row_count": len(rows),
        "sample_count": len({row.sample_id for row in rows}),
        "side_counts": side_counts,
        "mean_visible_side_error": float(np.mean(errors)),
        "p95_visible_side_error": float(np.percentile(errors, 95)),
        "mean_transform_regret": float(np.mean(regrets)),
        "p95_transform_regret": float(np.percentile(regrets, 95)),
        "oracle_row_count": int(sum(1 for row in rows if row.profile39_is_oracle)),
    }


__all__ = [
    "Profile39Row",
    "build_profile39_rows_for_sample",
    "is_profile39_sample",
    "load_profile39_rows",
    "profile39_mix_report",
    "profile39_samples",
]
