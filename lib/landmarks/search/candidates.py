#!/usr/bin/env python3
"""Candidate dataclass and enumeration surface for the ensemble search.

Splitting candidate construction out of :mod:`lib.landmarks.search.candidate_search`
keeps the search-engine module focused on the fit / evaluate / select loop
and gives the rest of the codebase a small, dependency-light import target
for the dataclass and the cartesian-product helpers.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import typing as T
from dataclasses import dataclass

from lib.landmarks.ensemble.strategies import (
    canonical_strategy,
    strategy_uses_threshold,
    validate_threshold,
)
from lib.landmarks.ensemble.weight_generators import get_generator

SUBSET_PRESETS: tuple[str, ...] = ("all", "pairs", "triples")
CandidateProgress = T.Callable[[T.Sequence["Candidate"]], T.Iterable["Candidate"]]


@dataclass(frozen=True)
class Candidate:
    """One fully-specified ensemble configuration to evaluate."""

    models: tuple[str, ...]
    weight_generator: str
    weight_generator_params: tuple[tuple[str, float], ...] = ()
    strategy: str = "static_weighted"
    outlier_threshold: float | None = None
    bbox_source: str = "manifest"
    crop_scale: float = 1.6

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("Candidate must include at least one model")
        if len(set(self.models)) != len(self.models):
            raise ValueError(f"Candidate.models must be unique: {self.models!r}")
        canonical = canonical_strategy(self.strategy)
        object.__setattr__(self, "strategy", canonical)
        validate_threshold(canonical, self.outlier_threshold)
        # Generator name validated lazily inside fit; do a fast existence check now.
        get_generator(self.weight_generator)

    def generator_params_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.weight_generator_params}


def candidate_id(
    candidate: Candidate,
    *,
    weights_hash: str,
    split_assignment_hash: str,
    objective: str,
) -> str:
    """Return a stable ``sha256:...`` identifier for a candidate evaluation.

    The hash covers every dimension that changes how a candidate behaves: the
    model subset (order-insensitive), generator name + params, weights hash,
    strategy, outlier threshold (when used), bbox source and crop scale, and
    the split assignment + objective version (since changing the splits or
    objective invalidates prior candidate IDs).
    """
    payload = {
        "models": sorted(candidate.models),
        "weight_generator": {
            "name": candidate.weight_generator,
            "params": candidate.generator_params_dict(),
        },
        "strategy": candidate.strategy,
        "outlier_threshold": candidate.outlier_threshold,
        "bbox_source": candidate.bbox_source,
        "crop_scale": candidate.crop_scale,
        "weights_hash": weights_hash,
        "split_assignment_hash": split_assignment_hash,
        "objective": objective,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return "sha256:" + digest


def weights_hash(weights: T.Mapping[str, T.Sequence[float]]) -> str:
    """Return a stable ``sha256:...`` hash of normalized per-landmark weights."""
    payload = {model: [float(value) for value in values] for model, values in weights.items()}
    return (
        "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    )


def expand_model_subsets(
    models: T.Sequence[str], presets: T.Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    """Expand subset preset names into actual model-tuple subsets.

    Supported presets: ``all`` (the full set), ``pairs`` (every 2-combination),
    ``triples`` (every 3-combination). Unknown names raise ``ValueError`` with
    the supported list.
    """
    model_tuple = tuple(models)
    if not model_tuple:
        raise ValueError("models must be non-empty")
    unknown = [name for name in presets if name not in SUBSET_PRESETS]
    if unknown:
        raise ValueError(
            f"unknown model-subset preset(s) {unknown!r}; supported: {', '.join(SUBSET_PRESETS)}"
        )
    subsets: list[tuple[str, ...]] = []
    if "all" in presets:
        subsets.append(model_tuple)
    if "pairs" in presets:
        subsets.extend(tuple(combo) for combo in itertools.combinations(model_tuple, 2))
    if "triples" in presets and len(model_tuple) >= 3:
        subsets.extend(tuple(combo) for combo in itertools.combinations(model_tuple, 3))
    seen: set[tuple[str, ...]] = set()
    ordered: list[tuple[str, ...]] = []
    for subset in subsets:
        key = tuple(sorted(subset))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(subset)
    return tuple(ordered)


def single_model_baseline_candidates(
    models: T.Sequence[str],
    *,
    bbox_source: str = "manifest",
    crop_scale: float = 1.6,
) -> list[Candidate]:
    """Return one ``plain_average`` Candidate per model so each runs as a baseline.

    Single-model ``plain_average`` reduces to that model's cached prediction at
    fusion time, so these candidates score the model as-is. They are tagged
    via ``is_single_model_baseline`` on :class:`CandidateResult` so the
    promotion report can compare ensembles against the obvious alternative.
    """
    return [
        Candidate(
            models=(model,),
            weight_generator="equal",
            strategy="plain_average",
            outlier_threshold=None,
            bbox_source=bbox_source,
            crop_scale=crop_scale,
        )
        for model in models
    ]


def enumerate_candidates(
    *,
    models: T.Sequence[str],
    model_subset_presets: T.Sequence[str],
    weight_generators: T.Sequence[str],
    strategies: T.Sequence[str],
    outlier_thresholds: T.Sequence[float],
    bbox_source: str = "manifest",
    crop_scale: float | None = None,
    crop_scales: T.Sequence[float] | None = None,
    include_single_model_baselines: bool = False,
) -> list[Candidate]:
    """Cartesian product of search dimensions, honoring strategy-scoped threshold rules.

    For strategies that do not consume a threshold the threshold dimension is
    skipped (one candidate per strategy/subset/generator). For strategies that
    do consume a threshold a candidate is emitted per supplied threshold.

    ``crop_scales`` controls the crop_scale fanout: the runtime extract crop
    scale stamped into ``best_setup.json`` is a real search dimension (the
    GT-derived geometry path consumes it as the AlignedFace coverage ratio,
    so crop-coverage signals shift with the candidate's crop scale). Pass a
    single-element sequence for back-compat. ``crop_scale`` (scalar) is
    honored only when ``crop_scales`` is omitted.
    """
    subsets = expand_model_subsets(models, model_subset_presets)
    if not subsets:
        raise ValueError("no model subsets produced from the requested presets")
    if not weight_generators:
        raise ValueError("at least one weight generator is required")
    if not strategies:
        raise ValueError("at least one strategy is required")
    if crop_scales is None:
        crop_scales = (1.6 if crop_scale is None else float(crop_scale),)
    if not crop_scales:
        raise ValueError("at least one crop_scale is required")
    crop_scale_values = tuple(float(value) for value in crop_scales)
    canonical_list = [canonical_strategy(name) for name in strategies]
    candidates: list[Candidate] = []
    for subset, generator, strategy, scale in itertools.product(
        subsets, weight_generators, canonical_list, crop_scale_values
    ):
        if strategy_uses_threshold(strategy):
            if not outlier_thresholds:
                raise ValueError(
                    f"strategy {strategy!r} requires --outlier-thresholds but none were supplied"
                )
            for threshold in outlier_thresholds:
                candidates.append(
                    Candidate(
                        models=tuple(subset),
                        weight_generator=generator,
                        strategy=strategy,
                        outlier_threshold=float(threshold),
                        bbox_source=bbox_source,
                        crop_scale=scale,
                    )
                )
        else:
            candidates.append(
                Candidate(
                    models=tuple(subset),
                    weight_generator=generator,
                    strategy=strategy,
                    outlier_threshold=None,
                    bbox_source=bbox_source,
                    crop_scale=scale,
                )
            )
    if include_single_model_baselines:
        existing = {
            (
                tuple(sorted(c.models)),
                c.strategy,
                c.weight_generator,
                c.outlier_threshold,
                c.crop_scale,
            )
            for c in candidates
        }
        for scale in crop_scale_values:
            for baseline in single_model_baseline_candidates(
                models, bbox_source=bbox_source, crop_scale=scale
            ):
                key = (
                    tuple(sorted(baseline.models)),
                    baseline.strategy,
                    baseline.weight_generator,
                    baseline.outlier_threshold,
                    baseline.crop_scale,
                )
                if key in existing:
                    continue
                existing.add(key)
                candidates.append(baseline)
    return candidates


__all__ = [
    "Candidate",
    "CandidateProgress",
    "SUBSET_PRESETS",
    "candidate_id",
    "enumerate_candidates",
    "expand_model_subsets",
    "single_model_baseline_candidates",
    "weights_hash",
]
