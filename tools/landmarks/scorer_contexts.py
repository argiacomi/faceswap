#!/usr/bin/env python3
"""Shared context loading for runtime resolver scorer tools."""

from __future__ import annotations

import logging
import typing as T
from pathlib import Path

from lib.landmarks.datasets.manifest_io import load_manifest
from tools.landmarks.pipeline_conventions import (
    SOURCE_GT_HARD,
    SOURCE_PRODUCTION_VALIDATED,
    ManifestCachePair,
    load_resolver_metadata_sidecar,
    normalize_source_label,
    require_manifest_cache_pair,
    validate_resolver_metadata_for_samples,
)
from tools.landmarks.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    SampleCandidateContext,
    load_contexts,
)

logger = logging.getLogger(__name__)


def scorer_manifest_cache_pairs(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
) -> tuple[ManifestCachePair, ...]:
    """Return explicit scorer manifest/cache pairs in canonical source order."""
    pairs = []
    gt_pair = require_manifest_cache_pair(
        source=SOURCE_GT_HARD,
        manifest_path=gt_manifest,
        cache_dir=gt_cache_dir,
    )
    if gt_pair is not None:
        pairs.append(gt_pair)
    production_pair = require_manifest_cache_pair(
        source=SOURCE_PRODUCTION_VALIDATED,
        manifest_path=production_manifest,
        cache_dir=production_cache_dir,
    )
    if production_pair is not None:
        pairs.append(production_pair)
    return tuple(pairs)


def load_scorer_contexts(
    *,
    gt_manifest: Path | None,
    gt_cache_dir: Path | None,
    production_manifest: Path | None,
    production_cache_dir: Path | None,
    weights_path: Path,
    candidates: T.Sequence[str],
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    allow_image_backfill: bool = False,
    gt_hard_resolver_metadata: Path | None = None,
    require_gt_hard_metadata: bool = True,
) -> list[SampleCandidateContext]:
    """Load scorer contexts across GT-hard and production sources.

    GT-hard contexts use the explicit `gt_hard` source label. When a GT-hard
    manifest is supplied, its resolver metadata sidecar is validated against
    `(sample_id, face_index)` keys before sample contexts are built. This makes
    wrong sample IDs and mismatched face indexes fail before they can silently
    produce derived runtime buckets.
    """
    pairs = scorer_manifest_cache_pairs(
        gt_manifest=gt_manifest,
        gt_cache_dir=gt_cache_dir,
        production_manifest=production_manifest,
        production_cache_dir=production_cache_dir,
    )
    if not pairs:
        raise ValueError("at least one scorer manifest/cache pair is required")

    gt_hard_metadata = load_resolver_metadata_sidecar(gt_hard_resolver_metadata)
    contexts: list[SampleCandidateContext] = []
    for pair in pairs:
        source = normalize_source_label(pair.source)
        resolver_metadata = None
        if source == SOURCE_GT_HARD:
            samples = load_manifest(pair.manifest_path)
            validate_resolver_metadata_for_samples(
                samples,
                gt_hard_metadata,
                source=SOURCE_GT_HARD,
                require_complete=require_gt_hard_metadata,
            )
            resolver_metadata = gt_hard_metadata
        logger.info("Loading %s scorer contexts from %s", source, pair.manifest_path)
        contexts.extend(
            load_contexts(
                manifest_path=pair.manifest_path,
                cache_dir=pair.cache_dir,
                weights_path=weights_path,
                candidates=candidates,
                source=source,
                resolver_metadata=resolver_metadata,
                failure_threshold=failure_threshold,
                outlier_threshold=outlier_threshold,
                allow_image_backfill=allow_image_backfill,
            )
        )
    if not contexts:
        raise ValueError("no scorer contexts were loaded")
    return contexts


__all__ = ["load_scorer_contexts", "scorer_manifest_cache_pairs"]
