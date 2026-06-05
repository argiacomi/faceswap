#!/usr/bin/env python3
"""Context-cache helpers for stacked residual regressor train/eval sweeps.

The expensive part of stacked-regressor sweeps is rebuilding
``SampleCandidateContext`` objects from manifests and prediction caches. This
module supports:

* pickle-backed context caches
* optional parallel context construction
* one shared implementation for train/eval/prebuild CLIs
"""

from __future__ import annotations

import logging
import os
import pickle
import typing as T
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from lib.landmarks.datasets.manifest_io import filter_canonical_68_samples, load_manifest
from lib.landmarks.ensemble.runtime_resolver_scorer_data import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_OUTLIER_THRESHOLD,
    build_sample_context,
)
from lib.landmarks.ensemble.weights import load_optional_weight_blocks, load_weights

CACHE_SCHEMA_VERSION = 2

ContextBuilder = T.Callable[[], list[T.Any]]
ContextProgress = T.Callable[[T.Sequence[T.Any], str], T.Iterable[T.Any]]


def load_stacked_context_cache(path: Path) -> list[T.Any]:
    """Load prebuilt stacked-regressor contexts from ``path``."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local trusted pipeline cache

    if isinstance(payload, dict):
        version = int(payload.get("schema_version", 0))
        if version not in {1, CACHE_SCHEMA_VERSION}:
            raise ValueError(
                f"stacked context cache {path} has schema_version={version}; "
                f"expected {CACHE_SCHEMA_VERSION}"
            )
        contexts = payload.get("contexts")
    else:
        contexts = payload

    if not isinstance(contexts, list):
        raise TypeError(f"stacked context cache {path} did not contain a context list")
    return contexts


def write_stacked_context_cache(
    path: Path,
    contexts: list[T.Any],
    *,
    metadata: T.Mapping[str, T.Any] | None = None,
) -> Path:
    """Atomically write ``contexts`` to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "contexts": contexts,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, target)
    return target


def load_or_build_stacked_contexts(
    *,
    context_cache: Path | None,
    rebuild_context_cache: bool,
    build: ContextBuilder,
    logger: logging.Logger,
    metadata: T.Mapping[str, T.Any] | None = None,
) -> list[T.Any]:
    """Load contexts from cache, or build and optionally cache them."""
    if context_cache is not None:
        cache_path = Path(context_cache)
        if cache_path.is_file() and not rebuild_context_cache:
            logger.info("Loading stacked regressor contexts from %s", cache_path)
            contexts = load_stacked_context_cache(cache_path)
            logger.info("Loaded %d stacked regressor context(s)", len(contexts))
            return contexts

    logger.info("Building stacked regressor contexts")
    contexts = build()
    logger.info("Built %d stacked regressor context(s)", len(contexts))

    if context_cache is not None:
        cache_path = write_stacked_context_cache(
            Path(context_cache),
            contexts,
            metadata=metadata,
        )
        logger.info("Wrote stacked regressor context cache to %s", cache_path)

    return contexts


def _context_worker(payload: tuple[T.Any, ...]) -> tuple[bool, T.Any, str, str]:
    """Build one context in a subprocess.

    Returns:
        ``(True, context, sample_id, "")`` on success.
        ``(False, None, sample_id, reason)`` on a skip-worthy sample error.
    """
    (
        sample,
        cache_dir,
        requested_candidates,
        weights,
        source,
        failure_threshold,
        outlier_threshold,
        bucket_weights,
        region_weights,
        include_adaptive_candidates,
        allow_image_backfill,
        image_backfill_crop_scale,
        image_backfill_crop_size,
    ) = payload

    sample_id = str(getattr(sample, "sample_id", ""))
    try:
        from lib.landmarks.cache.prediction_cache import DiskPredictionCache

        context = build_sample_context(
            sample,
            cache=DiskPredictionCache(Path(cache_dir)),
            requested_candidates=tuple(requested_candidates),
            weights=weights,
            source=str(source),
            failure_threshold=float(failure_threshold),
            outlier_threshold=float(outlier_threshold),
            bucket_weights=bucket_weights,
            region_weights=region_weights,
            include_adaptive_candidates=bool(include_adaptive_candidates),
            allow_image_backfill=bool(allow_image_backfill),
            image_backfill_crop_scale=float(image_backfill_crop_scale),
            image_backfill_crop_size=int(image_backfill_crop_size),
        )
        return True, context, sample_id, ""
    except (FileNotFoundError, ValueError) as err:
        return False, None, sample_id, str(err)


def load_contexts_maybe_parallel(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    candidates: T.Sequence[str] | None = None,
    source: str = "",
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    allow_image_backfill: bool = False,
    image_backfill_crop_scale: float = 1.6,
    image_backfill_crop_size: int = 256,
    workers: int = 0,
    progress: ContextProgress | None = None,
    logger: logging.Logger | None = None,
) -> list[T.Any]:
    """Load sample contexts, optionally in parallel.

    ``workers <= 1`` preserves the existing serial implementation. ``workers > 1``
    builds each sample in a subprocess. The output order follows the manifest
    order, not completion order.
    """
    log = logger or logging.getLogger(__name__)
    if workers <= 1:
        from lib.landmarks.ensemble.runtime_resolver_scorer_data import load_contexts

        return load_contexts(
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            weights_path=weights_path,
            candidates=candidates,
            source=source,
            failure_threshold=failure_threshold,
            outlier_threshold=outlier_threshold,
            allow_image_backfill=allow_image_backfill,
            image_backfill_crop_scale=image_backfill_crop_scale,
            image_backfill_crop_size=image_backfill_crop_size,
            progress=progress,
        )

    weights = load_weights(weights_path)
    bucket_weights, region_weights = load_optional_weight_blocks(weights_path)
    explicit_candidates = bool(candidates)
    include_adaptive_candidates = not explicit_candidates

    from lib.landmarks.ensemble.runtime_resolver_scorer_data import parse_candidates

    requested = tuple(candidates or parse_candidates(None, weights))
    samples = filter_canonical_68_samples(
        load_manifest(manifest_path),
        context="stacked regressor context cache",
        progress=progress,
    )

    if not samples:
        return []

    payloads = [
        (
            sample,
            Path(cache_dir),
            requested,
            weights,
            source,
            failure_threshold,
            outlier_threshold,
            bucket_weights,
            region_weights,
            include_adaptive_candidates,
            allow_image_backfill,
            image_backfill_crop_scale,
            image_backfill_crop_size,
        )
        for sample in samples
    ]

    contexts_by_index: dict[int, T.Any] = {}
    skipped = 0
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        future_to_index = {
            executor.submit(_context_worker, payload): index
            for index, payload in enumerate(payloads)
        }
        futures: T.Iterable[T.Any] = as_completed(future_to_index)
        if progress is not None:
            futures = progress(
                list(future_to_index), f"Build contexts [{source or manifest_path.stem}]"
            )

        for future in futures:
            index = future_to_index[future]
            ok, context, sample_id, reason = future.result()
            if ok:
                contexts_by_index[index] = context
            else:
                skipped += 1
                log.warning("skipping sample %s: %s", sample_id, reason)

    contexts = [contexts_by_index[index] for index in sorted(contexts_by_index)]
    log.info(
        "Built %d context(s) from %s with %d worker(s); skipped=%d",
        len(contexts),
        manifest_path,
        workers,
        skipped,
    )
    return contexts


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "load_contexts_maybe_parallel",
    "load_or_build_stacked_contexts",
    "load_stacked_context_cache",
    "write_stacked_context_cache",
]
