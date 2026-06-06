#!/usr/bin/env python3
"""FaceQA-aware training sampler weighting."""

from __future__ import annotations

import logging
import typing as T
from dataclasses import dataclass, field

import numpy as np

from lib.training.faceqa_diagnostics import FaceQASampleMetadata
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

SamplerMode = T.Literal["random", "faceqa_weighted", "faceqa_curriculum"]
MinQuality = T.Literal["off", "usable"]

_DIMENSION_ALIASES: dict[str, str] = {
    "pose": "yaw_pose",
    "yaw": "yaw_pose",
    "yaw_pose": "yaw_pose",
    "pitch": "pitch",
    "expression": "expression",
    "lighting": "lighting",
    "blur": "blur",
    "resolution": "resolution",
    "mask_qa": "mask_qa",
}
_QUALITY_BUCKETS = {
    "blur": {"unusable"},
    "resolution": {"tiny"},
    "mask_qa": {"missing"},
}


def normalize_dimensions(value: str | T.Iterable[str]) -> tuple[str, ...]:
    """Return stable FaceQA sampler dimensions from config text or an iterable."""
    raw = value.split(",") if isinstance(value, str) else list(value)
    dimensions: list[str] = []
    for item in raw:
        key = str(item).strip().lower()
        if not key:
            continue
        normalized = _DIMENSION_ALIASES.get(key)
        if normalized is None:
            raise ValueError(f"Unsupported FaceQA sampler dimension: {item!r}")
        if normalized not in dimensions:
            dimensions.append(normalized)
    return tuple(dimensions)


@dataclass(frozen=True)
class FaceQASamplerConfig:
    """Configuration for FaceQA-aware sample weighting."""

    mode: SamplerMode = "random"
    strength: float = 0.0
    dimensions: tuple[str, ...] = ("yaw_pose", "expression", "lighting")
    downweight_duplicates: bool = True
    downweight_outliers: bool = True
    min_quality: MinQuality = "usable"
    downweight_factor: float = 0.25
    max_weight: float = 4.0

    def __post_init__(self) -> None:
        if self.mode not in ("random", "faceqa_weighted", "faceqa_curriculum"):
            raise ValueError(f"Unsupported FaceQA sampler mode: {self.mode!r}")
        if self.strength < 0.0:
            raise ValueError(f"FaceQA sampler strength must be >= 0.0. Got {self.strength}")
        if self.min_quality not in ("off", "usable"):
            raise ValueError(f"Unsupported FaceQA minimum quality: {self.min_quality!r}")
        if not 0.0 < self.downweight_factor <= 1.0:
            raise ValueError(
                "FaceQA sampler downweight factor must be in the range (0.0, 1.0]. "
                f"Got {self.downweight_factor}"
            )
        if self.max_weight < 1.0:
            raise ValueError(f"FaceQA sampler max weight must be >= 1.0. Got {self.max_weight}")

    @property
    def active(self) -> bool:
        """``True`` when the sampler can alter random side permutations."""
        return self.mode != "random" and self.strength > 0.0

    def with_strength_multiplier(self, multiplier: float) -> FaceQASamplerConfig:
        """Return a copy with strength scaled by a non-negative scheduler multiplier."""
        if multiplier < 0.0:
            raise ValueError(
                f"FaceQA sampler strength multiplier must be >= 0.0. Got {multiplier}"
            )
        return FaceQASamplerConfig(
            mode=self.mode,
            strength=self.strength * multiplier,
            dimensions=self.dimensions,
            downweight_duplicates=self.downweight_duplicates,
            downweight_outliers=self.downweight_outliers,
            min_quality=self.min_quality,
            downweight_factor=self.downweight_factor,
            max_weight=self.max_weight,
        )


@dataclass(frozen=True)
class FaceQASamplerSummary:
    """Compact diagnostics for one side's sampler weights."""

    side: str
    metadata_count: int
    total_count: int
    effective_sample_count: float
    top_upweighted: list[tuple[str, float]] = field(default_factory=list)
    top_downweighted: list[tuple[str, float]] = field(default_factory=list)


def _sample_bucket(sample: FaceQASampleMetadata, dimension: str) -> str:
    """Return a sample's bucket label for a normalized sampler dimension."""
    return str(sample.dimensions().get(dimension, "unknown"))


def _is_bad_quality(sample: FaceQASampleMetadata, min_quality: MinQuality) -> bool:
    """Return whether quality metadata marks a sample as below the sampler threshold."""
    if min_quality == "off":
        return False
    dimensions = sample.dimensions()
    return any(
        dimensions.get(dimension) in buckets for dimension, buckets in _QUALITY_BUCKETS.items()
    )


def _is_protected_sample(sample: FaceQASampleMetadata, config: FaceQASamplerConfig) -> bool:
    """Return whether a sample should be downweighted and never amplified."""
    return sample.has_faceqa and (
        (config.downweight_duplicates and sample.duplicate_bucket == "duplicate")
        or (config.downweight_outliers and sample.identity_outlier_bucket in ("outlier", "reject"))
        or _is_bad_quality(sample, config.min_quality)
    )


def _rarity_weights(
    samples: list[FaceQASampleMetadata],
    config: FaceQASamplerConfig,
    bucket_loss_scores: T.Mapping[tuple[str, str, str], float] | None = None,
) -> np.ndarray:
    """Return raw rarity weights from selected FaceQA dimensions."""
    raw = np.ones(len(samples), dtype=np.float64)
    useful = np.array(
        [sample.has_faceqa and not _is_protected_sample(sample, config) for sample in samples],
        dtype=bool,
    )
    if not useful.any():
        return raw

    for dimension in config.dimensions:
        buckets = [_sample_bucket(sample, dimension) for sample in samples]
        counts: dict[str, int] = {}
        for bucket, is_useful in zip(buckets, useful, strict=True):
            if not is_useful or bucket == "unknown":
                continue
            counts[bucket] = counts.get(bucket, 0) + 1
        if not counts:
            continue
        mean_count = float(np.mean(list(counts.values())))
        for idx, bucket in enumerate(buckets):
            if not useful[idx] or bucket not in counts:
                continue
            sample = samples[idx]
            raw[idx] *= np.sqrt(mean_count / counts[bucket])
            if bucket_loss_scores is not None:
                loss_score = bucket_loss_scores.get((sample.side, dimension, bucket))
                if loss_score is None or loss_score <= 0.0:
                    continue
                dimension_scores = [
                    score
                    for (side, dim, _bucket), score in bucket_loss_scores.items()
                    if side == sample.side and dim == dimension and score > 0.0
                ]
                if not dimension_scores:
                    continue
                mean_loss = float(np.mean(dimension_scores))
                if mean_loss > 0.0:
                    raw[idx] *= np.sqrt(loss_score / mean_loss)
    return raw


def _blend_and_normalize(
    raw: np.ndarray, samples: list[FaceQASampleMetadata], config: FaceQASamplerConfig
) -> np.ndarray:
    """Blend raw weights by strength, apply safeguards, and normalize metadata weights."""
    weights = 1.0 + config.strength * (raw - 1.0)
    protected = np.array([_is_protected_sample(sample, config) for sample in samples], dtype=bool)
    weights[protected] = np.minimum(weights[protected], 1.0) * config.downweight_factor
    weights = np.clip(weights, config.downweight_factor, config.max_weight)

    metadata = np.array([sample.has_faceqa for sample in samples], dtype=bool)
    if metadata.any():
        mean = float(weights[metadata].mean())
        if mean > 0.0:
            weights[metadata] /= mean
            weights[protected] = np.minimum(weights[protected], 1.0)
            weights = np.clip(weights, config.downweight_factor, config.max_weight)
    return weights.astype(np.float64, copy=False)


def _summary(
    side: str,
    samples: list[FaceQASampleMetadata],
    weights: np.ndarray,
    *,
    limit: int = 5,
) -> FaceQASamplerSummary:
    """Return compact sampler diagnostics for logging."""
    metadata_count = sum(sample.has_faceqa for sample in samples)
    effective = float(weights.sum() ** 2 / np.square(weights).sum()) if weights.size else 0.0
    labelled = [
        (f"{sample.source_id}:{sample.filename}", float(weight))
        for sample, weight in zip(samples, weights, strict=True)
        if sample.has_faceqa
    ]
    top_up = sorted(labelled, key=lambda item: item[1], reverse=True)[:limit]
    top_down = sorted(labelled, key=lambda item: item[1])[:limit]
    return FaceQASamplerSummary(
        side=side,
        metadata_count=metadata_count,
        total_count=len(samples),
        effective_sample_count=effective,
        top_upweighted=top_up,
        top_downweighted=top_down,
    )


def compute_faceqa_sample_weights(
    side: str,
    samples: list[FaceQASampleMetadata],
    config: FaceQASamplerConfig,
    bucket_loss_scores: T.Mapping[tuple[str, str, str], float] | None = None,
) -> tuple[np.ndarray | None, FaceQASamplerSummary]:
    """Return per-sample weights for one training side.

    ``None`` weights mean callers should retain exact random sampling behavior.
    """
    if not config.active or not samples:
        weights = np.ones(len(samples), dtype=np.float64)
        return None, _summary(side, samples, weights)
    if not any(sample.has_faceqa for sample in samples):
        weights = np.ones(len(samples), dtype=np.float64)
        return None, _summary(side, samples, weights)

    raw = _rarity_weights(samples, config, bucket_loss_scores)
    weights = _blend_and_normalize(raw, samples, config)
    logger.debug(
        "[FaceQASampler] side=%s metadata=%s/%s effective=%.2f top_up=%s top_down=%s",
        side,
        sum(sample.has_faceqa for sample in samples),
        len(samples),
        float(weights.sum() ** 2 / np.square(weights).sum()) if weights.size else 0.0,
        sorted(weights, reverse=True)[:5],
        sorted(weights)[:5],
    )
    return weights, _summary(side, samples, weights)


__all__ = get_module_objects(__name__)
