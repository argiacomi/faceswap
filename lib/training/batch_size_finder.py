#!/usr/bin/env python3
"""Training batch-size finder."""

from __future__ import annotations

import logging
import math
import typing as T
from dataclasses import dataclass

from tqdm import tqdm

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from . import train

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchSizeProbe:
    """The result of probing a single training batch size."""

    batch_size: int
    """The probed per-side batch size."""
    success: bool
    """``True`` when the training step completed without accelerator OOM."""
    vram_allocated: int = 0
    """Peak accelerator memory allocated, in bytes."""
    vram_reserved: int = 0
    """Peak accelerator memory reserved, in bytes."""
    error: str | None = None
    """Optional OOM error detail."""

    @property
    def vram_reserved_mb(self) -> int:
        """Peak reserved accelerator memory in MiB."""
        return int(round(self.vram_reserved / (1024 * 1024)))

    def to_state(self) -> dict[str, T.Any]:
        """Return a JSON-serializable representation."""
        return {
            "batch_size": self.batch_size,
            "success": self.success,
            "vram_allocated": self.vram_allocated,
            "vram_reserved": self.vram_reserved,
            "error": self.error,
        }


@dataclass(frozen=True)
class BatchSizeRecommendation:
    """The final training batch-size finder recommendation."""

    configured_batch_size: int
    """The batch size requested by the user."""
    max_safe_batch_size: int
    """Largest probed batch size that completed successfully."""
    suggested_batch_size: int
    """Conservative batch-size recommendation."""
    estimated_vram_reserved: int
    """Estimated reserved VRAM for :attr:`suggested_batch_size`, in bytes."""
    gradient_accumulation_recommended: bool
    """``True`` when accumulation is recommended to reach the target effective batch size."""
    gradient_accumulation_steps: int
    """Recommended gradient accumulation steps."""
    effective_batch_size: int
    """Effective per-side batch size with accumulation."""
    target_effective_batch_size: int
    """Minimum effective per-side batch size the recommendation targets."""
    probes: tuple[BatchSizeProbe, ...]
    """All probes performed by the finder."""

    def to_state(self) -> dict[str, T.Any]:
        """Return a JSON-serializable representation."""
        return {
            "configured_batch_size": self.configured_batch_size,
            "max_safe_batch_size": self.max_safe_batch_size,
            "suggested_batch_size": self.suggested_batch_size,
            "estimated_vram_reserved": self.estimated_vram_reserved,
            "gradient_accumulation_recommended": self.gradient_accumulation_recommended,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "target_effective_batch_size": self.target_effective_batch_size,
            "probes": [probe.to_state() for probe in self.probes],
        }


class TrainingBatchSizeFinder:
    """Find a safe training batch size using the real training path.

    Parameters
    ----------
    trainer
        The loaded training loop to probe.
    max_batch_size
        The maximum per-side batch size to test.
    target_effective_batch_size
        The minimum effective per-side batch size to aim for with gradient accumulation.
    safety_margin
        Fraction of the maximum safe batch size to recommend for normal training.
    """

    def __init__(
        self,
        trainer: train.Trainer,
        max_batch_size: int,
        target_effective_batch_size: int = 4,
        safety_margin: float = 0.85,
    ) -> None:
        logger.debug(parse_class_init(locals()))
        self._trainer = trainer
        self._configured_batch_size = trainer.batch_size
        self._max_batch_size = max(1, int(max_batch_size))
        self._target_effective_batch_size = max(1, int(target_effective_batch_size))
        self._safety_margin = min(1.0, max(0.1, safety_margin))
        self._probes: dict[int, BatchSizeProbe] = {}

    @property
    def probes(self) -> tuple[BatchSizeProbe, ...]:
        """All probes performed by the finder in ascending batch-size order."""
        return tuple(self._probes[key] for key in sorted(self._probes))

    def _probe(self, batch_size: int, progress_bar: tqdm) -> BatchSizeProbe:
        """Probe a candidate batch size and record the result."""
        batch_size = max(1, min(batch_size, self._max_batch_size))
        existing = self._probes.get(batch_size)
        if existing is not None:
            return existing

        progress_bar.set_description(f"Testing batch size {batch_size}")
        probe = self._trainer.probe_training_batch_size(batch_size)
        self._probes[batch_size] = probe
        logger.info(
            "Batch size %s: %s%s",
            batch_size,
            "safe" if probe.success else "OOM",
            (
                ""
                if not probe.success or probe.vram_reserved == 0
                else f" ({probe.vram_reserved_mb}MB reserved)"
            ),
        )
        progress_bar.update()
        return probe

    def _find_bounds(self, progress_bar: tqdm) -> tuple[int, int | None]:
        """Probe outward until a successful lower and failing upper bound are known."""
        candidate = min(max(1, self._configured_batch_size), self._max_batch_size)
        lower = 0
        upper: int | None = None

        while True:
            probe = self._probe(candidate, progress_bar)
            if probe.success:
                lower = candidate
                if candidate >= self._max_batch_size:
                    return lower, None
                candidate = min(self._max_batch_size, candidate * 2)
                continue

            upper = candidate
            if lower > 0 or candidate == 1:
                return lower, upper
            candidate = max(1, candidate // 2)

    def _refine_bounds(self, lower: int, upper: int | None, progress_bar: tqdm) -> int:
        """Binary search the gap between known safe and failed batch sizes."""
        if upper is None:
            return lower

        while lower + 1 < upper:
            candidate = (lower + upper) // 2
            probe = self._probe(candidate, progress_bar)
            if probe.success:
                lower = candidate
            else:
                upper = candidate
        return lower

    def _estimate_vram(self, batch_size: int) -> int:
        """Estimate reserved VRAM for a successful batch size from probe results."""
        if batch_size < 1:
            return 0

        exact = self._probes.get(batch_size)
        if exact is not None and exact.success:
            return exact.vram_reserved

        successes = [probe for probe in self._probes.values() if probe.success]
        if not successes:
            return 0

        nearest = max(successes, key=lambda probe: probe.batch_size)
        if nearest.vram_reserved == 0:
            return 0
        return int(round(nearest.vram_reserved * (batch_size / nearest.batch_size)))

    def recommend(self, max_safe_batch_size: int) -> BatchSizeRecommendation:
        """Return the conservative batch-size recommendation."""
        if max_safe_batch_size < 1:
            suggested = 0
        elif max_safe_batch_size == 1:
            suggested = 1
        else:
            suggested = max(
                1,
                min(max_safe_batch_size, math.floor(max_safe_batch_size * self._safety_margin)),
            )

        accumulation = (
            1
            if suggested < 1
            else max(1, math.ceil(self._target_effective_batch_size / suggested))
        )
        effective = suggested * accumulation
        return BatchSizeRecommendation(
            configured_batch_size=self._configured_batch_size,
            max_safe_batch_size=max_safe_batch_size,
            suggested_batch_size=suggested,
            estimated_vram_reserved=self._estimate_vram(suggested),
            gradient_accumulation_recommended=accumulation > 1,
            gradient_accumulation_steps=accumulation,
            effective_batch_size=effective,
            target_effective_batch_size=self._target_effective_batch_size,
            probes=self.probes,
        )

    def _output_recommendation(self, recommendation: BatchSizeRecommendation) -> None:
        """Log and print a compact recommendation summary."""
        if recommendation.max_safe_batch_size < 1:
            logger.error("No safe training batch size was found.")
            print("Training batch-size finder: no safe batch size found.")
            return

        vram = (
            "N/A"
            if recommendation.estimated_vram_reserved == 0
            else f"{int(round(recommendation.estimated_vram_reserved / (1024 * 1024)))}MB"
        )
        lines = [
            "Training batch-size finder recommendation:",
            f"  Max safe batch size: {recommendation.max_safe_batch_size}",
            f"  Suggested batch size: {recommendation.suggested_batch_size}",
            f"  Estimated reserved VRAM: {vram}",
        ]
        if recommendation.gradient_accumulation_recommended:
            lines.extend(
                [
                    "  Gradient accumulation: recommended",
                    f"  Suggested accumulation steps: "
                    f"{recommendation.gradient_accumulation_steps}",
                    f"  Effective batch size: {recommendation.effective_batch_size}",
                ]
            )
        else:
            lines.append("  Gradient accumulation: not required")

        for line in lines:
            logger.info(line)
        print("\n".join(lines))

    def find(self) -> BatchSizeRecommendation:
        """Run the batch-size search."""
        logger.info("Finding safe training batch size...")
        with tqdm(desc="Testing batch size", leave=False, smoothing=0) as progress_bar:
            lower, upper = self._find_bounds(progress_bar)
            max_safe = self._refine_bounds(lower, upper, progress_bar)

        recommendation = self.recommend(max_safe)
        self._output_recommendation(recommendation)
        return recommendation


__all__ = get_module_objects(__name__)
