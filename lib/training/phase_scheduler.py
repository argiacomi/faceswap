#!/usr/bin/env python3
"""Deterministic training phase scheduler.

This module is intentionally pure. It does not import torch, keras, trainer config,
or model state. The trainer can feed it stable inputs, then decide whether to apply
or only log the returned schedule.
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass

AutomationMode = T.Literal["off", "dry_run", "conservative", "balanced", "aggressive"]
PresetMode = T.Literal["conservative", "balanced", "aggressive"]
PhaseName = T.Literal["off", "broad_reconstruction", "detail_ramp", "final_refinement"]

_PHASE_INDEX: dict[PhaseName, float] = {
    "off": 0.0,
    "broad_reconstruction": 1.0,
    "detail_ramp": 2.0,
    "final_refinement": 3.0,
}
"""Stable numeric phase identifiers for TensorBoard step charts."""


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the inclusive 0.0 to 1.0 range."""
    return max(0.0, min(1.0, value))


def _linear_ramp(iteration: int, start: int, length: int) -> float:
    """Return deterministic linear ramp progress for ``iteration``."""
    if iteration < start:
        return 0.0
    if length <= 0:
        return 1.0
    return _clamp01((iteration - start) / length)


@dataclass(frozen=True)
class ScheduledMultipliers:
    """Scheduled multipliers for configurable training loss components.

    Config values remain the user requested maximums. These multipliers are applied
    on top of those config values by the training loop or loss collator.
    """

    primary_loss: float = 1.0
    secondary_loss: float = 1.0
    boundary_loss: float = 1.0
    region_perceptual_loss: float = 1.0
    identity_loss: float = 1.0

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0.0. Got {value}")

    @classmethod
    def identity(cls) -> ScheduledMultipliers:
        """Return no-op multipliers."""
        return cls()

    def as_dict(self) -> dict[str, float]:
        """Return a stable dict representation for logging or TensorBoard."""
        return {
            "primary_loss": self.primary_loss,
            "secondary_loss": self.secondary_loss,
            "boundary_loss": self.boundary_loss,
            "region_perceptual_loss": self.region_perceptual_loss,
            "identity_loss": self.identity_loss,
        }


@dataclass(frozen=True)
class PhasePreset:
    """Deterministic phase timing and ramp targets for one automation preset."""

    mode: PresetMode
    detail_start_iteration: int
    final_start_iteration: int
    detail_ramp_iterations: int
    final_ramp_iterations: int
    detail_target_multiplier: float = 1.0
    identity_target_multiplier: float = 1.0

    def __post_init__(self) -> None:
        ints = {
            "detail_start_iteration": self.detail_start_iteration,
            "final_start_iteration": self.final_start_iteration,
            "detail_ramp_iterations": self.detail_ramp_iterations,
            "final_ramp_iterations": self.final_ramp_iterations,
        }
        for name, value in ints.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0. Got {value}")

        if self.final_start_iteration < self.detail_start_iteration:
            raise ValueError(
                "final_start_iteration must be >= detail_start_iteration. "
                f"Got final={self.final_start_iteration}, detail={self.detail_start_iteration}"
            )

        floats = {
            "detail_target_multiplier": self.detail_target_multiplier,
            "identity_target_multiplier": self.identity_target_multiplier,
        }
        for name, float_value in floats.items():
            if float_value < 0.0:
                raise ValueError(f"{name} must be >= 0.0. Got {float_value}")


@dataclass(frozen=True)
class ScheduleState:
    """Scheduler output for a single iteration."""

    iteration: int
    mode: AutomationMode
    phase: PhaseName
    scheduled_multipliers: ScheduledMultipliers
    apply: bool
    transition_reason: str
    dry_run: bool = False
    """``True`` when values are calculated for logging only and never applied."""

    @property
    def effective_multipliers(self) -> ScheduledMultipliers:
        """Return multipliers that should affect training behavior."""
        if not self.apply:
            return ScheduledMultipliers.identity()
        return self.scheduled_multipliers

    def tensorboard_scalars(self) -> dict[str, float]:
        """Return stable scalar logs for TensorBoard."""
        return {
            f"phase_scheduler/{key}": value
            for key, value in self.scheduled_multipliers.as_dict().items()
        } | {
            "phase_scheduler/phase": _PHASE_INDEX.get(self.phase, 0.0),
            "phase_scheduler/apply": 1.0 if self.apply else 0.0,
            "phase_scheduler/dry_run": 1.0 if self.dry_run else 0.0,
        }


_MODE_PRESETS: dict[PresetMode, PhasePreset] = {
    "conservative": PhasePreset(
        mode="conservative",
        detail_start_iteration=40_000,
        final_start_iteration=160_000,
        detail_ramp_iterations=40_000,
        final_ramp_iterations=40_000,
        detail_target_multiplier=0.75,
        identity_target_multiplier=0.50,
    ),
    "balanced": PhasePreset(
        mode="balanced",
        detail_start_iteration=20_000,
        final_start_iteration=80_000,
        detail_ramp_iterations=20_000,
        final_ramp_iterations=20_000,
    ),
    "aggressive": PhasePreset(
        mode="aggressive",
        detail_start_iteration=5_000,
        final_start_iteration=40_000,
        detail_ramp_iterations=10_000,
        final_ramp_iterations=10_000,
    ),
}


class TrainingPhaseScheduler:
    """Pure deterministic scheduler for v1 training automation."""

    def __init__(
        self,
        mode: AutomationMode = "off",
        preset: PhasePreset | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self._mode = mode
        # ``dry_run`` previews the schedule for the selected preset without applying it.
        # The legacy ``"dry_run"`` mode is kept as an alias that previews the balanced preset.
        self._dry_run = bool(dry_run) or mode == "dry_run"
        self._preset = self._resolve_preset(mode, preset)

    @classmethod
    def for_mode(cls, mode: AutomationMode, *, dry_run: bool = False) -> T.Self:  # type: ignore[name-defined]
        """Build a scheduler from a built-in automation mode."""
        return cls(mode=mode, dry_run=dry_run)

    @property
    def mode(self) -> AutomationMode:
        """Configured automation mode."""
        return self._mode

    @property
    def preset(self) -> PhasePreset:
        """Resolved deterministic phase preset."""
        return self._preset

    @staticmethod
    def _resolve_preset(mode: AutomationMode, preset: PhasePreset | None) -> PhasePreset:
        if preset is not None:
            return preset
        if mode in ("off", "dry_run"):
            return _MODE_PRESETS["balanced"]
        return _MODE_PRESETS[T.cast(PresetMode, mode)]

    def at(self, iteration: int) -> ScheduleState:
        """Return deterministic schedule state for ``iteration``."""
        iteration = max(0, int(iteration))
        if self._mode == "off":
            return ScheduleState(
                iteration=iteration,
                mode=self._mode,
                phase="off",
                scheduled_multipliers=ScheduledMultipliers.identity(),
                apply=False,
                transition_reason="automation disabled",
                dry_run=False,
            )

        phase, multipliers, reason = self._phase_state(iteration)
        return ScheduleState(
            iteration=iteration,
            mode=self._mode,
            phase=phase,
            scheduled_multipliers=multipliers,
            apply=not self._dry_run,
            transition_reason=reason,
            dry_run=self._dry_run,
        )

    def _phase_state(self, iteration: int) -> tuple[PhaseName, ScheduledMultipliers, str]:
        preset = self._preset

        if iteration < preset.detail_start_iteration:
            return (
                "broad_reconstruction",
                ScheduledMultipliers(
                    primary_loss=1.0,
                    secondary_loss=0.0,
                    boundary_loss=0.0,
                    region_perceptual_loss=0.0,
                    identity_loss=0.0,
                ),
                (f"iteration {iteration} is before detail start {preset.detail_start_iteration}"),
            )

        if iteration < preset.final_start_iteration:
            progress = _linear_ramp(
                iteration,
                preset.detail_start_iteration,
                preset.detail_ramp_iterations,
            )
            detail = preset.detail_target_multiplier * progress
            return (
                "detail_ramp",
                ScheduledMultipliers(
                    primary_loss=1.0,
                    secondary_loss=detail,
                    boundary_loss=detail,
                    region_perceptual_loss=detail,
                    identity_loss=0.0,
                ),
                f"detail start {preset.detail_start_iteration} reached",
            )

        detail = preset.detail_target_multiplier
        identity_progress = _linear_ramp(
            iteration,
            preset.final_start_iteration,
            preset.final_ramp_iterations,
        )
        identity = preset.identity_target_multiplier * identity_progress
        return (
            "final_refinement",
            ScheduledMultipliers(
                primary_loss=1.0,
                secondary_loss=detail,
                boundary_loss=detail,
                region_perceptual_loss=detail,
                identity_loss=identity,
            ),
            f"final start {preset.final_start_iteration} reached",
        )


__all__ = [
    "AutomationMode",
    "PhaseName",
    "PhasePreset",
    "PresetMode",
    "ScheduledMultipliers",
    "ScheduleState",
    "TrainingPhaseScheduler",
]
