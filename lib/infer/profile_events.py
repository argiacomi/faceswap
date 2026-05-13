#!/usr/bin/env python3
"""Structured contracts for extract profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ProfileEvent:
    """Canonical profiling event emitted by extract profiling helpers."""

    stage: str
    plugin: str
    frame_index: int | None
    face_index: int | None
    start_ns: int
    end_ns: int
    device: str
    bytes_in: int = 0
    bytes_out: int = 0
    transfer_direction: str = ""

    @property
    def duration_ms(self) -> float:
        """Return the event duration in milliseconds."""
        return max(0, self.end_ns - self.start_ns) / 1_000_000.0

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Serialize the event as a plain dictionary."""
        retval = asdict(self)
        retval["duration_ms"] = self.duration_ms
        return retval


@dataclass(slots=True)
class ExtractProfileReport:
    """Aggregated extract pipeline profiling report."""

    frames_per_sec: float
    faces_per_sec: float
    queue_occupancy: dict[str, float]
    stage_times_ms: dict[str, float]
    transfer_times_ms: dict[str, float]
    memory: dict[str, float | str]
    bottleneck_stage: str
    recommended_batch_sizes: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Serialize the report as a plain dictionary."""
        return asdict(self)


@dataclass(slots=True)
class SyntheticProfileConfig:
    """Synthetic workload configuration for pipeline profiling."""

    frames: int = 200
    width: int = 640
    height: int = 360
    face_count_pattern: tuple[int, ...] = (1, 2, 0, 3)
    seed: int = 1337

    @property
    def shape(self) -> tuple[int, int]:
        """Return the configured frame shape as ``(height, width)``."""
        return (self.height, self.width)


@dataclass(slots=True)
class RealProfileConfig:
    """Real-media workload configuration for pipeline profiling."""

    input_locations: tuple[str, ...]
    profile_frames: int | None = None
    extract_every: int = 1
