#!/usr/bin/env python3
"""Hard-case slicing for landmark validation manifests (#82).

For AFLW2000-3D the per-sample ``Pose_Para`` vector (preserved in
``metadata.Pose_Para`` by :mod:`lib.landmarks.datasets.aflw2000_3d`) carries
the 3DDFA pose: ``[pitch, yaw, roll, tx, ty, tz, scale]`` with angles in
radians. We use the yaw component to slice into profile / large-yaw / frontal
buckets without ever requiring manual labels. The bucket name is written into
each sample as a ``hard_slice`` field so downstream geometry tooling can group
by it just like any other condition tag.

Buckets (default thresholds, in degrees of |yaw|):

* ``frontal``         — |yaw| < 15°
* ``profile_left``    — yaw between -60° and -30°
* ``profile_right``   — yaw between  30° and  60°
* ``large_yaw_left``  — yaw < -60°
* ``large_yaw_right`` — yaw >  60°
* ``intermediate``    — 15° ≤ |yaw| < 30° (kept for reference; not "hard")

Samples missing pose annotations fall into the ``no_pose`` bucket and are
included only when ``include_unposed=True`` because their hard-case status
cannot be determined from the manifest alone.
"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass

DEFAULT_FRONTAL_YAW_DEGREES: float = 15.0
DEFAULT_PROFILE_MIN_DEGREES: float = 30.0
DEFAULT_PROFILE_MAX_DEGREES: float = 60.0

HARD_SLICES: tuple[str, ...] = (
    "profile_left",
    "profile_right",
    "large_yaw_left",
    "large_yaw_right",
)


@dataclass(frozen=True)
class HardSliceThresholds:
    """Yaw-degree thresholds used to label a sample's hard-case bucket."""

    frontal_degrees: float = DEFAULT_FRONTAL_YAW_DEGREES
    profile_min_degrees: float = DEFAULT_PROFILE_MIN_DEGREES
    profile_max_degrees: float = DEFAULT_PROFILE_MAX_DEGREES

    def __post_init__(self) -> None:
        if not 0.0 < self.frontal_degrees < self.profile_min_degrees:
            raise ValueError(
                "frontal_degrees must satisfy 0 < frontal_degrees < profile_min_degrees"
            )
        if self.profile_min_degrees >= self.profile_max_degrees:
            raise ValueError("profile_min_degrees must be strictly less than profile_max_degrees")


def _yaw_radians(sample: T.Mapping[str, T.Any]) -> float | None:
    """Return the yaw angle in radians from an AFLW2000-3D-style manifest entry."""
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    pose = metadata.get("Pose_Para") or sample.get("Pose_Para")
    if pose is None:
        return None
    try:
        return float(pose[1])
    except (IndexError, TypeError, ValueError):
        return None


def yaw_degrees(sample: T.Mapping[str, T.Any]) -> float | None:
    """Return yaw in degrees for a manifest sample, or ``None`` when unknown."""
    radians = _yaw_radians(sample)
    if radians is None:
        return None
    return math.degrees(radians)


def hard_slice_label(
    yaw_deg: float | None,
    *,
    thresholds: HardSliceThresholds | None = None,
) -> str:
    """Return the hard-slice bucket for ``yaw_deg`` (signed degrees)."""
    if yaw_deg is None:
        return "no_pose"
    if thresholds is None:
        thresholds = HardSliceThresholds()
    magnitude = abs(yaw_deg)
    if magnitude < thresholds.frontal_degrees:
        return "frontal"
    if magnitude < thresholds.profile_min_degrees:
        return "intermediate"
    if magnitude < thresholds.profile_max_degrees:
        return "profile_left" if yaw_deg < 0 else "profile_right"
    return "large_yaw_left" if yaw_deg < 0 else "large_yaw_right"


def is_hard_slice(label: str) -> bool:
    """Return True when ``label`` names one of the hard-case buckets."""
    return label in HARD_SLICES


def slice_manifest_samples(
    samples: T.Sequence[T.Mapping[str, T.Any]],
    *,
    thresholds: HardSliceThresholds | None = None,
    include_unposed: bool = False,
    hard_only: bool = True,
) -> tuple[list[dict[str, T.Any]], dict[str, int]]:
    """Tag and optionally filter manifest samples by hard-slice bucket.

    Returns a list of new sample dicts with a ``hard_slice`` field added (and
    the existing ``condition`` overwritten to match) plus a histogram of
    samples per bucket. Samples without pose annotations are kept only when
    ``include_unposed=True`` because their hard-case status cannot be inferred
    from the manifest.
    """
    if thresholds is None:
        thresholds = HardSliceThresholds()
    counts: dict[str, int] = {}
    sliced: list[dict[str, T.Any]] = []
    for sample in samples:
        yaw_deg = yaw_degrees(sample)
        label = hard_slice_label(yaw_deg, thresholds=thresholds)
        counts[label] = counts.get(label, 0) + 1
        if label == "no_pose" and not include_unposed:
            continue
        if hard_only and not is_hard_slice(label):
            continue
        tagged = dict(sample)
        tagged["hard_slice"] = label
        tagged["condition"] = label
        if yaw_deg is not None:
            tagged["yaw_degrees"] = float(yaw_deg)
        sliced.append(tagged)
    return sliced, counts


__all__ = [
    "DEFAULT_FRONTAL_YAW_DEGREES",
    "DEFAULT_PROFILE_MAX_DEGREES",
    "DEFAULT_PROFILE_MIN_DEGREES",
    "HARD_SLICES",
    "HardSliceThresholds",
    "hard_slice_label",
    "is_hard_slice",
    "slice_manifest_samples",
    "yaw_degrees",
]
