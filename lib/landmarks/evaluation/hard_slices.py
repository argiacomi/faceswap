#!/usr/bin/env python3
"""Hard-case slicing for landmark validation manifests (#82).

For AFLW2000-3D the per-sample ``Pose_Para`` vector (preserved in
``metadata.Pose_Para`` by :mod:`lib.landmarks.datasets.aflw2000_3d`) carries
the 3DDFA pose: ``[pitch, yaw, roll, tx, ty, tz, scale]`` with angles in
radians. We use yaw plus in-plane roll to slice into profile / large-yaw /
rolled buckets without ever requiring manual labels. The bucket name is written
into each sample as a ``hard_slice`` field so downstream geometry tooling can
group by it like any other condition tag.

Buckets (default thresholds, in degrees):

* ``frontal``                  — |yaw| < 15° and |roll| < 30°
* ``intermediate``             — 15° ≤ |yaw| < 30° and |roll| < 30°
* ``profile_left``             — -60° < yaw ≤ -30° and |roll| < 30°
* ``profile_right``            —  30° ≤ yaw <  60° and |roll| < 30°
* ``large_yaw_left``           — yaw ≤ -60° and |roll| < 30°
* ``large_yaw_right``          — yaw ≥  60° and |roll| < 30°
* ``large_roll``               — |roll| ≥ 30° on frontal/intermediate yaw
* ``extreme_roll``             — |roll| ≥ 45° on frontal/intermediate yaw
* ``rolled_profile_left``      — profile-left yaw and |roll| ≥ 30°
* ``rolled_profile_right``     — profile-right yaw and |roll| ≥ 30°
* ``rolled_large_yaw_left``    — large-left yaw and |roll| ≥ 30°
* ``rolled_large_yaw_right``   — large-right yaw and |roll| ≥ 30°

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
DEFAULT_ROLL_DEGREES: float = 30.0
DEFAULT_EXTREME_ROLL_DEGREES: float = 45.0

REFERENCE_SLICES: tuple[str, ...] = (
    "frontal",
    "intermediate",
    "no_pose",
)

YAW_HARD_SLICES: tuple[str, ...] = (
    "profile_left",
    "profile_right",
    "large_yaw_left",
    "large_yaw_right",
)

ROLL_HARD_SLICES: tuple[str, ...] = (
    "large_roll",
    "extreme_roll",
    "rolled_profile_left",
    "rolled_profile_right",
    "rolled_large_yaw_left",
    "rolled_large_yaw_right",
)

HARD_SLICES: tuple[str, ...] = (*YAW_HARD_SLICES, *ROLL_HARD_SLICES)
ALL_SLICES: tuple[str, ...] = (*REFERENCE_SLICES, *HARD_SLICES)

#: Buckets written by the hard-negative manifest builder (#217). When a sample
#: already carries one of these we trust the mined selection and skip AFLW pose
#: slicing entirely - mined manifests are themselves the selected hard source.
HARD_NEGATIVE_BUCKETS: frozenset[str] = frozenset(
    {"profile_occlusion", "profile", "occlusion", "anchor"}
)


def hard_negative_bucket(sample: T.Mapping[str, T.Any]) -> str | None:
    """Return the mined hard-negative bucket for a sample, or ``None``.

    Reads ``metadata.hard_negative_bucket`` (set by the hard-negative manifest
    builder), tolerating the value at the sample top level too. Only the four
    canonical buckets are recognized so arbitrary condition strings do not
    bypass pose slicing.
    """
    raw_metadata = sample.get("metadata")
    metadata: dict[str, T.Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    raw = (
        metadata.get("hard_negative_bucket")
        or sample.get("hard_negative_bucket")
        or sample.get("hard_negative_bucket_name")
    )
    bucket = str(raw or "").strip().lower()
    return bucket if bucket in HARD_NEGATIVE_BUCKETS else None


@dataclass(frozen=True)
class HardSliceThresholds:
    """Pose-degree thresholds used to label a sample's hard-case bucket."""

    frontal_degrees: float = DEFAULT_FRONTAL_YAW_DEGREES
    profile_min_degrees: float = DEFAULT_PROFILE_MIN_DEGREES
    profile_max_degrees: float = DEFAULT_PROFILE_MAX_DEGREES
    roll_degrees: float = DEFAULT_ROLL_DEGREES
    extreme_roll_degrees: float = DEFAULT_EXTREME_ROLL_DEGREES

    def __post_init__(self) -> None:
        if not 0.0 < self.frontal_degrees < self.profile_min_degrees:
            raise ValueError(
                "frontal_degrees must satisfy 0 < frontal_degrees < profile_min_degrees"
            )
        if self.profile_min_degrees >= self.profile_max_degrees:
            raise ValueError("profile_min_degrees must be strictly less than profile_max_degrees")
        if self.roll_degrees <= 0.0:
            raise ValueError("roll_degrees must be > 0")
        if self.extreme_roll_degrees < self.roll_degrees:
            raise ValueError("extreme_roll_degrees must be >= roll_degrees")


def _pose_radians(sample: T.Mapping[str, T.Any]) -> T.Sequence[T.Any] | None:
    """Return AFLW2000-3D-style pose radians from a manifest sample."""
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    pose = metadata.get("Pose_Para") or sample.get("Pose_Para")  # type: ignore[union-attr]
    if pose is None:
        return None
    return pose  # type: ignore[no-any-return]


def _pose_angle_degrees(sample: T.Mapping[str, T.Any], index: int) -> float | None:
    pose = _pose_radians(sample)
    if pose is None:
        return None
    try:
        return math.degrees(float(pose[index]))
    except (IndexError, TypeError, ValueError):
        return None


def yaw_degrees(sample: T.Mapping[str, T.Any]) -> float | None:
    """Return yaw in degrees for a manifest sample, or ``None`` when unknown."""
    return _pose_angle_degrees(sample, 1)


def roll_degrees(sample: T.Mapping[str, T.Any]) -> float | None:
    """Return in-plane roll in degrees for a manifest sample, or ``None`` when unknown."""
    return _pose_angle_degrees(sample, 2)


def yaw_slice_label(
    yaw_deg: float | None,
    *,
    thresholds: HardSliceThresholds | None = None,
) -> str:
    """Return the yaw-only bucket for ``yaw_deg`` (signed degrees)."""
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


def hard_slice_label(
    yaw_deg: float | None,
    *,
    roll_deg: float | None = None,
    thresholds: HardSliceThresholds | None = None,
) -> str:
    """Return the pose hard-slice bucket for signed yaw and roll degrees.

    Roll labels take precedence only when roll crosses the configured threshold.
    Yaw-hard samples with large roll get combined labels such as
    ``rolled_profile_left`` so the geometry evaluator can report catastrophics
    for rotated profile / large-yaw cases separately from ordinary profile cases.
    Frontal/intermediate samples with large roll use ``large_roll`` or
    ``extreme_roll``.
    """
    yaw_label = yaw_slice_label(yaw_deg, thresholds=thresholds)
    if yaw_label == "no_pose":
        return yaw_label
    if thresholds is None:
        thresholds = HardSliceThresholds()
    if roll_deg is None:
        return yaw_label
    roll_magnitude = abs(roll_deg)
    if roll_magnitude < thresholds.roll_degrees:
        return yaw_label
    if yaw_label in {"profile_left", "profile_right"}:
        return f"rolled_{yaw_label}"
    if yaw_label in {"large_yaw_left", "large_yaw_right"}:
        return f"rolled_{yaw_label}"
    if roll_magnitude >= thresholds.extreme_roll_degrees:
        return "extreme_roll"
    return "large_roll"


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
    """Tag and optionally filter manifest samples by pose hard-slice bucket.

    Returns a list of new sample dicts with a ``hard_slice`` field added (and
    the existing ``condition`` overwritten to match) plus a histogram of
    samples per bucket. Samples without pose annotations are kept only when
    ``include_unposed=True`` because their hard-case status cannot be inferred
    from the manifest.
    """
    if thresholds is None:
        thresholds = HardSliceThresholds()
    counts: dict[str, int] = dict.fromkeys(ALL_SLICES, 0)
    sliced: list[dict[str, T.Any]] = []
    for sample in samples:
        # Mined / hard-negative-annotated samples bypass AFLW pose slicing: the
        # mined manifest is already the selected hard source, so the bucket is
        # trusted directly and Pose_Para is not required. Anchors are kept (they
        # are deliberate controls in the #217 mix) regardless of ``hard_only``.
        bucket = hard_negative_bucket(sample)
        if bucket is not None:
            counts[bucket] = counts.get(bucket, 0) + 1
            tagged = dict(sample)
            tagged["hard_slice"] = bucket
            tagged["condition"] = bucket
            conditions = list(tagged.get("conditions") or [])
            if bucket not in conditions:
                conditions.insert(0, bucket)
            tagged["conditions"] = conditions
            sliced.append(tagged)
            continue
        yaw_deg = yaw_degrees(sample)
        roll_deg = roll_degrees(sample)
        yaw_label = yaw_slice_label(yaw_deg, thresholds=thresholds)
        label = hard_slice_label(yaw_deg, roll_deg=roll_deg, thresholds=thresholds)
        counts[label] = counts.get(label, 0) + 1
        if label == "no_pose" and not include_unposed:
            continue
        if hard_only and not is_hard_slice(label):
            continue
        tagged = dict(sample)
        tagged["hard_slice"] = label
        tagged["condition"] = label
        tagged["yaw_slice"] = yaw_label
        if yaw_deg is not None:
            tagged["yaw_degrees"] = float(yaw_deg)
        if roll_deg is not None:
            tagged["roll_degrees"] = float(roll_deg)
            tagged["abs_roll_degrees"] = abs(float(roll_deg))
        sliced.append(tagged)
    return sliced, counts


__all__ = [
    "ALL_SLICES",
    "DEFAULT_EXTREME_ROLL_DEGREES",
    "DEFAULT_FRONTAL_YAW_DEGREES",
    "DEFAULT_PROFILE_MAX_DEGREES",
    "DEFAULT_PROFILE_MIN_DEGREES",
    "DEFAULT_ROLL_DEGREES",
    "HARD_NEGATIVE_BUCKETS",
    "HARD_SLICES",
    "REFERENCE_SLICES",
    "ROLL_HARD_SLICES",
    "YAW_HARD_SLICES",
    "HardSliceThresholds",
    "hard_negative_bucket",
    "hard_slice_label",
    "is_hard_slice",
    "roll_degrees",
    "slice_manifest_samples",
    "yaw_degrees",
    "yaw_slice_label",
]
