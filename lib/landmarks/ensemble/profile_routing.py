#!/usr/bin/env python3
"""Profile/occlusion routing helpers for the runtime resolver scorer (#218).

The general ``learned_quality_v3`` scorer handles normal/frontal/intermediate
faces well, but profile/occlusion faces have distinct failure modes that
warrant a specialist policy (``learned_quality_v3_profile``) and a profile-safe
validity-first selector. These helpers centralize the "is this a profile or
occlusion context" decision and the policy route so the selector, evaluator,
reports, and repair candidate generator (#219) all agree.
"""

from __future__ import annotations

import typing as T

SCORER_POLICY_GENERAL = "learned_quality_v3"
SCORER_POLICY_PROFILE = "learned_quality_v3_profile"

#: Substrings that mark a tag as a profile/large-yaw/rolled/occlusion route.
PROFILE_ROUTE_HINTS: tuple[str, ...] = (
    "profile",
    "large_yaw",
    "yaw_",
    "rolled",
    "occlusion",
    "occluded",
)

#: Condition labels included when training/evaluating the profile specialist.
PROFILE_SPECIALIST_SPLITS: tuple[str, ...] = (
    "profile",
    "profile_left",
    "profile_right",
    "large_yaw_left",
    "large_yaw_right",
    "rolled_profile_left",
    "rolled_profile_right",
    "rolled_large_yaw_left",
    "rolled_large_yaw_right",
    "occlusion",
    "profile_occlusion",
    "single_eye_visible",
    "mouth_or_jaw_occluded",
    "large_yaw_occlusion",
)


def _normalize_tag(value: T.Any) -> str:
    return str(value or "").strip().lower()


def condition_tags(context_or_bucket: T.Any) -> tuple[str, ...]:
    """Return the lowercased condition/bucket/hard-case tags for a context.

    Accepts a context-like object exposing ``condition`` / ``runtime_bucket`` /
    ``hard_case_tags``, a ``(runtime_bucket, condition)`` tuple, a mapping, or a
    plain string. Unknown shapes yield an empty tuple.
    """
    if context_or_bucket is None:
        return ()
    if isinstance(context_or_bucket, str):
        tag = _normalize_tag(context_or_bucket)
        return (tag,) if tag else ()
    if isinstance(context_or_bucket, T.Mapping):
        values: list[T.Any] = [
            context_or_bucket.get("condition"),
            context_or_bucket.get("runtime_bucket"),
            *(context_or_bucket.get("hard_case_tags") or ()),
        ]
    elif isinstance(context_or_bucket, (tuple, list)):
        values = list(context_or_bucket)
    else:
        values = [
            getattr(context_or_bucket, "condition", None),
            getattr(context_or_bucket, "runtime_bucket", None),
            *(getattr(context_or_bucket, "hard_case_tags", ()) or ()),
        ]
    tags = [_normalize_tag(value) for value in values]
    return tuple(dict.fromkeys(tag for tag in tags if tag))


def is_profile_or_occlusion_context(context_or_bucket: T.Any) -> bool:
    """Return ``True`` when any tag marks a profile/large-yaw/occlusion route."""
    tags = condition_tags(context_or_bucket)
    return any(hint in tag for tag in tags for hint in PROFILE_ROUTE_HINTS)


def scorer_route_for_context(context_or_bucket: T.Any) -> str:
    """Return the scorer policy that should score a context.

    Profile/large-yaw/rolled/occlusion routes use the profile specialist;
    everything else uses the general scorer.
    """
    if is_profile_or_occlusion_context(context_or_bucket):
        return SCORER_POLICY_PROFILE
    return SCORER_POLICY_GENERAL


__all__ = [
    "PROFILE_ROUTE_HINTS",
    "PROFILE_SPECIALIST_SPLITS",
    "SCORER_POLICY_GENERAL",
    "SCORER_POLICY_PROFILE",
    "condition_tags",
    "is_profile_or_occlusion_context",
    "scorer_route_for_context",
]
