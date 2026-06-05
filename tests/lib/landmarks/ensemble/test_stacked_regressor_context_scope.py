from __future__ import annotations

from types import SimpleNamespace

from lib.landmarks.ensemble.stacked_regressor_training import (
    CONTEXT_SCOPE_FRONTAL_INTERMEDIATE_ONLY,
    CONTEXT_SCOPE_LARGE_YAW_LEFT_ONLY,
    CONTEXT_SCOPE_LARGE_YAW_ONLY,
    CONTEXT_SCOPE_LARGE_YAW_RIGHT_ONLY,
    CONTEXT_SCOPE_NON_PROFILE,
    CONTEXT_SCOPE_PROFILE_ONLY,
    filter_contexts_for_scope,
)


def _ctx(bucket: str) -> SimpleNamespace:
    return SimpleNamespace(runtime_bucket=bucket)


def test_context_scope_yaw_and_frontal_filters() -> None:
    rows = [
        _ctx("frontal"),
        _ctx("intermediate"),
        _ctx("large_yaw_left"),
        _ctx("large_yaw_right"),
        _ctx("rolled_large_yaw_right"),
        _ctx("profile_left"),
        _ctx("rolled_profile_right"),
    ]

    assert len(filter_contexts_for_scope(rows, context_scope=CONTEXT_SCOPE_NON_PROFILE)) == 5
    assert len(filter_contexts_for_scope(rows, context_scope=CONTEXT_SCOPE_PROFILE_ONLY)) == 2
    assert len(filter_contexts_for_scope(rows, context_scope=CONTEXT_SCOPE_LARGE_YAW_ONLY)) == 3
    assert (
        len(filter_contexts_for_scope(rows, context_scope=CONTEXT_SCOPE_LARGE_YAW_LEFT_ONLY)) == 1
    )
    assert (
        len(filter_contexts_for_scope(rows, context_scope=CONTEXT_SCOPE_LARGE_YAW_RIGHT_ONLY)) == 2
    )
    assert (
        len(filter_contexts_for_scope(rows, context_scope=CONTEXT_SCOPE_FRONTAL_INTERMEDIATE_ONLY))
        == 2
    )
