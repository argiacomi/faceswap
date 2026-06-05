from __future__ import annotations

import pytest

from lib.landmarks.ensemble.stacked_regressor import (
    RUNTIME_CONTEXT_SCOPE_ALL,
    RUNTIME_CONTEXT_SCOPE_FRONTAL_INTERMEDIATE_ONLY,
    RUNTIME_CONTEXT_SCOPE_LARGE_YAW_LEFT_ONLY,
    RUNTIME_CONTEXT_SCOPE_LARGE_YAW_ONLY,
    RUNTIME_CONTEXT_SCOPE_LARGE_YAW_RIGHT_ONLY,
    RUNTIME_CONTEXT_SCOPE_NON_PROFILE,
    RUNTIME_CONTEXT_SCOPE_PROFILE_ONLY,
    StackedRegressorInvalid,
    runtime_bucket_supported_by_scope,
)


def test_runtime_bucket_supported_by_scope() -> None:
    assert runtime_bucket_supported_by_scope("profile_left", RUNTIME_CONTEXT_SCOPE_ALL)
    assert not runtime_bucket_supported_by_scope("profile_left", RUNTIME_CONTEXT_SCOPE_NON_PROFILE)
    assert runtime_bucket_supported_by_scope(
        "rolled_profile_right", RUNTIME_CONTEXT_SCOPE_PROFILE_ONLY
    )
    assert runtime_bucket_supported_by_scope(
        "rolled_large_yaw_right", RUNTIME_CONTEXT_SCOPE_LARGE_YAW_ONLY
    )
    assert runtime_bucket_supported_by_scope(
        "large_yaw_left", RUNTIME_CONTEXT_SCOPE_LARGE_YAW_LEFT_ONLY
    )
    assert not runtime_bucket_supported_by_scope(
        "large_yaw_right", RUNTIME_CONTEXT_SCOPE_LARGE_YAW_LEFT_ONLY
    )
    assert runtime_bucket_supported_by_scope(
        "large_yaw_right", RUNTIME_CONTEXT_SCOPE_LARGE_YAW_RIGHT_ONLY
    )
    assert runtime_bucket_supported_by_scope(
        "frontal", RUNTIME_CONTEXT_SCOPE_FRONTAL_INTERMEDIATE_ONLY
    )
    assert runtime_bucket_supported_by_scope(
        "intermediate", RUNTIME_CONTEXT_SCOPE_FRONTAL_INTERMEDIATE_ONLY
    )
    assert not runtime_bucket_supported_by_scope(
        "large_yaw_left", RUNTIME_CONTEXT_SCOPE_FRONTAL_INTERMEDIATE_ONLY
    )


def test_runtime_bucket_supported_by_scope_rejects_unknown_scope() -> None:
    with pytest.raises(StackedRegressorInvalid):
        runtime_bucket_supported_by_scope("frontal", "unknown_scope")
