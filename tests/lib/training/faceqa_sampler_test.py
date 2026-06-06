#!/usr/bin/env python3
"""Tests for FaceQA-aware training sampler weighting."""

from __future__ import annotations

import numpy as np
import pytest

from lib.training.data.data_set import MultiDataset
from lib.training.faceqa_diagnostics import FaceQASampleMetadata
from lib.training.faceqa_sampler import (
    FaceQASamplerConfig,
    compute_faceqa_sample_weights,
    normalize_dimensions,
)
from plugins.train.trainer import trainer_config as trn_cfg


def _sample(
    idx: int,
    *,
    yaw: str = "frontal",
    expression: str = "neutral",
    lighting: str = "balanced",
    blur: str = "good",
    resolution: str = "good",
    duplicate: str = "unique",
    outlier: str = "inlier",
    has_faceqa: bool = True,
) -> FaceQASampleMetadata:
    """Build compact FaceQA sample metadata for sampler tests."""
    return FaceQASampleMetadata(
        side="A",
        filename=f"face_{idx}.png",
        source_file=f"src_{idx}.png",
        source_id=f"src_{idx}.png:0",
        face_index=0,
        has_faceqa=has_faceqa,
        yaw_pose_bucket=yaw,
        expression_bucket=expression,
        lighting_bucket=lighting,
        blur_bucket=blur,
        resolution_bucket=resolution,
        duplicate_bucket=duplicate,
        identity_outlier_bucket=outlier,
    )


class _DummySet:
    """Minimal dataset for MultiDataset shuffle tests."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value = np.array(index, dtype=np.int64)
        return value, value, np.array(_sample(index), dtype=object)


def test_normalize_dimensions_accepts_aliases_and_preserves_order() -> None:
    """Config text should support stable aliases without duplicate dimensions."""
    assert normalize_dimensions("pose, expression, yaw, lighting") == (
        "yaw_pose",
        "expression",
        "lighting",
    )


def test_faceqa_weighting_upweights_underrepresented_useful_bucket() -> None:
    """Rare useful buckets should receive higher sampling weight than common buckets."""
    samples = [
        _sample(0, yaw="frontal"),
        _sample(1, yaw="frontal"),
        _sample(2, yaw="frontal"),
        _sample(3, yaw="right_profile"),
    ]
    weights, summary = compute_faceqa_sample_weights(
        "A",
        samples,
        FaceQASamplerConfig(
            mode="faceqa_weighted",
            strength=1.0,
            dimensions=("yaw_pose",),
        ),
    )

    assert weights is not None
    assert weights[3] > weights[0]
    assert weights.mean() == pytest.approx(1.0)
    assert summary.metadata_count == 4
    assert summary.effective_sample_count < 4.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duplicate": "duplicate"},
        {"outlier": "outlier"},
        {"blur": "unusable"},
        {"resolution": "tiny"},
    ],
)
def test_bad_samples_are_downweighted_and_not_amplified(kwargs: dict[str, str]) -> None:
    """Known duplicate/outlier/low-quality samples should never be amplified."""
    samples = [
        _sample(0, yaw="frontal"),
        _sample(1, yaw="frontal"),
        _sample(2, yaw="right_profile", **kwargs),
    ]
    weights, _summary = compute_faceqa_sample_weights(
        "A",
        samples,
        FaceQASamplerConfig(
            mode="faceqa_weighted",
            strength=1.0,
            dimensions=("yaw_pose",),
        ),
    )

    assert weights is not None
    assert weights[2] <= 1.0
    assert weights[2] < weights[0]


def test_missing_metadata_falls_back_to_random_sampling() -> None:
    """No FaceQA metadata should return ``None`` weights for exact random fallback."""
    samples = [_sample(0, has_faceqa=False), _sample(1, has_faceqa=False)]
    weights, summary = compute_faceqa_sample_weights(
        "A",
        samples,
        FaceQASamplerConfig(mode="faceqa_weighted", strength=1.0),
    )

    assert weights is None
    assert summary.metadata_count == 0
    assert summary.total_count == 2


def test_curriculum_strength_multiplier_changes_weights() -> None:
    """Phase scheduler multipliers should scale curriculum sampler strength."""
    samples = [
        _sample(0, yaw="frontal"),
        _sample(1, yaw="frontal"),
        _sample(2, yaw="right_profile"),
    ]
    base = FaceQASamplerConfig(
        mode="faceqa_curriculum",
        strength=1.0,
        dimensions=("yaw_pose",),
    )

    off_weights, _ = compute_faceqa_sample_weights(
        "A", samples, base.with_strength_multiplier(0.0)
    )
    on_weights, _ = compute_faceqa_sample_weights("A", samples, base.with_strength_multiplier(1.0))

    assert off_weights is None
    assert on_weights is not None
    assert on_weights[2] > on_weights[0]


def test_curriculum_bucket_loss_scores_emphasize_valid_high_loss_bucket() -> None:
    """Rolling diagnostics can emphasize valid high-loss buckets."""
    samples = [
        _sample(0, yaw="frontal"),
        _sample(1, yaw="right_profile"),
        _sample(2, yaw="right_profile", duplicate="duplicate"),
    ]
    config = FaceQASamplerConfig(
        mode="faceqa_curriculum",
        strength=1.0,
        dimensions=("yaw_pose",),
    )

    weights, _ = compute_faceqa_sample_weights(
        "A",
        samples,
        config,
        {
            ("A", "yaw_pose", "frontal"): 1.0,
            ("A", "yaw_pose", "right_profile"): 4.0,
        },
    )

    assert weights is not None
    assert weights[1] > weights[0]
    assert weights[2] <= 1.0


def test_multidataset_uses_side_specific_sample_weights() -> None:
    """Weighted side shuffling should draw high-weighted side samples more often."""
    np.random.seed(0)
    weights = np.full(100, 0.01, dtype=np.float64)
    weights[7] = 100.0
    dataset = MultiDataset(
        (  # type: ignore[arg-type]
            _DummySet(100),
        ),
        is_random=True,
        sample_weights=(weights,),
    )

    assert np.count_nonzero(dataset._indices[0] == 7) > 90  # pylint:disable=protected-access


def test_training_sampler_config_defaults_are_random_fallback() -> None:
    """Sampler config defaults should preserve existing random behavior."""
    assert trn_cfg.Automation.training_sampler() == "random"
    assert trn_cfg.Automation.faceqa_sampler_strength() == "auto"
    assert trn_cfg.Automation.faceqa_sampler_dimensions() == "pose,expression,lighting"
    assert trn_cfg.Automation.faceqa_downweight_duplicates() is True
    assert trn_cfg.Automation.faceqa_downweight_outliers() is True
    assert trn_cfg.Automation.faceqa_min_quality() == "usable"
