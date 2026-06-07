#!/usr/bin/env python3
"""Tests for FaceQA-aware training sampler weighting."""

from __future__ import annotations

import typing as T

import numpy as np
import numpy.typing as npt
import pytest

from lib.training import faceqa_sampler
from lib.training.data.data_set import MultiDataset, TrainSet
from lib.training.data.loader import TrainLoader
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
    mask_qa: str = "present",
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
        mask_qa_bucket=mask_qa,
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
        {"mask_qa": "missing"},
    ],
)
def test_bad_samples_are_downweighted_and_not_amplified(kwargs: dict[str, T.Any]) -> None:
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


def test_uniform_metadata_preserves_random_sampling_fallback() -> None:
    """Uniform FaceQA signals should not switch MultiDataset into replacement sampling."""
    samples = [_sample(idx) for idx in range(4)]
    weights, summary = compute_faceqa_sample_weights(
        "A",
        samples,
        FaceQASamplerConfig(
            mode="faceqa_weighted",
            strength=1.0,
            dimensions=("yaw_pose",),
        ),
    )

    assert weights is None
    assert summary.metadata_count == 4
    assert summary.effective_sample_count == pytest.approx(4.0)


def test_protected_samples_stay_below_missing_metadata() -> None:
    """Protected rows should not be normalized back to neutral when mixed with missing rows."""
    samples = [
        _sample(0, duplicate="duplicate"),
        _sample(1, outlier="outlier"),
        _sample(2, has_faceqa=False),
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
    assert weights[0] < weights[2]
    assert weights[1] < weights[2]
    assert weights[2] == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs", [{"duplicate": "duplicate"}, {"outlier": "outlier"}])
def test_all_protected_uniform_samples_preserve_random_fallback(kwargs: dict[str, T.Any]) -> None:
    """Uniformly protected datasets should not switch to replacement sampling."""
    samples = [_sample(idx, **kwargs) for idx in range(4)]
    weights, summary = compute_faceqa_sample_weights(
        "A",
        samples,
        FaceQASamplerConfig(
            mode="faceqa_weighted",
            strength=1.0,
            dimensions=("yaw_pose",),
        ),
    )

    assert weights is None
    assert summary.effective_sample_count == pytest.approx(4.0)


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


def test_non_finite_curriculum_bucket_loss_preserves_random_fallback() -> None:
    """Invalid diagnostic scores should not create replacement weighted sampling."""
    samples = [
        _sample(0, yaw="frontal"),
        _sample(1, yaw="right_profile"),
    ]
    weights, summary = compute_faceqa_sample_weights(
        "A",
        samples,
        FaceQASamplerConfig(
            mode="faceqa_curriculum",
            strength=1.0,
            dimensions=("yaw_pose",),
        ),
        {
            ("A", "yaw_pose", "frontal"): 1.0,
            ("A", "yaw_pose", "right_profile"): float("inf"),
        },
    )

    assert weights is None
    assert summary.metadata_count == 2


def test_invalid_final_weights_return_finite_random_fallback_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal invalid-weight fallback should not leak NaN diagnostics."""
    samples = [
        _sample(0, yaw="frontal"),
        _sample(1, yaw="right_profile"),
    ]
    monkeypatch.setattr(
        faceqa_sampler,
        "_blend_and_normalize",
        lambda *_args, **_kwargs: np.array([np.nan, 1.0], dtype=np.float64),
    )

    weights, summary = compute_faceqa_sample_weights(
        "A",
        samples,
        FaceQASamplerConfig(
            mode="faceqa_curriculum",
            strength=1.0,
            dimensions=("yaw_pose",),
        ),
    )

    assert weights is None
    assert np.isfinite(summary.effective_sample_count)
    assert summary.effective_sample_count == pytest.approx(2.0)
    assert all(np.isfinite(weight) for _label, weight in summary.top_upweighted)
    assert all(np.isfinite(weight) for _label, weight in summary.top_downweighted)


def test_multidataset_uses_side_specific_sample_weights() -> None:
    """Weighted side shuffling should draw high-weighted side samples more often."""
    np.random.seed(0)
    weights: npt.NDArray[np.float64] = np.full(100, 0.01, dtype=np.float64)
    weights[7] = 100.0
    dataset = MultiDataset(
        (  # type: ignore[arg-type]
            _DummySet(100),
        ),
        is_random=True,
        sample_weights=(weights,),
    )

    assert np.count_nonzero(dataset._indices[0] == 7) > 90  # pylint:disable=protected-access


def test_multidataset_preserves_unweighted_remainder_when_weights_change() -> None:
    """Sampler changes should not discard no-replacement coverage for unweighted sides."""
    dataset = MultiDataset(
        (  # type: ignore[arg-type]
            _DummySet(3),
            _DummySet(5),
        ),
        is_random=True,
    )
    dataset._remainder = [  # pylint:disable=protected-access
        np.array([1, 2], dtype=np.int64),
        np.array([3, 4], dtype=np.int64),
    ]

    dataset.set_sample_weights((np.array([1.0, 2.0, 3.0], dtype=np.float64), None))

    assert dataset._remainder[0].size == 0  # pylint:disable=protected-access
    assert np.array_equal(dataset._remainder[1], np.array([3, 4]))  # pylint:disable=protected-access


def test_trainset_sampling_metadata_is_cached(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampler recomputes should reuse parsed metadata instead of scanning PNG headers again."""
    filenames = [tmp_path / f"face_{idx}.png" for idx in range(3)]
    for filename in filenames:
        filename.touch()
    calls = 0

    def fake_read_image_meta_batch(file_list: list[str]):
        nonlocal calls
        calls += 1
        return [(filename, {}) for filename in file_list]

    monkeypatch.setattr(
        "lib.training.data.data_set.read_image_meta_batch",
        fake_read_image_meta_batch,
    )
    data_set = TrainSet("A", str(tmp_path), 64, include_faceqa=True)

    first = data_set.faceqa_metadata_for_sampling()
    second = data_set.faceqa_metadata_for_sampling()

    assert first is second
    assert calls == 1


def test_curriculum_strength_update_is_queued_for_epoch_boundary() -> None:
    """Phase changes should be queued so active DataLoader workers keep a stable epoch view."""
    loader = TrainLoader.__new__(TrainLoader)
    loader._faceqa_sampler = FaceQASamplerConfig(  # pylint:disable=protected-access
        mode="faceqa_curriculum",
        strength=1.0,
    )
    loader._faceqa_sampler_multiplier = 0.0  # pylint:disable=protected-access
    loader._faceqa_sampler_weights_dirty = False  # pylint:disable=protected-access
    loader._sample_weights = pytest.fail  # type: ignore[method-assign]  # pylint:disable=protected-access

    loader.set_faceqa_sampler_strength_multiplier(1.0)

    assert loader._faceqa_sampler_multiplier == 1.0  # pylint:disable=protected-access
    assert loader._faceqa_sampler_weights_dirty is True  # pylint:disable=protected-access


@pytest.mark.parametrize("multiplier", [float("nan"), float("inf"), -1.0])
def test_curriculum_strength_update_rejects_non_finite_multiplier(multiplier: float) -> None:
    """TrainLoader should enforce the same multiplier validation as the sampler config."""
    loader = TrainLoader.__new__(TrainLoader)
    loader._faceqa_sampler = FaceQASamplerConfig(  # pylint:disable=protected-access
        mode="faceqa_curriculum",
        strength=1.0,
    )

    with pytest.raises(ValueError, match="finite"):
        loader.set_faceqa_sampler_strength_multiplier(multiplier)


def test_curriculum_loss_updates_are_queued_for_epoch_boundary() -> None:
    """Diagnostic bucket-loss updates should not recompute active worker weights mid-epoch."""
    loader = TrainLoader.__new__(TrainLoader)
    loader._faceqa_sampler = FaceQASamplerConfig(  # pylint:disable=protected-access
        mode="faceqa_curriculum",
        strength=1.0,
    )
    loader._faceqa_sampler_bucket_losses = {}  # pylint:disable=protected-access
    loader._faceqa_sampler_weights_dirty = False  # pylint:disable=protected-access
    loader._sample_weights = pytest.fail  # type: ignore[method-assign]  # pylint:disable=protected-access

    loader.update_faceqa_sampler_loss_logs({"bucket/A/yaw_pose/frontal/ema": 1.25})

    assert loader._faceqa_sampler_bucket_losses == {  # pylint:disable=protected-access
        ("A", "yaw_pose", "frontal"): 1.25
    }
    assert loader._faceqa_sampler_weights_dirty is True  # pylint:disable=protected-access


def test_curriculum_loss_updates_ignore_non_finite_bucket_scores() -> None:
    """Non-finite diagnostics should be ignored before sampler weight calculation."""
    logs = {
        "bucket/A/yaw_pose/frontal/ema": 1.0,
        "bucket/A/yaw_pose/right_profile/ema": float("inf"),
        "bucket/A/expression/smile/ema": float("nan"),
    }

    assert TrainLoader._bucket_loss_logs(logs) == {  # pylint:disable=protected-access
        ("A", "yaw_pose", "frontal"): 1.0
    }


def test_next_applies_queued_sampler_update_at_epoch_boundary() -> None:
    """Pending sampler updates should apply once before the next iterator is created."""
    calls: list[str] = []
    fake_input = type("FakeTensor", (), {"shape": (1,)})()
    fake_target = type("FakeTensor", (), {"shape": (1,)})()
    fake_batch = ([fake_input], [fake_target], "meta")

    class FakeDataset:
        """Small dataset recording epoch-bound sampler calls."""

        datasets: tuple[TrainSet, ...] = ()

        def set_sample_weights(self, weights: object) -> None:
            calls.append(f"set_sample_weights:{weights}")

        def shuffle(self) -> None:
            calls.append("shuffle")

    class FakeLoader:
        """Small loader whose iterator records when worker state would be rebuilt."""

        sampler = object()

        def __init__(self) -> None:
            self.dataset = FakeDataset()

        def __iter__(self):
            calls.append("iter")
            return iter([fake_batch])

    loader = TrainLoader.__new__(TrainLoader)
    loader._iterator = iter(())  # pylint:disable=protected-access
    loader._loader = FakeLoader()  # pylint:disable=protected-access
    loader._epoch = 0  # pylint:disable=protected-access
    loader._learn_mask = False  # pylint:disable=protected-access
    loader._faceqa_sampler_weights_dirty = True  # pylint:disable=protected-access

    def sample_weights(
        _datasets: tuple[TrainSet, ...],
        *,
        log_summary: bool = False,  # noqa: ARG001
    ) -> tuple[npt.NDArray[np.float64] | None, ...] | None:
        return None

    loader._sample_weights = sample_weights  # type: ignore[assignment]  # pylint:disable=protected-access

    assert next(loader) == fake_batch

    assert calls == ["set_sample_weights:None", "shuffle", "iter"]
    assert loader._faceqa_sampler_weights_dirty is False  # pylint:disable=protected-access
    assert loader._epoch == 1  # pylint:disable=protected-access


def test_training_sampler_config_defaults_are_random_fallback() -> None:
    """Sampler config defaults should preserve existing random behavior."""
    assert trn_cfg.Automation.training_sampler() == "random"
    assert trn_cfg.Automation.faceqa_sampler_strength() == "auto"
    assert trn_cfg.Automation.faceqa_sampler_dimensions() == "pose,expression,lighting"
    assert trn_cfg.Automation.faceqa_downweight_duplicates() is True
    assert trn_cfg.Automation.faceqa_downweight_outliers() is True
    assert trn_cfg.Automation.faceqa_min_quality() == "usable"
