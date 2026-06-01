#!/usr/bin/env python3
"""Tests for the training batch-size finder."""

from __future__ import annotations

import typing as T
from pathlib import Path

from lib.training.batch_size_finder import BatchSizeProbe, TrainingBatchSizeFinder
from lib.training.train import Trainer


class _State:
    """Minimal model state for trainer integration tests."""

    def __init__(self) -> None:
        self.recommendation: dict[str, T.Any] | None = None
        self.saved = False
        self.batch_size = 0

    def add_training_batch_size_finder(self, recommendation: dict[str, T.Any]) -> None:
        """Store the recommendation."""
        self.recommendation = recommendation

    def add_session_batchsize(self, batch_size: int) -> None:
        """Store the applied batch size."""
        self.batch_size = batch_size

    def save(self) -> None:
        """Record that state was saved."""
        self.saved = True


class _Model:
    """Minimal model object."""

    def __init__(self) -> None:
        self.state = _State()


class _Config:
    """Minimal train config."""

    def __init__(self, batch_size: int, folders: list[str]) -> None:
        self.batch_size = batch_size
        self.folders = folders


class _Plugin:
    """Minimal trainer plugin."""

    def __init__(self, batch_size: int, folders: list[str]) -> None:
        self.batch_size = batch_size
        self.config = _Config(batch_size, folders)


class _Trainer:
    """Fake trainer that exposes the batch-size finder probe surface."""

    def __init__(self, batch_size: int, safe_limit: int, folders: list[str] | None = None) -> None:
        self._plugin = _Plugin(batch_size, folders or [])
        self._model = _Model()
        self.safe_limit = safe_limit
        self.probed: list[int] = []

    @property
    def batch_size(self) -> int:
        """Current batch size."""
        return self._plugin.config.batch_size

    def _set_training_batch_size(self, batch_size: int) -> None:
        """Set the current batch size."""
        self._plugin.batch_size = batch_size
        self._plugin.config.batch_size = batch_size

    def probe_training_batch_size(self, batch_size: int) -> BatchSizeProbe:
        """Return success for batch sizes up to ``safe_limit``."""
        self.probed.append(batch_size)
        success = batch_size <= self.safe_limit
        return BatchSizeProbe(
            batch_size=batch_size,
            success=success,
            vram_reserved=0 if not success else batch_size * 1024 * 1024,
            error=None if success else "out of memory",
        )


def _folders(tmp_path: Path, count: int = 32) -> list[str]:
    """Create two fake training folders containing PNG filenames."""
    retval = []
    for side in ("a", "b"):
        folder = tmp_path / side
        folder.mkdir()
        for idx in range(count):
            (folder / f"{idx:04d}.png").touch()
        retval.append(str(folder))
    return retval


def test_finder_probes_up_then_bisects_after_oom() -> None:
    """The finder should grow until OOM then binary search the failure gap."""
    trainer = _Trainer(batch_size=2, safe_limit=6)
    finder = TrainingBatchSizeFinder(T.cast("Trainer", trainer), max_batch_size=10)

    recommendation = finder.find()

    assert recommendation.max_safe_batch_size == 6
    assert trainer.probed == [2, 4, 8, 6, 7]
    assert [probe.batch_size for probe in recommendation.probes] == [2, 4, 6, 7, 8]


def test_finder_steps_down_when_configured_batch_ooms() -> None:
    """If the configured batch size fails, the finder should search downward first."""
    trainer = _Trainer(batch_size=16, safe_limit=3)
    finder = TrainingBatchSizeFinder(T.cast("Trainer", trainer), max_batch_size=20)

    recommendation = finder.find()

    assert recommendation.max_safe_batch_size == 3
    assert trainer.probed == [16, 8, 4, 2, 3]


def test_recommendation_uses_conservative_batch_and_accumulation() -> None:
    """Small safe batches should recommend accumulation to reach the target effective batch."""
    trainer = _Trainer(batch_size=2, safe_limit=3)
    finder = TrainingBatchSizeFinder(
        T.cast("Trainer", trainer),
        max_batch_size=8,
        target_effective_batch_size=4,
    )

    recommendation = finder.recommend(max_safe_batch_size=3)

    assert recommendation.suggested_batch_size == 2
    assert recommendation.gradient_accumulation_recommended is True
    assert recommendation.gradient_accumulation_steps == 2
    assert recommendation.effective_batch_size == 4


def test_trainer_find_batch_size_does_not_auto_apply(tmp_path: Path) -> None:
    """Finder mode should restore the configured batch size unless auto apply is requested."""
    trainer = _Trainer(batch_size=2, safe_limit=6, folders=_folders(tmp_path))

    Trainer.find_batch_size(
        T.cast("Trainer", trainer),
        max_batch_size=10,
        target_effective_batch_size=4,
        auto_apply=False,
    )

    assert trainer.batch_size == 2
    assert trainer._model.state.saved is True
    assert trainer._model.state.recommendation is not None
    assert trainer._model.state.recommendation["max_safe_batch_size"] == 6
    assert trainer._model.state.batch_size == 0


def test_trainer_find_batch_size_auto_apply_uses_recommendation(tmp_path: Path) -> None:
    """Auto apply should update the active training batch size to the conservative value."""
    trainer = _Trainer(batch_size=2, safe_limit=6, folders=_folders(tmp_path))

    Trainer.find_batch_size(
        T.cast("Trainer", trainer),
        max_batch_size=10,
        target_effective_batch_size=4,
        auto_apply=True,
    )

    assert trainer.batch_size == 5
    assert trainer._model.state.batch_size == 5
