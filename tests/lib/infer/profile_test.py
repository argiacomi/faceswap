#!/usr/bin/env python3
"""Tests for Profiler torch-module discovery, including composite-adapter plugins."""

from __future__ import annotations

import typing as T

import numpy as np
import pytest
import torch

from lib.infer import profile as profile_module
from lib.infer.profile import DataTracker, Profiler

if T.TYPE_CHECKING:
    from plugins.extract.base import ExtractPlugin


class _TorchAligner(torch.nn.Module):
    """Minimal torch module used as a stand-in for a real plugin model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)


class _StaticAdapter:
    """Adapter wrapper with no torch state, mirroring StaticLandmarkAdapter."""

    def __init__(self) -> None:
        self.plugin: object | None = None  # static adapters have no wrapped plugin


class _WrappedTorchPlugin:
    """Plugin object exposed via ``FaceswapAlignerAdapter.plugin`` in the real pipeline."""

    def __init__(self) -> None:
        self.model = _TorchAligner()


class _FaceswapAdapter:
    """Adapter that wraps a torch-backed plugin, mirroring FaceswapAlignerAdapter."""

    def __init__(self, plugin: _WrappedTorchPlugin) -> None:
        self.plugin = plugin


class _FakeTorchPlugin:
    """Plugin whose ``load_model`` returns a torch module directly (normal aligner shape)."""

    name = "FakeTorchPlugin"
    input_size = 8
    batch_size = 1
    is_rgb = False
    dtype = np.float32
    scale = (0, 1)
    process_calls = 0

    def __init__(self) -> None:
        self._model = _TorchAligner()

    def load_model(self) -> torch.nn.Module:
        return self._model

    def process(self, batch: np.ndarray) -> np.ndarray:
        self.process_calls += 1
        return batch


class _FakeEnsemblePlugin:
    """Plugin whose ``load_model`` returns adapters and that exposes the profile hook."""

    name = "FakeEnsemble"
    input_size = 8
    batch_size = 1
    is_rgb = False
    dtype = np.float32
    scale = (0, 1)

    def __init__(self, adapters: list[T.Any], hook_modules: list[T.Any]) -> None:
        self._adapters = adapters
        self._hook_modules = hook_modules
        self.process_calls = 0
        self.hook_calls = 0

    def load_model(self) -> list[T.Any]:
        return self._adapters

    def profile_torch_modules(self, loaded: list[T.Any]) -> list[T.Any]:
        self.hook_calls += 1
        # Confirm the profiler hands us the same list ``load_model`` returned.
        assert loaded is self._adapters
        return self._hook_modules

    def process(self, batch: np.ndarray) -> np.ndarray:
        self.process_calls += 1
        return batch


def _make_profiler() -> Profiler:
    """Build a Profiler without touching ``__init__`` so the discovery helpers can be tested."""
    profiler = Profiler.__new__(Profiler)
    profiler._channels_last = []
    return profiler


def _make_data_tracker(monkeypatch: pytest.MonkeyPatch, size: int = 2) -> DataTracker:
    """Build a DataTracker with deterministic fake VRAM capacity."""
    monkeypatch.setattr(profile_module, "accelerator_total_memory", lambda: 1000)
    return DataTracker(size=size, max_vram=0.9, face_scaling=1, has_detector=False)


def _record_profile_result(
    tracker: DataTracker,
    iterations: tuple[int | None, ...],
    reserved_vram: int = 100,
) -> None:
    """Record one simulated profiling result.

    ``None`` leaves the iteration count at ``-1`` to mirror an OOM worker.
    """
    for idx, count in enumerate(iterations):
        if count is not None:
            tracker.update_iterations(count, idx)
    tracker.vram.append((reserved_vram, reserved_vram))


def _schedule_next(tracker: DataTracker) -> None:
    """Schedule the next search row and prepare its iteration storage."""
    tracker.add_next_batch_sizes()
    if not tracker.combos_exhausted:
        tracker.add_iterations_row()


def test_check_for_torch_uses_profile_hook_when_provided() -> None:
    """Composite plugins should be classified Torch-backed via the explicit hook."""
    profiler = _make_profiler()
    wrapped = _WrappedTorchPlugin()
    plugin = _FakeEnsemblePlugin(
        adapters=[_StaticAdapter(), _FaceswapAdapter(wrapped)],
        hook_modules=[wrapped.model],
    )

    assert profiler._check_for_torch(T.cast("ExtractPlugin", plugin)) is True
    assert plugin.hook_calls == 1
    # warmup_plugin probes channel order by calling ``process``; the ensemble must execute.
    assert plugin.process_calls >= 1
    assert profiler._channels_last == [False]


def test_check_for_torch_skips_when_hook_returns_empty() -> None:
    """Ensembles that wrap only static adapters should be skipped without warmup."""
    profiler = _make_profiler()
    plugin = _FakeEnsemblePlugin(adapters=[_StaticAdapter()], hook_modules=[])

    assert profiler._check_for_torch(T.cast("ExtractPlugin", plugin)) is False
    assert plugin.hook_calls == 1
    assert plugin.process_calls == 0
    assert profiler._channels_last == []


def test_check_for_torch_falls_back_to_get_torch_modules() -> None:
    """Plugins without the hook keep the legacy ``get_torch_modules`` walk."""
    profiler = _make_profiler()
    plugin = _FakeTorchPlugin()

    assert profiler._check_for_torch(T.cast("ExtractPlugin", plugin)) is True
    assert plugin.process_calls >= 1
    assert profiler._channels_last == [False]


def test_ensemble_profile_torch_modules_collects_from_wrapped_plugins() -> None:
    """The real Ensemble hook returns torch modules from each adapter's wrapped plugin."""
    from plugins.extract.align.ensemble import Ensemble

    wrapped_a = _WrappedTorchPlugin()
    wrapped_b = _WrappedTorchPlugin()
    adapters = [
        _StaticAdapter(),
        _FaceswapAdapter(wrapped_a),
        _FaceswapAdapter(wrapped_b),
    ]

    modules = Ensemble.profile_torch_modules(T.cast(T.Any, adapters))

    assert wrapped_a.model in modules
    assert wrapped_b.model in modules
    assert all(isinstance(m, torch.nn.Module) for m in modules)


def test_ensemble_profile_torch_modules_returns_empty_for_static_adapters() -> None:
    """The real Ensemble hook returns nothing when every adapter is static."""
    from plugins.extract.align.ensemble import Ensemble

    adapters = [_StaticAdapter(), _StaticAdapter()]

    assert Ensemble.profile_torch_modules(T.cast(T.Any, adapters)) == []


def test_check_for_torch_runs_warmup_via_ensemble_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Channel-order warmup must call the ensemble's ``process``, not bypass adapter work."""
    profiler = _make_profiler()
    wrapped = _WrappedTorchPlugin()
    plugin = _FakeEnsemblePlugin(
        adapters=[_FaceswapAdapter(wrapped)],
        hook_modules=[wrapped.model],
    )

    process_inputs: list[np.ndarray] = []
    original_process = plugin.process

    def _spy(batch: np.ndarray) -> np.ndarray:
        process_inputs.append(batch)
        return original_process(batch)

    monkeypatch.setattr(plugin, "process", _spy)

    assert profiler._check_for_torch(T.cast("ExtractPlugin", plugin)) is True
    assert process_inputs, "warmup should call Ensemble.process at least once"


def test_data_tracker_expands_slowest_plugin_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The search should spend the next benchmark on the throughput bottleneck."""
    tracker = _make_data_tracker(monkeypatch)

    _record_profile_result(tracker, (20, 10))
    tracker.add_next_batch_sizes()

    assert tracker._all_batch_sizes[-1].tolist() == [1, 2]


def test_data_tracker_refines_failed_plugin_to_unit_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a failure, search should bisect down to the tightest untested batch."""
    tracker = _make_data_tracker(monkeypatch, size=1)

    _record_profile_result(tracker, (10,))
    _schedule_next(tracker)
    assert tracker._all_batch_sizes[-1].tolist() == [2]

    _record_profile_result(tracker, (10,))
    _schedule_next(tracker)
    assert tracker._all_batch_sizes[-1].tolist() == [4]

    _record_profile_result(tracker, (10,))
    _schedule_next(tracker)
    assert tracker._all_batch_sizes[-1].tolist() == [8]

    _record_profile_result(tracker, (None,))
    _schedule_next(tracker)
    assert (8,) in tracker._failed_batch_sizes
    assert tracker._all_batch_sizes[-1].tolist() == [6]

    _record_profile_result(tracker, (10,))
    _schedule_next(tracker)
    assert tracker._all_batch_sizes[-1].tolist() == [7]

    _record_profile_result(tracker, (None,))
    tracker.add_next_batch_sizes()

    assert tracker.combos_exhausted is True
    assert tracker.batch_sizes[-1].tolist() == [6]


def test_data_tracker_tries_next_plugin_when_slowest_is_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exhausted bottleneck should not prevent other plugins from being explored."""
    tracker = _make_data_tracker(monkeypatch)
    tracker._failed_batch_sizes.add((2, 1))

    _record_profile_result(tracker, (1, 10))
    tracker.add_next_batch_sizes()

    assert tracker._all_batch_sizes[-1].tolist() == [1, 2]


def test_data_tracker_failed_combo_does_not_globally_cap_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed high-memory path should not globally cap another plugin's batch size."""
    tracker = _make_data_tracker(monkeypatch)

    _record_profile_result(tracker, (10, 4))
    _schedule_next(tracker)
    assert tracker._all_batch_sizes[-1].tolist() == [1, 2]

    _record_profile_result(tracker, (10, 4))
    _schedule_next(tracker)
    assert tracker._all_batch_sizes[-1].tolist() == [1, 4]

    _record_profile_result(tracker, (10, 4))
    _schedule_next(tracker)
    assert tracker._all_batch_sizes[-1].tolist() == [2, 4]

    _record_profile_result(tracker, (10, 4))
    _schedule_next(tracker)
    assert tracker._all_batch_sizes[-1].tolist() == [2, 8]

    _record_profile_result(tracker, (None, None))
    _schedule_next(tracker)

    assert (2, 8) in tracker._failed_batch_sizes
    assert tracker._all_batch_sizes[-1].tolist() == [2, 6]

    older_success = np.array([1, 4], dtype=np.int64)
    candidate = tracker._next_batch_candidate(older_success, 1)
    assert candidate is not None
    assert candidate.tolist() == [1, 8]
