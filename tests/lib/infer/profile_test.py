#!/usr/bin/env python3
"""Tests for Profiler torch-module discovery, including composite-adapter plugins."""

from __future__ import annotations

import typing as T

import numpy as np
import pytest
import torch

from lib.infer.profile import Profiler


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
    profiler._channels_last = []  # type:ignore[attr-defined]
    return profiler


def test_check_for_torch_uses_profile_hook_when_provided() -> None:
    """Composite plugins should be classified Torch-backed via the explicit hook."""
    profiler = _make_profiler()
    wrapped = _WrappedTorchPlugin()
    plugin = _FakeEnsemblePlugin(
        adapters=[_StaticAdapter(), _FaceswapAdapter(wrapped)],
        hook_modules=[wrapped.model],
    )

    assert profiler._check_for_torch(plugin) is True
    assert plugin.hook_calls == 1
    # warmup_plugin probes channel order by calling ``process``; the ensemble must execute.
    assert plugin.process_calls >= 1
    assert profiler._channels_last == [False]  # type:ignore[attr-defined]


def test_check_for_torch_skips_when_hook_returns_empty() -> None:
    """Ensembles that wrap only static adapters should be skipped without warmup."""
    profiler = _make_profiler()
    plugin = _FakeEnsemblePlugin(adapters=[_StaticAdapter()], hook_modules=[])

    assert profiler._check_for_torch(plugin) is False
    assert plugin.hook_calls == 1
    assert plugin.process_calls == 0
    assert profiler._channels_last == []  # type:ignore[attr-defined]


def test_check_for_torch_falls_back_to_get_torch_modules() -> None:
    """Plugins without the hook keep the legacy ``get_torch_modules`` walk."""
    profiler = _make_profiler()
    plugin = _FakeTorchPlugin()

    assert profiler._check_for_torch(plugin) is True
    assert plugin.process_calls >= 1
    assert profiler._channels_last == [False]  # type:ignore[attr-defined]


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

    modules = Ensemble.profile_torch_modules(adapters)

    assert wrapped_a.model in modules
    assert wrapped_b.model in modules
    assert all(isinstance(m, torch.nn.Module) for m in modules)


def test_ensemble_profile_torch_modules_returns_empty_for_static_adapters() -> None:
    """The real Ensemble hook returns nothing when every adapter is static."""
    from plugins.extract.align.ensemble import Ensemble

    assert Ensemble.profile_torch_modules([_StaticAdapter(), _StaticAdapter()]) == []


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

    assert profiler._check_for_torch(plugin) is True
    assert process_inputs, "warmup should call Ensemble.process at least once"
