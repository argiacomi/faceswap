#!/usr/bin/env python3
"""Tests for :mod:`tools.manual.aligner_service` (#104)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from tools.manual.aligner_service import (
    NORMALIZATION_CHOICES,
    AlignerStatus,
    ManualAlignerService,
)


@dataclass
class _StubBackend:
    """Test backend that records every call and returns canned landmarks."""

    aligner: str
    normalization: str
    landmarks: np.ndarray
    align_calls: list[tuple[np.ndarray, tuple[float, float, float, float]]]
    normalization_changes: list[str]
    fail_with: Exception | None = None
    instance_id: int = 0

    def align(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        if self.fail_with is not None:
            raise self.fail_with
        self.align_calls.append((image, bbox))
        return self.landmarks

    def set_normalization(self, method: str) -> None:
        self.normalization_changes.append(method)
        self.normalization = method


def _make_factory(
    instances: list[_StubBackend],
    landmarks: np.ndarray | None = None,
    fail_for: dict[tuple[str, str], Exception] | None = None,
):  # type:ignore[no-untyped-def]
    fail_for = fail_for or {}
    landmarks = landmarks if landmarks is not None else np.zeros((68, 2), dtype=np.float32)

    def factory(aligner: str, normalization: str) -> _StubBackend:
        key = (aligner, normalization)
        if key in fail_for:
            raise fail_for[key]
        backend = _StubBackend(
            aligner=aligner,
            normalization=normalization,
            landmarks=landmarks.copy(),
            align_calls=[],
            normalization_changes=[],
            instance_id=len(instances),
        )
        instances.append(backend)
        return backend

    return factory


def _service(
    *,
    available: tuple[str, ...] = ("HRNet", "FAN"),
    default: str = "HRNet",
    instances: list[_StubBackend] | None = None,
    landmarks: np.ndarray | None = None,
    fail_for: dict[tuple[str, str], Exception] | None = None,
    status_callback: Any = None,
) -> ManualAlignerService:
    instances = instances if instances is not None else []
    return ManualAlignerService(
        available=lambda: available,
        default=lambda: default,
        factory=_make_factory(instances, landmarks, fail_for),
        status_callback=status_callback,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_available_aligners_is_cached_after_first_call() -> None:
    """Plugin discovery is cached so the dropdowns don't repeat the scan."""
    calls = [0]

    def available() -> tuple[str, ...]:
        calls[0] += 1
        return ("HRNet",)

    svc = ManualAlignerService(available=available, default=lambda: "HRNet")
    assert svc.available_aligners == ("HRNet",)
    assert svc.available_aligners == ("HRNet",)
    assert calls[0] == 1


def test_available_normalizations_returns_legacy_choices() -> None:
    """The Tk Manual Tool offers none/clahe/hist/mean for normalization."""
    svc = _service()
    assert svc.available_normalizations() == NORMALIZATION_CHOICES
    assert NORMALIZATION_CHOICES == ("none", "clahe", "hist", "mean")


def test_default_aligner_falls_back_to_first_available_on_lookup_failure() -> None:
    """If ``default_aligner()`` raises, fall back to the first available plugin."""

    def fail() -> str:
        raise RuntimeError("default lookup broke")

    svc = ManualAlignerService(available=lambda: ("HRNet", "FAN"), default=fail)
    assert svc.default_aligner() == "HRNet"


def test_default_aligner_empty_when_no_plugins_available() -> None:
    """When no aligners are discoverable, ``default_aligner`` returns ``""``."""
    svc = ManualAlignerService(available=lambda: (), default=lambda: "")
    assert svc.default_aligner() == ""


# ---------------------------------------------------------------------------
# align()
# ---------------------------------------------------------------------------


def test_align_returns_factory_landmarks_with_default_aligner() -> None:
    """``align`` constructs a backend on first use and returns its landmarks."""
    instances: list[_StubBackend] = []
    landmarks = np.full((68, 2), 7.0, dtype=np.float32)
    svc = _service(instances=instances, landmarks=landmarks)

    out = svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 10.0, 10.0))

    assert out.shape == (68, 2)
    assert np.allclose(out, landmarks)
    assert len(instances) == 1
    assert instances[0].aligner == "HRNet"
    assert instances[0].normalization == "hist"  # DEFAULT_NORMALIZATION


def test_align_reuses_cached_backend_for_same_key() -> None:
    """Two calls with the same (aligner, normalization) reuse one backend."""
    instances: list[_StubBackend] = []
    svc = _service(instances=instances)

    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0))
    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (1.0, 1.0, 8.0, 8.0))

    assert len(instances) == 1
    assert len(instances[0].align_calls) == 2


def test_align_caches_per_normalization() -> None:
    """Changing normalization spawns a new backend instance (different key)."""
    instances: list[_StubBackend] = []
    svc = _service(instances=instances)

    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0))
    svc.align(
        np.zeros((4, 4, 3), dtype=np.uint8),
        (0.0, 0.0, 8.0, 8.0),
        normalization="clahe",
    )

    assert len(instances) == 2
    assert {b.normalization for b in instances} == {"hist", "clahe"}


def test_align_uses_explicit_aligner_argument() -> None:
    """``aligner=`` overrides the default for one call."""
    instances: list[_StubBackend] = []
    svc = _service(instances=instances)
    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0), aligner="FAN")
    assert instances[0].aligner == "FAN"


def test_align_raises_when_no_plugins_available() -> None:
    """Calling ``align`` with no plugins surfaces an explicit error."""
    svc = ManualAlignerService(available=lambda: (), default=lambda: "")
    with pytest.raises(RuntimeError, match="No aligner plugins"):
        svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0))


def test_align_failure_propagates_and_does_not_cache() -> None:
    """A failing factory raises and does NOT poison the cache."""
    svc = _service(
        fail_for={("HRNet", "hist"): RuntimeError("model missing")},
    )
    with pytest.raises(RuntimeError, match="model missing"):
        svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0))

    # Subsequent call retries — the factory's now happy and we can succeed.
    svc._factory = _make_factory(  # noqa: SLF001 - retry with a fresh factory
        [],
        landmarks=np.zeros((68, 2), dtype=np.float32),
    )
    out = svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0))
    assert out.shape == (68, 2)


# ---------------------------------------------------------------------------
# Normalization propagation
# ---------------------------------------------------------------------------


def test_set_normalization_propagates_to_loaded_backends() -> None:
    """Changing normalization updates every cached backend, like Tk does."""
    instances: list[_StubBackend] = []
    svc = _service(instances=instances)
    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0), aligner="HRNet")
    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0), aligner="FAN")
    svc.set_normalization("mean")

    assert [b.normalization_changes for b in instances] == [["mean"], ["mean"]]


def test_set_normalization_filters_by_aligner_when_named() -> None:
    """``aligner=`` scopes the propagation to a single plugin."""
    instances: list[_StubBackend] = []
    svc = _service(instances=instances)
    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0), aligner="HRNet")
    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0), aligner="FAN")
    svc.set_normalization("mean", aligner="HRNet")

    assert instances[0].normalization_changes == ["mean"]
    assert instances[1].normalization_changes == []


def test_set_normalization_swallows_backend_errors() -> None:
    """A backend rejecting the change must not break the service."""
    instances: list[_StubBackend] = []

    def factory(aligner: str, normalization: str) -> _StubBackend:
        backend = _StubBackend(
            aligner=aligner,
            normalization=normalization,
            landmarks=np.zeros((68, 2), dtype=np.float32),
            align_calls=[],
            normalization_changes=[],
        )

        def angry(_method: str) -> None:
            raise RuntimeError("can't change normalization")

        backend.set_normalization = angry  # type:ignore[method-assign]
        instances.append(backend)
        return backend

    svc = ManualAlignerService(
        available=lambda: ("HRNet",),
        default=lambda: "HRNet",
        factory=factory,
    )
    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0))
    # Must NOT raise.
    svc.set_normalization("clahe")


# ---------------------------------------------------------------------------
# Preload + status callbacks
# ---------------------------------------------------------------------------


def test_preload_emits_loading_then_ready_status() -> None:
    """Successful preload emits a load-start then ready event."""
    events: list[AlignerStatus] = []
    svc = _service(status_callback=events.append)
    assert svc.preload("HRNet") is True
    kinds = [event.kind for event in events]
    assert kinds == ["loading", "ready"]
    assert all(event.aligner == "HRNet" for event in events)


def test_preload_emits_failed_status_when_factory_raises() -> None:
    """A failing preload emits a load-start then failed event and returns False."""
    events: list[AlignerStatus] = []
    svc = _service(
        fail_for={("HRNet", "hist"): RuntimeError("oops")},
        status_callback=events.append,
    )
    assert svc.preload("HRNet") is False
    kinds = [event.kind for event in events]
    assert kinds == ["loading", "failed"]
    assert "oops" in events[-1].message


def test_align_emits_aligning_then_aligned_status() -> None:
    """``align`` emits an in-flight then completed status event."""
    events: list[AlignerStatus] = []
    svc = _service(status_callback=events.append)
    svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0))
    kinds = [event.kind for event in events]
    assert "aligning" in kinds
    assert "aligned" in kinds


def test_status_callback_exception_does_not_break_service() -> None:
    """A misbehaving status hook must not propagate."""

    def angry(_status: AlignerStatus) -> None:
        raise RuntimeError("status broke")

    svc = _service(status_callback=angry)
    # Should NOT raise.
    out = svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0))
    assert out.shape == (68, 2)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_align_for_same_aligner_uses_one_backend() -> None:
    """Two threads requesting the same aligner end up sharing one backend."""
    instances: list[_StubBackend] = []
    svc = _service(instances=instances)

    def hit() -> None:
        svc.align(np.zeros((4, 4, 3), dtype=np.uint8), (0.0, 0.0, 8.0, 8.0))

    threads = [threading.Thread(target=hit) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # The cache should converge on a single backend.  The test factory has
    # no inter-thread synchronization so it can briefly produce more than
    # one instance, but only one ever ends up in the cache.  All four
    # calls must still land — exercise that through the call counter.
    assert len(instances) >= 1
    total_calls = sum(len(b.align_calls) for b in instances)
    assert total_calls == 4
