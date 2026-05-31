#!/usr/bin/env python3
"""Targeted regression tests for the #194 lib/infer cleanup commit.

Covers:

* **P1** — ``_format_images`` does NOT mutate the caller's uint8 batch
  when the plugin's scale requires in-place rescaling.
* **P2 high** — ``torch_normalize`` reuses the cached mean/std tensors
  across calls with the same ``(mean, std, device)`` triple.
* **P2 high** — ``_affine_grid_from_matrices`` reuses the cached
  scaffolding (identity + meshgrid) across calls with the same
  ``(device, output_size)`` pair.
* **P2 medium** — ``ExtractIterator._fifo`` is a ``deque`` so the
  drain path doesn't pay ``list.pop(0)``'s O(N) shift.

All tests are CPU-only and exercise the helpers directly without
loading any extract plugin.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lib.infer import handler as handler_mod  # noqa: E402
from lib.infer import iterator as iterator_mod  # noqa: E402
from lib.infer import torch_preprocess as tp  # noqa: E402

# ---------------------------------------------------------------------------
# P1 — _format_images aliasing guard
# ---------------------------------------------------------------------------


class _PluginStub:
    def __init__(self, dtype, scale):
        self.dtype = dtype
        self.scale = scale


class _ConcreteHandler(handler_mod.ExtractHandler):
    """Minimal concrete subclass to escape the abstract-method gate."""

    processors = ()

    def pre_process(self, batch):  # noqa: D401, ANN001
        return None

    def process(self, batch):  # noqa: D401, ANN001
        return None

    def post_process(self, batch):  # noqa: D401, ANN001
        return None


def _make_handler(*, dtype, scale):
    """Build a minimally-initialised ExtractHandler for ``_format_images``."""
    h = _ConcreteHandler.__new__(_ConcreteHandler)
    h.plugin = _PluginStub(dtype=dtype, scale=scale)
    return h


def test_format_images_rejects_uint8_with_non_passthrough_scale() -> None:
    """The legacy shape silently mutated the caller's uint8 batch
    when the plugin asked for ``scale != (0, 255)``. The #194 P1 fix
    surfaces that as an explicit ``ValueError`` at the boundary so
    the configuration mistake fails fast instead of corrupting the
    upstream batch."""
    handler = _make_handler(dtype=np.uint8, scale=(-1, 1))
    images = np.full((2, 4, 4, 3), 128, dtype=np.uint8)
    snapshot = images.copy()

    with pytest.raises(ValueError, match=r"uint8 plugins must use scale=\(0, 255\)"):
        handler._format_images(images)

    # The caller's batch is untouched — the error fires BEFORE any
    # in-place work.
    np.testing.assert_array_equal(images, snapshot)


def test_format_images_aliases_uint8_when_scale_is_passthrough() -> None:
    """When the plugin's scale IS ``(0, 255)``, the legacy fast path
    is preserved: we return the caller's array directly (no copy)."""
    handler = _make_handler(dtype=np.uint8, scale=(0, 255))
    images = np.full((2, 4, 4, 3), 128, dtype=np.uint8)
    assert handler._format_images(images) is images


def test_format_images_returns_float_array_for_non_uint8_plugin() -> None:
    """The pre-existing float dtype path is unchanged."""
    handler = _make_handler(dtype=np.float32, scale=(-1, 1))
    images = np.full((1, 2, 2, 3), 255, dtype=np.uint8)

    retval = handler._format_images(images)

    assert retval.dtype == np.float32
    # 255 maps to ``-1 + 255 * (1 - -1) / 255`` = ``-1 + 2`` = ``1``.
    np.testing.assert_allclose(retval, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# P2 high — torch_normalize tensor cache
# ---------------------------------------------------------------------------


def test_torch_normalize_reuses_cached_stats() -> None:
    """Two calls with the same (mean, std, device) produce the SAME
    cached tensors — verified via the ``_cached_normalization_stats``
    cache info BEFORE and AFTER the second call."""
    tp._cached_normalization_stats.cache_clear()
    image = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    tp.torch_normalize(image, mean, std)
    after_first = tp._cached_normalization_stats.cache_info()
    tp.torch_normalize(image, mean, std)
    after_second = tp._cached_normalization_stats.cache_info()

    # First call was a miss + insertion; second call must be a hit.
    assert after_second.hits == after_first.hits + 1
    assert after_second.misses == after_first.misses


def test_torch_normalize_correct_result() -> None:
    """Cached path still produces the right per-channel normalisation."""
    image = torch.full((1, 3, 2, 2), 0.5, dtype=torch.float32)
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
    out = tp.torch_normalize(image, mean, std)
    torch.testing.assert_close(out, image)


# ---------------------------------------------------------------------------
# P2 high — affine_grid scaffolding cache
# ---------------------------------------------------------------------------


def test_affine_grid_reuses_cached_scaffold() -> None:
    """Two warps with the same output size + device hit the
    ``_cached_affine_scaffold`` cache on the second call."""
    tp._cached_affine_scaffold.cache_clear()
    matrix = torch.eye(2, 3, dtype=torch.float32).unsqueeze(0)
    tp._affine_grid_from_matrices(
        matrix, input_size=(8, 8), output_size=(4, 4), align_corners=False
    )
    after_first = tp._cached_affine_scaffold.cache_info()
    tp._affine_grid_from_matrices(
        matrix, input_size=(8, 8), output_size=(4, 4), align_corners=False
    )
    after_second = tp._cached_affine_scaffold.cache_info()

    assert after_second.hits == after_first.hits + 1
    assert after_second.misses == after_first.misses


def test_affine_grid_handles_distinct_output_sizes() -> None:
    """A different output size must take a fresh cache slot (miss)
    rather than reusing the previous scaffold."""
    tp._cached_affine_scaffold.cache_clear()
    matrix = torch.eye(2, 3, dtype=torch.float32).unsqueeze(0)
    tp._affine_grid_from_matrices(
        matrix, input_size=(8, 8), output_size=(4, 4), align_corners=False
    )
    info_a = tp._cached_affine_scaffold.cache_info()
    tp._affine_grid_from_matrices(
        matrix, input_size=(8, 8), output_size=(8, 8), align_corners=False
    )
    info_b = tp._cached_affine_scaffold.cache_info()

    assert info_b.misses == info_a.misses + 1


# ---------------------------------------------------------------------------
# P2 medium — deque-backed FIFO
# ---------------------------------------------------------------------------


class _ConcreteIterator(iterator_mod.ExtractIterator):
    """Minimal concrete subclass to satisfy the ABC."""

    def __next__(self):  # noqa: D401
        raise StopIteration


def test_iterator_fifo_is_deque_for_O1_pop() -> None:
    """``ExtractIterator._fifo`` must be a ``collections.deque`` so
    ``popleft()`` runs in O(1). The old ``list.pop(0)`` was O(N)."""
    queue = MagicMock()
    error_state = MagicMock()
    instance = _ConcreteIterator(
        queue=queue,
        name="test",
        plugin_type="detect",
        batch_size=4,
        error_state=error_state,
    )
    assert isinstance(instance._fifo, deque)


class _StubBatch:
    """Minimal stand-in exposing the surface ``_rebatch_data`` touches:
    ``__len__``, ``filenames`` and slice support for ``batch[i:end]``."""

    def __init__(self, size: int) -> None:
        self._size = size
        self.filenames = [f"f{idx}" for idx in range(size)]

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, sl: slice) -> _StubBatch:
        return _StubBatch(len(range(*sl.indices(self._size))))


def test_rebatch_data_trace_does_not_slice_deque() -> None:
    """``_rebatch_data`` logs the rebatched tail via ``self._fifo[-count:]``.
    Since #194 made ``_fifo`` a ``deque`` (no slice support), that line
    raised ``TypeError: sequence index must be integer, not 'slice'`` on
    every extract. The join is a *call argument*, so it is evaluated
    eagerly regardless of log level. This guards the deque-safe tail
    access and pins the regression."""
    instance = iterator_mod.InboundIterator(
        queue=MagicMock(),
        name="test",
        plugin_type="align",
        batch_size=4,
        error_state=MagicMock(),
    )
    # Pre-seed the FIFO with more items than ``count`` to exercise the tail.
    instance._fifo.extend(_StubBatch(2) for _ in range(3))
    # Drive the loop without real batching machinery: report 6 boxes and
    # make the fifo/no-box appends no-ops so only the trace line is under test.
    instance._handle_non_split_batch = lambda batch: (6, instance._batch_size)  # type: ignore[method-assign]
    instance._batch_to_fifo = lambda in_batch: None  # type: ignore[method-assign]
    instance._append_no_boxes = lambda batch: None  # type: ignore[method-assign]

    # Force the trace record to actually format, belt-and-braces against a
    # future lazy-logging refactor of the call.
    iterator_mod.logger.setLevel(5)  # TRACE

    instance._rebatch_data(_StubBatch(6))  # must not raise
