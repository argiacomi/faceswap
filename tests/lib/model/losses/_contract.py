#!/usr/bin/env python3
"""Shared helper for loss output-shape / dtype / sanity contracts.

Most loss tests in this package were repeating the same body: build a pair of
random ``(B, C, H, W)`` tensors, run the loss, and assert that the result is a
finite ``float32`` scalar below some loss-specific upper bound.  Putting that
contract in one place keeps the per-loss test files small and prevents the
generic checks from drifting apart over time.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def assert_loss_contract(
    loss_factory: Callable[[], torch.nn.Module],
    *,
    max_value: float,
    shape: tuple[int, int, int, int] = (2, 3, 32, 32),
) -> None:
    """Run ``loss_factory()`` against random tensors and check the basic contract.

    The contract:

    - the output dtype is ``float32``,
    - no element is ``NaN``,
    - the mean is at or below ``max_value`` (loss-specific upper bound).

    Specialised behaviour (gradient flow, edge cases, mask handling, ...) lives
    in dedicated tests next to the loss it belongs to.
    """
    y_a = torch.from_numpy(np.random.random(shape).astype("float32")).cpu()
    y_b = torch.from_numpy(np.random.random(shape).astype("float32")).cpu()
    metric = loss_factory().cpu()
    output = metric(y_a, y_b).detach().numpy()
    assert output.dtype == np.float32
    assert not np.any(np.isnan(output))
    assert output.mean() <= max_value
