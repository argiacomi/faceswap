#!/usr/bin/env python3
"""Tests for batch-relative loss weighting."""

from __future__ import annotations

import pytest
import torch

from lib.training.loss import BatchLoss, BatchRelativeLossWeighting, LossCollator


def _loss(
    values: torch.Tensor,
    brlw: BatchRelativeLossWeighting | None = None,
    *,
    mask: torch.Tensor | None = None,
) -> BatchLoss:
    """Build a one-output BatchLoss from per-sample weighted values."""
    return BatchLoss(
        unweighted=[{"mae": values}],
        weighted=[{"mae": values}],
        mask=mask,
        brlw=BatchRelativeLossWeighting() if brlw is None else brlw,
    )


def test_disabled_matches_batch_mean_reduction_exactly() -> None:
    """Disabled BRLW must preserve the existing component mean reducer."""
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([0.1, 0.2, 0.3, 0.4])
    loss = _loss(values, mask=mask)

    assert torch.equal(loss.total, values.mean() + mask.mean())
    assert loss.brlw_stats is None


def test_enabled_weights_higher_loss_samples_more_heavily() -> None:
    """Higher-loss samples should receive higher BRLW weights within the batch."""
    values = torch.tensor([1.0, 2.0, 4.0, 8.0])
    loss = _loss(
        values,
        BatchRelativeLossWeighting(
            enabled=True,
            strength=1.0,
            min_batch_size=4,
            min_weight=0.1,
            max_weight=10.0,
        ),
    )

    assert loss.total > values.mean()
    assert loss.brlw_stats is not None
    weights = loss.brlw_stats.weights
    assert torch.all(weights[1:] > weights[:-1])
    assert torch.isclose(weights.mean(), torch.tensor(1.0))


def test_weights_are_clamped_and_normalized() -> None:
    """Outlier-driven weights should stay bounded while retaining mean 1.0."""
    values = torch.tensor([0.0, 0.0, 0.0, 100.0])
    loss = _loss(
        values,
        BatchRelativeLossWeighting(
            enabled=True,
            strength=1.0,
            min_batch_size=4,
            min_weight=0.5,
            max_weight=2.0,
        ),
    )

    _ = loss.total
    assert loss.brlw_stats is not None
    weights = loss.brlw_stats.weights
    assert weights.min().item() >= 0.5
    assert weights.max().item() <= 2.0
    assert weights.mean().item() == pytest.approx(1.0)
    assert loss.brlw_stats.effective_batch_size < values.numel()


def test_min_batch_size_uses_standard_mean() -> None:
    """BRLW is a no-op below the configured minimum batch size."""
    values = torch.tensor([1.0, 2.0, 10.0])
    loss = _loss(
        values,
        BatchRelativeLossWeighting(enabled=True, strength=1.0, min_batch_size=4),
    )

    assert torch.equal(loss.total, values.mean())
    assert loss.brlw_stats is None


@pytest.mark.parametrize(
    "values",
    [
        torch.zeros(4, dtype=torch.float16),
        torch.full((4,), torch.finfo(torch.float16).tiny, dtype=torch.float16),
    ],
)
def test_fp16_zero_and_tiny_losses_do_not_nan(values: torch.Tensor) -> None:
    """BRLW should compute weights in a dtype that keeps fp16 zero/tiny batches finite."""
    loss = _loss(
        values,
        BatchRelativeLossWeighting(
            enabled=True,
            strength=1.0,
            min_batch_size=4,
            min_weight=0.5,
            max_weight=2.0,
        ),
    )

    assert torch.isfinite(loss.total)
    assert loss.brlw_stats is not None
    assert torch.isfinite(loss.brlw_stats.weights).all()


def test_protected_samples_are_not_overweighted() -> None:
    """Metadata-protected samples may contribute but should not receive >1.0 weight."""
    values = torch.tensor([1.0, 1.0, 1.0, 100.0])
    protected = torch.tensor([False, False, False, True])
    loss = _loss(
        values,
        BatchRelativeLossWeighting(
            enabled=True,
            strength=1.0,
            min_batch_size=4,
            min_weight=0.5,
            max_weight=2.0,
            protected_samples=protected,
        ),
    )

    _ = loss.total

    assert loss.brlw_stats is not None
    assert loss.brlw_stats.weights[-1].item() <= 1.0
    assert loss.brlw_stats.weights.mean().item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strength", float("nan")),
        ("strength", float("inf")),
        ("min_weight", float("nan")),
        ("max_weight", float("inf")),
    ],
)
def test_non_finite_brlw_config_values_are_rejected(field: str, value: float) -> None:
    """Non-finite BRLW config values should fail before producing NaN weights."""
    kwargs = {
        "enabled": True,
        "strength": 1.0,
        "min_weight": 0.5,
        "max_weight": 2.0,
        field: value,
    }

    with pytest.raises(ValueError, match="finite"):
        BatchRelativeLossWeighting(**kwargs)


def test_detached_weights_do_not_backpropagate_through_weight_calculation() -> None:
    """Default detached weights should make the sample gradient equal weight / batch."""
    values = torch.tensor([1.0, 2.0, 4.0, 8.0], requires_grad=True)
    loss = _loss(
        values,
        BatchRelativeLossWeighting(
            enabled=True,
            strength=1.0,
            min_batch_size=4,
            min_weight=0.1,
            max_weight=10.0,
        ),
    )

    loss.total.backward()

    assert loss.brlw_stats is not None
    assert torch.allclose(values.grad, loss.brlw_stats.weights.detach() / values.numel())


def test_collator_auto_strength_uses_iteration_warmup() -> None:
    """Without a phase schedule, auto strength follows the deterministic warmup."""
    collator = LossCollator(
        ["mae", "none", "none", "none"],
        [1.0, 0.0, 0.0, 0.0],
        "bgr",
        False,
        1.0,
        1.0,
        8,
        brlw_enabled=True,
        brlw_strength=None,
        brlw_warmup_iterations=10,
    )

    collator.set_iteration(5)

    assert collator.batch_relative_loss_weighting.strength == pytest.approx(0.125)


def test_collator_auto_strength_uses_phase_schedule_multiplier() -> None:
    """Injected phase multipliers override the standalone warmup ramp for auto strength."""
    collator = LossCollator(
        ["mae", "none", "none", "none"],
        [1.0, 0.0, 0.0, 0.0],
        "bgr",
        False,
        1.0,
        1.0,
        8,
        brlw_enabled=True,
        brlw_strength=None,
        brlw_warmup_iterations=10,
    )

    collator.set_iteration(1)
    collator.set_schedule({"secondary_loss": 0.5})

    assert collator.batch_relative_loss_weighting.strength == pytest.approx(0.125)
