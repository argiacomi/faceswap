#!/usr/bin/env python3
"""Tests for :mod:`lib.training.optimizer`, focused on the schedule-free optimizers (#185).

Covers construction through the optimizer factory, the ``train()`` / ``eval()`` mode toggling
that schedule-free optimizers require, a simple update step that reduces a loss, state
serialization / round-trip, default-parity for standard optimizers, and the Learning Rate Finder
guard.
"""

from __future__ import annotations

import typing as T

import keras
import torch
from schedulefree import AdamWScheduleFree, ScheduleFreeWrapper

from lib.training.optimizer import (
    SCHEDULE_FREE_ADAMW,
    SCHEDULE_FREE_LION,
    Optimizer,
)


class _FakeIO:
    """Minimal model.io surface used by the optimizer state loader."""

    model_exists = False


class _FakeModel:
    """Wraps a real keras model with the attributes the optimizer wrapper reads."""

    def __init__(self, keras_model: keras.Model) -> None:
        self.model = keras_model
        self.io = _FakeIO()


def _build_model() -> keras.Model:
    """Build and materialize a tiny keras model so its torch parameters exist."""
    model = keras.Sequential(
        [keras.layers.Input((4,)), keras.layers.Dense(4), keras.layers.Dense(2)]
    )
    # Forward once to ensure variables (and their backing torch parameters) are created.
    model(keras.ops.zeros((1, 4)))
    return model


def _make_config(name: str, **overrides: T.Any) -> type:
    """Return a config-like class exposing the methods the optimizer wrapper queries."""
    values: dict[str, T.Any] = {
        "optimizer": name,
        "learning_rate": 0.05,
        "weight_decay": 0.01,
        "gradient_accumulation": 1,
        "gradient_clipping": "none",
        "clipping_value": 1.0,
        "autoclip_history": 10000,
        "ada_beta_1": 0.9,
        "ada_beta_2": 0.999,
        "epsilon_exponent": -8,
        "ada_amsgrad": False,
    }
    values.update(overrides)

    return type(
        "_OptConfig",
        (),
        {key: staticmethod((lambda v: lambda: v)(value)) for key, value in values.items()},
    )


def _params(optimizer: Optimizer) -> list[torch.nn.Parameter]:
    return [p for group in optimizer._optimizer.param_groups for p in group["params"]]


def _reduce_param_loss(optimizer: Optimizer, steps: int = 25) -> tuple[float, float]:
    """Run ``steps`` updates minimizing the sum of squared parameters; return (first, last)."""
    params = _params(optimizer)
    first = last = 0.0
    for step in range(steps):
        optimizer.train()
        optimizer.zero_grad()
        for param in params:
            if param.grad is not None:
                param.grad = None
        loss = torch.stack([param.pow(2).sum() for param in params]).sum()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if step == 0:
            first = value
        last = value
    return first, last


def test_factory_builds_schedule_free_adamw() -> None:
    optimizer = Optimizer(_FakeModel(_build_model()), _make_config(SCHEDULE_FREE_ADAMW))
    assert optimizer.is_schedule_free is True
    assert isinstance(optimizer._optimizer, AdamWScheduleFree)


def test_factory_builds_schedule_free_lion() -> None:
    optimizer = Optimizer(_FakeModel(_build_model()), _make_config(SCHEDULE_FREE_LION))
    assert optimizer.is_schedule_free is True
    assert isinstance(optimizer._optimizer, ScheduleFreeWrapper)


def test_standard_optimizer_is_not_schedule_free_and_toggles_are_noops() -> None:
    optimizer = Optimizer(_FakeModel(_build_model()), _make_config("adam"))
    assert optimizer.is_schedule_free is False
    before = [param.detach().clone() for param in _params(optimizer)]
    # train()/eval() must be safe no-ops for standard optimizers and not mutate parameters.
    optimizer.eval()
    optimizer.train()
    after = _params(optimizer)
    assert all(torch.allclose(b, a) for b, a in zip(before, after, strict=True))


def test_schedule_free_train_eval_swaps_parameters() -> None:
    optimizer = Optimizer(_FakeModel(_build_model()), _make_config(SCHEDULE_FREE_ADAMW))
    _reduce_param_loss(optimizer, steps=5)
    optimizer.train()
    train_weights = [param.detach().clone() for param in _params(optimizer)]
    optimizer.eval()
    eval_weights = [param.detach().clone() for param in _params(optimizer)]
    optimizer.train()
    restored = _params(optimizer)
    # Eval (averaged) weights differ from train weights, and train mode round-trips.
    assert any(not torch.allclose(t, e) for t, e in zip(train_weights, eval_weights, strict=True))
    assert all(torch.allclose(t, r) for t, r in zip(train_weights, restored, strict=True))


def test_schedule_free_adamw_reduces_loss() -> None:
    optimizer = Optimizer(_FakeModel(_build_model()), _make_config(SCHEDULE_FREE_ADAMW))
    first, last = _reduce_param_loss(optimizer)
    assert last < first


def test_schedule_free_lion_reduces_loss() -> None:
    optimizer = Optimizer(_FakeModel(_build_model()), _make_config(SCHEDULE_FREE_LION))
    first, last = _reduce_param_loss(optimizer)
    assert last < first


def test_schedule_free_state_dict_round_trips() -> None:
    optimizer = Optimizer(_FakeModel(_build_model()), _make_config(SCHEDULE_FREE_ADAMW))
    _reduce_param_loss(optimizer, steps=5)
    state = optimizer.state_dict()
    assert state["version"] == 1.0

    other = Optimizer(_FakeModel(_build_model()), _make_config(SCHEDULE_FREE_ADAMW))
    other.load_state_dict(state)
    # The momentum buffers (z) restore onto the fresh optimizer.
    assert other._optimizer.state_dict()["state"].keys() == (
        optimizer._optimizer.state_dict()["state"].keys()
    )


def test_learning_rate_finder_is_skipped_for_schedule_free() -> None:
    optimizer = Optimizer(_FakeModel(_build_model()), _make_config(SCHEDULE_FREE_ADAMW))
    result = optimizer.find_learning_rate(
        trainer=T.cast("T.Any", None),
        steps=10,
        start_lr=1e-6,
        end_lr=1.0,
        strength="default",
        mode="set",
    )
    assert result is False
