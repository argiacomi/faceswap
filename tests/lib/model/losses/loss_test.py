#!/usr/bin/env python3
"""Output-shape / dtype contract tests for ``lib.model.losses.loss``.

The generic per-loss contract is enforced via :func:`assert_loss_contract`.
Specialised behaviour belongs in dedicated tests next to each loss.
"""

import pytest

from lib.model.losses.loss import (
    FocalFrequencyLoss,
    GeneralizedLoss,
    GradientLoss,
    LaplacianPyramidLoss,
    LInfNorm,
)
from lib.utils import get_backend
from tests.lib.model.losses._contract import assert_loss_contract

_PARAMS = (
    (FocalFrequencyLoss, 1.0),
    (GeneralizedLoss, 1.0),
    (GradientLoss, 200.0),
    (LaplacianPyramidLoss, 1.0),
    (LInfNorm, 1.0),
)
_IDS = [f"{loss.__name__}[{get_backend().upper()}]" for loss, _ in _PARAMS]


@pytest.mark.parametrize(("loss_func", "max_target"), _PARAMS, ids=_IDS)
def test_loss_output(loss_func, max_target):
    assert_loss_contract(loss_func, max_value=max_target)
