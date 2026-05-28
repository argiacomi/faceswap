#!/usr/bin/env python3
"""Output-shape / dtype contract tests for ``lib.model.losses.feature_loss``."""

import pytest

# pylint:disable=import-error
from lib.model.losses.feature_loss import LPIPSLoss
from lib.utils import get_backend
from tests.lib.model.losses._contract import assert_loss_contract

_NETS = ("alex", "squeeze", "vgg16")
_IDS = [f"LPIPS_{net}[{get_backend().upper()}]" for net in _NETS]


@pytest.mark.parametrize("net", _NETS, ids=_IDS)
def test_loss_output(net):
    # LPIPS output is reduced 10x relative to a typical perceptual loss.
    assert_loss_contract(lambda: LPIPSLoss(net), max_value=0.1)
