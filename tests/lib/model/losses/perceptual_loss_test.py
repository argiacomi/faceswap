#!/usr/bin/env python3
"""Output-shape / dtype contract tests for the perceptual loss family."""

import pytest

from lib.model.losses.flip import LDRFLIPLoss

# pylint:disable=import-error,duplicate-code
from lib.model.losses.perceptual_loss import GMSDLoss, MSSIMLoss, SSIMLoss
from lib.utils import get_backend
from tests.lib.model.losses._contract import assert_loss_contract

_PARAMS = [SSIMLoss, GMSDLoss, LDRFLIPLoss, MSSIMLoss]
_IDS = [f"{loss.__name__}[{get_backend().upper()}]" for loss in _PARAMS]


@pytest.mark.parametrize("loss_func", _PARAMS, ids=_IDS)
def test_loss_output(loss_func):
    # SSIM family uses 128x128 patches so the structural metric is meaningful.
    assert_loss_contract(loss_func, max_value=1.0, shape=(2, 3, 128, 128))
