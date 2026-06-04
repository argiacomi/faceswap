#!/usr/bin/env python3
"""Tests for optional stacked regressor wiring in the ensemble aligner."""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.landmarks.ensemble.production_artifacts import ProductionBundleMissing
from plugins.extract.align import ensemble as ensemble_module
from plugins.extract.align.ensemble import Ensemble


class _BundleWithoutStackedRegressor:
    """Minimal production bundle stub with no stacked regressor installed."""

    setup_path = Path("setup.json")

    def scorer_path_for(self, _policy: str):
        """roll_aware_veto uses no scorer."""
        return None

    def scorer_paths_for(self, _policy: str):
        """No routed scorers in this test."""
        return {}

    def stacked_regressor_path_for(self, _name: str):
        """No stacked regressor is installed."""
        return None


def test_explicit_stacked_regressor_requires_installed_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicitly enabling stacked regression should not silently disable it."""
    monkeypatch.setattr(
        ensemble_module.Ensemble,
        "_load_bundle_or_none",
        staticmethod(lambda **_kwargs: _BundleWithoutStackedRegressor()),
    )
    monkeypatch.setattr(
        ensemble_module.Ensemble,
        "_load_promoted_setup",
        lambda self, _path, _mode: None,
    )

    with pytest.raises(ProductionBundleMissing, match="stacked regressor"):
        Ensemble(
            adapters=[],
            use_alignment_resolver=True,
            use_stacked_landmark_regressor=True,
            stacked_landmark_regressor_policy="stacked_residual_v1",
        )
