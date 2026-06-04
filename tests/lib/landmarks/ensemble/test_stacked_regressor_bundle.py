#!/usr/bin/env python3
"""Tests for stacked regressor install/load in the production bundle (#223)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.landmarks.ensemble import production_artifacts as pa


def _seed_sources(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    setup_src = root / "src_setup.json"
    weights_src = root / "src_weights.json"
    scorer_v3 = root / "scorer_v3.json"
    regressor = root / "stacked_residual_v1.json"
    setup_src.write_text(json.dumps({"weights_path": "best_weights.json"}))
    weights_src.write_text(json.dumps({"hrnet": [1.0]}))
    scorer_v3.write_text(json.dumps({"model_type": "lightgbm_lambdarank"}))
    regressor.write_text(json.dumps({"model_type": "numpy_linear_residual_v1"}))
    return {
        "setup_src": setup_src,
        "weights_src": weights_src,
        "learned_quality_v3": scorer_v3,
        "stacked_residual_v1": regressor,
    }


def test_install_and_load_stacked_regressor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _seed_sources(tmp_path / "src")
    bundle = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle))

    pa.install_production_bundle(
        setup_src=sources["setup_src"],
        weights_src=sources["weights_src"],
        scorer_sources={"learned_quality_v3": sources["learned_quality_v3"]},
        active_policy="learned_quality_v3",
        stacked_regressor_sources={"stacked_residual_v1": sources["stacked_residual_v1"]},
        created_by="test",
    )

    installed = bundle / pa.STACKED_REGRESSORS_SUBDIR / "stacked_residual_v1.json"
    assert installed.is_file()
    manifest = json.loads((bundle / pa.MANIFEST_FILENAME).read_text())
    assert manifest["stacked_regressors"] == {
        "stacked_residual_v1": "regressors/stacked_residual_v1.json"
    }

    loaded = pa.load_production_bundle()
    assert loaded.stacked_regressor_path_for("stacked_residual_v1") == installed.resolve()
    assert loaded.stacked_regressor_path_for("missing") is None


def test_missing_stacked_regressor_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle without any stacked regressor loads cleanly (feature is optional)."""
    sources = _seed_sources(tmp_path / "src")
    bundle = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle))

    pa.install_production_bundle(
        setup_src=sources["setup_src"],
        weights_src=sources["weights_src"],
        scorer_sources={"learned_quality_v3": sources["learned_quality_v3"]},
        active_policy="learned_quality_v3",
        created_by="test",
    )
    loaded = pa.load_production_bundle()
    assert loaded.stacked_regressors == {}
    assert loaded.stacked_regressor_path_for("stacked_residual_v1") is None


def test_install_skips_missing_regressor_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _seed_sources(tmp_path / "src")
    bundle = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle))

    pa.install_production_bundle(
        setup_src=sources["setup_src"],
        weights_src=sources["weights_src"],
        scorer_sources={"learned_quality_v3": sources["learned_quality_v3"]},
        active_policy="learned_quality_v3",
        stacked_regressor_sources={"stacked_residual_v1": tmp_path / "src" / "nope.json"},
        created_by="test",
    )
    manifest = json.loads((bundle / pa.MANIFEST_FILENAME).read_text())
    assert "stacked_regressors" not in manifest


def test_load_rejects_missing_referenced_regressor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest that references a regressor file that is absent is invalid."""
    sources = _seed_sources(tmp_path / "src")
    bundle = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle))

    pa.install_production_bundle(
        setup_src=sources["setup_src"],
        weights_src=sources["weights_src"],
        scorer_sources={"learned_quality_v3": sources["learned_quality_v3"]},
        active_policy="learned_quality_v3",
        stacked_regressor_sources={"stacked_residual_v1": sources["stacked_residual_v1"]},
        created_by="test",
    )
    # Remove the installed regressor file so the manifest dangles.
    (bundle / pa.STACKED_REGRESSORS_SUBDIR / "stacked_residual_v1.json").unlink()
    with pytest.raises(pa.ProductionBundleInvalid):
        pa.load_production_bundle()
