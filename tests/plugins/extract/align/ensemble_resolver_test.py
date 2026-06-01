#!/usr/bin/env python3
"""Plugin-level tests for the geometry-risk alignment resolver wiring (#78)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.adapters import LandmarkAdapterConfig, StaticLandmarkAdapter
from plugins.extract.align.ensemble import Ensemble


def _face() -> np.ndarray:
    """Plausible iBUG-style 68-point face used so AlignedFace returns sane geometry."""
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    points[0:17, 0] = np.linspace(40, 160, 17)
    points[0:17, 1] = 120 + 30 * np.sin(np.linspace(0, np.pi, 17))
    points[17:22, 0] = np.linspace(50, 90, 5)
    points[17:22, 1] = 70
    points[22:27, 0] = np.linspace(110, 150, 5)
    points[22:27, 1] = 70
    points[27:36, 0] = 100
    points[27:36, 1] = np.linspace(75, 110, 9)
    points[36:42, 0] = np.linspace(60, 80, 6)
    points[36:42, 1] = 85
    points[42:48, 0] = np.linspace(120, 140, 6)
    points[42:48, 1] = 85
    points[48:60, 0] = np.linspace(70, 130, 12)
    points[48:60, 1] = 130
    points[60:68, 0] = np.linspace(80, 120, 8)
    points[60:68, 1] = 130
    return points


_TEST_NO_BUNDLE_KWARGS: dict[str, str] = {
    "setup_path": "",
    "resolver_scorer_path": "",
}
"""Sentinel kwargs that bypass the production bundle requirement.

The plugin requires a bundle whenever ``use_alignment_resolver=True`` and
neither path kwarg is supplied. Tests construct adapters directly and have
no setup/scorer file on disk, so they pass these explicit empty strings to
opt out of the bundle lookup.
"""


def _three_adapters(offsets: tuple[float, float, float]) -> list[StaticLandmarkAdapter]:
    """Three static adapters returning the canonical face shifted by ``offsets``."""
    base = _face()
    return [
        StaticLandmarkAdapter(
            LandmarkAdapterConfig(name, coordinate_space="frame", weight=1.0),
            base + offset,
        )
        for name, offset in zip(("hrnet", "spiga", "orformer"), offsets, strict=True)
    ]


def test_resolver_disabled_uses_legacy_fusion(tmp_path) -> None:
    """``use_alignment_resolver=False`` (default) preserves existing fusion."""
    adapters = _three_adapters((0.0, 0.5, 1.0))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=False,
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))
    debug = plugin.last_debug_metadata[0]
    assert debug["weight_source"] in {"adapter_config", "promoted_setup"}
    assert "resolver" not in debug


def test_resolver_enabled_routes_low_risk_path(tmp_path) -> None:
    """Closely-agreeing adapters take the resolver's low-risk path."""
    adapters = _three_adapters((0.0, 0.3, 0.6))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=True,
        **_TEST_NO_BUNDLE_KWARGS,  # type: ignore[arg-type]  # bypass production bundle requirement
        hard_case_strategy="static_weighted_downweight",
        hard_disagreement_px=50.0,  # generous so we land in low_risk
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))
    debug = plugin.last_debug_metadata[0]
    assert debug["weight_source"] == "runtime_resolver"
    assert debug["resolver"]["risk_route"] == "low_risk"
    assert debug["selected_candidate"] in debug["candidate_priority"]
    assert "active_models" in debug


def test_resolver_metadata_carries_per_model_disagreement(tmp_path) -> None:
    """Per-model disagreement values appear in ``last_debug_metadata`` for debugging."""
    adapters = _three_adapters((0.0, 0.5, 1.0))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=True,
        **_TEST_NO_BUNDLE_KWARGS,  # type: ignore[arg-type]  # bypass production bundle requirement
    )
    plugin.model = plugin.load_model()
    plugin.predict_landmarks_68(np.zeros((256, 256, 3), dtype="float32"))
    debug = plugin.last_debug_metadata[0]
    assert "max_disagreement_px" in debug["resolver"]
    assert debug["resolver"]["max_disagreement_px"] >= 0.0
    assert "landmark_consensus_distance" in debug["resolver"]


def test_strict_resolver_error_hard_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict mode hard-fails at load time when a learned policy has no scorer.

    The plugin preloads the LightGBM Booster in ``load_model`` (so it lands
    before Torch initializes libomp on macOS); a misconfigured learned policy
    therefore surfaces at load time rather than on the first face. The error
    message still names the missing knob so the operator can fix the
    deployment without reading code.
    """
    adapters = _three_adapters((0.0, 0.5, 1.0))
    plugin = Ensemble(
        adapters=adapters,
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=True,
        # Explicit empty setup/scorer kwargs trigger the strict-failure path
        # without needing a real bundle on disk; the test is specifically
        # validating that load_model raises when the learned policy has no
        # scorer artifact.
        setup_path="",
        resolver_policy="learned_quality_v2",
        resolver_scorer_path="",
        strict=True,
    )

    with pytest.raises(ValueError, match="resolver_scorer_path"):
        plugin.load_model()


# ---------------------------------------------------------------------------
# Production bundle resolution (Phase 3)
# ---------------------------------------------------------------------------


def _write_valid_setup_and_weights_pair(src_dir: Path) -> tuple[Path, Path]:
    """Materialize a valid promoted_setup pair so load_promoted_setup is happy."""
    from lib.landmarks.ensemble.promoted_setup import (
        SETUP_FILENAME,
        WEIGHTS_FILENAME,
        write_best_setup,
        write_best_weights,
    )
    from lib.landmarks.ensemble.weights import LANDMARK_COUNT

    src_dir.mkdir(parents=True, exist_ok=True)
    weights_path = src_dir / WEIGHTS_FILENAME
    setup_path = src_dir / SETUP_FILENAME
    write_best_weights(
        weights_path,
        {
            "hrnet": [1.0 / 3] * LANDMARK_COUNT,
            "spiga": [1.0 / 3] * LANDMARK_COUNT,
            "orformer": [1.0 / 3] * LANDMARK_COUNT,
        },
        models=("hrnet", "spiga", "orformer"),
    )
    write_best_setup(
        setup_path,
        candidate_id="sha256:0123abc",
        models=("hrnet", "spiga", "orformer"),
        strategy="static_weighted",
        outlier_threshold=None,
        weight_generator_name="inverse_mean_error",
        weight_generator_params={"epsilon": 1e-6},
        crop_scale=1.6,
        bbox_source="manifest",
        regression_epsilon_nme=0.001,
        reproducibility={
            "split_assignment_hash": "sha256:abc",
            "candidate_search_seed": 1337,
            "objective": "extract_alignment_v1",
        },
        fit={"sample_count": 12, "scenario_buckets": ["fixture:clean"]},
        selection_metrics={"sample_count": 4},
        report_metrics={"sample_count": 4},
        evaluation_log_path="candidate_results.json",
        weights_path=WEIGHTS_FILENAME,
    )
    return setup_path, weights_path


def test_init_resolves_setup_and_scorer_from_production_bundle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no path kwargs are supplied, the plugin reads paths from the bundle.

    Phase 3 contract: ``setup_path`` and ``resolver_scorer_path`` are no
    longer read from extract.ini. They are resolved from the installed
    production bundle (or its env override) so changing ``resolver_policy``
    in config automatically selects the matching scorer.
    """
    from lib.landmarks.ensemble import production_artifacts as pa

    src_dir = tmp_path / "src"
    setup_src, weights_src = _write_valid_setup_and_weights_pair(src_dir)
    scorer_src = src_dir / "scorer_v1_1.json"
    scorer_src.write_text('{"model_type": "linear_regression"}\n', encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle_dir))
    pa.install_production_bundle(
        setup_src=setup_src,
        weights_src=weights_src,
        scorer_sources={"learned_quality_v2": scorer_src},
        active_policy="learned_quality_v2",
    )

    plugin = Ensemble(
        adapters=_three_adapters((0.0, 0.5, 1.0)),
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=True,
        resolver_policy="learned_quality_v2",
    )

    assert plugin._setup_path == str((bundle_dir / "best_setup.json").resolve())
    assert plugin._resolver_scorer_path == str(
        (bundle_dir / "scorers" / "learned_quality_v1_1.json").resolve()
    )
    # weights_path is no longer carried in config; promoted_setup resolves it.
    assert plugin._weights_path == ""
    # Bundle-supplied setup ⇒ strict mode by default.
    assert plugin._setup_mode == "strict"


def test_init_kwargs_override_bundle(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit path kwargs still win over the installed bundle.

    Tests inject a custom setup/scorer pair without installing a real
    bundle. The bundle lookup is short-circuited when both kwargs are
    supplied so callers don't need to mock the bundle filesystem.
    """
    from lib.landmarks.ensemble import production_artifacts as pa

    bundle_dir = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle_dir))
    # Intentionally do NOT install_production_bundle — both kwargs are
    # supplied so the plugin should never look the bundle up.

    plugin = Ensemble(
        adapters=_three_adapters((0.0, 0.5, 1.0)),
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=True,
        setup_path="",  # explicit empty: no setup file
        resolver_policy="roll_aware_veto",  # no scorer required
        resolver_scorer_path="",
    )

    assert plugin._setup_path == ""
    assert plugin._resolver_scorer_path == ""
    # No setup → effective setup_mode is "off" via the existing resolver.
    assert plugin._setup_mode == "off"


def test_init_resolver_enabled_without_bundle_or_kwargs_is_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing bundle is fatal when use_alignment_resolver=True and no kwargs.

    Previously this case silently degraded (no setup, no scorer) which let
    roll_aware_veto run on a misconfigured deployment without the promoted
    setup/weights the operator intended. The new contract treats the bundle
    as authoritative whenever the resolver is enabled.
    """
    from lib.landmarks.ensemble import production_artifacts as pa

    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(tmp_path / "no_bundle_here"))

    with pytest.raises(pa.ProductionBundleMissing, match="use_alignment_resolver=True"):
        Ensemble(
            adapters=_three_adapters((0.0, 0.5, 1.0)),
            crop_scale=1.0,
            strategy="plain_average",
            use_alignment_resolver=True,
            resolver_policy="roll_aware_veto",
        )


def test_init_resolver_disabled_without_bundle_is_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling the resolver makes the bundle optional.

    Without ``use_alignment_resolver=True`` the plugin is in legacy fusion
    mode and the production bundle is not load-bearing, so a missing
    bundle should not block construction.
    """
    from lib.landmarks.ensemble import production_artifacts as pa

    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(tmp_path / "no_bundle_here"))

    plugin = Ensemble(
        adapters=_three_adapters((0.0, 0.5, 1.0)),
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=False,
    )

    assert plugin._setup_path == ""
    assert plugin._resolver_scorer_path == ""


def test_init_resolver_enabled_with_test_kwargs_skips_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit setup/scorer kwargs (even empty) bypass the bundle requirement.

    Tests construct plugins with adapters directly and have no real bundle
    on disk. Passing explicit path kwargs (the test-injection hatch) opts
    out of the bundle check so unit tests don't need to materialize one.
    """
    from lib.landmarks.ensemble import production_artifacts as pa

    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(tmp_path / "no_bundle_here"))

    plugin = Ensemble(
        adapters=_three_adapters((0.0, 0.5, 1.0)),
        crop_scale=1.0,
        strategy="plain_average",
        use_alignment_resolver=True,
        resolver_policy="roll_aware_veto",
        **_TEST_NO_BUNDLE_KWARGS,  # type: ignore[arg-type]
    )

    assert plugin._setup_path == ""
    assert plugin._resolver_scorer_path == ""
