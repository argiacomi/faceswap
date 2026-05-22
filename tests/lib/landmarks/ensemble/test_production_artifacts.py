"""Tests for the production landmark-ensemble bundle module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.landmarks.ensemble import production_artifacts as pa


def _seed_pipeline_sources(root: Path) -> dict[str, Path]:
    """Produce minimal pipeline-shaped source files for install tests."""
    root.mkdir(parents=True, exist_ok=True)
    setup_src = root / "src_setup.json"
    weights_src = root / "src_weights.json"
    scorer_v1 = root / "scorer_v1.json"
    scorer_v1_1 = root / "scorer_v1_1.json"
    scorer_v2 = root / "scorer_v2.json"
    setup_src.write_text(json.dumps({"weights_path": "best_weights.json"}))
    weights_src.write_text(json.dumps({"hrnet": [1.0]}))
    scorer_v1.write_text(json.dumps({"model_type": "logistic_regression"}))
    scorer_v1_1.write_text(json.dumps({"model_type": "linear_regression"}))
    scorer_v2.write_text(json.dumps({"model_type": "lightgbm_lambdarank"}))
    return {
        "setup_src": setup_src,
        "weights_src": weights_src,
        "learned_quality_v1": scorer_v1,
        "learned_quality_v1_1": scorer_v1_1,
        "learned_quality_v2": scorer_v2,
    }


def test_bundle_dir_respects_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The env var should override the default ``.fs_cache`` location."""
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(tmp_path / "alt"))
    resolved = pa.bundle_dir()
    assert resolved == (tmp_path / "alt").resolve()
    assert pa.manifest_path() == resolved / pa.MANIFEST_FILENAME


def test_bundle_dir_defaults_to_project_fs_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the env override, the bundle lives under .fs_cache/."""
    monkeypatch.delenv(pa.BUNDLE_DIR_ENV, raising=False)
    resolved = pa.bundle_dir()
    assert resolved == pa.DEFAULT_BUNDLE_DIR
    assert resolved.parts[-3:] == (".fs_cache", "landmark_ensemble", "current")


def test_load_production_bundle_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean directory raises ProductionBundleMissing with a useful hint."""
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(tmp_path / "empty"))
    with pytest.raises(pa.ProductionBundleMissing) as info:
        pa.load_production_bundle()
    assert "manifest" in str(info.value)


def test_install_and_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """install_production_bundle writes the expected layout; load reads it."""
    sources = _seed_pipeline_sources(tmp_path / "src")
    bundle = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle))

    pa.install_production_bundle(
        setup_src=sources["setup_src"],
        weights_src=sources["weights_src"],
        scorer_sources={
            "learned_quality_v1": sources["learned_quality_v1"],
            "learned_quality_v1_1": sources["learned_quality_v1_1"],
            "learned_quality_v2": sources["learned_quality_v2"],
        },
        active_policy="learned_quality_v2",
        source_output_root=tmp_path / "src",
        created_by="test",
    )

    # Files landed where the manifest schema promises.
    assert (bundle / "best_setup.json").is_file()
    assert (bundle / "best_weights.json").is_file()
    assert (bundle / "scorers" / "learned_quality_v1.json").is_file()
    assert (bundle / "scorers" / "learned_quality_v1_1.json").is_file()
    assert (bundle / "scorers" / "learned_quality_v2.json").is_file()
    assert (bundle / "manifest.json").is_file()

    loaded = pa.load_production_bundle()
    assert loaded.active_policy == "learned_quality_v2"
    assert loaded.setup_path == (bundle / "best_setup.json").resolve()
    assert (
        loaded.scorer_path_for("learned_quality_v2")
        == (bundle / "scorers" / "learned_quality_v2.json").resolve()
    )
    # roll_aware_veto returns None — no scorer needed for that path.
    assert loaded.scorer_path_for(pa.ROLL_AWARE_VETO_POLICY) is None


def test_install_skips_missing_scorer_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only present scorer sources are installed; the manifest reflects that."""
    sources = _seed_pipeline_sources(tmp_path / "src")
    bundle = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle))

    pa.install_production_bundle(
        setup_src=sources["setup_src"],
        weights_src=sources["weights_src"],
        scorer_sources={
            "learned_quality_v1": tmp_path / "src" / "does_not_exist.json",
            "learned_quality_v2": sources["learned_quality_v2"],
        },
        active_policy="learned_quality_v2",
    )

    loaded = pa.load_production_bundle()
    assert set(loaded.scorers) == {"learned_quality_v2"}
    with pytest.raises(pa.ProductionBundleInvalid):
        loaded.scorer_path_for("learned_quality_v1")


def test_install_rejects_unknown_policy(tmp_path: Path) -> None:
    """Unknown policies in scorer_sources are a programming error."""
    sources = _seed_pipeline_sources(tmp_path / "src")
    with pytest.raises(ValueError, match="unsupported policies"):
        pa.install_production_bundle(
            setup_src=sources["setup_src"],
            weights_src=sources["weights_src"],
            scorer_sources={"learned_quality_v3": sources["learned_quality_v2"]},
            active_policy="learned_quality_v2",
            dest=tmp_path / "bundle",
        )


def test_install_rejects_active_policy_with_no_scorer(tmp_path: Path) -> None:
    """active_policy must have an installed scorer for learned policies."""
    sources = _seed_pipeline_sources(tmp_path / "src")
    with pytest.raises(ValueError, match="no scorer was installed"):
        pa.install_production_bundle(
            setup_src=sources["setup_src"],
            weights_src=sources["weights_src"],
            scorer_sources={"learned_quality_v1": sources["learned_quality_v1"]},
            active_policy="learned_quality_v2",
            dest=tmp_path / "bundle",
        )


def test_load_rejects_old_schema_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An older artifact_schema_version surfaces as ProductionBundleInvalid."""
    sources = _seed_pipeline_sources(tmp_path / "src")
    bundle = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle))
    pa.install_production_bundle(
        setup_src=sources["setup_src"],
        weights_src=sources["weights_src"],
        scorer_sources={"learned_quality_v2": sources["learned_quality_v2"]},
        active_policy="learned_quality_v2",
    )
    manifest_file = bundle / "manifest.json"
    payload = json.loads(manifest_file.read_text())
    payload["artifact_schema_version"] = 999
    manifest_file.write_text(json.dumps(payload))

    with pytest.raises(pa.ProductionBundleInvalid, match="artifact_schema_version"):
        pa.load_production_bundle()


def test_load_rejects_missing_referenced_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If best_setup.json is deleted post-install, load fails loudly."""
    sources = _seed_pipeline_sources(tmp_path / "src")
    bundle = tmp_path / "bundle"
    monkeypatch.setenv(pa.BUNDLE_DIR_ENV, str(bundle))
    pa.install_production_bundle(
        setup_src=sources["setup_src"],
        weights_src=sources["weights_src"],
        scorer_sources={"learned_quality_v2": sources["learned_quality_v2"]},
        active_policy="learned_quality_v2",
    )
    (bundle / "best_setup.json").unlink()

    with pytest.raises(pa.ProductionBundleInvalid, match="setup file is missing"):
        pa.load_production_bundle()


# ---------------------------------------------------------------------------
# Policy ↔ scorer-runtime_policy consistency
# ---------------------------------------------------------------------------


def test_resolve_runtime_rejects_scorer_runtime_policy_mismatch(tmp_path: Path) -> None:
    """The runtime refuses to score with a scorer trained for a different policy.

    Regression for the bug where the v1.1 continuous scorer was installed
    under the v1 manifest slot. With the corrected scorer-training output
    this can no longer happen via the pipeline, but operators editing
    extract.ini or manifests by hand could still produce a mismatch; the
    runtime check makes that loud instead of silent.
    """
    from lib.landmarks.ensemble.runtime_resolver import (
        ModelPrediction,
        RuntimeResolverConfig,
        RuntimeResolverError,
        resolve_runtime,
    )
    from lib.landmarks.ensemble.runtime_resolver_scorer import (
        RuntimeResolverScorer,
        write_runtime_resolver_scorer,
    )

    # Minimal one-feature scorer artifact carrying runtime_policy=v1.
    scorer = RuntimeResolverScorer(
        features=("candidate_name=hrnet",),
        coefficients=(1.0,),
        intercept=0.0,
        runtime_policy="learned_quality_v1",
    )
    scorer_path = write_runtime_resolver_scorer(scorer, tmp_path / "scorer.json")

    # Three synthetic predictions with simple frame-space landmarks. The
    # actual scoring details don't matter — the test just needs the runtime
    # to reach the policy check; the mismatch must raise before that.
    import numpy as np

    # 68 landmarks spread across a 200x200 region so the resolver's
    # frame-space validator accepts them.
    base = np.column_stack([np.linspace(20.0, 220.0, 68), np.linspace(20.0, 220.0, 68)]).astype(
        "float32"
    )
    predictions = [
        ModelPrediction(model=name, landmarks=base + offset, weight=1.0)
        for name, offset in (("hrnet", 0.0), ("spiga", 1.0), ("orformer", -1.0))
    ]
    config = RuntimeResolverConfig(
        policy="learned_quality_v1_1",  # mismatch vs scorer's "learned_quality_v1"
        scorer_path=str(scorer_path),
        general_strategy="plain_average",
        hard_case_strategy="plain_average",
        secondary_hard_case_strategy="plain_average",
        fallback_strategy="plain_average",
        fallback_model="hrnet",
        outlier_threshold=3.5,
        weights=None,
        adapter_weights={"hrnet": 1.0, "spiga": 1.0, "orformer": 1.0},
        hard_disagreement_px=12.0,
        roll_veto_degrees=15.0,
        hard_roll_degrees=30.0,
        strict=True,
    )

    with pytest.raises(RuntimeResolverError, match="runtime_policy"):
        resolve_runtime(predictions, config, detector_bbox=(0.0, 0.0, 256.0, 256.0))
