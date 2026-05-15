#!/usr/bin/env python3
"""CLI generator-selection tests for ``compute_static_weights`` (issue #68)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.ensemble.weight_generators import GENERATOR_NAMES
from lib.landmarks.ensemble.weights import LANDMARK_COUNT, MODEL_NAMES, load_weights
from lib.landmarks.eval.prediction_cache import DiskPredictionCache
from lib.landmarks.schema import LandmarkPrediction
from tools.landmarks.compute_static_weights import (
    DEFAULT_GENERATOR,
    fit_static_weights,
    main,
    save_weight_artifact,
)


def _points(offset: float = 0.0) -> np.ndarray:
    points = np.stack(
        (
            np.linspace(0, 67, LANDMARK_COUNT, dtype="float32"),
            np.linspace(10, 77, LANDMARK_COUNT, dtype="float32"),
        ),
        axis=1,
    )
    return points + offset


def _write_manifest(tmp_path: Path, sample_ids: tuple[str, ...]) -> Path:
    landmark_dir = tmp_path / "truth"
    landmark_dir.mkdir()
    samples = []
    for sample_id in sample_ids:
        path = landmark_dir / f"{sample_id}.npy"
        np.save(str(path), _points())
        samples.append(
            {
                "sample_id": sample_id,
                "image": f"{sample_id}.png",
                "landmarks": str(path.relative_to(tmp_path)),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}) + "\n", encoding="utf-8")
    return manifest


def _seed_cache(tmp_path: Path) -> Path:
    cache = DiskPredictionCache(tmp_path / "cache")
    for sample_id in ("a", "b"):
        for model, offset in zip(MODEL_NAMES, (1.0, 2.0, 4.0), strict=True):
            cache.write(sample_id, LandmarkPrediction(_points(offset), model_name=model))
    return tmp_path / "cache"


def test_default_generator_is_inverse_mean_error() -> None:
    """The CLI default preserves the legacy inverse-error behavior name."""
    assert DEFAULT_GENERATOR == "inverse_mean_error"


@pytest.mark.parametrize("generator", GENERATOR_NAMES)
def test_cli_supports_every_registered_generator(tmp_path: Path, generator: str) -> None:
    """Every generator name is acceptable as ``--generator`` and writes valid weights."""
    manifest = _write_manifest(tmp_path, ("a", "b"))
    cache = _seed_cache(tmp_path)
    output = tmp_path / "weights.json"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--cache-dir",
            str(cache),
            "--generator",
            generator,
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "2d_68"
    assert payload["generator"]["name"] == generator
    weights = load_weights(output)
    assert tuple(weights) == MODEL_NAMES
    for landmark_index in range(LANDMARK_COUNT):
        assert sum(weights[model][landmark_index] for model in MODEL_NAMES) == pytest.approx(1.0)


def test_cli_rejects_unknown_generator(tmp_path: Path) -> None:
    """argparse rejects unsupported generator names with a clear error."""
    manifest = _write_manifest(tmp_path, ("a",))
    cache = _seed_cache(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "--manifest",
                str(manifest),
                "--cache-dir",
                str(cache),
                "--generator",
                "made_up_generator",
                "--output",
                str(tmp_path / "weights.json"),
            ]
        )


def test_cli_forwards_generator_params_to_constructor(tmp_path: Path) -> None:
    """``--generator-params epsilon=...`` is forwarded into the generator instance."""
    manifest = _write_manifest(tmp_path, ("a", "b"))
    cache = _seed_cache(tmp_path)
    output = tmp_path / "weights.json"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--cache-dir",
            str(cache),
            "--generator",
            "inverse_mean_error",
            "--generator-params",
            "epsilon=1e-3",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generator"]["name"] == "inverse_mean_error"
    assert payload["generator"]["diagnostics"]["epsilon"] == pytest.approx(1e-3)


def test_fit_static_weights_returns_diagnostics_and_table(tmp_path: Path) -> None:
    """The diagnostics-aware API returns both WeightFitResult and ErrorTable."""
    manifest = _write_manifest(tmp_path, ("a", "b"))
    cache = _seed_cache(tmp_path)

    result, table = fit_static_weights(manifest, cache, MODEL_NAMES)

    assert result.name == DEFAULT_GENERATOR
    assert tuple(result.weights) == MODEL_NAMES
    assert table.models == MODEL_NAMES
    assert "mean_errors" in result.diagnostics


def test_save_weight_artifact_embeds_generator_block(tmp_path: Path) -> None:
    """The saved artifact stores generator name and diagnostics for promotion."""
    manifest = _write_manifest(tmp_path, ("a", "b"))
    cache = _seed_cache(tmp_path)
    result, _table = fit_static_weights(
        manifest, cache, MODEL_NAMES, generator="regularized_inverse_error"
    )
    output = tmp_path / "weights.json"
    save_weight_artifact(output, result)

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema"] == "2d_68"
    assert payload["generator"]["name"] == "regularized_inverse_error"
    diagnostics = payload["generator"]["diagnostics"]
    assert diagnostics["models"] == list(MODEL_NAMES)
    assert {"per_landmark_weight", "per_region_weight", "global_weight"}.issubset(
        diagnostics["components"]
    )
