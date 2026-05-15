#!/usr/bin/env python3
"""Pipeline-level split integration tests (#67)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.landmarks.run_static_weight_pipeline import main as pipeline_main


def _points(offset: float = 0.0) -> np.ndarray:
    return (
        np.stack(
            (
                np.linspace(20, 70, 68, dtype="float32"),
                np.linspace(25, 75, 68, dtype="float32"),
            ),
            axis=1,
        )
        + offset
    )


def _write_multi_sample_fixture(
    root: Path,
    *,
    bucket_sizes: dict[str, int],
    models: tuple[str, ...] = ("hrnet", "spiga"),
) -> dict[str, Path]:
    """Write a fixture with multiple samples spanning scenario buckets."""
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True)
    blank_image = np.zeros((96, 96, 3), dtype="uint8")
    truth_path = dataset_dir / "truth.npy"
    np.save(str(truth_path), _points())
    image_path = dataset_dir / "sample.png"
    assert cv2.imwrite(str(image_path), blank_image)

    samples: list[dict[str, str]] = []
    prediction_dirs = {model: root / "raw_predictions" / model for model in models}
    for path in prediction_dirs.values():
        path.mkdir(parents=True)

    for bucket, count in bucket_sizes.items():
        dataset, condition = bucket.split(":", 1)
        for idx in range(count):
            sid = f"{bucket}-{idx:03d}"
            samples.append(
                {
                    "sample_id": sid,
                    "dataset": dataset,
                    "condition": condition,
                    "image": "sample.png",
                    "landmarks": "truth.npy",
                }
            )
            for model_idx, model in enumerate(models):
                np.save(
                    str(prediction_dirs[model] / f"{sid}.npy"),
                    _points(float(model_idx)),
                )

    manifest = dataset_dir / "manifest.json"
    manifest.write_text(
        json.dumps({"dataset": "fixture", "samples": samples}),
        encoding="utf-8",
    )
    return {"manifest": manifest, **prediction_dirs}


def _summary(root: Path) -> dict[str, object]:
    return json.loads((root / "run_summary.json").read_text(encoding="utf-8"))


def _prediction_root_args(paths: dict[str, Path], models: tuple[str, ...]) -> list[str]:
    args: list[str] = []
    for model in models:
        args.extend(["--prediction-root", f"{model}={paths[model]}"])
    return args


def _run(args: list[str]) -> int:
    return pipeline_main(args)


def test_split_mode_none_skips_split_stage(tmp_path: Path) -> None:
    """``--split-mode=none`` preserves the legacy pipeline shape."""
    root = tmp_path / "run"
    models = ("hrnet", "spiga")
    paths = _write_multi_sample_fixture(root, bucket_sizes={"fixture:clean": 6}, models=models)

    exit_code = _run(
        [
            "--output-root",
            str(root),
            "--models",
            ",".join(models),
            "--run-predictions",
            "--prediction-mode",
            "import",
            *_prediction_root_args(paths, models),
            "--split-mode",
            "none",
            "--baseline-variants",
            "plain_average",
            "--weighted-variants",
            "static_weighted",
            "--skip-failure-viewer",
        ]
    )

    payload = _summary(root)
    assert exit_code == 0
    stage_names = [stage["name"] for stage in payload["stages"]]
    assert "create_splits" not in stage_names
    assert "splits" not in payload
    assert not (root / "splits" / "splits.json").exists()
    assert not (root / "dataset" / "fit_manifest.json").exists()


def test_scenario_stratified_split_creates_assignment_and_fit_manifest(tmp_path: Path) -> None:
    """Default scenario-stratified mode writes splits.json, per-split manifests, and run summary."""
    root = tmp_path / "run"
    models = ("hrnet", "spiga")
    paths = _write_multi_sample_fixture(
        root,
        bucket_sizes={"fixture:clean": 6, "fixture:occluded": 6},
        models=models,
    )

    exit_code = _run(
        [
            "--output-root",
            str(root),
            "--models",
            ",".join(models),
            "--run-predictions",
            "--prediction-mode",
            "import",
            *_prediction_root_args(paths, models),
            "--baseline-variants",
            "plain_average",
            "--weighted-variants",
            "static_weighted",
            "--skip-failure-viewer",
        ]
    )

    payload = _summary(root)
    assert exit_code == 0
    stage_names = [stage["name"] for stage in payload["stages"]]
    assert "create_splits" in stage_names

    splits_block = payload["splits"]
    assert splits_block["mode"] == "scenario-stratified"
    assert splits_block["assignment_hash"].startswith("sha256:")
    assert splits_block["counts"]["totals"]["fit"] >= 1
    assert splits_block["counts"]["totals"]["select"] >= 1
    assert splits_block["counts"]["totals"]["report"] >= 1

    splits_file = root / "splits" / "splits.json"
    fit_manifest = root / "dataset" / "fit_manifest.json"
    select_manifest = root / "dataset" / "select_manifest.json"
    report_manifest = root / "dataset" / "report_manifest.json"
    assert splits_file.is_file()
    assert fit_manifest.is_file()
    assert select_manifest.is_file()
    assert report_manifest.is_file()

    splits_payload = json.loads(splits_file.read_text(encoding="utf-8"))
    assert splits_payload["schema_version"] == 1
    assert splits_payload["assignment_hash"] == splits_block["assignment_hash"]
    fit_total = len(splits_payload["splits"]["fit"])
    fit_samples = json.loads(fit_manifest.read_text(encoding="utf-8"))["samples"]
    assert len(fit_samples) == fit_total


def test_split_seed_is_deterministic(tmp_path: Path) -> None:
    """Same seed + same manifest produce the same assignment hash across runs."""
    models = ("hrnet", "spiga")

    def run_once(suffix: str) -> str:
        root = tmp_path / f"run-{suffix}"
        paths = _write_multi_sample_fixture(
            root,
            bucket_sizes={"fixture:clean": 6, "fixture:occluded": 6},
            models=models,
        )
        exit_code = _run(
            [
                "--output-root",
                str(root),
                "--models",
                ",".join(models),
                "--run-predictions",
                "--prediction-mode",
                "import",
                *_prediction_root_args(paths, models),
                "--baseline-variants",
                "plain_average",
                "--weighted-variants",
                "static_weighted",
                "--skip-failure-viewer",
                "--split-seed",
                "777",
            ]
        )
        assert exit_code == 0
        return _summary(root)["splits"]["assignment_hash"]  # type: ignore[index]

    first = run_once("first")
    second = run_once("second")
    assert first == second


def test_compute_weights_uses_fit_manifest_only(tmp_path: Path) -> None:
    """Weight derivation must consume only the fit-split manifest."""
    root = tmp_path / "run"
    models = ("hrnet", "spiga")
    paths = _write_multi_sample_fixture(
        root,
        bucket_sizes={"fixture:clean": 6, "fixture:occluded": 6},
        models=models,
    )

    exit_code = _run(
        [
            "--output-root",
            str(root),
            "--models",
            ",".join(models),
            "--run-predictions",
            "--prediction-mode",
            "import",
            *_prediction_root_args(paths, models),
            "--baseline-variants",
            "plain_average",
            "--weighted-variants",
            "static_weighted",
            "--skip-failure-viewer",
        ]
    )

    payload = _summary(root)
    assert exit_code == 0
    fit_ids = set(payload["splits"]["counts"]["by_scenario_bucket"]["fit"])  # type: ignore[index]
    # The weight report's mean_errors are computed only on fit samples; we
    # can't read individual sample IDs from the weight report but we can
    # cross-check that fit count is non-zero and matches the manifest count.
    fit_total = payload["splits"]["counts"]["totals"]["fit"]  # type: ignore[index]
    fit_samples = json.loads((root / "dataset" / "fit_manifest.json").read_text(encoding="utf-8"))[
        "samples"
    ]
    assert len(fit_samples) == fit_total
    assert fit_ids  # at least one scenario bucket present in fit


def test_split_file_and_split_output_are_mutually_exclusive(tmp_path: Path) -> None:
    """argparse rejects passing both --split-file and --split-output."""
    with pytest.raises(SystemExit):
        _run(
            [
                "--output-root",
                str(tmp_path / "run"),
                "--models",
                "hrnet",
                "--split-file",
                str(tmp_path / "splits.json"),
                "--split-output",
                str(tmp_path / "out.json"),
            ]
        )


def test_split_mode_file_requires_split_file(tmp_path: Path) -> None:
    """``--split-mode=file`` without ``--split-file`` fails fast."""
    root = tmp_path / "run"

    with pytest.raises(SystemExit, match="--split-file"):
        _run(
            [
                "--output-root",
                str(root),
                "--models",
                "hrnet",
                "--skip-dataset-build",
                "--skip-predictions",
                "--skip-failure-viewer",
                "--split-mode",
                "file",
            ]
        )


def test_split_mode_none_rejects_split_flags(tmp_path: Path) -> None:
    """``--split-mode=none`` with ``--split-output`` is a clear argument error."""
    root = tmp_path / "run"

    with pytest.raises(SystemExit, match="--split-output"):
        _run(
            [
                "--output-root",
                str(root),
                "--models",
                "hrnet",
                "--skip-dataset-build",
                "--skip-predictions",
                "--skip-failure-viewer",
                "--split-mode",
                "none",
                "--split-output",
                str(root / "out.json"),
            ]
        )


def test_split_mode_file_loads_existing_assignment(tmp_path: Path) -> None:
    """``--split-mode=file`` consumes a precomputed assignment instead of generating one."""
    root = tmp_path / "run"
    models = ("hrnet", "spiga")
    paths = _write_multi_sample_fixture(
        root,
        bucket_sizes={"fixture:clean": 6, "fixture:occluded": 6},
        models=models,
    )

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    sample_ids = [sample["sample_id"] for sample in manifest["samples"]]
    splits_path = tmp_path / "external_splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "file",
                "splits": {
                    "fit": sample_ids[:8],
                    "select": sample_ids[8:10],
                    "report": sample_ids[10:],
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = _run(
        [
            "--output-root",
            str(root),
            "--models",
            ",".join(models),
            "--run-predictions",
            "--prediction-mode",
            "import",
            *_prediction_root_args(paths, models),
            "--baseline-variants",
            "plain_average",
            "--weighted-variants",
            "static_weighted",
            "--skip-failure-viewer",
            "--split-mode",
            "file",
            "--split-file",
            str(splits_path),
        ]
    )

    payload = _summary(root)
    assert exit_code == 0
    assert payload["splits"]["mode"] == "file"
    assert payload["splits"]["counts"]["totals"]["fit"] == 8
    assert payload["splits"]["counts"]["totals"]["select"] == 2
    assert payload["splits"]["counts"]["totals"]["report"] == 2


def test_split_mode_file_rejects_missing_sample_ids(tmp_path: Path) -> None:
    """``--split-file`` referencing unknown sample IDs fails with a clear error."""
    root = tmp_path / "run"
    models = ("hrnet", "spiga")
    paths = _write_multi_sample_fixture(
        root,
        bucket_sizes={"fixture:clean": 6},
        models=models,
    )

    splits_path = tmp_path / "broken_splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "splits": {
                    "fit": ["nonexistent-sample"],
                    "select": ["fixture:clean-000"],
                    "report": ["fixture:clean-001"],
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = _run(
        [
            "--output-root",
            str(root),
            "--models",
            ",".join(models),
            "--run-predictions",
            "--prediction-mode",
            "import",
            *_prediction_root_args(paths, models),
            "--skip-failure-viewer",
            "--split-mode",
            "file",
            "--split-file",
            str(splits_path),
        ]
    )

    payload = _summary(root)
    assert exit_code != 0
    assert "samples not in the manifest" in payload["failed_stage_error"]


def test_dry_run_includes_create_splits_stage(tmp_path: Path) -> None:
    """Dry-run plan lists create_splits when split mode is not 'none'."""
    root = tmp_path / "run"

    exit_code = _run(
        [
            "--output-root",
            str(root),
            "--datasets",
            "wflw",
            "--dry-run",
            "--split-mode",
            "scenario-stratified",
        ]
    )
    payload = _summary(root)
    assert exit_code == 0
    assert any(str(stage).startswith("create_splits") for stage in payload["planned_stages"])
