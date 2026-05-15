#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.eval.splits` (issue #67)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.landmarks.eval.splits import (
    SCENARIO_STRATIFIED,
    SPLIT_NAMES,
    BucketDiagnostic,
    SplitAssignment,
    SplitRatios,
    filter_manifest_samples,
    load_split_file,
    save_split_file,
    scenario_bucket,
    split_assignment_hash,
    split_manifest_samples,
    split_summary_counts,
    write_split_manifest,
)


def _make_samples(bucket_sizes: dict[str, int]) -> list[dict[str, str]]:
    """Build a synthetic manifest sample list with the given per-bucket counts."""
    samples: list[dict[str, str]] = []
    for bucket, count in bucket_sizes.items():
        dataset, condition = bucket.split(":", 1)
        for index in range(count):
            samples.append(
                {
                    "sample_id": f"{bucket}-{index:03d}",
                    "image": f"{bucket}-{index:03d}.png",
                    "landmarks": f"{bucket}-{index:03d}.npy",
                    "dataset": dataset,
                    "condition": condition,
                }
            )
    return samples


def test_split_ratios_must_sum_to_one() -> None:
    """Ratios that don't sum to 1.0 raise immediately."""
    with pytest.raises(ValueError):
        SplitRatios(fit=0.5, select=0.5, report=0.5)


def test_split_ratios_require_positive_components() -> None:
    """Zero-weight splits are rejected."""
    with pytest.raises(ValueError):
        SplitRatios(fit=1.0, select=0.0, report=0.0)


def test_scenario_bucket_falls_back_to_unspecified() -> None:
    """Missing dataset/condition fields produce the ``unspecified`` bucket."""
    assert scenario_bucket({}) == "unspecified:unspecified"
    assert scenario_bucket({"dataset": "wflw"}) == "wflw:unspecified"
    assert scenario_bucket({"condition": "occluded"}) == "unspecified:occluded"


def test_scenario_stratified_split_is_deterministic_for_same_seed() -> None:
    """Same seed + same manifest must yield the same assignment."""
    samples = _make_samples({"wflw:occluded": 10, "wflw:clean": 10, "cofw:default": 6})
    ratios = SplitRatios(fit=0.6, select=0.2, report=0.2)
    first, _ = split_manifest_samples(samples, mode=SCENARIO_STRATIFIED, ratios=ratios, seed=1337)
    second, _ = split_manifest_samples(samples, mode=SCENARIO_STRATIFIED, ratios=ratios, seed=1337)
    assert first == second
    # Different seed produces a different assignment.
    third, _ = split_manifest_samples(samples, mode=SCENARIO_STRATIFIED, ratios=ratios, seed=42)
    assert third != first


def test_scenario_stratified_split_preserves_bucket_coverage() -> None:
    """Every bucket with enough samples appears in all three splits."""
    samples = _make_samples({"wflw:occluded": 10, "wflw:clean": 8, "cofw:default": 5})
    ratios = SplitRatios(fit=0.6, select=0.2, report=0.2)
    assignment, diagnostics = split_manifest_samples(
        samples, mode=SCENARIO_STRATIFIED, ratios=ratios, seed=1
    )

    for diagnostic in diagnostics:
        assert diagnostic.fit + diagnostic.select + diagnostic.report == diagnostic.total
        if diagnostic.total >= 3:
            assert diagnostic.fit >= 1
            assert diagnostic.select >= 1
            assert diagnostic.report >= 1
            assert diagnostic.too_small is False

    assignment_size = len(assignment.fit) + len(assignment.select) + len(assignment.report)
    assert assignment_size == len(samples)


def test_scenario_stratified_split_records_too_small_buckets() -> None:
    """Buckets with fewer than three samples mark themselves too_small in diagnostics."""
    samples = _make_samples({"wflw:occluded": 1, "wflw:clean": 2, "cofw:default": 5})
    ratios = SplitRatios(fit=0.6, select=0.2, report=0.2)
    assignment, diagnostics = split_manifest_samples(
        samples, mode=SCENARIO_STRATIFIED, ratios=ratios, seed=7
    )
    by_bucket = {diagnostic.bucket: diagnostic for diagnostic in diagnostics}

    occluded = by_bucket["wflw:occluded"]
    clean = by_bucket["wflw:clean"]
    default = by_bucket["cofw:default"]

    assert occluded.too_small is True
    assert occluded.fit == 1 and occluded.select == 0 and occluded.report == 0
    assert clean.too_small is True
    assert clean.fit == 1 and clean.select == 1 and clean.report == 0
    assert default.too_small is False
    assert assignment.sample_count == len(samples)


def test_random_split_is_single_bucket_diagnostic() -> None:
    """Random mode reports one ``__all__`` diagnostic bucket."""
    samples = _make_samples({"wflw:clean": 12})
    assignment, diagnostics = split_manifest_samples(
        samples, mode="random", ratios=SplitRatios(0.6, 0.2, 0.2), seed=3
    )

    assert len(diagnostics) == 1
    assert diagnostics[0].bucket == "__all__"
    assert diagnostics[0].total == 12
    assert assignment.sample_count == 12


def test_split_rejects_unknown_mode() -> None:
    """``file`` and ``none`` are pipeline-level modes and not valid here."""
    samples = _make_samples({"wflw:clean": 6})
    with pytest.raises(ValueError):
        split_manifest_samples(samples, mode="file", ratios=SplitRatios(0.6, 0.2, 0.2), seed=1)
    with pytest.raises(ValueError):
        split_manifest_samples(samples, mode="bogus", ratios=SplitRatios(0.6, 0.2, 0.2), seed=1)


def test_split_rejects_duplicate_sample_ids() -> None:
    """Duplicate sample IDs in the manifest cause a clear failure."""
    samples = _make_samples({"wflw:clean": 3})
    samples.append(dict(samples[0]))  # duplicate sample_id
    with pytest.raises(ValueError, match="duplicates"):
        split_manifest_samples(
            samples, mode=SCENARIO_STRATIFIED, ratios=SplitRatios(0.6, 0.2, 0.2), seed=1
        )


def test_split_assignment_hash_is_stable_and_order_independent() -> None:
    """The hash depends on the sample set per split, not their order."""
    base = SplitAssignment(fit=("a", "b", "c"), select=("d",), report=("e",))
    permuted = SplitAssignment(fit=("c", "a", "b"), select=("d",), report=("e",))
    different = SplitAssignment(fit=("a", "b"), select=("c", "d"), report=("e",))
    assert split_assignment_hash(base) == split_assignment_hash(permuted)
    assert split_assignment_hash(base) != split_assignment_hash(different)
    assert split_assignment_hash(base).startswith("sha256:")


def test_save_and_load_split_file_round_trip(tmp_path: Path) -> None:
    """Saved splits files round-trip back to the same assignment."""
    samples = _make_samples({"wflw:clean": 6})
    assignment, diagnostics = split_manifest_samples(
        samples, mode=SCENARIO_STRATIFIED, ratios=SplitRatios(0.6, 0.2, 0.2), seed=11
    )
    path = tmp_path / "splits.json"

    save_split_file(
        path,
        assignment,
        mode=SCENARIO_STRATIFIED,
        ratios=SplitRatios(0.6, 0.2, 0.2),
        seed=11,
        diagnostics=diagnostics,
    )
    loaded = load_split_file(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["assignment_hash"] == split_assignment_hash(assignment)
    assert payload["counts"]["fit"] == len(assignment.fit)
    assert loaded == assignment


def test_filter_manifest_samples_preserves_order_and_full_metadata() -> None:
    """Filtering keeps the manifest's existing order and all sample fields."""
    samples = _make_samples({"wflw:clean": 3, "cofw:default": 2})
    wanted = (samples[0]["sample_id"], samples[3]["sample_id"])

    filtered = filter_manifest_samples(samples, wanted)

    assert [sample["sample_id"] for sample in filtered] == [
        samples[0]["sample_id"],
        samples[3]["sample_id"],
    ]
    assert filtered[0]["dataset"] == "wflw"
    assert filtered[1]["dataset"] == "cofw"


def test_filter_manifest_samples_raises_on_missing_id() -> None:
    """Filtering must hard-fail when a requested ID is not in the manifest."""
    samples = _make_samples({"wflw:clean": 3})
    with pytest.raises(ValueError, match="missing samples"):
        filter_manifest_samples(samples, ["wflw:clean-000", "does-not-exist"])


def test_write_split_manifest_preserves_top_level_metadata(tmp_path: Path) -> None:
    """Top-level keys other than ``samples`` are forwarded into the filtered manifest."""
    samples = _make_samples({"wflw:clean": 3})
    base = {"version": 7, "schema": "2d_68", "samples": samples}

    path = write_split_manifest(tmp_path / "fit.json", base, [samples[0]["sample_id"]])

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 7
    assert payload["schema"] == "2d_68"
    assert len(payload["samples"]) == 1
    assert payload["samples"][0]["sample_id"] == samples[0]["sample_id"]


def test_split_summary_counts_groups_by_dataset_condition_and_bucket() -> None:
    """``split_summary_counts`` reports per-split bucket coverage for the run summary."""
    samples = _make_samples({"wflw:occluded": 6, "wflw:clean": 6, "cofw:default": 6})
    assignment, _ = split_manifest_samples(
        samples, mode=SCENARIO_STRATIFIED, ratios=SplitRatios(0.5, 0.25, 0.25), seed=5
    )

    counts = split_summary_counts(assignment, samples)

    assert set(counts) == {"by_dataset", "by_condition", "by_scenario_bucket", "totals"}
    for name in SPLIT_NAMES:
        assert counts["totals"][name] == len(assignment.ids_for(name))
        per_bucket = counts["by_scenario_bucket"][name]
        assert sum(per_bucket.values()) == len(assignment.ids_for(name))


def test_split_assignment_rejects_overlap() -> None:
    """A sample ID cannot appear in more than one split."""
    with pytest.raises(ValueError, match="more than one split"):
        SplitAssignment(fit=("a", "b"), select=("b",), report=("c",))


def test_bucket_diagnostic_serializes_round_trip() -> None:
    """BucketDiagnostic round-trips through its payload form."""
    diagnostic = BucketDiagnostic(
        bucket="wflw:occluded", fit=4, select=1, report=1, total=6, too_small=False
    )
    payload = diagnostic.to_payload()
    assert payload == {
        "bucket": "wflw:occluded",
        "fit": 4,
        "select": 1,
        "report": 1,
        "total": 6,
        "too_small": False,
    }
