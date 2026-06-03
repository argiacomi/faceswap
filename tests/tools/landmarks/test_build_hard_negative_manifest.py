#!/usr/bin/env python3
"""Tests for the hard-negative manifest builder tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tools.landmarks.build_hard_negative_manifest import build_hard_negative_manifest


def _write_manifest(path: Path, samples: list[dict]) -> Path:
    path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return path


@pytest.fixture
def manifests(tmp_path: Path) -> dict[str, Path]:
    wflw = _write_manifest(
        tmp_path / "wflw.json",
        [
            {
                "sample_id": "w_po",
                "image": "po.png",
                "landmarks": "po.npy",
                "conditions": ["pose", "occlusion"],
            },
            {
                "sample_id": "w_p",
                "image": "p.png",
                "landmarks": "p.npy",
                "conditions": ["pose"],
            },
            {
                "sample_id": "w_o",
                "image": "o.png",
                "landmarks": "o.npy",
                "conditions": ["occlusion"],
            },
        ],
    )
    cofw = _write_manifest(
        tmp_path / "cofw.json",
        [
            {"sample_id": "c1", "image": "c1.png", "landmarks": "c1.npy"},
            {"sample_id": "c2", "image": "c2.png", "landmarks": "c2.npy"},
        ],
    )
    w300 = _write_manifest(
        tmp_path / "w300.json",
        [{"sample_id": "a1", "image": "a1.png", "landmarks": "a1.npy"}],
    )
    return {"wflw": wflw, "cofw": cofw, "300w": w300}


def _load(path: Path) -> dict:
    return cast(dict[Any, Any], json.loads(path.read_text(encoding="utf-8")))


def test_merges_and_classifies_all_datasets(manifests, tmp_path: Path) -> None:
    out = tmp_path / "out"
    report = build_hard_negative_manifest(manifests=manifests, output_dir=out, write_audit=True)
    assert report["counts"]["profile_occlusion"] == 1
    assert report["counts"]["profile"] == 1
    # cofw defaults to occlusion (2 samples) plus the explicit wflw occlusion sample
    assert report["counts"]["occlusion"] == 3
    assert report["counts"]["anchor"] == 1
    assert (out / "manifest.json").exists()
    assert (out / "hard_negative_mix.json").exists()
    assert (out / "dataset_audit.json").exists()


def test_output_preserves_fields_and_adds_weight(manifests, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build_hard_negative_manifest(manifests=manifests, output_dir=out)
    samples = _load(out / "manifest.json")["samples"]
    by_id = {sample["sample_id"]: sample for sample in samples}
    profile_occ = by_id["w_po"]
    assert profile_occ["image"] == "po.png"
    assert profile_occ["landmarks"] == "po.npy"
    assert profile_occ["metadata"]["hard_negative_weight"] == 5.0
    assert profile_occ["metadata"]["hard_negative_bucket"] == "profile_occlusion"
    # cofw default occlusion
    assert by_id["c1"]["metadata"]["hard_negative_weight"] == 2.0
    # 300w default anchor
    assert by_id["a1"]["metadata"]["hard_negative_weight"] == 1.0


def test_priority_quota_enforced(manifests, tmp_path: Path) -> None:
    out = tmp_path / "out"
    report = build_hard_negative_manifest(
        manifests=manifests,
        output_dir=out,
        max_occlusion=1,
        max_anchors=0,
    )
    assert report["counts"]["occlusion"] == 1
    assert report["counts"]["anchor"] == 0
    buckets = {
        sample["metadata"]["hard_negative_bucket"]
        for sample in _load(out / "manifest.json")["samples"]
    }
    assert "anchor" not in buckets


def test_dedupe_by_source_key(tmp_path: Path) -> None:
    # The same physical sample (identical dataset + source id) can appear in two
    # overlapping manifests; it must dedupe to a single entry unless overlap is
    # explicitly allowed.
    dup = {
        "sample_id": "dup",
        "image": "dup.png",
        "landmarks": "dup.npy",
        "conditions": ["pose"],
        "source": {"dataset": "wflw", "source_id": "dup"},
    }
    first = _write_manifest(tmp_path / "a.json", [dict(dup)])
    second = _write_manifest(tmp_path / "b.json", [dict(dup)])
    out = tmp_path / "out"
    report = build_hard_negative_manifest(
        manifests={"wflw": first, "wflw_extra": second}, output_dir=out
    )
    assert report["counts"]["profile"] == 1

    out_overlap = tmp_path / "out_overlap"
    report_overlap = build_hard_negative_manifest(
        manifests={"wflw": first, "wflw_extra": second},
        output_dir=out_overlap,
        allow_overlap=True,
    )
    assert report_overlap["counts"]["profile"] == 2


def test_quota_sampling_deterministic_with_seed(tmp_path: Path) -> None:
    samples = [
        {
            "sample_id": f"p{idx}",
            "image": f"p{idx}.png",
            "landmarks": f"p{idx}.npy",
            "conditions": ["pose"],
        }
        for idx in range(10)
    ]
    manifest = _write_manifest(tmp_path / "wflw.json", samples)

    def _ids(out: Path, seed: int) -> list[str]:
        build_hard_negative_manifest(
            manifests={"wflw": manifest}, output_dir=out, max_profile=3, seed=seed
        )
        return [s["sample_id"] for s in _load(out / "manifest.json")["samples"]]

    first = _ids(tmp_path / "o1", seed=7)
    second = _ids(tmp_path / "o2", seed=7)
    assert first == second
    assert len(first) == 3


def test_audit_tracks_dataset_defaults(manifests, tmp_path: Path) -> None:
    out = tmp_path / "out"
    report = build_hard_negative_manifest(manifests=manifests, output_dir=out, write_audit=True)
    audit = _load(out / "dataset_audit.json")

    assert audit["300w"]["dataset_default"] == 1
    assert audit["300w"]["anchor"] == 1
    assert audit["cofw"]["dataset_default"] == 2
    assert audit["cofw"]["occlusion"] == 2
    assert report["dataset_default_buckets"]["300w"] == "anchor"
    assert report["anchor_count"] == report["counts"]["anchor"]
    assert "anchor" in report["quota_fill_rates"]
