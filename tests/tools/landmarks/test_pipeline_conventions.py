#!/usr/bin/env python3
"""Tests for shared landmark pipeline path and sidecar conventions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.landmarks.datasets.manifest_io import LandmarkSample
from lib.landmarks.pipeline_conventions import (
    SOURCE_GT_HARD,
    SOURCE_PRODUCTION_EXTRACTION,
    SOURCE_PRODUCTION_VALIDATED,
    expected_metadata_keys,
    load_resolver_metadata_sidecar,
    metadata_key,
    normalize_source_label,
    validate_resolver_metadata_for_samples,
)


def _sample(sample_id: str = "s1", *, face_index: int = 0) -> LandmarkSample:
    return LandmarkSample(
        sample_id=sample_id,
        image=f"{sample_id}.png",
        landmarks=f"{sample_id}.npy",
        dataset="fixture",
        condition="profile_left",
        metadata={"face_index": face_index},
    )


def _sidecar_row(sample_id: str = "s1", *, face_index: int = 0) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "face_index": face_index,
        "landmark_ensemble": {
            "runtime_bucket": "profile_left",
            "runtime_bucket_source": "stored_resolver_sidecar",
            "selected_candidate": "hrnet",
        },
    }


def test_source_label_aliases_are_canonical() -> None:
    assert normalize_source_label("gt") == SOURCE_GT_HARD
    assert normalize_source_label("production") == SOURCE_PRODUCTION_VALIDATED
    assert normalize_source_label("production-extract") == SOURCE_PRODUCTION_EXTRACTION


def test_expected_metadata_keys_include_face_index() -> None:
    assert expected_metadata_keys([_sample("a", face_index=2)]) == {metadata_key("a", 2)}


def test_load_resolver_metadata_sidecar_rejects_missing_sample_id(tmp_path: Path) -> None:
    path = tmp_path / "resolver_metadata.jsonl"
    path.write_text(json.dumps({"face_index": 0}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing sample_id"):
        load_resolver_metadata_sidecar(path)


def test_validate_resolver_metadata_rejects_missing_manifest_metadata() -> None:
    with pytest.raises(ValueError, match="missing 1 manifest key"):
        validate_resolver_metadata_for_samples(
            [_sample("s1", face_index=0)],
            {},
            source=SOURCE_GT_HARD,
            require_complete=True,
        )


def test_validate_resolver_metadata_rejects_wrong_sample_id() -> None:
    metadata = {metadata_key("wrong", 0): _sidecar_row("wrong", face_index=0)}

    with pytest.raises(ValueError, match="not present in manifest"):
        validate_resolver_metadata_for_samples(
            [_sample("s1", face_index=0)],
            metadata,
            source=SOURCE_GT_HARD,
            require_complete=False,
        )


def test_validate_resolver_metadata_rejects_mismatched_face_index() -> None:
    metadata = {metadata_key("s1", 1): _sidecar_row("s1", face_index=1)}

    with pytest.raises(ValueError, match="not present in manifest"):
        validate_resolver_metadata_for_samples(
            [_sample("s1", face_index=0)],
            metadata,
            source=SOURCE_GT_HARD,
            require_complete=False,
        )


def test_validate_resolver_metadata_requires_runtime_bucket() -> None:
    metadata = {
        metadata_key("s1", 0): {
            "sample_id": "s1",
            "face_index": 0,
            "landmark_ensemble": {"selected_candidate": "hrnet"},
        }
    }

    with pytest.raises(ValueError, match="has no runtime bucket"):
        validate_resolver_metadata_for_samples(
            [_sample("s1", face_index=0)],
            metadata,
            source=SOURCE_GT_HARD,
            require_complete=True,
        )
