#!/usr/bin/env python3
"""Landmark quality dataset manifest helpers."""

from __future__ import annotations

import json
import typing as T
from pathlib import Path

SUPPORTED_DATASETS = ("wflw", "cofw")


def build_manifest(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    scenario: str = "default",
) -> Path:
    """Build a simple manifest from ``*.npy`` landmarks and matching images.

    This MVP supports local WFLW/COFW-style prepared folders where each sample
    has ``<id>.npy`` landmarks and an image with the same stem.
    """
    dataset_name = dataset.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset '{dataset}'")
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, T.Any]] = []
    for landmarks in sorted(src.glob("*.npy")):
        image = next(
            (
                src / f"{landmarks.stem}{ext}"
                for ext in (".png", ".jpg", ".jpeg")
                if (src / f"{landmarks.stem}{ext}").is_file()
            ),
            None,
        )
        if image is None:
            continue
        samples.append(
            {
                "sample_id": landmarks.stem,
                "dataset": dataset_name,
                "condition": scenario,
                "image": str(image.resolve()),
                "landmarks": str(landmarks.resolve()),
            }
        )
    manifest = out / "manifest.json"
    audit = out / "dataset_audit.json"
    manifest.write_text(
        json.dumps({"dataset": dataset_name, "samples": samples}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "total_entries": len(samples),
                "condition_counts": {scenario: len(samples)},
                "supported_datasets": SUPPORTED_DATASETS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
