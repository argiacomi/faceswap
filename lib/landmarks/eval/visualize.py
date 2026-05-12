#!/usr/bin/env python3
"""Debug output for landmark evaluation."""

from __future__ import annotations

import csv
import json
import typing as T
from pathlib import Path

import cv2
import numpy as np

from lib.landmarks.visualization import make_debug_overlay


def write_overlay(
    image_path: str | Path,
    predictions: T.Mapping[str, np.ndarray],
    output_path: str | Path,
) -> Path:
    """Write one prediction overlay image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    overlay = make_debug_overlay(image, predictions)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overlay)
    return output


def write_debug_records(records: list[dict[str, T.Any]], output_dir: str | Path) -> None:
    """Write JSON and CSV debug records."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "debug_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record})
    with (out_dir / "debug_records.csv").open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_contact_sheet(
    image_paths: T.Sequence[str | Path],
    output_path: str | Path,
    *,
    columns: int = 4,
) -> Path:
    """Write a simple worst-first contact sheet."""
    if columns <= 0:
        raise ValueError("columns must be greater than zero")
    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(cv2.resize(image, (160, 160)))
    if not images:
        raise ValueError("no readable images for contact sheet")
    rows = []
    for start in range(0, len(images), columns):
        chunk = images[start : start + columns]
        while len(chunk) < columns:
            chunk.append(np.zeros_like(images[0]))
        rows.append(np.hstack(chunk))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))
    return output
