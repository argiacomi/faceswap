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
    *,
    rejected_landmarks: T.Mapping[str, T.Sequence[int]] | None = None,
) -> Path:
    """Write one prediction overlay image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    overlay = make_debug_overlay(image, predictions)
    if rejected_landmarks:
        overlay = draw_rejected_landmarks(overlay, predictions, rejected_landmarks)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overlay)
    return output


def draw_rejected_landmarks(
    image: np.ndarray,
    predictions: T.Mapping[str, np.ndarray],
    rejected_landmarks: T.Mapping[str, T.Sequence[int]],
    *,
    color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draw cross marks on rejected model/landmark pairs."""
    output = np.array(image, copy=True)
    height, width = output.shape[:2]
    for name, indexes in rejected_landmarks.items():
        points = predictions.get(name)
        if points is None:
            continue
        array = np.asarray(points, dtype="float32")
        for index in indexes:
            if index < 0 or index >= len(array):
                continue
            x_val = int(round(float(array[index, 0])))
            y_val = int(round(float(array[index, 1])))
            if x_val < 0 or y_val < 0 or x_val >= width or y_val >= height:
                continue
            cv2.line(output, (x_val - 4, y_val - 4), (x_val + 4, y_val + 4), color, 1)
            cv2.line(output, (x_val - 4, y_val + 4), (x_val + 4, y_val - 4), color, 1)
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
