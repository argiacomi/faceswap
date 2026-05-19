#!/usr/bin/env python3
"""Export a landmark manifest from reviewed Faceswap production alignments."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import typing as T
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.align.alignments import Alignments
from lib.align.objects import FileAlignments
from lib.logger import get_loglevel

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "production_validated"
DEFAULT_SOURCE = "faceswap_extraction_plugin_reviewed"
DEFAULT_LABEL_QUALITY = "human_validated"
LOG_LEVELS = ("INFO", "VERBOSE", "DEBUG", "TRACE", "WARNING", "ERROR")
VALID_REVIEW_STATUSES = frozenset(("accepted", "rejected", "needs_review"))
VALID_ISSUE_TYPES = frozenset(
    (
        "",
        "bad_profile_alignment",
        "bad_roll",
        "occlusion",
        "wrong_face",
        "partial_face",
        "blur",
        "expression",
        "detector_bbox_bad",
    )
)


@dataclass(frozen=True)
class ReviewLabel:
    """Human review label for one production alignment sample."""

    sample_id: str
    image_path: str
    face_index: int
    review_status: str
    issue_type: str = ""
    notes: str = ""


def _json_safe(value: T.Any) -> T.Any:
    """Convert numpy/scalar payloads from alignments metadata to JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _safe_sample_id(frame_name: str, face_index: int) -> str:
    """Return a stable sample id from a Faceswap frame key and face index."""
    stem = Path(frame_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return f"{safe_stem or 'frame'}_face{face_index}"


def _resolve_image_path(images_dir: Path, frame_name: str) -> Path:
    """Resolve a frame key from alignments against the image root."""
    candidate = images_dir / frame_name
    if candidate.is_file():
        return candidate.resolve()
    basename = images_dir / Path(frame_name).name
    if basename.is_file():
        return basename.resolve()
    return candidate.resolve()


def _load_review_labels(path: Path | None) -> dict[str, ReviewLabel]:
    """Load optional review labels keyed by sample id."""
    if path is None or not path.is_file():
        logger.debug("No review labels file selected")
        return {}
    labels: dict[str, ReviewLabel] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "image_path", "face_index", "review_status"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"review labels missing columns: {sorted(missing)}")
        for row_num, row in enumerate(reader, start=2):
            sample_id = str(row.get("sample_id", "")).strip()
            status = str(row.get("review_status", "")).strip()
            issue_type = str(row.get("issue_type", "")).strip()
            if not sample_id:
                raise ValueError(f"review labels row {row_num} has empty sample_id")
            if status not in VALID_REVIEW_STATUSES:
                raise ValueError(
                    f"review labels row {row_num} has invalid review_status: {status!r}"
                )
            if issue_type not in VALID_ISSUE_TYPES:
                raise ValueError(f"review labels row {row_num} has invalid issue_type: {issue_type!r}")
            try:
                face_index = int(str(row.get("face_index", "")).strip())
            except ValueError as err:
                raise ValueError(f"review labels row {row_num} has invalid face_index") from err
            labels[sample_id] = ReviewLabel(
                sample_id=sample_id,
                image_path=str(row.get("image_path", "")).strip(),
                face_index=face_index,
                review_status=status,
                issue_type=issue_type,
                notes=str(row.get("notes", "")).strip(),
            )
            logger.trace(  # type:ignore[attr-defined]
                "Loaded review label: sample_id=%s face_index=%d status=%s issue_type=%s",
                sample_id,
                face_index,
                status,
                issue_type or "<none>",
            )
    logger.info("Loaded %d review labels from %s", len(labels), path)
    return labels


def _bbox(face: FileAlignments) -> list[float]:
    """Return face bbox in canonical ltrb order."""
    return [
        float(face.x),
        float(face.y),
        float(face.x + face.w),
        float(face.y + face.h),
    ]


def _normalizer(face: FileAlignments) -> float:
    """Return bbox diagonal normalizer for landmark metrics."""
    return float(np.hypot(float(face.w), float(face.h)))


def _review_for_sample(
    sample_id: str,
    image_path: Path,
    face_index: int,
    review_labels: dict[str, ReviewLabel],
) -> ReviewLabel:
    """Return human review label, defaulting to accepted for reviewed extraction exports."""
    label = review_labels.get(sample_id)
    if label is not None:
        return label
    return ReviewLabel(
        sample_id=sample_id,
        image_path=str(image_path),
        face_index=face_index,
        review_status="accepted",
    )


def _write_landmarks(output_dir: Path, sample_id: str, face: FileAlignments) -> str:
    """Write one reviewed landmark array and return its manifest-relative path."""
    landmarks_dir = output_dir / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    landmarks_path = landmarks_dir / f"{sample_id}.npy"
    np.save(str(landmarks_path), np.asarray(face.landmarks_xy, dtype="float32"))
    logger.trace("Wrote landmarks for %s to %s", sample_id, landmarks_path)  # type:ignore[attr-defined]
    return landmarks_path.relative_to(output_dir).as_posix()


def _metadata_summary(
    face: FileAlignments,
    review: ReviewLabel,
    frame_name: str,
    alignments_path: Path,
) -> dict[str, T.Any]:
    """Build compact manifest metadata for one sample."""
    ensemble_metadata = face.metadata.get("landmark_ensemble", {})
    bucket = ensemble_metadata.get("bucket") if isinstance(ensemble_metadata, dict) else None
    metadata: dict[str, T.Any] = {
        "review_status": review.review_status,
        "label_quality": DEFAULT_LABEL_QUALITY,
        "source": DEFAULT_SOURCE,
        "frame": frame_name,
        "face_index": review.face_index,
        "alignments_file": str(alignments_path.resolve()),
    }
    if review.issue_type:
        metadata["issue_type"] = review.issue_type
    if review.notes:
        metadata["notes"] = review.notes
    if bucket:
        metadata["runtime_bucket"] = str(bucket)
    if isinstance(ensemble_metadata, dict):
        for key in (
            "selected_candidate",
            "bucket",
            "roll_estimate",
            "yaw_estimate",
            "risk_route",
            "max_disagreement_px",
        ):
            if key in ensemble_metadata:
                metadata[f"landmark_ensemble_{key}"] = ensemble_metadata[key]
    return _json_safe(metadata)


def _log_audit_summary(audit: dict[str, T.Any]) -> None:
    """Log a compact manifest build summary."""
    skipped = T.cast(Counter[str], audit["skipped"])
    review_status_counts = T.cast(Counter[str], audit["review_status_counts"])
    condition_counts = T.cast(Counter[str], audit["condition_counts"])
    missing_images = T.cast(list[str], audit["missing_images"])
    missing_ensemble = T.cast(list[str], audit["missing_landmark_ensemble_metadata"])

    logger.info(
        "Manifest build summary: frames=%d faces=%d samples=%d skipped=%d",
        audit["frames_total"],
        audit["faces_total"],
        audit["samples_written"],
        sum(skipped.values()),
    )
    if review_status_counts:
        logger.info("Review status counts: %s", dict(sorted(review_status_counts.items())))
    if condition_counts:
        logger.info("Condition counts: %s", dict(sorted(condition_counts.items())))
    if skipped:
        logger.info("Skip breakdown: %s", dict(sorted(skipped.items())))
    if missing_images:
        logger.debug(
            "Missing images: count=%d first=%s",
            len(missing_images),
            missing_images[:10],
        )
    if missing_ensemble:
        logger.debug(
            "Missing landmark ensemble metadata: count=%d first=%s",
            len(missing_ensemble),
            missing_ensemble[:10],
        )


def build_manifest(
    images_dir: Path,
    alignments_path: Path,
    output_dir: Path,
    *,
    dataset_name: str = DEFAULT_DATASET,
    review_labels_path: Path | None = None,
    only_reviewed: str = "accepted",
    require_landmark_ensemble_metadata: bool = False,
) -> dict[str, T.Any]:
    """Export manifest, resolver JSONL sidecar and audit from Faceswap alignments."""
    if only_reviewed != "all" and only_reviewed not in VALID_REVIEW_STATUSES:
        raise ValueError(f"invalid --only-reviewed value: {only_reviewed!r}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not alignments_path.is_file():
        raise FileNotFoundError(f"alignments file not found: {alignments_path}")

    logger.info(
        "Building manifest: dataset=%s images=%s alignments=%s output=%s",
        dataset_name,
        images_dir,
        alignments_path,
        output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    default_review_labels = output_dir / "review_labels.csv"
    labels_path = review_labels_path or (default_review_labels if default_review_labels.is_file() else None)
    logger.debug(
        "Resolved paths: images=%s alignments=%s output=%s review_labels=%s",
        images_dir.resolve(),
        alignments_path.resolve(),
        output_dir.resolve(),
        labels_path.resolve() if labels_path else None,
    )
    review_labels = _load_review_labels(labels_path)
    if labels_path is None:
        logger.info("No review labels file found; unlabeled samples default to accepted")
    alignments = Alignments(str(alignments_path.parent), alignments_path.name)
    logger.info(
        "Loaded alignments: frames=%d faces=%d",
        alignments.frames_count,
        alignments.faces_count,
    )

    samples: list[dict[str, T.Any]] = []
    resolver_records: list[dict[str, T.Any]] = []
    audit: dict[str, T.Any] = {
        "schema_version": 1,
        "dataset": dataset_name,
        "images": str(images_dir.resolve()),
        "alignments": str(alignments_path.resolve()),
        "review_labels": str(labels_path.resolve()) if labels_path else None,
        "options": {
            "only_reviewed": only_reviewed,
            "require_landmark_ensemble_metadata": require_landmark_ensemble_metadata,
        },
        "frames_total": alignments.frames_count,
        "faces_total": alignments.faces_count,
        "samples_written": 0,
        "skipped": Counter(),
        "review_status_counts": Counter(),
        "condition_counts": Counter(),
        "issue_type_counts": Counter(),
        "missing_images": [],
        "missing_landmark_ensemble_metadata": [],
    }

    used_ids: Counter[str] = Counter()
    for frame_name, entry in sorted(alignments.data.items()):
        logger.trace(  # type:ignore[attr-defined]
            "Processing frame %s with %d faces", frame_name, len(entry.faces)
        )
        image_path = _resolve_image_path(images_dir, frame_name)
        if not image_path.is_file():
            logger.debug(
                "Skipping frame %s: resolved image missing at %s (faces=%d)",
                frame_name,
                image_path,
                len(entry.faces),
            )
            audit["missing_images"].append(str(image_path))
            audit["skipped"]["missing_image"] += len(entry.faces)
            continue

        for face_index, face in enumerate(entry.faces):
            sample_id = _safe_sample_id(frame_name, face_index)
            used_ids[sample_id] += 1
            if used_ids[sample_id] > 1:
                logger.debug(
                    "Sample id collision for %s; using suffix %d",
                    sample_id,
                    used_ids[sample_id],
                )
                sample_id = f"{sample_id}_{used_ids[sample_id]}"

            review = _review_for_sample(sample_id, image_path, face_index, review_labels)
            logger.trace(  # type:ignore[attr-defined]
                "Evaluating sample %s: frame=%s face_index=%d image=%s review_status=%s",
                sample_id,
                frame_name,
                face_index,
                image_path,
                review.review_status,
            )
            audit["review_status_counts"][review.review_status] += 1
            if review.issue_type:
                audit["issue_type_counts"][review.issue_type] += 1
            if only_reviewed != "all" and review.review_status != only_reviewed:
                logger.debug(
                    "Skipping sample %s: review_status=%s required=%s",
                    sample_id,
                    review.review_status,
                    only_reviewed,
                )
                audit["skipped"][f"review_status:{review.review_status}"] += 1
                continue

            ensemble_metadata = face.metadata.get("landmark_ensemble")
            logger.trace(  # type:ignore[attr-defined]
                "Sample %s landmark_ensemble metadata keys: %s",
                sample_id,
                sorted(ensemble_metadata) if isinstance(ensemble_metadata, dict) else "<missing>",
            )
            if require_landmark_ensemble_metadata and not isinstance(ensemble_metadata, dict):
                logger.debug(
                    "Skipping sample %s: missing landmark_ensemble metadata",
                    sample_id,
                )
                audit["missing_landmark_ensemble_metadata"].append(sample_id)
                audit["skipped"]["missing_landmark_ensemble_metadata"] += 1
                continue

            condition = "unknown"
            if isinstance(ensemble_metadata, dict):
                condition = str(ensemble_metadata.get("bucket") or condition)
            audit["condition_counts"][condition] += 1
            landmark_path = _write_landmarks(output_dir, sample_id, face)
            metadata = _metadata_summary(face, review, frame_name, alignments_path)
            sample = {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "condition": condition,
                "conditions": [condition],
                "source_schema": "2d_68",
                "image": str(image_path),
                "landmarks": landmark_path,
                "face_bbox": _bbox(face),
                "normalizer": _normalizer(face),
                "source": {"dataset": dataset_name, "source_id": sample_id},
                "metadata": metadata,
            }
            samples.append(sample)
            logger.trace(  # type:ignore[attr-defined]
                "Accepted sample %s: condition=%s bbox=%s normalizer=%.6f",
                sample_id,
                condition,
                sample["face_bbox"],
                sample["normalizer"],
            )
            resolver_records.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "face_index": face_index,
                    "review_status": review.review_status,
                    "condition": condition,
                    "landmark_ensemble": _json_safe(ensemble_metadata or {}),
                }
            )

    audit["samples_written"] = len(samples)
    _log_audit_summary(audit)
    manifest_payload = {
        "dataset": dataset_name,
        "metadata": {
            "review_status": only_reviewed,
            "label_quality": DEFAULT_LABEL_QUALITY,
            "source": DEFAULT_SOURCE,
        },
        "samples": sorted(samples, key=lambda item: str(item["sample_id"])),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "resolver_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for record in sorted(resolver_records, key=lambda item: str(item["sample_id"])):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    serializable_audit = dict(audit)
    for key in ("skipped", "review_status_counts", "condition_counts", "issue_type_counts"):
        serializable_audit[key] = dict(sorted(audit[key].items()))
    (output_dir / "audit.json").write_text(
        json.dumps(serializable_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.debug(
        "Wrote manifest artifacts: manifest=%s resolver=%s audit=%s",
        output_dir / "manifest.json",
        output_dir / "resolver_metadata.jsonl",
        output_dir / "audit.json",
    )
    return serializable_audit


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a production validated landmark manifest from Faceswap alignments."
    )
    parser.add_argument("--images", type=Path, required=True, help="Directory containing source images")
    parser.add_argument("--alignments", type=Path, required=True, help="Faceswap .fsa alignments file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET, help="Dataset name for manifest")
    parser.add_argument(
        "--review-labels",
        type=Path,
        default=None,
        help="Optional review_labels.csv. Defaults to output-dir/review_labels.csv when present.",
    )
    parser.add_argument(
        "--require-landmark-ensemble-metadata",
        action="store_true",
        help="Skip samples that do not carry landmark_ensemble metadata.",
    )
    parser.add_argument(
        "--only-reviewed",
        default="accepted",
        choices=sorted(VALID_REVIEW_STATUSES | {"all"}),
        help="Review status to export, or 'all'.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=get_loglevel(args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    audit = build_manifest(
        args.images,
        args.alignments,
        args.output_dir,
        dataset_name=args.dataset_name,
        review_labels_path=args.review_labels,
        only_reviewed=args.only_reviewed,
        require_landmark_ensemble_metadata=args.require_landmark_ensemble_metadata,
    )
    logger.info(
        "Wrote %d samples to %s",
        audit["samples_written"],
        args.output_dir / "manifest.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
