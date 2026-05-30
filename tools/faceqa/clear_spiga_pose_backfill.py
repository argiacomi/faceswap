#!/usr/bin/env python3
"""Clear SPIGA pose metadata from .fsa files so FaceQA re-runs SPIGA backfill.

Dry-run by default. Use --write to modify files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from lib.faceqa.coverage import load_alignments_envelope, save_alignments_envelope

POSE_RECORD_FIELDS = {
    # Selected pose fields that can cause _valid_spiga_pose() to skip backfill
    "yaw",
    "pitch",
    "roll",
    "pose_source",
    "pose_model",
    "pose_confidence",
    # Native SPIGA pose fields
    "spiga_yaw",
    "spiga_pitch",
    "spiga_roll",
    "spiga_pose_source",
    "spiga_pose_model",
    "spiga_pose_error",
    # Delta/confidence fields derived from stale SPIGA pose
    "pose_delta_yaw",
    "pose_delta_pitch",
    "pose_delta_roll",
    "pose_max_abs_delta",
}

# Pre-sorted view used by ``clear_face_metadata`` so the iteration order is
# stable AND the sort cost is paid once at import time rather than per face
# (issue #196).
_SORTED_POSE_FIELDS: tuple[str, ...] = tuple(sorted(POSE_RECORD_FIELDS))


def _remove_if_empty(parent: dict[str, Any], key: str) -> None:
    value = parent.get(key)
    if isinstance(value, dict) and not value:
        parent.pop(key, None)


def _source_is_spiga(payload: dict[str, Any]) -> bool:
    source = str(payload.get("source") or payload.get("pose_source") or "").lower()
    model = str(payload.get("model") or payload.get("pose_model") or "").lower()
    return source in {"spiga", "spiga_backfill"} or model == "spiga"


def clear_face_metadata(metadata: dict[str, Any]) -> int:
    """Remove SPIGA pose locations from one face metadata dict.

    Returns the number of removed/changed fields.
    """
    changes = 0

    # 1. Native SPIGA extractor metadata: metadata["spiga"]["pose"]
    spiga = metadata.get("spiga")
    if isinstance(spiga, dict) and isinstance(spiga.get("pose"), dict):
        spiga.pop("pose", None)
        changes += 1
        _remove_if_empty(metadata, "spiga")

    # 2. FaceQA backfill metadata: metadata["faceqa"]["pose"]
    faceqa = metadata.get("faceqa")
    if isinstance(faceqa, dict):
        if isinstance(faceqa.get("pose"), dict):
            faceqa.pop("pose", None)
            changes += 1

        # Remove stale selected/native pose fields that can make
        # _valid_spiga_pose() return True before backfill runs.
        for key in _SORTED_POSE_FIELDS:
            if key in faceqa:
                faceqa.pop(key, None)
                changes += 1

        _remove_if_empty(metadata, "faceqa")

    # 3. Older/direct metadata["pose"] style. Only remove if it is SPIGA-like.
    pose = metadata.get("pose")
    if isinstance(pose, dict) and _source_is_spiga(pose):
        metadata.pop("pose", None)
        changes += 1

    # 4. Ensemble adapter pose metadata:
    # metadata["landmark_ensemble"]["adapter_pose"]["spiga"]
    ensemble = metadata.get("landmark_ensemble")
    if isinstance(ensemble, dict):
        adapter_pose = ensemble.get("adapter_pose")
        if isinstance(adapter_pose, dict) and "spiga" in adapter_pose:
            adapter_pose.pop("spiga", None)
            changes += 1
            if not adapter_pose:
                ensemble.pop("adapter_pose", None)
        _remove_if_empty(metadata, "landmark_ensemble")

    return changes


def backup_path(path: Path) -> Path:
    """Return a non-colliding backup path next to ``path``.

    ``path.with_suffix(path.suffix + ".bak")`` is intentional: it preserves
    the original extension (``foo.fsa`` becomes ``foo.fsa.bak``, not
    ``foo.bak``) so the relationship to the source file is obvious on disk.
    Subsequent collisions append a numeric suffix (``foo.fsa.bak1``,
    ``foo.fsa.bak2``, ...) — verified per issue #196.
    """
    candidate = path.with_suffix(path.suffix + ".bak")
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        candidate = path.with_suffix(path.suffix + f".bak{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def process_file(path: Path, *, write: bool, backup: bool) -> tuple[int, int]:
    raw, entries = load_alignments_envelope(path)

    changed_faces = 0
    removed_fields = 0

    for entry in entries.values():
        for face in entry.faces:
            metadata = getattr(face, "metadata", None)
            if not isinstance(metadata, dict):
                continue

            removed = clear_face_metadata(metadata)
            if removed:
                changed_faces += 1
                removed_fields += removed

    if write and removed_fields:
        if backup:
            dest = backup_path(path)
            shutil.copy2(path, dest)
            print(f"  backup: {dest}")
        save_alignments_envelope(path, raw, entries)

    return changed_faces, removed_fields


def find_fsa_files(root: Path, *, recursive: bool) -> list[Path]:
    pattern = "**/*.fsa" if recursive else "*.fsa"
    return sorted(path for path in root.glob(pattern) if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear SPIGA pose metadata from .fsa files to force FaceQA SPIGA backfill."
    )
    parser.add_argument("directory", type=Path, help="Directory containing .fsa files.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Modify files in place. Without this flag, runs as dry-run.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top-level directory.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups when writing.",
    )

    args = parser.parse_args()
    root = args.directory.expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    files = find_fsa_files(root, recursive=not args.no_recursive)
    if not files:
        print(f"No .fsa files found under: {root}")
        return 0

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"{mode}: scanning {len(files)} .fsa file(s) under {root}")

    total_faces = 0
    total_fields = 0

    for path in files:
        try:
            changed_faces, removed_fields = process_file(
                path,
                write=args.write,
                backup=not args.no_backup,
            )
        except Exception as err:  # noqa: BLE001
            print(f"[ERROR] {path}: {err}")
            continue

        total_faces += changed_faces
        total_fields += removed_fields

        status = "updated" if args.write and removed_fields else "would update"
        if not removed_fields:
            status = "no changes"

        print(f"{path}: {status}; changed_faces={changed_faces}, removed_fields={removed_fields}")

    print(
        f"Done. files={len(files)}, changed_faces={total_faces}, "
        f"removed_fields={total_fields}, write={args.write}"
    )

    if not args.write:
        print("Dry-run only. Re-run with --write to modify files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
