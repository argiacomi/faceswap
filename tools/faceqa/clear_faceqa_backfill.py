#!/usr/bin/env python3
"""Clear FaceQA backfill metadata from .fsa files so FaceQA re-runs the
matching backfill stage on the next ``faceqa coverage`` invocation.

Two clear modes, controlled by CLI flags:

* ``--pose``: removes the SPIGA-derived pose backfill metadata
  (``metadata["spiga"]["pose"]``, ``metadata["faceqa"]["pose"]``, the
  top-level ``metadata["pose"]`` when it is SPIGA-flavoured, the
  ``landmark_ensemble.adapter_pose.spiga`` block, and the FaceQA
  envelope's selected/native pose fields). Forces SPIGA backfill to
  rerun on the next coverage pass.
* ``--identity``: removes the identity embeddings stored on
  ``face.identity`` AND the FaceQA identity quality decisions stored
  under ``metadata["faceqa"]["identity"]`` (and the flat
  ``identity_*`` fields when present). Forces identity backfill +
  identity quality classification to rerun.

``--all`` is a shortcut for both. Dry-run by default; pass ``--write``
to modify files in place. A ``.bak`` is written next to each modified
file unless ``--no-backup`` is supplied.

Renamed from ``clear_spiga_pose_backfill.py`` so the tool can clear
identity backfills too, not just SPIGA pose.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.faceqa.coverage import load_alignments_envelope, save_alignments_envelope

# ---------------------------------------------------------------------------
# Field constants
# ---------------------------------------------------------------------------

POSE_RECORD_FIELDS: frozenset[str] = frozenset(
    {
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
)

IDENTITY_RECORD_FIELDS: frozenset[str] = frozenset(
    {
        # Quality / decision fields written by compute_identity_quality
        "identity_score",
        "identity_cluster",
        "identity_model",
        "identity_consensus",
        "identity_threshold_used",
        "identity_quality_flag",
        "identity_final_decision",
        # Verifier-pipeline fields (kept here so the verifier rerun is
        # equally forced by --identity).
        "identity_primary_model",
        "identity_verifier_model",
        "identity_primary_score",
        "identity_verifier_score",
        "identity_agreement",
        "identity_decision_reason",
        "identity_verifier_trigger",
    }
)

# Pre-sorted tuples used by the per-face clear helpers so the iteration
# order is stable AND the sort cost is paid once at import time rather
# than per face (issue #196).
_SORTED_POSE_FIELDS: tuple[str, ...] = tuple(sorted(POSE_RECORD_FIELDS))
_SORTED_IDENTITY_FIELDS: tuple[str, ...] = tuple(sorted(IDENTITY_RECORD_FIELDS))


# ---------------------------------------------------------------------------
# Per-face clear helpers
# ---------------------------------------------------------------------------


def _remove_if_empty(parent: dict[str, Any], key: str) -> None:
    """Pop ``parent[key]`` when the value is an empty dict."""
    value = parent.get(key)
    if isinstance(value, dict) and not value:
        parent.pop(key, None)


def _source_is_spiga(payload: dict[str, Any]) -> bool:
    """Return ``True`` if a generic ``metadata["pose"]`` block looks SPIGA."""
    source = str(payload.get("source") or payload.get("pose_source") or "").lower()
    model = str(payload.get("model") or payload.get("pose_model") or "").lower()
    return source in {"spiga", "spiga_backfill"} or model == "spiga"


def clear_pose_metadata(metadata: dict[str, Any]) -> int:
    """Remove every SPIGA-derived pose location from one face's metadata.

    Returns the number of removed / changed fields.
    """
    changes = 0

    # 1. Native SPIGA extractor metadata: metadata["spiga"]["pose"]
    spiga = metadata.get("spiga")
    if isinstance(spiga, dict) and isinstance(spiga.get("pose"), dict):
        spiga.pop("pose", None)
        changes += 1
        _remove_if_empty(metadata, "spiga")

    # 2. FaceQA backfill metadata: metadata["faceqa"]["pose"] + flat pose
    # fields stored at the envelope root.
    faceqa = metadata.get("faceqa")
    if isinstance(faceqa, dict):
        if isinstance(faceqa.get("pose"), dict):
            faceqa.pop("pose", None)
            changes += 1
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


def clear_identity_metadata(face: Any, metadata: dict[str, Any]) -> int:
    """Remove identity embeddings + FaceQA identity quality decisions.

    Returns the number of removed / changed fields. Each face.identity entry
    counts as one change so the per-file summary reports a meaningful number
    of changed faces. ``face`` is the ``FileAlignments`` object itself
    (``face.identity`` is the embedding dict); ``metadata`` is its
    ``face.metadata`` payload.
    """
    changes = 0

    # 1. face.identity — the actual embeddings.
    identity = getattr(face, "identity", None)
    if isinstance(identity, dict) and identity:
        changes += len(identity)
        identity.clear()

    # 2. FaceQA identity-quality decisions: metadata["faceqa"]["identity"]
    # plus the flat ``identity_*`` fields that older records used.
    faceqa = metadata.get("faceqa")
    if isinstance(faceqa, dict):
        if isinstance(faceqa.get("identity"), dict):
            faceqa.pop("identity", None)
            changes += 1
        for key in _SORTED_IDENTITY_FIELDS:
            if key in faceqa:
                faceqa.pop(key, None)
                changes += 1
        _remove_if_empty(metadata, "faceqa")

    return changes


def clear_face(
    face: Any,
    *,
    clear_pose: bool,
    clear_identity: bool,
) -> int:
    """Run the requested clears against one ``FileAlignments`` instance.

    Returns the total number of removed / changed fields across the
    requested clear modes. Faces with no ``metadata`` dict are skipped.
    """
    metadata = getattr(face, "metadata", None)
    if not isinstance(metadata, dict):
        return 0

    changes = 0
    if clear_pose:
        changes += clear_pose_metadata(metadata)
    if clear_identity:
        changes += clear_identity_metadata(face, metadata)
    return changes


# ---------------------------------------------------------------------------
# Filesystem driver
# ---------------------------------------------------------------------------


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


def process_file(
    path: Path,
    *,
    write: bool,
    backup: bool,
    clear_pose: bool,
    clear_identity: bool,
) -> tuple[int, int]:
    """Apply the requested clears to one ``.fsa`` file.

    Returns ``(changed_faces, removed_fields)`` for the per-file summary.
    """
    raw, entries = load_alignments_envelope(path)

    changed_faces = 0
    removed_fields = 0

    for entry in entries.values():
        for face in entry.faces:
            removed = clear_face(
                face,
                clear_pose=clear_pose,
                clear_identity=clear_identity,
            )
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
    """Return the sorted list of ``.fsa`` files under ``root``."""
    pattern = "**/*.fsa" if recursive else "*.fsa"
    return sorted(path for path in root.glob(pattern) if path.is_file())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clear FaceQA backfill metadata (SPIGA pose and/or identity) from "
            ".fsa files so the next FaceQA coverage run re-runs the matching "
            "backfill stage. Dry-run by default."
        )
    )
    parser.add_argument("directory", type=Path, help="Directory containing .fsa files.")
    parser.add_argument(
        "--pose",
        action="store_true",
        help=(
            "Clear SPIGA pose backfill metadata (native SPIGA + FaceQA "
            "envelope + ensemble adapter pose entries)."
        ),
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help=(
            "Clear identity embeddings (face.identity) AND FaceQA identity "
            "quality decisions (metadata['faceqa']['identity'] + flat "
            "identity_* fields)."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Shortcut for --pose --identity.",
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    clear_pose = args.pose or args.all
    clear_identity = args.identity or args.all
    if not (clear_pose or clear_identity):
        parser.error("must specify at least one of --pose, --identity, or --all")

    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    files = find_fsa_files(root, recursive=not args.no_recursive)
    if not files:
        print(f"No .fsa files found under: {root}")
        return 0

    mode = "WRITE" if args.write else "DRY-RUN"
    targets = []
    if clear_pose:
        targets.append("pose")
    if clear_identity:
        targets.append("identity")
    print(
        f"{mode}: clearing {'+'.join(targets)} backfill across "
        f"{len(files)} .fsa file(s) under {root}"
    )

    total_faces = 0
    total_fields = 0

    for path in files:
        try:
            changed_faces, removed_fields = process_file(
                path,
                write=args.write,
                backup=not args.no_backup,
                clear_pose=clear_pose,
                clear_identity=clear_identity,
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
