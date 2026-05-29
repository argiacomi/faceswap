#!/usr/bin/env python3
"""Tests for identity backfilling in lib/faceqa/coverage.py."""

from __future__ import annotations

import numpy as np

from lib.align.objects import AlignmentsEntry, FileAlignments
from lib.faceqa.coverage import (
    IdentityBackfiller,
    backfill_identity,
    majority_identity_model,
)
from lib.serializer import get_serializer


def _face(**identity: np.ndarray) -> FileAlignments:
    landmarks: np.ndarray = np.zeros((68, 2), dtype="float32")
    face = FileAlignments(x=0, y=0, w=80, h=90, landmarks_xy=landmarks)
    if identity:
        face.identity = {key: vec.astype("float32") for key, vec in identity.items()}
    return face


def _save_alignments(path, frames: dict[str, list[FileAlignments]]) -> None:
    get_serializer("compressed").save(
        str(path),
        {
            "__meta__": {"version": 2.4},
            "__data__": {
                frame: AlignmentsEntry(faces=faces).to_dict() for frame, faces in frames.items()
            },
        },
    )


class _FakeFrames:
    """Minimal frames loader that hands back a synthetic BGR uint8 image."""

    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load_image(self, frame: str) -> np.ndarray:
        self.loaded.append(frame)
        image: np.ndarray = np.full((128, 128, 3), 90, dtype=np.uint8)
        return image


class _FakePlugin:
    """Stand-in identity plugin matching FacePlugin's surface."""

    input_size = 64
    centering = "face"

    def __init__(self) -> None:
        self.calls = 0

    def load_model(self) -> object:  # pragma: no cover - never invoked here
        return object()

    def process(self, batch: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.ones((batch.shape[0], 512), dtype=np.float32)

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Majority-model selection
# ---------------------------------------------------------------------------


def test_majority_identity_model_picks_broadest_coverage() -> None:
    """The model used by the most faces wins; ties break by name."""
    faces = [
        _face(insightface=np.ones(512, dtype="float32")),
        _face(insightface=np.ones(512, dtype="float32")),
        _face(adaface=np.ones(512, dtype="float32")),
        _face(),  # no embeddings
    ]

    chosen, counts = majority_identity_model(faces)

    assert chosen == "insightface"
    assert counts == {"insightface": 2, "adaface": 1}


def test_majority_identity_model_handles_no_embeddings() -> None:
    chosen, counts = majority_identity_model([_face(), _face()])
    assert chosen is None
    assert counts == {}


def test_majority_identity_model_can_restrict_to_candidates() -> None:
    """Passing ``candidates`` filters out unsupported keys."""
    faces = [
        _face(t_face_irse50=np.ones(512, dtype="float32")),
        _face(insightface=np.ones(512, dtype="float32")),
    ]
    chosen, _ = majority_identity_model(faces, candidates={"insightface", "adaface"})
    assert chosen == "insightface"


# ---------------------------------------------------------------------------
# Per-face backfiller
# ---------------------------------------------------------------------------


def test_identity_backfiller_writes_vector_into_face(monkeypatch) -> None:
    """A successful backfill mutates ``face.identity[model]`` in place."""
    backfiller = IdentityBackfiller(_FakeFrames(), model="insightface")
    plugin = _FakePlugin()
    backfiller._plugin = plugin  # pylint:disable=protected-access
    face = _face()

    result = backfiller(face, frame="frame_000001.png", face_index=0)

    assert plugin.calls == 1
    assert result is not None
    assert "insightface" in face.identity
    np.testing.assert_allclose(face.identity["insightface"], result)


def test_identity_backfiller_self_disables_on_plugin_error() -> None:
    """Plugin exceptions disable the backfiller so per-frame loops do not cascade."""

    class _ExplodingPlugin(_FakePlugin):
        def process(self, batch: np.ndarray) -> np.ndarray:
            raise RuntimeError("boom")

    backfiller = IdentityBackfiller(_FakeFrames(), model="insightface")
    backfiller._plugin = _ExplodingPlugin()  # pylint:disable=protected-access
    face = _face()

    assert backfiller(face, frame="frame_000001.png", face_index=0) is None
    assert backfiller.disabled_reason is not None
    # Subsequent calls are no-ops.
    assert backfiller(face, frame="frame_000002.png", face_index=0) is None


def test_identity_backfiller_skips_when_frame_cannot_load() -> None:
    """A frame-load failure returns None without disabling the backfiller."""

    class _BadFrames:
        def load_image(self, frame: str) -> np.ndarray:
            raise FileNotFoundError(frame)

    backfiller = IdentityBackfiller(_BadFrames(), model="insightface")
    backfiller._plugin = _FakePlugin()  # pylint:disable=protected-access

    assert backfiller(_face(), frame="frame_000001.png", face_index=0) is None
    assert backfiller.disabled_reason is None


# ---------------------------------------------------------------------------
# backfill_identity (whole-file runner)
# ---------------------------------------------------------------------------


def test_backfill_identity_persists_new_vectors(tmp_path, monkeypatch) -> None:
    """Mutated alignments should be written back to disk on success."""
    alignments = tmp_path / "alignments.fsa"
    seeded = _face(insightface=np.ones(512, dtype="float32"))
    missing = _face()
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [seeded],
            "frame_000002.png": [missing],
        },
    )

    # Replace plugin loading so we never touch the real model.
    def _fake_loader(self, _model_name: str) -> object:  # noqa: ANN001
        self._plugin = _FakePlugin()
        return self._plugin

    monkeypatch.setattr(IdentityBackfiller, "_load_plugin", _fake_loader)

    report = backfill_identity(alignments, frames_loader=_FakeFrames())

    assert report.model == "insightface"
    assert report.total_faces == 2
    assert report.already_present == 1
    assert report.backfilled == 1
    assert report.persisted is True

    # Reload alignments to confirm the vector was persisted.
    raw = get_serializer("compressed").load(str(alignments))
    persisted = AlignmentsEntry.from_dict(raw["__data__"]["frame_000002.png"])
    assert "insightface" in persisted.faces[0].identity
    assert persisted.faces[0].identity["insightface"].shape == (512,)


def test_backfill_identity_no_op_when_nothing_to_fill(tmp_path) -> None:
    """An alignments file with full coverage produces a no-op report."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [_face(insightface=np.ones(512, dtype="float32"))],
        },
    )

    report = backfill_identity(alignments, frames_loader=_FakeFrames())

    assert report.model == "insightface"
    assert report.backfilled == 0
    assert report.already_present == 1
    assert report.persisted is False


def test_backfill_identity_reports_unsupported_model(tmp_path) -> None:
    """When every face uses an unsupported model, the report flags it cleanly."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [
                _face(t_face_irse50=np.ones(512, dtype="float32")),
            ],
        },
    )

    report = backfill_identity(alignments, frames_loader=_FakeFrames())

    assert report.model == "t_face_irse50"
    assert report.disabled_reason is not None
    assert "unsupported" in report.disabled_reason
    assert report.backfilled == 0
    assert report.persisted is False


def test_backfill_identity_reports_no_identity_embeddings(tmp_path) -> None:
    """A faceset with zero identity vectors is reported but not an error."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(alignments, {"frame_000001.png": [_face()]})

    report = backfill_identity(alignments, frames_loader=_FakeFrames())

    assert report.model is None
    assert report.disabled_reason and "no identity" in report.disabled_reason
    assert report.persisted is False


def test_backfill_identity_explicit_model_override(tmp_path, monkeypatch) -> None:
    """Passing ``model=...`` bypasses the majority-model selection."""
    alignments = tmp_path / "alignments.fsa"
    _save_alignments(
        alignments,
        {
            "frame_000001.png": [_face(adaface=np.ones(512, dtype="float32"))],
            "frame_000002.png": [_face()],
        },
    )

    def _fake_loader(self, _model_name: str) -> object:  # noqa: ANN001
        self._plugin = _FakePlugin()
        return self._plugin

    monkeypatch.setattr(IdentityBackfiller, "_load_plugin", _fake_loader)

    report = backfill_identity(alignments, frames_loader=_FakeFrames(), model="insightface")

    # Even the seeded face has no "insightface" key (it had "adaface"), so the
    # explicit override should backfill both faces.
    assert report.model == "insightface"
    assert report.backfilled == 2
    assert report.already_present == 0
