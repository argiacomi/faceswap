#!/usr/bin/env python3
"""Identity-aware spatial tracking tests."""

from __future__ import annotations

import typing as T
from argparse import Namespace

import numpy as np

from tools.alignments.jobs import Spatial
from tools.alignments.media import AlignmentData


class _Face:
    """Minimal alignment face stub for Spatial tracking tests."""

    def __init__(
        self,
        *,
        x: float,
        identity: np.ndarray | None,
        landmarks_offset: float = 0.0,
    ) -> None:
        self.x = x
        self.y = 0.0
        self.w = 20.0
        self.h = 20.0
        self.landmarks_xy = np.zeros((68, 2), dtype=np.float32) + landmarks_offset
        self.identity: dict[str, np.ndarray] = (
            {} if identity is None else {"test": identity.astype(np.float32)}
        )


class _Frame:
    """Minimal frame alignment stub."""

    def __init__(self, faces: list[_Face]) -> None:
        self.faces = faces


class _Alignments:
    """Minimal alignments container for Spatial tracking tests."""

    def __init__(self, data: dict[str, _Frame]) -> None:
        self.data = data
        self.saved = False

    def save(self) -> None:
        self.saved = True


def _spatial(data: dict[str, _Frame]) -> Spatial:
    return Spatial(T.cast(AlignmentData, _Alignments(data)), Namespace())


def _identity(index: int) -> np.ndarray:
    value: np.ndarray = np.zeros(4, dtype=np.float32)
    value[index] = 1.0
    return value


def test_identity_tracks_follow_people_when_face_slots_swap() -> None:
    """Different identities should follow identity, not face index."""
    identity_a = _identity(0)
    identity_b = _identity(1)
    spatial = _spatial(
        {
            "frame_001.png": _Frame(
                [
                    _Face(x=0.0, identity=identity_a),
                    _Face(x=100.0, identity=identity_b),
                ]
            ),
            "frame_002.png": _Frame(
                [
                    _Face(x=0.0, identity=identity_b),
                    _Face(x=100.0, identity=identity_a),
                ]
            ),
        }
    )

    tracks = spatial._build_identity_instance_tracks()  # pylint:disable=protected-access
    paths = sorted(
        tuple((obs.frame_key, obs.face_index) for obs in track.observations) for track in tracks
    )

    assert paths == [
        (("frame_001.png", 0), ("frame_002.png", 1)),
        (("frame_001.png", 1), ("frame_002.png", 0)),
    ]


def test_same_identity_side_by_side_instances_do_not_collapse() -> None:
    """Same-person SBS duplicates should remain separate left/right instance tracks."""
    identity_a = _identity(0)
    spatial = _spatial(
        {
            "frame_001.png": _Frame(
                [
                    _Face(x=0.0, identity=identity_a),
                    _Face(x=100.0, identity=identity_a),
                ]
            ),
            "frame_002.png": _Frame(
                [
                    _Face(x=5.0, identity=identity_a),
                    _Face(x=105.0, identity=identity_a),
                ]
            ),
        }
    )

    tracks = spatial._build_identity_instance_tracks()  # pylint:disable=protected-access
    paths = sorted(tuple(obs.face_index for obs in track.observations) for track in tracks)

    assert paths == [(0, 0), (1, 1)]


def test_no_identity_embeddings_uses_slot_fallback() -> None:
    """Missing identity embeddings should leave the identity path disabled."""
    spatial = _spatial(
        {
            "frame_001.png": _Frame([_Face(x=0.0, identity=None)]),
            "frame_002.png": _Frame([_Face(x=5.0, identity=None)]),
        }
    )

    assert spatial._has_identity_embeddings() is False  # pylint:disable=protected-access


def test_identity_tracks_split_on_missing_frame_gaps() -> None:
    """Observed samples separated by missing frames should smooth as separate segments."""
    identity_a = _identity(0)
    spatial = _spatial(
        {
            "frame_001.png": _Frame([_Face(x=0.0, identity=identity_a)]),
            "frame_004.png": _Frame([_Face(x=5.0, identity=identity_a)]),
        }
    )

    tracks = spatial._build_identity_instance_tracks()  # pylint:disable=protected-access
    assert len(tracks) == 1

    segments = spatial._split_track_segments(tracks[0])  # pylint:disable=protected-access
    assert [[obs.frame_key for obs in segment.observations] for segment in segments] == [
        ["frame_001.png"],
        ["frame_004.png"],
    ]


def test_identity_tracks_keep_contiguous_observations_in_one_segment() -> None:
    """Adjacent observed frames should remain one smoothing segment."""
    identity_a = _identity(0)
    spatial = _spatial(
        {
            "frame_001.png": _Frame([_Face(x=0.0, identity=identity_a)]),
            "frame_002.png": _Frame([_Face(x=2.0, identity=identity_a)]),
            "frame_003.png": _Frame([_Face(x=4.0, identity=identity_a)]),
        }
    )

    tracks = spatial._build_identity_instance_tracks()  # pylint:disable=protected-access
    assert len(tracks) == 1

    segments = spatial._split_track_segments(tracks[0])  # pylint:disable=protected-access
    assert [[obs.frame_key for obs in segment.observations] for segment in segments] == [
        ["frame_001.png", "frame_002.png", "frame_003.png"],
    ]
