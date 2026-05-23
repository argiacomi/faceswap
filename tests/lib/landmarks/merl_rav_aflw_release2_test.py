#!/usr/bin/env python3
"""Tests for MERL-RAV manifest building over AFLW release-2 cropped images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lib.landmarks.datasets.merl_rav import (
    _build_aflw_release2_index,
    _estimate_crop_origin_xy,
    _landmarks68_to_5anchors_xy,
    _parse_pts_signed,
    _select_best_crop_candidate,
    _source_valid_xy,
    _translate_to_crop,
    _visibility_for_crop,
    build_merl_rav_manifest,
)

IMAGE00002_PTS = """\
version: 1
n_points:  68
{
-128.31632653061226 -122.44897959183675
-127.80612244897961 -153.06122448979596
135.45918367346943 187.2448979591837
-140.0510204081633 -212.75510204081635
-146.68367346938777 -243.36734693877554
-156.3775510204082 -270.40816326530614
-167.60204081632656 -298.97959183673476
185.96938775510205 315.8163265306123
213.01020408163268 330.10204081632656
242.60204081632656 325.00000000000006
267.0918367346939 306.1224489795919
-289.030612244898 -286.2244897959184
-302.80612244897964 -260.20408163265313
-313.52040816326536 -232.65306122448985
324.2346938775511 200.51020408163268
-334.9489795918368 -171.42857142857147
-340.56122448979596 -139.2857142857143
147.41253644314875 140.3061224489796
-164.54081632653066 -138.11953352769683
181.30466472303212 142.128279883382
-197.33965014577262 -149.78134110787175
-210.45918367346945 -159.2565597667639
253.09766763848404 160.71428571428578
268.7682215743441 155.61224489795924
284.43877551020415 151.60349854227408
301.5670553935861 148.3236151603499
319.0597667638485 154.15451895043734
229.3367346938776 179.5918367346939
227.29591836734699 202.55102040816328
225.25510204081635 218.8775510204082
222.7040816326531 237.75510204081635
202.8061224489796 236.73469387755105
210.45918367346943 242.34693877551024
221.6836734693878 247.44897959183677
232.3979591836735 244.8979591836735
242.60204081632656 243.36734693877554
164.1763848396502 160.3498542274053
177.66034985422746 153.06122448979596
-194.05976676384844 -157.79883381924202
205.3571428571429 170.91836734693885
189.68658892128283 171.64723032069978
174.74489795918373 169.46064139941694
253.82653061224497 177.8425655976677
267.2063723448564 168.1591003748439
282.0439400249897 166.0766347355269
298.6516034985424 173.4693877551021
286.98979591836746 181.1224489795919
270.95481049562693 181.1224489795919
184.9489795918368 260.308204914619
196.66284881299467 257.44481466055817
208.1164098292379 259.78758850478977
216.4462723865057 264.7334443981675
225.55705955851735 263.1715951686798
240.9152436484799 263.69221157850905
250.5466472303208 270.19991670137455
244.81986672219918 285.2977925864224
233.10599750104134 293.6276551436903
215.66534777176184 296.23073719283644
198.22469804248237 291.2848812994587
187.29175343606838 278.790087463557
187.03144523115375 261.6097459391921
206.03394418992093 267.0762182423991
215.66534777176184 270.46022490628917
225.55705955851738 270.7205331112038
248.72448979591846 272.02207413577685
227.1189087880051 280.0916284881301
216.4462723865057 280.8725531028739
207.07517700957942 277.4885464389839
}
"""

GT5_XY_IMAGE00002 = np.array(
    [
        [127.0, 140.0],
        [219.0, 149.0],
        [165.0, 223.0],
        [128.0, 241.0],
        [195.0, 251.0],
    ],
    dtype=np.float64,
)
GT5_YX_IMAGE00002 = GT5_XY_IMAGE00002[:, ::-1]
EXPECTED_ORIGIN = np.array([57, 19], dtype=int)
EXPECTED_HW = (338, 337)
EXPECTED_SCORE_VISIBLE = 53
EXPECTED_COORDINATE_VALID = 68
EXPECTED_SCORE_INVISIBLE_INDICES = [0, 1, 3, 4, 5, 6, 11, 12, 13, 15, 16, 18, 20, 21, 38]


def _write_release2_fixture(
    base: Path,
    *,
    relative: str = "0/image00002_46712.jpg",
    gt5_yx: np.ndarray = GT5_YX_IMAGE00002,
    mat_hw: tuple[int, int] = EXPECTED_HW,
    image_hw: tuple[int, int] | None = None,
) -> Path:
    """Write a single-row AFLW release-2 cropped fixture."""
    cv2 = pytest.importorskip("cv2")
    scipy_io = pytest.importorskip("scipy.io")
    base.mkdir(parents=True, exist_ok=True)
    output_dir = base / "output"
    image_path = output_dir / relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image_hw or mat_hw
    image = np.zeros((height, width, 3), dtype="uint8")
    image[..., 1] = 128
    assert cv2.imwrite(str(image_path), image)
    (base / "aflw_train_images.txt").write_text(relative + "\n", encoding="utf-8")
    scipy_io.savemat(
        str(base / "aflw_train_keypoints.mat"),
        {
            "gt": gt5_yx.reshape(1, 5, 2).astype(np.float64),
            "hw": np.asarray([[mat_hw[0], mat_hw[1]]], dtype=np.float64),
        },
    )
    return image_path


def test_parse_pts_signed_preserves_negative_coordinates() -> None:
    """The signed parser keeps externally-occluded landmarks for crop translation."""
    signed = _parse_pts_signed_text(IMAGE00002_PTS)
    assert signed.shape == (68, 2)
    assert signed[0, 0] < 0 and signed[0, 1] < 0
    assert signed[2, 0] > 0 and signed[2, 1] > 0


def test_image00002_anchor_and_crop_origin_match_proof_case(tmp_path: Path) -> None:
    """The 5-anchor estimator reproduces the documented [57, 19] crop origin."""
    pts_path = tmp_path / "image00002.pts"
    pts_path.write_text(IMAGE00002_PTS, encoding="utf-8")
    signed = _parse_pts_signed(pts_path)
    visibility, src, score_visible = _visibility_for_crop(signed)
    assert visibility.count("visible") == EXPECTED_SCORE_VISIBLE
    assert int(score_visible.sum()) == EXPECTED_SCORE_VISIBLE
    assert sum(1 for v in visibility if v in ("externally_occluded", "self_occluded")) == 15
    src5 = _landmarks68_to_5anchors_xy(src)
    origin, used, residuals = _estimate_crop_origin_xy(src5, GT5_XY_IMAGE00002, min_anchors=3)
    assert origin.tolist() == EXPECTED_ORIGIN.tolist()
    assert used == 5
    assert residuals.shape == (5, 2)
    translated, src_valid, in_crop = _translate_to_crop(src, origin, EXPECTED_HW)
    assert int(src_valid.sum()) == EXPECTED_COORDINATE_VALID
    assert int(in_crop.sum()) == EXPECTED_COORDINATE_VALID
    invisible_indices = [int(i) for i, ok in enumerate(score_visible) if not bool(ok)]
    assert invisible_indices == EXPECTED_SCORE_INVISIBLE_INDICES
    valid_pts = translated[src_valid]
    assert np.all(valid_pts[:, 0] >= 0)
    assert np.all(valid_pts[:, 1] >= 0)
    assert np.all(valid_pts[:, 0] < EXPECTED_HW[1])
    assert np.all(valid_pts[:, 1] < EXPECTED_HW[0])


def test_build_merl_rav_aflw_release2_manifest_proof_case(tmp_path: Path) -> None:
    """End-to-end builder run reproduces the proof case manifest entry."""
    labels_root = tmp_path / "labels_repo" / "merl_rav_labels"
    label_path = labels_root / "frontal" / "trainset" / "labels" / "image00002.pts"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(IMAGE00002_PTS, encoding="utf-8")
    release2 = tmp_path / "aflw_release-2"
    _write_release2_fixture(release2)

    manifest = build_merl_rav_manifest(
        tmp_path / "out",
        aflw_release2_dir=release2,
        source_dir=tmp_path / "labels_repo",
        splits=("train",),
        validate_hw=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dataset"] == "merl-rav"
    assert len(payload["samples"]) == 1
    sample = payload["samples"][0]
    metadata = sample["metadata"]
    assert metadata["aflw_image_source"] == "aflw_release2_cropped"
    assert metadata["aflw_release2_split"] == "train"
    assert metadata["aflw_release2_row"] == 0
    assert metadata["crop_origin_xy"] == [57, 19]
    assert metadata["crop_height"] == EXPECTED_HW[0]
    assert metadata["crop_width"] == EXPECTED_HW[1]
    assert metadata["landmark_source_valid_count"] == EXPECTED_COORDINATE_VALID
    assert metadata["landmark_in_crop_count"] == EXPECTED_COORDINATE_VALID
    assert metadata["landmark_score_visible_count"] == EXPECTED_SCORE_VISIBLE
    invalid_from_mask = [
        idx for idx, ok in enumerate(metadata["landmark_score_visibility_mask"]) if not ok
    ]
    assert invalid_from_mask == EXPECTED_SCORE_INVISIBLE_INDICES
    assert sample["image"].endswith("image00002_46712.jpg")
    assert sample["sample_id"] == "frontal_trainset_labels_image00002__image00002_46712"

    landmarks_path = tmp_path / "out" / sample["landmarks"]
    landmarks = np.load(landmarks_path)
    assert landmarks.shape == (68, 2)
    assert np.all(np.isfinite(landmarks))
    valid_mask = np.asarray(metadata["landmark_coordinate_valid_mask"], dtype=bool)
    valid_pts = landmarks[valid_mask]
    assert np.all(valid_pts[:, 0] >= 0)
    assert np.all(valid_pts[:, 1] >= 0)
    assert np.all(valid_pts[:, 0] < EXPECTED_HW[1])
    assert np.all(valid_pts[:, 1] < EXPECTED_HW[0])
    assert valid_mask.all()


def test_build_release2_skips_when_too_few_anchors(tmp_path: Path) -> None:
    """Samples with insufficient valid anchors are skipped, not silently zeroed."""
    labels_root = tmp_path / "labels_repo" / "merl_rav_labels"
    label_path = labels_root / "frontal" / "trainset" / "labels" / "image00002.pts"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(IMAGE00002_PTS, encoding="utf-8")
    release2 = tmp_path / "aflw_release-2"
    _write_release2_fixture(release2)

    with pytest.raises(FileNotFoundError, match="No MERL-RAV/AFLW release-2 crop pairs"):
        build_merl_rav_manifest(
            tmp_path / "out_strict",
            aflw_release2_dir=release2,
            source_dir=tmp_path / "labels_repo",
            splits=("train",),
            min_anchors=99,
            validate_hw=True,
        )


def test_build_release2_validates_hw_mismatch(tmp_path: Path) -> None:
    """An image with hw mismatch against mat['hw'] is audited and skipped."""
    labels_root = tmp_path / "labels_repo" / "merl_rav_labels"
    label_path = labels_root / "frontal" / "trainset" / "labels" / "image00002.pts"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(IMAGE00002_PTS, encoding="utf-8")
    release2 = tmp_path / "aflw_release-2"
    _write_release2_fixture(release2, mat_hw=EXPECTED_HW, image_hw=(200, 200))

    with pytest.raises(FileNotFoundError, match="No MERL-RAV/AFLW release-2 crop pairs"):
        build_merl_rav_manifest(
            tmp_path / "out_hw",
            aflw_release2_dir=release2,
            source_dir=tmp_path / "labels_repo",
            splits=("train",),
            validate_hw=True,
        )


def test_release2_index_matches_imageNNNNN_stem(tmp_path: Path) -> None:
    """The release-2 index keys candidates by the AFLW source ``imageNNNNN`` stem."""
    release2 = tmp_path / "aflw_release-2"
    _write_release2_fixture(release2)
    index, crops_root, counts = _build_aflw_release2_index(release2, splits=("train",))
    assert counts == {"train": 1}
    assert "image00002" in index
    entry = index["image00002"][0]
    assert entry["relative"] == "0/image00002_46712.jpg"
    assert entry["stem"] == "image00002_46712"
    assert entry["absolute"].is_relative_to(crops_root)
    assert np.allclose(entry["gt5_xy"], GT5_XY_IMAGE00002)
    assert entry["hw"] == EXPECTED_HW


def _parse_pts_signed_text(text: str) -> np.ndarray:
    """Test helper to parse signed .pts directly from string data."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".pts", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return _parse_pts_signed(path)
    finally:
        path.unlink(missing_ok=True)


def test_source_valid_mask_rejects_negative_and_nan() -> None:
    """``_source_valid_xy`` flags only finite, non-negative landmarks."""
    pts = np.array(
        [
            [1.0, 2.0],
            [-1.0, -1.0],
            [-3.0, 5.0],
            [np.nan, 4.0],
            [10.0, np.inf],
        ],
        dtype=np.float64,
    )
    mask = _source_valid_xy(pts)
    assert mask.tolist() == [True, False, False, False, False]


def test_select_best_candidate_uses_min_residual(tmp_path: Path) -> None:
    """When multiple crops share a source stem, the lowest-residual fit wins."""
    pts_path = tmp_path / "image00002.pts"
    pts_path.write_text(IMAGE00002_PTS, encoding="utf-8")
    signed = _parse_pts_signed(pts_path)
    _, src, _score_visible = _visibility_for_crop(signed)
    src5 = _landmarks68_to_5anchors_xy(src)
    good = {"gt5_xy": GT5_XY_IMAGE00002, "stem": "good"}
    bad = {
        "gt5_xy": GT5_XY_IMAGE00002 + np.array([1000.0, 1000.0]),
        "stem": "bad",
    }
    chosen, origin, _, used = _select_best_crop_candidate(src5, [bad, good], min_anchors=3)
    assert chosen is good
    assert origin.tolist() == EXPECTED_ORIGIN.tolist()
    assert used == 5
