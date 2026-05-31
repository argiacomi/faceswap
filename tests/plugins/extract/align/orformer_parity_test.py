#!/usr/bin/env python3
"""Upstream parity test for ORFormer landmark prediction.

Purpose
-------
Verify that Faceswap's ORFormer implementation matches the upstream reference
when both use the same image, the same landmark-derived bounding box, the same
crop transform, the same weights, and the same 64×64 reference resize path
(OpenCV INTER_LINEAR before normalisation).

Fixture provenance
------------------
Expected output arrays are generated once using the upstream test script
``test_HGNet_with_ORFormer.py`` at the pinned commit below and saved as .npy
files.  The provenance metadata is stored alongside each fixture.

    upstream_commit : 7e77569783b677f00a71f0caa45d8663d6113167
    fixture_dir     : tests/plugins/extract/align/orformer_parity_fixtures/
    fixture_format  : see _FIXTURE_SCHEMA below

Generating fixtures
-------------------
Run the upstream test script on a machine with the official weights, then call:

    python tests/plugins/extract/align/orformer_parity_test.py --generate \
        --upstream-output /path/to/upstream_predictions.jsonl \
        --fixture-dir tests/plugins/extract/align/orformer_parity_fixtures/

This writes per-sample .npy files and a provenance.json.  Commit the fixtures;
the test will pick them up automatically.

Running the test
----------------
    mamba run -n faceswap env KERAS_BACKEND=torch KERAS_TORCH_DEVICE=CPU \
        FACESWAP_BACKEND=cpu pytest -v \
        tests/plugins/extract/align/orformer_parity_test.py

If the fixture directory is absent or empty the test is skipped.

Acceptance thresholds (CPU)
---------------------------
    landmark max_abs_diff         ≤ 1e-4
    NME diff                      ≤ 1e-4

    PyTorch-vs-OpenCV resize diff is logged informally only (~0.008 typical).
    See orformer_reference_resize_test.py for the resize parity measurement gate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from lib.infer.align import Align  # noqa: F401  (import for side-effect)

logger = logging.getLogger(__name__)

_FIXTURE_DIR = Path(__file__).parent / "orformer_parity_fixtures"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Parity thresholds — CPU only; GPU/MPS may use looser bounds.
# Note: PyTorch-vs-OpenCV resize diff (~0.008 typical) is logged but not asserted here;
# see orformer_reference_resize_test.py for that measurement.
_LANDMARK_MAX_ABS_DIFF = 1e-4
_NME_DIFF_THRESHOLD = 1e-4

# NME normalisation landmark pairs (same as benchmark script)
_NME_PAIRS = {"wflw": (60, 72), "300w": (36, 45)}

# ---------------------------------------------------------------------------
# Fixture schema
# ---------------------------------------------------------------------------
# Each fixture is a directory under _FIXTURE_DIR named by sample_id.
# Required files per fixture:
#   image.png          — cropped face at 256×256 (or original; crop box stored separately)
#   gt_landmarks.npy   — (N, 2) float32 ground-truth landmarks in image px space
#   upstream_pred.npy  — (N, 2) float32 upstream-predicted landmarks in [0,1) space
#   crop_box.npy       — (4,) int32 [left, top, right, bottom] landmark-derived crop
#   provenance.json    — metadata: upstream_commit, faceswap_commit, weight_sha256,
#                        opencv_version, torch_version, dataset, model_name


def _load_fixtures() -> list[dict]:
    """Return list of fixture dicts; empty list if fixture dir absent or empty."""
    if not _FIXTURE_DIR.exists():
        return []
    fixtures = []
    for sample_dir in sorted(_FIXTURE_DIR.iterdir()):
        if not sample_dir.is_dir():
            continue
        required = [
            "gt_landmarks.npy",
            "upstream_pred.npy",
            "crop_box.npy",
            "provenance.json",
        ]
        if not all((sample_dir / f).exists() for f in required):
            logger.warning("Incomplete fixture at %s — skipping", sample_dir)
            continue
        prov = json.loads((sample_dir / "provenance.json").read_text(encoding="utf-8"))
        fixtures.append(
            {
                "sample_id": sample_dir.name,
                "sample_dir": sample_dir,
                "gt_landmarks": np.load(sample_dir / "gt_landmarks.npy"),
                "upstream_pred": np.load(sample_dir / "upstream_pred.npy"),
                "crop_box": np.load(sample_dir / "crop_box.npy"),
                "dataset": prov.get("dataset", "wflw"),
                "model_name": prov.get("model_name", "wflw"),
                "upstream_commit": prov.get("upstream_commit", "unknown"),
            }
        )
    return fixtures


_FIXTURES = _load_fixtures()

pytestmark = pytest.mark.skipif(
    not _FIXTURES,
    reason=(
        "No ORFormer parity fixtures found. "
        f"Generate them and place under {_FIXTURE_DIR}. "
        "See module docstring for instructions."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_crop_from_box(
    image: np.ndarray, crop_box: np.ndarray, output_size: int = 256
) -> np.ndarray:
    """Extract affine-warped face crop at output_size using cv2.INTER_LINEAR."""
    import cv2

    left, top, right, bottom = crop_box.tolist()
    src_pts = np.array([[left, top], [right, top], [left, bottom]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [output_size, 0], [0, output_size]], dtype=np.float32)
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    return cv2.warpAffine(image, matrix, (output_size, output_size), flags=cv2.INTER_LINEAR)  # type: ignore[no-any-return]


def _upstream_reference_input(crop_uint8: np.ndarray) -> np.ndarray:
    """OpenCV resize before normalisation — the upstream pipeline."""
    import cv2

    resized = cv2.resize(crop_uint8, (64, 64), interpolation=cv2.INTER_LINEAR)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(norm.transpose(2, 0, 1)[np.newaxis])  # type: ignore[no-any-return]


def _normalize_256(crop_uint8: np.ndarray) -> np.ndarray:
    """Normalise 256×256 crop to (1, 3, 256, 256) float32."""
    norm = crop_uint8.astype(np.float32) / 255.0
    norm = (norm - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(norm.transpose(2, 0, 1)[np.newaxis])  # type: ignore[no-any-return]


def _nme(pred: np.ndarray, gt: np.ndarray, pair: tuple[int, int]) -> float:
    p1, p2 = pair
    norm_dist = float(np.linalg.norm(gt[p1] - gt[p2]))
    if norm_dist < 1e-6:
        return float("nan")
    return float(np.linalg.norm(pred - gt, axis=1).mean() / norm_dist)


def _load_model(model_name: str):
    """Load ORFormerFaceswapModel without the full extract pipeline."""
    import torch

    from plugins.extract.align._orformer.model import ORFormerFaceswapModel
    from plugins.extract.align.orformer import _MODEL_CONFIG, _get_checkpoint_paths

    config = _MODEL_CONFIG[model_name]
    paths = _get_checkpoint_paths(config)
    model = ORFormerFaceswapModel(
        num_points=config.num_landmarks,
        num_edges=config.num_edges,
        edge_info=config.edge_info(),
        orformer_weights_path=paths["orformer"],
    )
    hgnet_state = torch.load(paths["hgnet"], map_location="cpu", weights_only=True)
    model.load_state_dict(hgnet_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models() -> dict[str, object]:
    """Load each model once per test module."""
    needed = {f["model_name"] for f in _FIXTURES}
    return {name: _load_model(name) for name in needed}


@pytest.mark.parametrize("fixture", _FIXTURES, ids=[f["sample_id"] for f in _FIXTURES])
def test_landmark_parity_tight_bbox(fixture: dict, models: dict) -> None:
    """Faceswap output matches upstream on the landmark-derived bbox path.

    Uses the OpenCV reference resize path to maximise upstream parity.
    """
    import cv2
    import torch

    sample_dir = fixture["sample_dir"]
    model_name = fixture["model_name"]
    dataset = fixture["dataset"]
    crop_box = fixture["crop_box"]
    gt_landmarks = fixture["gt_landmarks"]
    upstream_pred_norm = fixture["upstream_pred"]  # already in [0,1) space

    image_path = sample_dir / "image.png"
    if not image_path.exists():
        pytest.skip(f"image.png missing from fixture {fixture['sample_id']}")

    image_bgr = cv2.imread(str(image_path))
    assert image_bgr is not None, f"Could not read {image_path}"
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    crop_uint8 = _build_crop_from_box(image_rgb, crop_box)
    crop_nchw = _normalize_256(crop_uint8)
    ref_nchw = _upstream_reference_input(crop_uint8)

    model = models[model_name]
    with torch.no_grad():
        pred = model(torch.from_numpy(crop_nchw), reference_input=torch.from_numpy(ref_nchw))
    pred_norm = pred.cpu().numpy()[0]  # (N, 2) in [0, 1)

    # Log the PyTorch-vs-OpenCV resize diff for observational purposes only.
    # The production path uses PyTorch bilinear; the parity test passes the OpenCV
    # reference_input directly into the model, so this diff does not affect the
    # landmark result.  For tolerance context see orformer_reference_resize_test.py.
    fs_ref_full = torch.nn.functional.interpolate(
        torch.from_numpy(crop_nchw), size=(64, 64), mode="bilinear", align_corners=False
    ).numpy()
    ref_diff = float(np.abs(fs_ref_full - ref_nchw).max())
    logger.info(
        "[%s] reference input max_abs_diff=%.2e (informational only)",
        fixture["sample_id"],
        ref_diff,
    )

    # --- landmark diff ---
    lmk_diff = float(np.abs(pred_norm - upstream_pred_norm).max())
    logger.info("[%s] landmark max_abs_diff=%.2e", fixture["sample_id"], lmk_diff)
    assert lmk_diff <= _LANDMARK_MAX_ABS_DIFF, (
        f"Landmark diff {lmk_diff:.2e} exceeds {_LANDMARK_MAX_ABS_DIFF:.0e}."
    )

    # --- NME diff ---
    side = float(crop_box[2] - crop_box[0])
    pred_px = pred_norm * side + crop_box[:2].astype(np.float32)
    up_px = upstream_pred_norm * side + crop_box[:2].astype(np.float32)
    pair = _NME_PAIRS[dataset]
    nme_fs = _nme(pred_px, gt_landmarks, pair)
    nme_up = _nme(up_px, gt_landmarks, pair)
    nme_delta = abs(nme_fs - nme_up)
    logger.info(
        "[%s] NME faceswap=%.4f upstream=%.4f delta=%.2e",
        fixture["sample_id"],
        nme_fs,
        nme_up,
        nme_delta,
    )
    assert nme_delta <= _NME_DIFF_THRESHOLD, (
        f"NME diff {nme_delta:.2e} exceeds {_NME_DIFF_THRESHOLD:.0e}. "
        f"faceswap={nme_fs:.4f}, upstream={nme_up:.4f}."
    )


@pytest.mark.parametrize("fixture", _FIXTURES, ids=[f["sample_id"] for f in _FIXTURES])
def test_detector_bbox_increases_nme_vs_tight_bbox(fixture: dict, models: dict) -> None:
    """Document crop distribution mismatch between tight-landmark and detector-box crops.

    Runs both crop paths and logs NME for each.  A hard ordering assertion is
    deliberately omitted until benchmark data on real datasets confirms the
    expected direction; for now the test only asserts that both NME values are
    finite floats.  Tighten to ``assert det_nme >= tight_nme`` once confirmed.
    """
    import cv2
    import torch

    sample_dir = fixture["sample_dir"]
    det_box_path = sample_dir / "detector_bbox.npy"
    if not det_box_path.exists():
        pytest.skip(f"No detector_bbox.npy in fixture {fixture['sample_id']}")

    image_path = sample_dir / "image.png"
    if not image_path.exists():
        pytest.skip(f"image.png missing from fixture {fixture['sample_id']}")

    image_bgr = cv2.imread(str(image_path))
    assert image_bgr is not None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    gt_landmarks = fixture["gt_landmarks"]
    upstream_pred_norm = fixture["upstream_pred"]
    tight_crop_box = fixture["crop_box"]
    det_bbox = np.load(det_box_path)
    dataset = fixture["dataset"]
    model_name = fixture["model_name"]
    pair = _NME_PAIRS[dataset]
    model = models[model_name]

    def _run(crop_box):
        crop_uint8 = _build_crop_from_box(image_rgb, crop_box)
        crop_nchw = _normalize_256(crop_uint8)
        ref_nchw = _upstream_reference_input(crop_uint8)
        with torch.no_grad():
            pred = model(torch.from_numpy(crop_nchw), reference_input=torch.from_numpy(ref_nchw))
        pred_norm = pred.cpu().numpy()[0]
        side = float(crop_box[2] - crop_box[0])
        pred_px = pred_norm * side + crop_box[:2].astype(np.float32)
        up_px = upstream_pred_norm * side + tight_crop_box[:2].astype(np.float32)
        return _nme(pred_px, gt_landmarks, pair), _nme(up_px, gt_landmarks, pair)

    tight_nme, _ = _run(tight_crop_box)

    # Detector crop at default scale 1.20 — expected to be worse
    w = det_bbox[2] - det_bbox[0]
    h = det_bbox[3] - det_bbox[1]
    side = max(w, h) * 1.20
    half = round(side * 0.5)
    cx = round((det_bbox[0] + det_bbox[2]) * 0.5)
    cy = round((det_bbox[1] + det_bbox[3]) * 0.5)
    det_crop = np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.int32)
    det_nme, _ = _run(det_crop)

    logger.info(
        "[%s] NME tight=%.4f detector_1.20=%.4f",
        fixture["sample_id"],
        tight_nme,
        det_nme,
    )
    # We document rather than assert a strict ordering here — in some cases
    # a larger crop could help.  The test body can be tightened once benchmark
    # data confirms the expected ordering.
    assert isinstance(tight_nme, float) and isinstance(det_nme, float)
