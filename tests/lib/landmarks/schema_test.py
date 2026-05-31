#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.core.schema`."""

import typing as T
from pathlib import Path

import numpy as np
import pytest
import yaml

from lib.align.constants import LANDMARK_PARTS, MAP_2D_68, LandmarkType
from lib.landmarks.core.schema import (
    LandmarkPrediction,
    canonicalize_schema,
    normalize_landmark_array,
    normalize_landmarks,
    to_canonical_68,
)


def test_canonicalize_schema_aliases() -> None:
    """Schema aliases normalize to local names."""
    assert canonicalize_schema("68pt") == "2d_68"
    assert canonicalize_schema(LandmarkType.LM_2D_98) == "2d_98"


def test_normalize_landmark_array_flat_input() -> None:
    """Flat x/y pairs are reshaped and converted to float32."""
    points = normalize_landmark_array([0, 1, 2, 3, 4, 5, 6, 7], schema="2d_4")  # type: ignore[list-item]
    assert points.dtype == np.float32
    np.testing.assert_array_equal(
        points,
        np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype="float32"),
    )


def test_normalize_landmark_array_rejects_non_finite() -> None:
    """NaN or infinite values are not accepted."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        normalize_landmark_array([[0, 1], [np.nan, 3]])


def test_landmark_prediction_validates_confidence_shape() -> None:
    """Confidence must match the point count."""
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    with pytest.raises(ValueError, match="one value per landmark"):
        LandmarkPrediction(points=points, confidence=np.zeros(67, dtype="float32"))


def test_landmark_prediction_records_adapter_metadata() -> None:
    """Predictions expose the metadata required by model adapters."""
    points = np.zeros((68, 2), dtype="float32")  # type: ignore[var-annotated]
    prediction = LandmarkPrediction(
        landmarks=points,
        model_name="hrnet",
        source_landmark_count=98,
        coordinate_space="frame",
        metadata={"checkpoint": "test"},
    )

    assert prediction.points is prediction.landmarks
    assert prediction.source == "hrnet"
    assert prediction.model_name == "hrnet"
    assert prediction.source_landmark_count == 98
    assert prediction.coordinate_space == "frame"
    assert prediction.metadata == {"checkpoint": "test"}


def test_to_canonical_68_from_98_point_schema() -> None:
    """98-point inputs reuse Faceswap's existing 98-to-68 mapping."""
    points = np.arange(196, dtype="float32").reshape((98, 2))  # type: ignore[var-annotated]
    expected = points[MAP_2D_68[LandmarkType.LM_2D_98]]
    np.testing.assert_array_equal(to_canonical_68(points, source_schema="2d_98"), expected)


@pytest.mark.parametrize("source_schema", ["2d_68", "2d_98"])
@pytest.mark.parametrize("normalizer", [to_canonical_68, normalize_landmarks])
def test_normalizers_accept_mixed_68_and_98_inputs(
    normalizer: T.Callable[..., np.ndarray],
    source_schema: str,
) -> None:
    """Public normalization helpers map 68 and 98 point inputs to canonical 68."""
    count = 68 if source_schema == "2d_68" else 98
    points = np.arange(count * 2, dtype="float32").reshape((count, 2))  # type: ignore[var-annotated]
    expected = points if source_schema == "2d_68" else points[MAP_2D_68[LandmarkType.LM_2D_98]]

    result = normalizer(points, source_schema=source_schema)

    assert result.shape == (68, 2)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_normalizers_infer_mixed_68_and_98_inputs_without_schema() -> None:
    """Shape inference keeps 68-point inputs and remaps 98-point inputs."""
    points_68 = np.arange(136, dtype="float32").reshape((68, 2))  # type: ignore[var-annotated]
    points_98 = np.arange(196, dtype="float32").reshape((98, 2))  # type: ignore[var-annotated]

    np.testing.assert_array_equal(to_canonical_68(points_68), points_68)
    np.testing.assert_array_equal(
        normalize_landmarks(points_98),
        points_98[MAP_2D_68[LandmarkType.LM_2D_98]],
    )


def test_canonical_68_order_matches_faceswap_ibug_landmark_parts() -> None:
    """Faceswap 68 parts use the expected iBUG contiguous index ranges."""
    parts = LANDMARK_PARTS[LandmarkType.LM_2D_68]

    assert parts["jaw"] == (0, 17, False)
    assert parts["right_eyebrow"] == (17, 22, False)
    assert parts["left_eyebrow"] == (22, 27, False)
    assert parts["nose"] == (27, 36, False)
    assert parts["right_eye"] == (36, 42, True)
    assert parts["left_eye"] == (42, 48, True)
    assert parts["mouth_outer"] == (48, 60, True)
    assert parts["mouth_inner"] == (60, 68, True)


def test_98_to_68_mapping_preserves_canonical_region_order() -> None:
    """The WFLW/Faceswap 98 mapping emits points in canonical 68 region order."""
    indexes = MAP_2D_68[LandmarkType.LM_2D_98]
    source_parts = LANDMARK_PARTS[LandmarkType.LM_2D_98]
    canonical_parts = LANDMARK_PARTS[LandmarkType.LM_2D_68]

    assert len(indexes) == 68
    assert len(set(indexes)) == 68
    for name, (start_68, end_68, _closed_68) in canonical_parts.items():
        source_start, source_end, _closed_98 = source_parts[name]
        assert all(source_start <= index < source_end for index in indexes[start_68:end_68])


def test_ibug_68_config_matches_canonical_schema() -> None:
    """The shipped canonical schema config mirrors Faceswap's 68-point parts."""
    config_path = Path(__file__).parents[3] / "configs" / "landmarks" / "ibug_68.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["schema"] == "2d_68"
    assert config["points"] == 68
    assert config["dimensions"] == 2
    assert config["coordinate_order"] == ["x", "y"]
    assert config["nme"]["interocular_indices"] == {
        "right_eye_outer": 36,
        "left_eye_outer": 45,
    }
    for name, (start, end, is_polygon) in LANDMARK_PARTS[LandmarkType.LM_2D_68].items():
        assert config["parts"][name] == {
            "start": start,
            "end": end,
            "closed": is_polygon,
        }
