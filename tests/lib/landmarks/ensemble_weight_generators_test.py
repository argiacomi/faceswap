#!/usr/bin/env python3
"""Tests for :mod:`lib.landmarks.ensemble.weight_generators` (issue #68)."""

from __future__ import annotations

import numpy as np
import pytest

from lib.landmarks.ensemble.weight_generators import (
    GENERATOR_NAMES,
    GENERATORS,
    EqualGenerator,
    ErrorTable,
    InverseMeanErrorGenerator,
    InverseMedianErrorGenerator,
    RegionInverseErrorGenerator,
    RegularizedInverseErrorGenerator,
    SoftmaxNegativeErrorGenerator,
    WinnerTakeLandmarkGenerator,
    get_generator,
    landmark_region_indices,
)
from lib.landmarks.ensemble.weights import LANDMARK_COUNT, weights_from_errors

MODELS = ("hrnet", "spiga", "orformer")


def _table_from_constant_errors(values: tuple[float, float, float]) -> ErrorTable:
    """ErrorTable where every sample has a uniform per-landmark error per model."""
    errors = {
        name: np.full((4, LANDMARK_COUNT), value, dtype="float32")
        for name, value in zip(MODELS, values, strict=True)
    }
    return ErrorTable.from_samples(errors)


def _table_from_varied_errors() -> ErrorTable:
    """ErrorTable with different errors per landmark range, useful for region tests."""
    errors: dict[str, np.ndarray] = {}
    for offset, name in enumerate(MODELS, start=1):
        matrix = np.zeros((3, LANDMARK_COUNT), dtype="float32")
        matrix[:, :17] = 1.0 * offset  # jaw
        matrix[:, 17:48] = 2.0 * offset  # brows, nose, eyes
        matrix[:, 48:] = 3.0 * offset  # mouth
        errors[name] = matrix
    return ErrorTable.from_samples(errors)


def test_generator_registry_lists_all_seven_named_generators() -> None:
    """Every Epic-listed generator is reachable by name through the registry."""
    assert set(GENERATOR_NAMES) == {
        "equal",
        "inverse_mean_error",
        "inverse_median_error",
        "region_inverse_error",
        "regularized_inverse_error",
        "winner_take_landmark",
        "softmax_negative_error",
    }
    for name in GENERATOR_NAMES:
        instance = GENERATORS[name]
        assert instance.name == name
        assert hasattr(instance, "fit")


def test_get_generator_rejects_unknown_with_supported_list() -> None:
    """Unknown generator names fail with the full supported list."""
    with pytest.raises(ValueError) as exc:
        get_generator("nope")
    message = str(exc.value)
    assert "nope" in message
    for name in GENERATOR_NAMES:
        assert name in message


@pytest.mark.parametrize("name", GENERATOR_NAMES)
def test_every_generator_returns_normalized_68_weights(name: str) -> None:
    """All generators output 68-length non-negative columns summing to 1.0."""
    table = _table_from_constant_errors((1.0, 2.0, 4.0))
    result = GENERATORS[name].fit(table)
    assert result.name == name
    assert tuple(result.weights) == MODELS
    for model in MODELS:
        column = np.asarray(result.weights[model], dtype="float32")
        assert column.shape == (LANDMARK_COUNT,)
        assert np.all(column >= 0)
    matrix = np.stack([result.weights[model] for model in MODELS], axis=0)
    np.testing.assert_allclose(matrix.sum(axis=0), np.ones(LANDMARK_COUNT), atol=1e-6)


def test_inverse_mean_error_parity_with_legacy_weights_from_errors() -> None:
    """``inverse_mean_error`` must match legacy ``weights_from_errors`` output."""
    table = _table_from_constant_errors((1.0, 2.0, 4.0))
    legacy = weights_from_errors(table.mean_errors())
    result = InverseMeanErrorGenerator().fit(table)
    for model in MODELS:
        np.testing.assert_allclose(result.weights[model], legacy[model], rtol=1e-6, atol=1e-6)


def test_equal_generator_distributes_evenly_across_models() -> None:
    """``equal`` puts ``1/M`` weight on every (model, landmark) cell."""
    table = _table_from_constant_errors((1.0, 2.0, 4.0))
    result = EqualGenerator().fit(table)
    for model in MODELS:
        np.testing.assert_allclose(
            result.weights[model],
            np.full(LANDMARK_COUNT, 1.0 / len(MODELS), dtype="float32"),
            rtol=1e-6,
        )


def test_inverse_median_error_diverges_from_mean_under_outlier_samples() -> None:
    """``inverse_median_error`` ignores a single high-error outlier sample."""
    errors: dict[str, np.ndarray] = {}
    base = np.full((9, LANDMARK_COUNT), 1.0, dtype="float32")
    errors["hrnet"] = base.copy()
    spiga = base.copy()
    spiga[0] = 100.0  # one bad sample
    errors["spiga"] = spiga
    orformer = base.copy() * 5.0
    errors["orformer"] = orformer
    table = ErrorTable.from_samples(errors)

    median_result = InverseMedianErrorGenerator().fit(table)
    mean_result = InverseMeanErrorGenerator().fit(table)

    # Median-based fit treats hrnet and spiga as equally accurate; mean-based
    # fit downweights spiga because of the single outlier sample.
    np.testing.assert_allclose(
        median_result.weights["hrnet"], median_result.weights["spiga"], rtol=1e-6
    )
    assert mean_result.weights["hrnet"][0] > mean_result.weights["spiga"][0]


def test_winner_take_landmark_gives_each_landmark_to_lowest_error_model() -> None:
    """``winner_take_landmark`` puts 100% weight on the lowest-error model per column."""
    errors = {
        "hrnet": np.full((1, LANDMARK_COUNT), 5.0, dtype="float32"),
        "spiga": np.full((1, LANDMARK_COUNT), 3.0, dtype="float32"),
        "orformer": np.full((1, LANDMARK_COUNT), 1.0, dtype="float32"),
    }
    errors["spiga"][0, :10] = 0.5  # spiga wins these landmarks
    errors["hrnet"][0, 30:35] = 0.25  # hrnet wins these landmarks
    table = ErrorTable.from_samples(errors)

    result = WinnerTakeLandmarkGenerator().fit(table)

    assert np.allclose(result.weights["spiga"][:10], 1.0)
    assert np.allclose(result.weights["hrnet"][30:35], 1.0)
    other_indices = [
        idx
        for idx in range(LANDMARK_COUNT)
        if idx not in set(list(range(10)) + list(range(30, 35)))
    ]
    assert np.allclose([result.weights["orformer"][idx] for idx in other_indices], 1.0)


def test_softmax_negative_error_temperature_concentrates_weight() -> None:
    """Higher temperature pushes more weight toward the lowest-error model."""
    table = _table_from_constant_errors((1.0, 2.0, 4.0))
    low_t = SoftmaxNegativeErrorGenerator(temperature=0.01).fit(table)
    high_t = SoftmaxNegativeErrorGenerator(temperature=10.0).fit(table)
    assert high_t.weights["hrnet"][0] > low_t.weights["hrnet"][0]
    assert high_t.weights["hrnet"][0] > 0.99


def test_softmax_negative_error_rejects_non_positive_temperature() -> None:
    """``temperature`` must be strictly positive."""
    with pytest.raises(ValueError):
        SoftmaxNegativeErrorGenerator(temperature=0)
    with pytest.raises(ValueError):
        SoftmaxNegativeErrorGenerator(temperature=-1.0)


def test_region_inverse_error_uses_region_means() -> None:
    """``region_inverse_error`` produces identical weights within each region."""
    table = _table_from_varied_errors()
    result = RegionInverseErrorGenerator().fit(table)
    regions = landmark_region_indices(LANDMARK_COUNT)
    for indices in regions.values():
        for model in MODELS:
            column = np.asarray(result.weights[model], dtype="float32")[indices]
            np.testing.assert_allclose(column, column[0], rtol=1e-6, atol=1e-6)


def test_regularized_inverse_error_default_mix_weights_components() -> None:
    """Default mix produces a blend strictly between equal and inverse-mean-error."""
    table = _table_from_constant_errors((1.0, 2.0, 4.0))
    regularized = RegularizedInverseErrorGenerator().fit(table)
    pure = InverseMeanErrorGenerator().fit(table)
    # Mixing in global (uniform across landmarks) softens the per-landmark
    # contrast: hrnet's per-landmark weight is below pure inverse-error.
    assert regularized.weights["hrnet"][0] < pure.weights["hrnet"][0]
    # But hrnet still has the most weight because its errors are lowest.
    assert (
        regularized.weights["hrnet"][0]
        > regularized.weights["spiga"][0]
        > regularized.weights["orformer"][0]
    )


def test_regularized_inverse_error_rejects_invalid_components() -> None:
    """Component weights must be non-negative and have a positive sum."""
    with pytest.raises(ValueError):
        RegularizedInverseErrorGenerator(
            per_landmark_weight=-0.1,
            per_region_weight=0.5,
            global_weight=0.5,
        )
    with pytest.raises(ValueError):
        RegularizedInverseErrorGenerator(
            per_landmark_weight=0.0,
            per_region_weight=0.0,
            global_weight=0.0,
        )


def test_error_table_rejects_invalid_inputs() -> None:
    """ErrorTable validates shape and finiteness at construction time."""
    with pytest.raises(ValueError):
        ErrorTable.from_samples({"hrnet": np.full((2, 67), 1.0, dtype="float32")})
    with pytest.raises(ValueError):
        ErrorTable.from_samples({"hrnet": np.full((2, 68), -1.0, dtype="float32")})
    bad = np.full((2, 68), 1.0, dtype="float32")
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        ErrorTable.from_samples({"hrnet": bad})


def test_error_table_from_means_round_trips_through_mean_errors() -> None:
    """from_means accepts precomputed means and exposes them via mean_errors()."""
    table = ErrorTable.from_means({"hrnet": [1.0] * LANDMARK_COUNT})
    mean = table.mean_errors()["hrnet"]
    assert mean.shape == (LANDMARK_COUNT,)
    np.testing.assert_allclose(mean, np.ones(LANDMARK_COUNT, dtype="float32"))


def test_generator_diagnostics_include_name_and_model_metadata() -> None:
    """Every generator records identifying metadata in ``diagnostics``."""
    table = _table_from_constant_errors((1.0, 2.0, 4.0))
    for name in GENERATOR_NAMES:
        result = GENERATORS[name].fit(table)
        assert result.name == name
        assert "models" in result.diagnostics
        assert result.diagnostics["models"] == list(MODELS)
