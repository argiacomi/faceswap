#!/usr/bin/env python3
"""Tests for AdaFace, ArcFace and InsightFace identity plugins."""

from __future__ import annotations

import typing as T

import numpy as np
import numpy.typing as npt
import pytest

from lib.align.detected_face import DetectedFace
from lib.infer.identity import Identity
from lib.infer.objects import ExtractBatch
from lib.utils import FaceswapError
from plugins.extract.identity import insightface as insightface_plugin
from plugins.extract.identity import insightface_defaults
from plugins.extract.identity._model_adapter import (
    IdentityModelAdapter,
    ModelProvenance,
    normalize_embeddings,
)
from plugins.extract.identity.adaface import AdaFace
from plugins.extract.identity.arcface import ArcFace
from plugins.extract.identity.insightface import InsightFace
from plugins.plugin_loader import PluginLoader


class _StubModel:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[tuple[int, ...]] = []
        self.dtypes: list[np.dtype] = []

    def embed(self, faces: np.ndarray) -> np.ndarray:
        self.calls.append(faces.shape)
        self.dtypes.append(faces.dtype)
        return np.ones((len(faces), self.dim), dtype=np.float32)


def _adapter(name: str, loader: T.Callable[[], _StubModel]) -> IdentityModelAdapter:
    return IdentityModelAdapter(
        name=name,
        embedding_dim=4,
        input_size=112,
        normalized=True,
        provenance=ModelProvenance(
            source="test/source",
            license="test",
            commercial_use=True,
            notes="test notes",
        ),
        loader=loader,
    )


def test_identity_plugins_are_selectable() -> None:
    """AdaFace, ArcFace and InsightFace are exposed as identity extract plugins."""
    plugins = PluginLoader.get_available_extractors("identity")

    assert "adaface" in plugins
    assert "arcface" in plugins
    assert "insightface" in plugins
    assert "insightface-antelopev2" not in plugins
    assert "insightface-buffalo-l" not in plugins
    assert "insightface-buffalo-sc" not in plugins


@pytest.mark.parametrize(
    ("plugin_cls", "storage_name"),
    [(AdaFace, "adaface"), (ArcFace, "arcface")],
)
def test_cvlface_plugins_use_stable_storage_key_and_adapter(
    monkeypatch: pytest.MonkeyPatch,
    plugin_cls: type[AdaFace] | type[ArcFace],
    storage_name: str,
) -> None:
    """CVLFace-backed plugins use stable storage keys and delegate loading to adapters."""
    stub = _StubModel()
    monkeypatch.setattr(IdentityModelAdapter, "load", lambda self: stub)

    plugin = plugin_cls()
    # The framework hands aligned faces to identity plugins in the plugin's declared
    # dtype (uint8). process is expected to forward the batch as-is.
    batch: npt.NDArray[np.uint8] = np.zeros(
        (2, plugin.input_size, plugin.input_size, 3), dtype=np.uint8
    )

    assert plugin.storage_name == storage_name
    assert plugin.load_model() is stub
    plugin.model = stub
    out = plugin.process(batch)
    assert out.shape == (2, 4)
    assert stub.dtypes == [np.dtype("uint8")]


@pytest.mark.parametrize("model_type", ["antelopev2", "buffalo_l", "buffalo_sc"])
def test_insightface_config_selects_adapter(
    monkeypatch: pytest.MonkeyPatch, model_type: str
) -> None:
    """The single InsightFace plugin selects the backend through config."""
    stub = _StubModel()
    adapter = _adapter(f"insightface-{model_type}", lambda: stub)
    seen: dict[str, T.Any] = {}

    def fake_insightface_adapter(
        selected_model_type: str, *, force_cpu: bool = False
    ) -> IdentityModelAdapter:
        seen["model_type"] = selected_model_type
        seen["force_cpu"] = force_cpu
        return adapter

    monkeypatch.setattr(insightface_defaults, "model_type", lambda: model_type)
    monkeypatch.setattr(insightface_defaults, "cpu", lambda: False)
    monkeypatch.setattr(insightface_plugin, "insightface_adapter", fake_insightface_adapter)

    plugin = InsightFace()
    batch: npt.NDArray[np.uint8] = np.zeros(
        (3, plugin.input_size, plugin.input_size, 3), dtype=np.uint8
    )

    assert seen == {"model_type": model_type, "force_cpu": False}
    assert plugin.storage_name == "insightface"
    assert plugin.model_type == model_type
    assert plugin.load_model() is stub
    assert plugin.identity_metadata["model_type"] == model_type
    plugin.model = stub
    out = plugin.process(batch)
    assert out.shape == (3, 4)
    assert stub.calls == [(3, 112, 112, 3)]
    assert stub.dtypes == [np.dtype("uint8")]


def test_insightface_passes_force_cpu_to_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The InsightFace CPU setting is passed into the adapter loader."""
    stub = _StubModel()
    adapter = _adapter("insightface-buffalo_l", lambda: stub)
    seen: dict[str, T.Any] = {}

    def fake_insightface_adapter(
        selected_model_type: str, *, force_cpu: bool = False
    ) -> IdentityModelAdapter:
        seen["model_type"] = selected_model_type
        seen["force_cpu"] = force_cpu
        return adapter

    monkeypatch.setattr(insightface_defaults, "model_type", lambda: "buffalo_l")
    monkeypatch.setattr(insightface_defaults, "cpu", lambda: True)
    monkeypatch.setattr(insightface_plugin, "insightface_adapter", fake_insightface_adapter)

    InsightFace()

    assert seen == {"model_type": "buffalo_l", "force_cpu": True}


def test_insightface_invalid_model_type_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid model_type values are rejected before model loading."""
    monkeypatch.setattr(insightface_defaults, "model_type", lambda: "bad_model")

    with pytest.raises(FaceswapError, match="Unsupported InsightFace model_type"):
        InsightFace()


def test_embeddings_are_l2_normalized_float32() -> None:
    """Stored identity embeddings are stable float32 unit vectors."""
    embeddings = normalize_embeddings(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64))

    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(embeddings[0], np.array([0.6, 0.8], dtype=np.float32))
    np.testing.assert_allclose(embeddings[1], np.zeros((2,), dtype=np.float32))


def test_identity_metadata_round_trips_through_alignment() -> None:
    """Identity embeddings and model provenance survive alignment serialization."""
    embedding: npt.NDArray[np.float32] = np.arange(4, dtype=np.float32)
    face = DetectedFace(
        left=1,
        width=10,
        top=2,
        height=12,
        landmarks_xy=np.zeros((68, 2), dtype=np.float32),
        identity={"insightface": embedding},
        metadata={"identity": {"insightface": {"model_type": "buffalo_l"}}},
    )

    loaded = DetectedFace().from_alignment(face.to_alignment())

    assert loaded.identity["insightface"].dtype == np.float32
    np.testing.assert_array_equal(loaded.identity["insightface"], embedding)
    assert loaded.metadata["identity"]["insightface"]["model_type"] == "buffalo_l"


def test_identity_handler_adds_plugin_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin provenance is copied onto every face in the batch."""
    metadata = {"model_type": "buffalo_l", "embedding_dim": 512}

    class _Plugin:
        storage_name = "insightface"
        identity_metadata = metadata

    handler = object.__new__(Identity)
    handler.plugin = T.cast(T.Any, _Plugin())
    handler.storage_name = "insightface"
    batch = ExtractBatch(
        filenames=["frame.png"],
        images=[np.zeros((32, 32, 3), dtype=np.uint8)],
        sources=["/tmp"],
    )
    batch.bboxes = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.int32)

    handler._add_identity_metadata(batch)  # pylint:disable=protected-access

    assert batch.aligned.metadata == [
        {"identity": {"insightface": metadata}},
        {"identity": {"insightface": metadata}},
    ]
