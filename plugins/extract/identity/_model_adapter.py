#!/usr/bin/env python3
"""Shared lazy loaders for optional identity-recognition models."""

from __future__ import annotations

import logging
import typing as T
from dataclasses import dataclass
from importlib import import_module

import numpy as np

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


class LoadedIdentityModel(T.Protocol):
    """Minimal identity model contract used by extract plugins."""

    def embed(self, faces: np.ndarray) -> np.ndarray:
        """Return ``(N, D)`` float32 embeddings for BGR uint8 face crops."""


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """License and source notes for a pretrained identity model."""

    source: str
    license: str
    commercial_use: bool
    notes: str


@dataclass(frozen=True, slots=True)
class IdentityModelAdapter:
    """Registry entry for one optional identity-recognition backend."""

    name: str
    embedding_dim: int
    input_size: int
    normalized: bool
    provenance: ModelProvenance
    loader: T.Callable[[], LoadedIdentityModel]

    def load(self) -> LoadedIdentityModel:
        """Load the model and surface dependency errors with model context."""
        logger.debug("Loading identity model adapter: %s", self.name)
        return self.loader()


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return contiguous float32 L2-normalized embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return T.cast(np.ndarray, np.ascontiguousarray(embeddings / norms, dtype=np.float32))


def _load_onnx_adaface() -> LoadedIdentityModel:
    """Load AdaFace ONNX weights from Hugging Face."""
    try:
        ort = import_module("onnxruntime")
    except ImportError as exc:
        raise ImportError(
            "AdaFace identity extraction requires 'onnxruntime'. Install a CPU or GPU "
            "onnxruntime package suitable for your Faceswap backend."
        ) from exc
    try:
        from huggingface_hub import hf_hub_download  # pylint:disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "AdaFace identity extraction requires 'huggingface_hub' to download weights."
        ) from exc

    weights = hf_hub_download(
        repo_id="mk-minchul/AdaFace", filename="adaface_ir101_webface12m.onnx"
    )
    session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    class _AdaFaceOnnx:
        def embed(self, faces: np.ndarray) -> np.ndarray:
            x = faces.astype(np.float32) / 127.5 - 1.0
            x = np.transpose(x, (0, 3, 1, 2))
            outputs = session.run(None, {input_name: x})
            return T.cast(np.ndarray, np.asarray(outputs[0], dtype=np.float32))

    return _AdaFaceOnnx()


def _make_insightface_loader(model_type: str) -> T.Callable[[], LoadedIdentityModel]:
    """Return a loader for one InsightFace recognition pack."""

    def _load() -> LoadedIdentityModel:
        try:
            face_analysis = import_module("insightface.app")
        except ImportError as exc:
            raise ImportError(
                "InsightFace identity extraction requires 'insightface' and a compatible "
                "'onnxruntime' package."
            ) from exc

        face_analysis_cls = face_analysis.FaceAnalysis
        app = face_analysis_cls(name=model_type, allowed_modules=["recognition"])
        app.prepare(ctx_id=-1)

        class _InsightFace:
            def embed(self, faces: np.ndarray) -> np.ndarray:
                outputs = [
                    np.asarray(app.models["recognition"].get_feat(face), dtype=np.float32).reshape(
                        -1
                    )
                    for face in faces
                ]
                if not outputs:
                    return np.zeros((0, 512), dtype=np.float32)
                return T.cast(np.ndarray, np.stack(outputs, axis=0))

        return _InsightFace()

    return _load


ADAFACE = IdentityModelAdapter(
    name="adaface",
    embedding_dim=512,
    input_size=112,
    normalized=True,
    provenance=ModelProvenance(
        source="mk-minchul/AdaFace",
        license="MIT",
        commercial_use=True,
        notes="AdaFace iResNet101 WebFace12M, Kim et al., CVPR 2022.",
    ),
    loader=_load_onnx_adaface,
)


INSIGHTFACE: dict[str, IdentityModelAdapter] = {
    "antelopev2": IdentityModelAdapter(
        name="insightface-antelopev2",
        embedding_dim=512,
        input_size=112,
        normalized=True,
        provenance=ModelProvenance(
            source="deepinsight/insightface",
            license="non-commercial",
            commercial_use=False,
            notes="InsightFace antelopev2 ArcFace pack. Check upstream terms before use.",
        ),
        loader=_make_insightface_loader("antelopev2"),
    ),
    "buffalo_l": IdentityModelAdapter(
        name="insightface-buffalo-l",
        embedding_dim=512,
        input_size=112,
        normalized=True,
        provenance=ModelProvenance(
            source="deepinsight/insightface",
            license="non-commercial",
            commercial_use=False,
            notes="InsightFace buffalo_l ArcFace R100 pack. Check upstream terms before use.",
        ),
        loader=_make_insightface_loader("buffalo_l"),
    ),
    "buffalo_sc": IdentityModelAdapter(
        name="insightface-buffalo-sc",
        embedding_dim=512,
        input_size=112,
        normalized=True,
        provenance=ModelProvenance(
            source="deepinsight/insightface",
            license="non-commercial",
            commercial_use=False,
            notes="InsightFace buffalo_sc MobileFaceNet pack. Check upstream terms before use.",
        ),
        loader=_make_insightface_loader("buffalo_sc"),
    ),
}


def insightface_adapter(model_type: str) -> IdentityModelAdapter:
    """Return the configured InsightFace adapter."""
    if model_type not in INSIGHTFACE:
        raise ValueError(
            f"Unsupported InsightFace model_type '{model_type}'. "
            f"Select from {sorted(INSIGHTFACE)}."
        )
    return INSIGHTFACE[model_type]


def metadata(
    adapter: IdentityModelAdapter, *, storage_name: str, model_type: str | None = None
) -> dict[str, T.Any]:
    """Return serializable provenance metadata for per-face storage."""
    retval: dict[str, T.Any] = {
        "model": adapter.name,
        "embedding_dim": adapter.embedding_dim,
        "normalized": adapter.normalized,
        "source": adapter.provenance.source,
        "license": adapter.provenance.license,
        "commercial_use": adapter.provenance.commercial_use,
        "notes": adapter.provenance.notes,
        "storage_key": storage_name,
    }
    if model_type is not None:
        retval["model_type"] = model_type
    return retval


__all__ = get_module_objects(__name__)
