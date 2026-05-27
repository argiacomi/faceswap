#!/usr/bin/env python3
"""Shared lazy loaders for optional identity-recognition models."""

from __future__ import annotations

import glob
import logging
import os.path as osp
import typing as T
from dataclasses import dataclass, replace
from importlib import import_module

import numpy as np

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

OnnxProvider = str | tuple[str, dict[str, T.Any]]


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


def _onnxruntime_providers(
    ort: T.Any, *, force_cpu: bool = False
) -> tuple[list[OnnxProvider], int]:
    """Return preferred ONNX Runtime providers and matching InsightFace ctx_id."""
    if force_cpu:
        return ["CPUExecutionProvider"], -1

    available = set(ort.get_available_providers())
    providers: list[OnnxProvider] = []

    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    elif "CoreMLExecutionProvider" in available:
        providers.append(("CoreMLExecutionProvider", {"MLComputeUnits": "ALL"}))

    providers.append("CPUExecutionProvider")

    # InsightFace ArcFaceONNX.prepare(ctx_id < 0) resets providers to CPU.
    # Use a non-negative ctx_id whenever an accelerator provider was selected.
    ctx_id = 0 if len(providers) > 1 else -1
    return providers, ctx_id


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


def _make_insightface_loader(
    model_type: str, *, force_cpu: bool = False
) -> T.Callable[[], LoadedIdentityModel]:
    """Return a loader for one InsightFace recognition pack."""

    def _load() -> LoadedIdentityModel:
        try:
            model_zoo = import_module("insightface.model_zoo.model_zoo")
            insightface_utils = import_module("insightface.utils")
            ort = import_module("onnxruntime")
        except ImportError as exc:
            raise ImportError(
                "InsightFace identity extraction requires 'insightface' and a compatible "
                "'onnxruntime' package."
            ) from exc

        providers, ctx_id = _onnxruntime_providers(ort, force_cpu=force_cpu)
        model_dir = insightface_utils.ensure_available("models", model_type)
        onnx_files = sorted(glob.glob(osp.join(model_dir, "*.onnx")))

        if not onnx_files:
            raise RuntimeError(f"No ONNX files found for InsightFace model pack '{model_type}'.")

        def _load_recognition(active_providers: list[OnnxProvider]) -> T.Any:
            for onnx_file in onnx_files:
                model = model_zoo.get_model(onnx_file, providers=active_providers)
                if model is None:
                    continue
                if getattr(model, "taskname", None) == "recognition":
                    return model
            raise RuntimeError(
                f"No recognition model found in InsightFace model pack '{model_type}'."
            )

        try:
            recognition = _load_recognition(providers)
            recognition.prepare(ctx_id=ctx_id)
        except Exception:  # pylint:disable=broad-except
            if force_cpu or providers == ["CPUExecutionProvider"]:
                raise
            logger.warning(
                "Failed to load InsightFace %s with ONNX Runtime providers %s. "
                "Falling back to CPUExecutionProvider.",
                model_type,
                providers,
                exc_info=True,
            )
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
            recognition = _load_recognition(providers)
            recognition.prepare(ctx_id=ctx_id)

        logger.info(
            "InsightFace %s recognition providers: %s",
            model_type,
            recognition.session.get_providers(),
        )

        class _InsightFace:
            def embed(self, faces: np.ndarray) -> np.ndarray:
                outputs = [
                    np.asarray(recognition.get_feat(face), dtype=np.float32).reshape(-1)
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
            notes="InsightFace antelopev2 ArcFace R100 pack. Check upstream terms before use.",
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
            notes="InsightFace buffalo_l ResNet50@WebFace600K pack. Check upstream terms before use.",
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


def insightface_adapter(model_type: str, *, force_cpu: bool = False) -> IdentityModelAdapter:
    """Return the configured InsightFace adapter."""
    if model_type not in INSIGHTFACE:
        raise ValueError(
            f"Unsupported InsightFace model_type '{model_type}'. "
            f"Select from {sorted(INSIGHTFACE)}."
        )
    adapter = INSIGHTFACE[model_type]
    return replace(adapter, loader=_make_insightface_loader(model_type, force_cpu=force_cpu))


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
