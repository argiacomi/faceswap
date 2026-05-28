#!/usr/bin/env python3
"""Shared lazy loaders for optional identity-recognition models."""

from __future__ import annotations

import glob
import importlib.util
import io
import logging
import os
import os.path as osp
import sys
import threading
import types
import typing as T
from contextlib import contextmanager, redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass, replace
from importlib import import_module

import numpy as np

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)

OnnxProvider = str | tuple[str, dict[str, T.Any]]
_CVLFACE_LOAD_LOCK = threading.Lock()


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


@contextmanager
def _third_party_output_to_debug(label: str) -> T.Iterator[None]:
    """Capture noisy dependency stdout/stderr and replay it through Faceswap DEBUG."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
    finally:
        for stream_name, stream in (("stdout", stdout), ("stderr", stderr)):
            for line in stream.getvalue().splitlines():
                if line.strip():
                    logger.debug("[%s %s] %s", label, stream_name, line)


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


@contextmanager
def _quiet_hf_transfer_logs() -> T.Iterator[None]:
    """Suppress noisy HTTP transfer INFO logs while preserving warnings/errors."""
    loggers = [
        logging.getLogger(name)
        for name in ("httpx", "httpcore", "huggingface_hub", "huggingface_hub.file_download")
    ]
    levels = {log: log.level for log in loggers}
    try:
        for log in loggers:
            if log.getEffectiveLevel() < logging.WARNING:
                log.setLevel(logging.WARNING)
        yield
    finally:
        for log, level in levels.items():
            log.setLevel(level)


def _download_cvlface_snapshot(repo_id: str, model_label: str) -> str:
    """Download the full CVLFace custom-code snapshot required by Transformers."""
    try:
        hub = import_module("huggingface_hub")
    except ImportError as exc:
        raise ImportError(
            f"{model_label} identity extraction requires 'huggingface_hub' to download "
            "CVLFace model code and weights."
        ) from exc

    # The CVLFace wrappers import a repo-local top-level ``models`` package and load
    # ``pretrained_model/model.pt`` with a relative path. Pull the repo code and weights
    # into a local snapshot so we can add the snapshot root to sys.path while loading.
    with _quiet_hf_transfer_logs():
        return T.cast(
            str,
            hub.snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "config.json",
                    "wrapper.py",
                    "models/**",
                    "pretrained_model/**",
                    "model.safetensors",
                ],
            ),
        )


def _pop_modules(prefix: str) -> dict[str, types.ModuleType]:
    """Remove and return loaded modules matching a top-level package prefix."""
    modules: dict[str, types.ModuleType] = {}
    for name in tuple(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            module = sys.modules.pop(name, None)
            if isinstance(module, types.ModuleType):
                modules[name] = module
    return modules


@contextmanager
def _cvlface_dependency_context() -> T.Iterator[None]:
    """Provide lightweight optional dependencies required only by CVLFace imports."""
    try:
        import_module("fvcore.nn")
        yield
        return
    except ImportError:
        pass

    fvcore = types.ModuleType("fvcore")
    fvcore_nn = types.ModuleType("fvcore.nn")

    def _flop_count(*_args: T.Any, **_kwargs: T.Any) -> tuple[dict[str, float], dict[str, float]]:
        return {}, {}

    fvcore_nn.flop_count = _flop_count  # type:ignore[attr-defined]
    fvcore.nn = fvcore_nn  # type:ignore[attr-defined]
    previous = {name: sys.modules.get(name) for name in ("fvcore", "fvcore.nn")}
    sys.modules["fvcore"] = fvcore
    sys.modules["fvcore.nn"] = fvcore_nn
    logger.debug("Using lightweight fvcore.nn.flop_count stub for CVLFace model loading")
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@contextmanager
def _cvlface_repo_context(model_dir: str) -> T.Iterator[None]:
    """Temporarily make one CVLFace snapshot importable as a local Python package."""
    cwd = os.getcwd()
    added_path = False
    old_models = _pop_modules("models")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
        added_path = True
    try:
        os.chdir(model_dir)
        with _cvlface_dependency_context():
            yield
    finally:
        _pop_modules("models")
        sys.modules.update(old_models)
        os.chdir(cwd)
        if added_path:
            with suppress(ValueError):
                sys.path.remove(model_dir)


def _load_cvlface_wrapper(model_dir: str, model_label: str) -> T.Any:
    """Load a CVLFace wrapper directly, bypassing Transformers model finalization."""
    wrapper_path = osp.join(model_dir, "wrapper.py")
    module_name = f"_faceswap_cvlface_{model_label.lower()}_{abs(hash(model_dir))}"
    spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {model_label} wrapper from {wrapper_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        config = module.ModelConfig()
        return module.CVLFaceRecognitionModel(config)
    finally:
        sys.modules.pop(module_name, None)


def _make_cvlface_loader(
    repo_id: str, *, model_label: str, embedding_dim: int, force_cpu: bool = False
) -> T.Callable[[], LoadedIdentityModel]:
    """Return a loader for a CVLFace Transformers/safetensors recognition model."""

    def _load() -> LoadedIdentityModel:
        try:
            torch = import_module("torch")
            import_module("transformers")
            torch_utils = import_module("lib.torch_utils")
        except ImportError as exc:
            raise ImportError(
                f"{model_label} identity extraction requires 'torch' and 'transformers'."
            ) from exc

        device = torch_utils.get_device(cpu=force_cpu)
        model_dir = _download_cvlface_snapshot(repo_id, model_label)
        with (
            _CVLFACE_LOAD_LOCK,
            _cvlface_repo_context(model_dir),
            _third_party_output_to_debug(f"{model_label} CVLFace"),
        ):
            model = _load_cvlface_wrapper(model_dir, model_label)
        model.to(device)
        model.eval()
        logger.info("%s CVLFace model loaded on %s from %s", model_label, device, repo_id)

        def _extract_tensor(output: T.Any) -> T.Any:
            """Extract the embedding tensor from common Transformers/custom-code outputs."""
            if hasattr(torch, "is_tensor") and torch.is_tensor(output):
                return output
            if isinstance(output, dict):
                for key in ("embeddings", "embedding", "features", "logits", "last_hidden_state"):
                    if key in output:
                        return _extract_tensor(output[key])
                for value in output.values():
                    try:
                        return _extract_tensor(value)
                    except TypeError:
                        continue
            if isinstance(output, (list, tuple)):
                for value in output:
                    try:
                        return _extract_tensor(value)
                    except TypeError:
                        continue
            for attr in ("embeddings", "embedding", "features", "logits", "last_hidden_state"):
                if hasattr(output, attr):
                    return _extract_tensor(getattr(output, attr))
            raise TypeError(f"Unsupported {model_label} model output type: {type(output)!r}")

        class _CVLFace:
            def __init__(self) -> None:
                self._device = device
                self._model = model

            def _prepare(self, faces: np.ndarray) -> T.Any:
                """Return RGB NCHW float32 tensor normalized to [-1, 1]."""
                if faces.ndim != 4:
                    raise ValueError(
                        f"{model_label} expects a 4D face batch, received shape {faces.shape}."
                    )
                # Prefer NHWC since the framework hands aligned faces in that layout; only
                # treat (N, 3, H, W) as NCHW when the trailing axis cannot plausibly be a
                # colour channel.
                if faces.shape[-1] == 3:
                    rgb = np.ascontiguousarray(faces[..., ::-1])
                    inputs = torch.from_numpy(rgb).to(dtype=torch.float32).permute(0, 3, 1, 2)
                elif faces.shape[1] == 3:
                    rgb = np.ascontiguousarray(faces[:, ::-1, :, :])
                    inputs = torch.from_numpy(rgb).to(dtype=torch.float32)
                else:
                    raise ValueError(
                        f"{model_label} expected BGR face batch in NCHW or NHWC layout, "
                        f"received shape {faces.shape}."
                    )
                inputs = inputs.mul_(1.0 / 255.0).sub_(0.5).div_(0.5)
                if self._device.type == "cuda":
                    inputs = inputs.pin_memory().to(self._device, non_blocking=True)
                else:
                    inputs = inputs.to(self._device)
                return inputs.contiguous()

            def _run(self, faces: np.ndarray) -> np.ndarray:
                if len(faces) == 0:
                    return np.zeros((0, embedding_dim), dtype=np.float32)
                with torch.inference_mode():
                    output = self._model(self._prepare(faces))
                    embeddings = _extract_tensor(output)
                    if embeddings.ndim == 1:
                        embeddings = embeddings.unsqueeze(0)
                    if embeddings.ndim > 2:
                        embeddings = embeddings.reshape(embeddings.shape[0], -1)
                    embeddings = embeddings.to("cpu", dtype=torch.float32).numpy()
                return T.cast(np.ndarray, np.ascontiguousarray(embeddings, dtype=np.float32))

            def embed(self, faces: np.ndarray) -> np.ndarray:
                try:
                    return self._run(faces)
                except RuntimeError as gpu_err:
                    if self._device.type == "cpu":
                        raise
                    # Try CPU once. If it also fails the issue is not device-specific
                    # (e.g. a warmup probe with a deliberately wrong channel layout) — restore
                    # the accelerator so real inference still uses it.
                    original_device = self._device
                    cpu = torch.device("cpu")
                    self._model.to(cpu)
                    self._device = cpu
                    try:
                        result = self._run(faces)
                    except RuntimeError:
                        self._model.to(original_device)
                        self._device = original_device
                        raise
                    logger.warning(
                        "%s failed on %s. Falling back to CPU.",
                        model_label,
                        original_device,
                        exc_info=gpu_err,
                    )
                    return result

        return _CVLFace()

    return _load


def _make_insightface_loader(
    model_type: str, *, embedding_dim: int, force_cpu: bool = False
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

        # ONNX Runtime writes CoreML partition warnings directly to stderr. Keep them out
        # of normal Faceswap logs; real model-load failures still raise exceptions below.
        with suppress(Exception):
            ort.set_default_logger_severity(3)  # ERROR

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
            with _third_party_output_to_debug(f"InsightFace {model_type}"):
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
            with _third_party_output_to_debug(f"InsightFace {model_type} CPU fallback"):
                recognition = _load_recognition(providers)
                recognition.prepare(ctx_id=ctx_id)

        logger.info(
            "InsightFace %s recognition providers: %s",
            model_type,
            recognition.session.get_providers(),
        )

        class _InsightFace:
            def embed(self, faces: np.ndarray) -> np.ndarray:
                if len(faces) == 0:
                    return np.zeros((0, embedding_dim), dtype=np.float32)
                # ``get_feat`` accepts a list of faces and runs one batched ONNX call,
                # which avoids paying per-image kernel launch + provider dispatch.
                output = np.asarray(recognition.get_feat(list(faces)), dtype=np.float32)
                return T.cast(np.ndarray, output.reshape(len(faces), -1))

        return _InsightFace()

    return _load


ADAFACE = IdentityModelAdapter(
    name="adaface",
    embedding_dim=512,
    input_size=112,
    normalized=True,
    provenance=ModelProvenance(
        source="minchul/cvlface_adaface_ir101_webface12m",
        license="MIT",
        commercial_use=True,
        notes="CVLFace AdaFace iResNet101 WebFace12M, Kim et al., CVPR 2022.",
    ),
    loader=_make_cvlface_loader(
        "minchul/cvlface_adaface_ir101_webface12m", model_label="AdaFace", embedding_dim=512
    ),
)

ARCFACE = IdentityModelAdapter(
    name="arcface",
    embedding_dim=512,
    input_size=112,
    normalized=True,
    provenance=ModelProvenance(
        source="minchul/cvlface_arcface_ir101_webface4m",
        license="non-commercial",
        commercial_use=False,
        notes="CVLFace ArcFace iResNet101 WebFace4M, Deng et al., CVPR 2019.",
    ),
    loader=_make_cvlface_loader(
        "minchul/cvlface_arcface_ir101_webface4m", model_label="ArcFace", embedding_dim=512
    ),
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
        loader=_make_insightface_loader("antelopev2", embedding_dim=512),
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
        loader=_make_insightface_loader("buffalo_l", embedding_dim=512),
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
        loader=_make_insightface_loader("buffalo_sc", embedding_dim=512),
    ),
}


def cvlface_adapter(
    adapter: IdentityModelAdapter, *, repo_id: str, model_label: str, force_cpu: bool = False
) -> IdentityModelAdapter:
    """Return a CVLFace adapter with the requested runtime device policy."""
    return replace(
        adapter,
        loader=_make_cvlface_loader(
            repo_id,
            model_label=model_label,
            embedding_dim=adapter.embedding_dim,
            force_cpu=force_cpu,
        ),
    )


def adaface_adapter(*, force_cpu: bool = False) -> IdentityModelAdapter:
    """Return the configured AdaFace adapter."""
    return cvlface_adapter(
        ADAFACE,
        repo_id="minchul/cvlface_adaface_ir101_webface12m",
        model_label="AdaFace",
        force_cpu=force_cpu,
    )


def arcface_adapter(*, force_cpu: bool = False) -> IdentityModelAdapter:
    """Return the configured ArcFace adapter."""
    return cvlface_adapter(
        ARCFACE,
        repo_id="minchul/cvlface_arcface_ir101_webface4m",
        model_label="ArcFace",
        force_cpu=force_cpu,
    )


def insightface_adapter(model_type: str, *, force_cpu: bool = False) -> IdentityModelAdapter:
    """Return the configured InsightFace adapter."""
    if model_type not in INSIGHTFACE:
        raise ValueError(
            f"Unsupported InsightFace model_type '{model_type}'. "
            f"Select from {sorted(INSIGHTFACE)}."
        )
    adapter = INSIGHTFACE[model_type]
    return replace(
        adapter,
        loader=_make_insightface_loader(
            model_type, embedding_dim=adapter.embedding_dim, force_cpu=force_cpu
        ),
    )


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
