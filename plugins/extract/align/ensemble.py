#!/usr/bin/env python3
"""Landmark ensemble aligner plugin.

The ensemble runs model adapters on a shared aligner crop, converts every
prediction into canonical 68-point original-frame pixels, fuses in that common
space, then maps the fused result back to normalized crop coordinates for
Faceswap's normal aligner post-processing path.
"""

from __future__ import annotations

import importlib
import logging
import typing as T

import numpy as np

from lib.landmarks.adapters import (
    FaceswapAlignerAdapter,
    LandmarkAdapter,
    LandmarkAdapterConfig,
)
from lib.landmarks.coordinates import (
    frame_to_normalized_crop,
    roi_to_matrix,
)
from lib.landmarks.ensemble.alignment_resolver import (
    AlignmentResolverConfig,
    AlignmentResolverError,
    CandidateInput,
    resolve_alignment_geometry,
)
from lib.landmarks.ensemble.outliers import weighted_median
from lib.landmarks.ensemble.promoted_setup import (
    PromotedSetup,
    PromotedSetupError,
    ensure_compatible_adapters,
    load_promoted_setup,
    strategy_supported_by_runtime,
)
from lib.landmarks.ensemble.strategies import (
    CANONICAL_STRATEGIES,
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
)
from lib.landmarks.fusion import (
    FusionResult,
    normalize_weight_matrix,
    plain_average,
    static_weighted,
)
from lib.landmarks.schema import CANONICAL_SCHEMA, LandmarkPrediction
from lib.utils import get_module_objects
from plugins.extract.base import ExtractPlugin

from . import ensemble_defaults as cfg

logger = logging.getLogger(__name__)


_PLUGIN_CLASSES = {
    "hrnet": ("plugins.extract.align.hrnet", "HRNet", "2d_68"),
    "spiga": ("plugins.extract.align.spiga", "SPIGA", None),
    "orformer": ("plugins.extract.align.orformer", "ORFormer", None),
}


class Ensemble(ExtractPlugin):
    """Faceswap aligner that fuses predictions from landmark adapters."""

    def __init__(
        self,
        adapters: T.Sequence[LandmarkAdapter] | None = None,
        *,
        crop_scale: float | None = None,
        reject_outliers: bool | None = None,
        outlier_threshold: float | None = None,
        min_models: int | None = None,
        strategy: str | None = None,
        setup_path: str | None = None,
        setup_mode: str | None = None,
        fallback_strategy: str | None = None,
        use_alignment_resolver: bool | None = None,
        resolver_hard_case_strategy: str | None = None,
        resolver_high_disagreement_px: float | None = None,
    ) -> None:
        super().__init__(
            input_size=256,
            batch_size=cfg.batch_size(),
            is_rgb=True,
            dtype="float32",
            scale=(0, 1),
        )
        self.realign_centering = "legacy"
        self._injected_adapters = list(adapters) if adapters is not None else None
        self._crop_scale = cfg.crop_scale() if crop_scale is None else crop_scale
        raw_strategy = cfg.strategy() if strategy is None else strategy
        raw_reject_outliers = cfg.reject_outliers() if reject_outliers is None else reject_outliers
        configured_strategy = self._resolve_strategy(raw_strategy, raw_reject_outliers)
        configured_threshold = (
            cfg.outlier_threshold() if outlier_threshold is None else outlier_threshold
        )

        raw_setup_path = cfg.setup_path() if setup_path is None else setup_path
        # When the caller passes ``setup_path`` explicitly but omits
        # ``setup_mode``, default to strict so misconfigured deployments fail
        # fast. ``setup_mode=None`` with no explicit path falls through to the
        # configured value (defaulting to ``off``).
        if setup_mode is None:
            raw_setup_mode = "strict" if setup_path is not None else cfg.setup_mode()
        else:
            raw_setup_mode = setup_mode
        raw_fallback_strategy = (
            cfg.fallback_strategy() if fallback_strategy is None else fallback_strategy
        )
        self._setup_path = str(raw_setup_path or "")
        self._setup_mode = self._resolve_setup_mode(self._setup_path, raw_setup_mode)
        self._fallback_strategy = self._resolve_fallback_strategy(
            raw_fallback_strategy, configured_strategy
        )
        self._promoted: PromotedSetup | None = None
        self._promoted_failure: str = ""
        if self._setup_mode != "off":
            self._promoted = self._load_promoted_setup(self._setup_path, self._setup_mode)

        if self._promoted is not None:
            self._strategy = self._promoted.strategy
            self._outlier_threshold = (
                self._promoted.outlier_threshold
                if self._promoted.outlier_threshold is not None
                else configured_threshold
            )
        elif self._setup_mode == "fallback":
            # Promoted load failed; honor the configured fallback strategy.
            self._strategy = self._fallback_strategy
            self._outlier_threshold = configured_threshold
        else:
            self._strategy = configured_strategy
            self._outlier_threshold = configured_threshold
        self._outlier_method = strategy_outlier_method(self._strategy)
        self._uses_threshold = strategy_uses_threshold(self._strategy)
        self._requires_weights = strategy_requires_weights(self._strategy)
        self._min_models = cfg.min_models() if min_models is None else min_models
        self._use_resolver = bool(
            cfg.use_alignment_resolver()
            if use_alignment_resolver is None
            else use_alignment_resolver
        )
        self._resolver_hard_case = (
            cfg.resolver_hard_case_strategy()
            if resolver_hard_case_strategy is None
            else resolver_hard_case_strategy
        )
        self._resolver_disagreement_px = float(
            cfg.resolver_high_disagreement_px()
            if resolver_high_disagreement_px is None
            else resolver_high_disagreement_px
        )
        self._last_matrices: np.ndarray | None = None
        self._last_detector_bboxes: np.ndarray | None = None
        self.last_debug_metadata: list[dict[str, T.Any]] = []
        self.model: list[LandmarkAdapter]

    @staticmethod
    def _resolve_setup_mode(setup_path: str, configured_mode: str | None) -> str:
        """Resolve effective setup_mode.

        - Empty ``setup_path`` always disables setup loading.
        - Non-empty ``setup_path`` with an omitted / empty ``setup_mode`` defaults
          to ``strict`` so misconfigured deployments fail fast.
        - Any other explicit value is honored and validated against the
          supported choices.
        """
        if not setup_path:
            return "off"
        mode = (configured_mode or "").strip() or "strict"
        if mode not in {"off", "strict", "fallback"}:
            raise ValueError(
                f"unsupported setup_mode {mode!r}; expected one of off, strict, fallback"
            )
        return mode

    @staticmethod
    def _resolve_fallback_strategy(fallback: str | None, configured: str) -> str:
        """Resolve which strategy to use when fallback mode triggers."""
        value = (fallback or "").strip() or "plain_average"
        if value == "adapter_config":
            return configured
        return canonical_strategy(value)

    def _load_promoted_setup(self, path: str, mode: str) -> PromotedSetup | None:
        """Load and validate the promoted setup; honor strict vs fallback semantics."""
        try:
            setup = load_promoted_setup(path)
            strategy_supported_by_runtime(setup.strategy, CANONICAL_STRATEGIES)
        except PromotedSetupError as err:
            self._promoted_failure = str(err)
            if mode == "strict":
                raise
            logger.warning(
                "[Ensemble] promoted setup %r failed strict validation; falling back to "
                "strategy %r: %s",
                path,
                self._fallback_strategy,
                err,
            )
            return None
        return setup

    @staticmethod
    def _resolve_strategy(strategy: str, reject_outliers: bool) -> str:
        """Resolve a configured strategy + legacy ``reject_outliers`` flag.

        ``reject_outliers`` is retained only as a compatibility flag for the
        ``static_weighted`` strategy. When both are set, the run is promoted to
        ``static_weighted_hard_drop`` and a deprecation note is logged. Every
        other strategy ignores ``reject_outliers``.
        """
        canonical = canonical_strategy(strategy)
        if not reject_outliers:
            return canonical
        if canonical == "static_weighted":
            logger.info(
                "[Ensemble] 'reject_outliers=True' with 'static_weighted' is deprecated; "
                "treating run as 'static_weighted_hard_drop'. Set 'strategy=static_weighted_hard_drop' "
                "directly to silence this notice."
            )
            return "static_weighted_hard_drop"
        logger.info(
            "[Ensemble] 'reject_outliers=True' is ignored; strategy %r already governs "
            "outlier behavior.",
            canonical,
        )
        return canonical

    def load_model(self) -> list[LandmarkAdapter]:
        """Load configured adapters.

        Injected adapters are returned as-is for tests. Real adapters are only
        imported when their plugin modules exist in the local tree.
        """
        adapters = (
            list(self._injected_adapters)
            if self._injected_adapters is not None
            else self._build_configured_adapters()
        )
        loaded = [adapter for adapter in adapters if adapter.config.enabled]
        for adapter in loaded:
            if hasattr(adapter, "load_model"):
                adapter.load_model()  # type: ignore[attr-defined]
        if not loaded:
            raise ValueError("No enabled landmark ensemble adapters are available")
        if self._promoted is not None:
            ensure_compatible_adapters(
                self._promoted,
                [adapter.config.name for adapter in loaded],
            )
        logger.info(
            "Loaded landmark ensemble adapters: %s",
            ", ".join(adapter.config.name for adapter in loaded),
        )
        return loaded

    def _build_configured_adapters(self) -> list[LandmarkAdapter]:
        """Create adapters for configured aligner plugins that are importable."""
        adapters: list[LandmarkAdapter] = []
        for name in cfg.models():
            if name not in _PLUGIN_CLASSES:
                logger.warning("[Ensemble] Unknown adapter '%s'; skipping", name)
                continue
            module_name, class_name, schema = _PLUGIN_CLASSES[name]
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                logger.info(
                    "[Ensemble] Optional adapter '%s' is not installed; skipping",
                    name,
                )
                continue
            plugin_cls = getattr(module, class_name)
            plugin = plugin_cls()
            adapter_schema = schema or self._schema_from_plugin(plugin)
            adapters.append(
                FaceswapAlignerAdapter(
                    LandmarkAdapterConfig(
                        name=name,
                        schema=adapter_schema,
                        coordinate_space="normalized_crop",
                    ),
                    plugin,
                    input_is_rgb=self.is_rgb,
                    input_scale=self.scale,
                )
            )
        return adapters

    @staticmethod
    def _schema_from_plugin(plugin: object) -> str:
        """Infer an adapter schema from known model configuration attributes."""
        model_config = getattr(plugin, "_model_config", None)
        count = getattr(model_config, "num_landmarks", 68)
        return f"2d_{count}"

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Format detection boxes into a shared square ensemble crop."""
        heights = batch[:, 3] - batch[:, 1]
        widths = batch[:, 2] - batch[:, 0]
        ctr_x = np.rint((batch[:, 0] + batch[:, 2]) * 0.5).astype("int32")
        ctr_y = np.rint((batch[:, 1] + batch[:, 3]) * 0.5).astype("int32")
        side = np.maximum(widths, heights) * self._crop_scale
        half = np.rint(side * 0.5).astype("int32")

        retval = np.empty((batch.shape[0], 4), dtype=np.int32)
        retval[:, 0] = ctr_x - half
        retval[:, 1] = ctr_y - half
        retval[:, 2] = ctr_x + half
        retval[:, 3] = ctr_y + half
        self._last_matrices = roi_to_matrix(retval)
        # Cache the raw detector bboxes per face so the geometry resolver can
        # consume bbox-aspect / ROI-ratio / outside-bbox signals at runtime.
        self._last_detector_bboxes = batch.astype("float32", copy=True)
        return retval

    def _active_adapters(self) -> list[LandmarkAdapter]:
        """Return loaded or injected adapters."""
        model = getattr(self, "model", None)
        if model is not None:
            return [adapter for adapter in model if adapter.config.enabled]
        if self._injected_adapters is not None:
            return [adapter for adapter in self._injected_adapters if adapter.config.enabled]
        raise ValueError("Ensemble adapters have not been loaded")

    def _matrices_for_batch(self, batch_size: int) -> np.ndarray:
        """Return crop-to-frame matrices, falling back to identity for warmup calls."""
        if self._last_matrices is not None and self._last_matrices.shape[0] == batch_size:
            return self._last_matrices
        matrices = np.repeat(np.eye(3, dtype="float32")[None], batch_size, axis=0)
        return matrices

    def _collect_predictions(
        self, batch: np.ndarray, matrices: np.ndarray
    ) -> tuple[list[list[tuple[LandmarkAdapter, LandmarkPrediction]]], list[str]]:
        """Run adapters and bucket successful predictions by face index."""
        per_face: list[list[tuple[LandmarkAdapter, LandmarkPrediction]]] = [
            [] for _ in range(batch.shape[0])
        ]
        errors: list[str] = []
        for adapter in self._active_adapters():
            try:
                predictions = adapter.predict_batch(batch, matrices=matrices)
            except Exception as err:  # pylint:disable=broad-except
                logger.warning("[Ensemble] Adapter '%s' failed: %s", adapter.config.name, err)
                errors.append(f"{adapter.config.name}: {err}")
                continue
            if len(predictions) != batch.shape[0]:
                message = (
                    f"{adapter.config.name}: expected {batch.shape[0]} predictions, "
                    f"got {len(predictions)}"
                )
                logger.warning("[Ensemble] %s", message)
                errors.append(message)
                continue
            for idx, prediction in enumerate(predictions):
                per_face[idx].append((adapter, prediction))
        return per_face, errors

    def _weights_for_face(self, adapters: T.Sequence[LandmarkAdapter]) -> np.ndarray:
        """Return the per-face weight vector or matrix for the active adapters.

        For a promoted setup the per-landmark weight matrix is subset to the
        successful adapters and renormalized so each landmark column still sums
        to 1.0; otherwise the legacy scalar-per-adapter vector is used.
        """
        if self._promoted is not None and self._promoted.weights:
            promoted_weights = self._promoted.weights
            subset = np.array(
                [promoted_weights[adapter.config.name] for adapter in adapters],
                dtype="float32",
            )
            totals = subset.sum(axis=0)
            safe = np.where(totals > 0, totals, 1.0)
            return subset / safe[None, :]
        return np.array([adapter.config.weight for adapter in adapters], dtype="float32")

    def _resolve_via_geometry(
        self,
        *,
        adapters: list[LandmarkAdapter],
        items: list[LandmarkPrediction],
        errors: list[str],
        threshold: float | None,
        detector_bbox: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray | None:
        """Route this face through the geometry-risk resolver (#78).

        Returns the fused points when the resolver makes a decision, or
        ``None`` to fall through to the regular dispatch path (e.g. when the
        resolver hard-fails and we want the existing fallback semantics).
        """
        weights_map: dict[str, list[float]] | None = None
        if self._promoted is not None and self._promoted.weights:
            weights_map = {model: list(values) for model, values in self._promoted.weights.items()}
        candidates = [
            CandidateInput(
                model=adapter.config.name,
                landmarks=prediction.canonical_68().points,
            )
            for adapter, prediction in zip(adapters, items, strict=True)
        ]
        resolver_config = AlignmentResolverConfig(
            general_strategy=self._strategy,
            hard_case_strategy=canonical_strategy(self._resolver_hard_case),
            fallback_strategy=canonical_strategy(self._fallback_strategy)
            if self._fallback_strategy
            else "plain_average",
            outlier_threshold=self._outlier_threshold,
            weights=weights_map,
            high_disagreement_px=self._resolver_disagreement_px,
            min_models_after_rejection=self._min_models,
        )
        try:
            result = resolve_alignment_geometry(
                candidates,
                config=resolver_config,
                detector_bbox=detector_bbox,
                image_shape=None,
            )
        except AlignmentResolverError as err:
            logger.warning("[Ensemble] geometry resolver hard-failed: %s", err)
            return None

        self.last_debug_metadata.append(
            {
                "sources": tuple(adapter.config.name for adapter in adapters),
                "weights": [],  # resolver does not surface a weight matrix
                "kept_indices": tuple(range(len(result.active_models))),
                "rejected_indices": tuple(
                    idx
                    for idx, adapter in enumerate(adapters)
                    if adapter.config.name in result.rejected_models
                ),
                "rejected_landmarks": 0,
                "adapter_errors": tuple(errors),
                "strategy": result.chosen_strategy,
                "outlier_method": strategy_outlier_method(result.chosen_strategy),
                "outlier_threshold": threshold,
                "setup_path": self._setup_path,
                "setup_mode": self._setup_mode,
                "promoted_candidate_id": (
                    self._promoted.candidate_id if self._promoted is not None else ""
                ),
                "weight_source": "geometry_resolver",
                "active_models": result.active_models,
                "resolver": {
                    "risk_route": result.risk_route,
                    "risk_score": result.risk_score,
                    "geometry_confidence": result.geometry_confidence,
                    "geometry_flags": list(result.geometry_flags),
                    "rejected_models": list(result.rejected_models),
                    "max_disagreement_px": result.debug_metadata.get("max_disagreement_px", 0.0),
                    "detector_bbox": list(detector_bbox) if detector_bbox is not None else None,
                },
            }
        )
        return result.alignment_landmarks.astype("float32", copy=False)

    def _fuse_face(
        self,
        predictions: list[tuple[LandmarkAdapter, LandmarkPrediction]],
        errors: list[str],
        *,
        detector_bbox: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray:
        """Fuse one face's adapter predictions and return frame-space points.

        ``detector_bbox`` is the per-face source-frame detection box that
        ``pre_process`` saw (left/top/right/bottom). When supplied, the
        geometry resolver uses it for bbox-aspect, ROI-ratio, and
        landmarks-outside-bbox signals; otherwise those signals stay inactive
        (e.g. for warmup calls that bypass ``pre_process``).
        """
        if len(predictions) < self._min_models:
            raise ValueError(
                "Not enough successful landmark adapters for ensemble face: "
                f"required {self._min_models}, got {len(predictions)}"
            )
        adapters = [adapter for adapter, _prediction in predictions]
        items = [prediction for _adapter, prediction in predictions]
        weights = self._weights_for_face(adapters)
        threshold = self._outlier_threshold if self._uses_threshold else None

        if self._use_resolver:
            resolver_points = self._resolve_via_geometry(
                adapters=adapters,
                items=items,
                errors=errors,
                threshold=threshold,
                detector_bbox=detector_bbox,
            )
            if resolver_points is not None:
                return resolver_points

        if not self._requires_weights:
            fused = plain_average(
                items,
                outlier_method=self._outlier_method,
                outlier_threshold=self._outlier_threshold,
            )
        elif self._strategy == "weighted_median":
            stack = np.stack([item.canonical_68().points for item in items], axis=0)
            normalized = normalize_weight_matrix(
                weights,
                model_count=len(items),
                landmark_count=stack.shape[1],
            )
            fused = FusionResult(
                points=weighted_median(stack, normalized),
                schema=CANONICAL_SCHEMA,
                strategy="weighted_median",
                weights=normalized,
                sources=tuple(adapter.config.name for adapter in adapters),
                kept_indices=tuple(range(len(items))),
            )
        else:
            fused = static_weighted(
                items,
                weights=weights,
                outlier_method=self._outlier_method,
                outlier_threshold=self._outlier_threshold,
            )

        self.last_debug_metadata.append(
            {
                "sources": fused.sources,
                "weights": fused.weights.tolist(),
                "kept_indices": fused.kept_indices,
                "rejected_indices": fused.rejected_indices,
                "rejected_landmarks": fused.rejected_landmarks,
                "adapter_errors": tuple(errors),
                "strategy": self._strategy,
                "outlier_method": self._outlier_method,
                "outlier_threshold": threshold,
                "setup_path": self._setup_path,
                "setup_mode": self._setup_mode,
                "promoted_candidate_id": (
                    self._promoted.candidate_id if self._promoted is not None else ""
                ),
                "weight_source": (
                    "promoted_setup" if self._promoted is not None else "adapter_config"
                ),
                "active_models": tuple(adapter.config.name for adapter in adapters),
            }
        )
        return fused.points

    def predict_landmarks_68(
        self,
        image: np.ndarray,
        *,
        matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return fused canonical ``(68, 2)`` landmarks in original-frame pixels.

        The input image is a prepared ensemble crop. ``matrix`` maps normalized
        crop coordinates for that crop into the original frame. If omitted, an
        identity matrix is used, matching warmup/test calls that already operate
        in frame space.
        """
        matrices = (
            np.eye(3, dtype="float32")[None]
            if matrix is None
            else np.asarray(matrix, dtype="float32")[None]
        )
        per_face, errors = self._collect_predictions(image[None], matrices)
        self.last_debug_metadata = []
        return self._fuse_face(per_face[0], errors, detector_bbox=None)

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Run adapter predictions, fuse in frame space and return normalized landmarks."""
        matrices = self._matrices_for_batch(batch.shape[0])
        per_face, errors = self._collect_predictions(batch, matrices)
        self.last_debug_metadata = []
        output = np.empty((batch.shape[0], 68, 2), dtype="float32")
        for idx, predictions in enumerate(per_face):
            output[idx] = frame_to_normalized_crop(
                self._fuse_face(predictions, errors, detector_bbox=self._bbox_for_face(idx)),
                matrices[idx],
            )
        return output

    def _bbox_for_face(self, face_index: int) -> tuple[float, float, float, float] | None:
        """Return the cached detector bbox for one face index, or ``None`` if unavailable."""
        if self._last_detector_bboxes is None:
            return None
        if face_index >= self._last_detector_bboxes.shape[0]:
            return None
        bbox = self._last_detector_bboxes[face_index]
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


__all__ = get_module_objects(__name__)
