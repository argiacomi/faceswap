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
    normalized_crop_to_frame,
    roi_to_matrix,
)
from lib.landmarks.core.fusion import (
    FusionResult,
    normalize_weight_matrix,
    plain_average,
    static_weighted,
)
from lib.landmarks.core.schema import CANONICAL_SCHEMA, LandmarkPrediction
from lib.landmarks.ensemble.outliers import weighted_median
from lib.landmarks.ensemble.promoted_setup import (
    PromotedSetup,
    PromotedSetupError,
    ensure_compatible_adapters,
    load_promoted_setup,
    strategy_supported_by_runtime,
)
from lib.landmarks.ensemble.runtime_resolver import (
    ModelPrediction,
    RuntimeResolverConfig,
    RuntimeResolverError,
    resolve_runtime,
)
from lib.landmarks.ensemble.runtime_resolver_scorer import load_runtime_resolver_scorer
from lib.landmarks.ensemble.strategies import (
    CANONICAL_STRATEGIES,
    canonical_strategy,
    strategy_outlier_method,
    strategy_requires_weights,
    strategy_uses_threshold,
)
from lib.utils import get_module_objects
from plugins.extract.base import ExtractPlugin

from . import ensemble_defaults as cfg

logger = logging.getLogger(__name__)


def _trace(message: str, *args: T.Any) -> None:
    """Log at Faceswap TRACE level when available."""
    trace = getattr(logger, "trace", None)
    if callable(trace):
        trace(message, *args)


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
        hard_case_strategy: str | None = None,
        hard_disagreement_px: float | None = None,
        resolver_policy: str | None = None,
        resolver_scorer_path: str | None = None,
        secondary_hard_case_strategy: str | None = None,
        fallback_model: str | None = None,
        strict: bool | None = None,
        roll_veto_degrees: float | None = None,
        hard_roll_degrees: float | None = None,
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
        if setup_mode is None:
            raw_setup_mode = "strict" if setup_path is not None else cfg.setup_mode()
        else:
            raw_setup_mode = setup_mode
        raw_fallback_strategy = (
            cfg.fallback_strategy() if fallback_strategy is None else fallback_strategy
        )
        self._setup_path = str(raw_setup_path or "")
        self._weights_path = str(cfg.weights_path() or "")
        self._setup_mode = self._resolve_setup_mode(self._setup_path, raw_setup_mode)
        self._fallback_strategy = self._resolve_fallback_strategy(
            raw_fallback_strategy, configured_strategy
        )
        self._resolver_policy = (
            cfg.resolver_policy() if resolver_policy is None else resolver_policy
        )
        self._resolver_scorer_path = (
            cfg.resolver_scorer_path() if resolver_scorer_path is None else resolver_scorer_path
        )
        self._secondary_hard_case = (
            cfg.secondary_hard_case_strategy()
            if secondary_hard_case_strategy is None
            else secondary_hard_case_strategy
        )
        self._fallback_model = cfg.fallback_model() if fallback_model is None else fallback_model
        self._strict = bool(cfg.strict() if strict is None else strict)
        self._roll_veto_degrees = float(
            cfg.roll_veto_degrees() if roll_veto_degrees is None else roll_veto_degrees
        )
        self._hard_roll_degrees = float(
            cfg.hard_roll_degrees() if hard_roll_degrees is None else hard_roll_degrees
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
            cfg.hard_case_strategy() if hard_case_strategy is None else hard_case_strategy
        )
        self._resolver_disagreement_px = float(
            cfg.hard_disagreement_px() if hard_disagreement_px is None else hard_disagreement_px
        )
        self._last_matrices: np.ndarray | None = None
        self._last_detector_bboxes: np.ndarray | None = None
        self.last_debug_metadata: list[dict[str, T.Any]] = []
        self._runtime_scorer: T.Any = None
        self.model: list[LandmarkAdapter]
        logger.debug(
            "[Ensemble] init strategy=%s setup_mode=%s setup_path=%s promoted=%s "
            "resolver=%s scorer=%s hard_case=%s secondary=%s fallback=%s strict=%s",
            self._strategy,
            self._setup_mode,
            self._setup_path,
            self._promoted.candidate_id if self._promoted is not None else "",
            self._resolver_policy,
            bool(self._resolver_scorer_path),
            self._resolver_hard_case,
            self._secondary_hard_case,
            self._fallback_strategy,
            self._strict,
        )

    @staticmethod
    def _resolve_setup_mode(setup_path: str, configured_mode: str | None) -> str:
        """Resolve effective setup_mode."""
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
        logger.debug(
            "[Ensemble] loaded promoted setup candidate=%s strategy=%s models=%s crop_scale=%s",
            setup.candidate_id,
            setup.strategy,
            setup.models,
            setup.crop_scale,
        )
        return setup

    @staticmethod
    def _resolve_strategy(strategy: str, reject_outliers: bool) -> str:
        """Resolve a configured strategy + legacy ``reject_outliers`` flag."""
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
        """Load configured adapters."""
        # Construct the LightGBM Booster BEFORE Torch/MPS adapter loads.
        # If the booster is built later (inside the ensemble worker, after
        # Torch/MPS is warm) it can hang on macOS — instead, build it up
        # front, then hand the preloaded scorer to resolve_runtime so the
        # live path skips re-construction.
        if self._use_resolver and self._resolver_policy.startswith("learned_quality"):
            if not self._resolver_scorer_path:
                raise ValueError(
                    f"{self._resolver_policy} requires resolver_scorer_path"
                )
            self._runtime_scorer = load_runtime_resolver_scorer(self._resolver_scorer_path)
            if hasattr(self._runtime_scorer, "_booster"):
                logger.debug(
                    "[Ensemble] warming runtime LightGBM scorer before Torch/MPS adapters"
                )
                self._runtime_scorer._booster()
                logger.debug("[Ensemble] warmed runtime LightGBM scorer")
        adapters = (
            list(self._injected_adapters)
            if self._injected_adapters is not None
            else self._build_configured_adapters()
        )
        loaded = [adapter for adapter in adapters if adapter.config.enabled]
        if not loaded:
            raise ValueError("No enabled landmark ensemble adapters are available")
        if self._promoted is not None:
            ensure_compatible_adapters(
                self._promoted,
                [adapter.config.name for adapter in loaded],
            )
            loaded = self._filter_promoted_adapters(loaded)
        for adapter in loaded:
            if hasattr(adapter, "load_model"):
                adapter.load_model()  # type: ignore[attr-defined]
        logger.info(
            "Loaded landmark ensemble adapters: %s",
            ", ".join(adapter.config.name for adapter in loaded),
        )
        logger.debug(
            "[Ensemble] adapter config=%s",
            {
                adapter.config.name: {
                    "weight": adapter.config.weight,
                    "schema": adapter.config.schema,
                    "coordinate_space": adapter.config.coordinate_space,
                }
                for adapter in loaded
            },
        )
        return loaded

    def _filter_promoted_adapters(
        self, adapters: T.Sequence[LandmarkAdapter]
    ) -> list[LandmarkAdapter]:
        """Return adapters in promoted setup order, excluding non-promoted extras."""
        if self._promoted is None:
            return list(adapters)
        by_name = {adapter.config.name: adapter for adapter in adapters}
        selected = [by_name[model] for model in self._promoted.models]
        skipped = sorted(set(by_name).difference(self._promoted.models))
        if skipped:
            logger.info(
                "[Ensemble] Promoted setup uses adapters %s; skipping configured extras: %s",
                ", ".join(self._promoted.models),
                ", ".join(skipped),
            )
        return selected

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
        self.set_crop_matrices(roi_to_matrix(retval), detector_bboxes=batch)
        _trace(
            "[Ensemble] pre_process detector_bboxes=%s crop_rois=%s crop_scale=%s",
            batch.tolist(),
            retval.tolist(),
            self._crop_scale,
        )
        return retval

    def set_crop_matrices(
        self,
        matrices: np.ndarray,
        *,
        detector_bboxes: np.ndarray | None = None,
    ) -> None:
        """Receive the runner's current crop-to-frame matrices for the next feed batch."""
        self._last_matrices = np.asarray(matrices, dtype="float32").copy()
        if detector_bboxes is None:
            self._last_detector_bboxes = None
            _trace("[Ensemble] cached crop matrices without detector bboxes: %s", matrices)
            return
        bboxes = np.asarray(detector_bboxes, dtype="float32")
        if bboxes.shape[0] != self._last_matrices.shape[0]:
            logger.debug(
                "[Ensemble] Ignoring detector bboxes with shape %s for crop matrices %s",
                bboxes.shape,
                self._last_matrices.shape,
            )
            self._last_detector_bboxes = None
            return
        self._last_detector_bboxes = bboxes.copy()
        _trace(
            "[Ensemble] cached crop matrices=%s detector_bboxes=%s",
            self._last_matrices.tolist(),
            self._last_detector_bboxes.tolist(),
        )

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
        matrices = np.repeat(np.eye(3, dtype="float32")[None], batch_size, axis=0)
        if self._last_matrices is None:
            return matrices

        cached_count = self._last_matrices.shape[0]
        copy_count = min(cached_count, batch_size)
        matrices[:copy_count] = self._last_matrices[:copy_count]
        if cached_count != batch_size:
            logger.debug(
                "[Ensemble] adjusted cached crop matrices for batch padding: "
                "cached=%s batch=%s copied=%s",
                cached_count,
                batch_size,
                copy_count,
            )
        return matrices

    @staticmethod
    def _looks_normalized_against_bbox(
        points: np.ndarray,
        detector_bbox: tuple[float, float, float, float] | None,
    ) -> bool:
        """Return ``True`` when tiny normalized coordinates are paired with a large bbox."""
        if detector_bbox is None:
            return False
        left, top, right, bottom = detector_bbox
        bbox_w = right - left
        bbox_h = bottom - top
        if max(bbox_w, bbox_h) <= 100.0:
            return False
        arr = np.asarray(points, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] < 2 or not np.all(np.isfinite(arr[:, :2])):
            return False
        extent_x = float(np.max(arr[:, 0]) - np.min(arr[:, 0]))
        extent_y = float(np.max(arr[:, 1]) - np.min(arr[:, 1]))
        return extent_x < 2.0 and extent_y < 2.0

    @staticmethod
    def _is_identity_matrix(matrix: np.ndarray | None) -> bool:
        """Return whether ``matrix`` is effectively identity."""
        if matrix is None:
            return True
        return bool(np.allclose(np.asarray(matrix, dtype="float32"), np.eye(3, dtype="float32")))

    def _matrix_from_detector_bbox(
        self,
        detector_bbox: tuple[float, float, float, float] | None,
    ) -> np.ndarray | None:
        """Rebuild the shared square crop matrix from the detector bbox."""
        if detector_bbox is None:
            return None
        left, top, right, bottom = detector_bbox
        width = right - left
        height = bottom - top
        if width <= 0.0 or height <= 0.0:
            return None
        center_x = round((left + right) * 0.5)
        center_y = round((top + bottom) * 0.5)
        half = round(max(width, height) * self._crop_scale * 0.5)
        roi = np.asarray(
            [center_x - half, center_y - half, center_x + half, center_y + half],
            dtype="float32",
        )
        return roi_to_matrix(roi)

    def _frame_points_for_resolver(
        self,
        *,
        adapter_name: str,
        points: np.ndarray,
        detector_bbox: tuple[float, float, float, float] | None,
        crop_to_frame_matrix: np.ndarray | None,
    ) -> np.ndarray:
        """Ensure runtime resolver candidates are in original-frame coordinates."""
        frame_points = np.asarray(points, dtype="float32")
        if not self._looks_normalized_against_bbox(frame_points, detector_bbox):
            return frame_points
        if not self._is_identity_matrix(crop_to_frame_matrix):
            converted = normalized_crop_to_frame(frame_points, crop_to_frame_matrix)
            if not self._looks_normalized_against_bbox(converted, detector_bbox):
                logger.debug(
                    "[Ensemble] converted normalized resolver candidate to frame space: "
                    "adapter=%s bbox=%s",
                    adapter_name,
                    detector_bbox,
                )
                return converted.astype("float32", copy=False)
        fallback_matrix = self._matrix_from_detector_bbox(detector_bbox)
        if fallback_matrix is not None:
            converted = normalized_crop_to_frame(frame_points, fallback_matrix)
            if not self._looks_normalized_against_bbox(converted, detector_bbox):
                logger.debug(
                    "[Ensemble] reconstructed crop matrix for normalized resolver candidate: "
                    "adapter=%s bbox=%s",
                    adapter_name,
                    detector_bbox,
                )
                return converted.astype("float32", copy=False)

        raise RuntimeResolverError(
            "Runtime resolver received non-frame-space landmarks from "
            f"{adapter_name!r}; detector_bbox={detector_bbox}"
        )

    def _collect_predictions(
        self, batch: np.ndarray, matrices: np.ndarray
    ) -> tuple[list[list[tuple[LandmarkAdapter, LandmarkPrediction]]], list[str]]:
        """Run adapters and bucket successful predictions by face index."""
        per_face: list[list[tuple[LandmarkAdapter, LandmarkPrediction]]] = [
            [] for _ in range(batch.shape[0])
        ]
        errors: list[str] = []
        for adapter in self._active_adapters():
            logger.debug(
                "[Ensemble] running adapter=%s batch_shape=%s",
                adapter.config.name,
                batch.shape,
            )
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
            logger.debug(
                "[Ensemble] adapter=%s produced %d predictions",
                adapter.config.name,
                len(predictions),
            )
            _trace(
                "[Ensemble] adapter=%s prediction schemas=%s",
                adapter.config.name,
                [prediction.schema for prediction in predictions],
            )
            for idx, prediction in enumerate(predictions):
                per_face[idx].append((adapter, prediction))
        logger.debug(
            "[Ensemble] collected predictions per_face=%s errors=%s",
            [len(face_predictions) for face_predictions in per_face],
            errors,
        )
        return per_face, errors

    def _weights_for_face(self, adapters: T.Sequence[LandmarkAdapter]) -> np.ndarray:
        """Return the per-face weight vector or matrix for the active adapters."""
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

    @staticmethod
    def _roll_estimate(points: np.ndarray) -> float | None:
        """Return an approximate in-plane roll from eye centers, in degrees."""
        if points.shape[0] < 48:
            return None
        left_eye = points[36:42].mean(axis=0)
        right_eye = points[42:48].mean(axis=0)
        vector = right_eye - left_eye
        if float(np.linalg.norm(vector)) <= 1e-6:
            return None
        return float(np.degrees(np.arctan2(float(vector[1]), float(vector[0]))))

    @staticmethod
    def _consensus_distances(items: T.Sequence[LandmarkPrediction]) -> dict[str, float]:
        """Return simple prediction-only consensus distances for debug output."""
        if not items:
            return {"mean_landmark_distance_px": 0.0, "max_landmark_distance_px": 0.0}
        stack = np.stack([item.canonical_68().points for item in items], axis=0)
        consensus = np.median(stack, axis=0)
        per_model = np.mean(np.linalg.norm(stack - consensus[None], axis=2), axis=1)
        return {
            "mean_landmark_distance_px": float(per_model.mean()),
            "max_landmark_distance_px": float(per_model.max()),
        }

    @staticmethod
    def _prediction_availability(
        adapters: T.Sequence[LandmarkAdapter], errors: T.Sequence[str]
    ) -> dict[str, bool]:
        """Return model availability flags for runtime debug metadata."""
        available = {adapter.config.name: True for adapter in adapters}
        for message in errors:
            model = message.split(":", maxsplit=1)[0].strip()
            if model:
                available.setdefault(model, False)
        return dict(sorted(available.items()))

    def _base_runtime_debug(
        self,
        *,
        selected_candidate: str,
        fallback_used: bool,
        veto_reasons: T.Sequence[str],
        points: np.ndarray,
        items: T.Sequence[LandmarkPrediction],
        adapters: T.Sequence[LandmarkAdapter],
        errors: T.Sequence[str],
        geometry_valid: bool,
        detector_bbox: tuple[float, float, float, float] | None,
    ) -> dict[str, T.Any]:
        """Build the stable per-face runtime resolver debug envelope."""
        return {
            "selected_candidate": selected_candidate,
            "fallback_used": bool(fallback_used),
            "veto_reasons": list(veto_reasons),
            "roll_estimate": self._roll_estimate(points),
            "consensus_distances": self._consensus_distances(items),
            "geometry_valid": bool(geometry_valid),
            "model_predictions_available": self._prediction_availability(adapters, errors),
            "detector_bbox": list(detector_bbox) if detector_bbox is not None else None,
            "resolver_policy": self._resolver_policy,
            "resolver_scorer_path": self._resolver_scorer_path,
            "hard_case_strategy": self._resolver_hard_case,
            "secondary_hard_case_strategy": self._secondary_hard_case,
            "fallback_model": self._fallback_model,
            "strict": self._strict,
            "roll_veto_degrees": self._roll_veto_degrees,
            "hard_roll_degrees": self._hard_roll_degrees,
            "setup_path": self._setup_path,
            "weights_path": (
                self._promoted.weights_path if self._promoted is not None else self._weights_path
            ),
            "setup_mode": self._setup_mode,
            "promoted_candidate_id": (
                self._promoted.candidate_id if self._promoted is not None else ""
            ),
        }

    def _append_debug_metadata(self, metadata: dict[str, T.Any]) -> None:
        """Store and log per-face resolver metadata."""
        self.last_debug_metadata.append(metadata)
        logger.debug(
            "[Ensemble] runtime metadata selected=%s bucket=%s fallback=%s vetoed=%s",
            metadata.get("selected_candidate"),
            metadata.get("runtime_bucket"),
            metadata.get("fallback_reason"),
            metadata.get("vetoed"),
        )
        _trace("[Ensemble] runtime resolver metadata: %s", metadata)
        bucket = metadata.get("runtime_bucket")
        if bucket not in (None, "", "frontal", "intermediate", "no_pose"):
            logger.debug("[Ensemble] runtime resolver hard/profile metadata: %s", metadata)

    def _resolve_via_geometry(
        self,
        *,
        adapters: list[LandmarkAdapter],
        items: list[LandmarkPrediction],
        errors: list[str],
        threshold: float | None,
        detector_bbox: tuple[float, float, float, float] | None = None,
        image_crop: np.ndarray | None = None,
        crop_to_frame_matrix: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Route this face through the production runtime resolver."""
        weights_map: dict[str, list[float]] | None = None
        if self._promoted is not None and self._promoted.weights:
            weights_map = {model: list(values) for model, values in self._promoted.weights.items()}
        model_predictions = [
            ModelPrediction(
                adapter.config.name,
                self._frame_points_for_resolver(
                    adapter_name=adapter.config.name,
                    points=prediction.canonical_68().points,
                    detector_bbox=detector_bbox,
                    crop_to_frame_matrix=crop_to_frame_matrix,
                ),
                weight=float(adapter.config.weight),
            )
            for adapter, prediction in zip(adapters, items, strict=True)
        ]
        resolver_config = RuntimeResolverConfig(
            policy=self._resolver_policy,
            scorer_path=str(self._resolver_scorer_path or ""),
            general_strategy=self._strategy,
            hard_case_strategy=canonical_strategy(self._resolver_hard_case),
            secondary_hard_case_strategy=canonical_strategy(self._secondary_hard_case),
            fallback_strategy=canonical_strategy(self._fallback_strategy)
            if self._fallback_strategy
            else "plain_average",
            fallback_model=self._fallback_model,
            outlier_threshold=self._outlier_threshold,
            weights=weights_map,
            adapter_weights={adapter.config.name: adapter.config.weight for adapter in adapters},
            hard_disagreement_px=self._resolver_disagreement_px,
            roll_veto_degrees=self._roll_veto_degrees,
            hard_roll_degrees=self._hard_roll_degrees,
            strict=self._strict,
        )
        logger.debug(
            "[Ensemble] resolving via runtime policy=%s models=%s detector_bbox=%s "
            "image_crop=%s matrix=%s",
            resolver_config.policy,
            [prediction.model for prediction in model_predictions],
            detector_bbox,
            None if image_crop is None else image_crop.shape,
            None if crop_to_frame_matrix is None else crop_to_frame_matrix.tolist(),
        )
        try:
            result = resolve_runtime(
                model_predictions,
                resolver_config,
                detector_bbox=detector_bbox,
                image_crop=image_crop,
                crop_to_frame_matrix=crop_to_frame_matrix,
                preloaded_scorer=self._runtime_scorer,
            )
        except RuntimeResolverError as err:
            logger.warning("[Ensemble] runtime resolver hard-failed: %s", err)
            if self._strict:
                raise
            return None
        except Exception:  # noqa: BLE001 - diagnostic: surface unexpected crashes
            # Temporary widened catch to surface stack traces when the
            # runtime resolver silently swallows an unexpected exception
            # (e.g. lightgbm loading, scorer payload mismatch). Keep
            # narrowing once the failure mode is identified.
            logger.exception("[Ensemble] runtime resolver crashed")
            if self._strict:
                raise
            return None

        veto_reasons = list(result.metadata.get("vetoed", ()))
        metadata = self._base_runtime_debug(
            selected_candidate=result.selected_candidate,
            fallback_used=bool(result.metadata.get("fallback_reason")),
            veto_reasons=veto_reasons,
            points=result.landmarks,
            items=items,
            adapters=adapters,
            errors=errors,
            geometry_valid=not bool(veto_reasons),
            detector_bbox=detector_bbox,
        )
        selected_strategy = (
            result.selected_candidate
            if result.selected_candidate in CANONICAL_STRATEGIES
            else self._strategy
        )
        metadata.update(
            {
                "sources": tuple(adapter.config.name for adapter in adapters),
                "weights": [],
                "kept_indices": tuple(range(len(adapters))),
                "rejected_indices": (),
                "rejected_landmarks": 0,
                "adapter_errors": tuple(errors),
                "strategy": selected_strategy,
                "outlier_method": strategy_outlier_method(selected_strategy),
                "outlier_threshold": threshold,
                "weight_source": "runtime_resolver",
                "active_models": tuple(adapter.config.name for adapter in adapters),
                "resolver": result.metadata,
            }
        )
        metadata.update(result.metadata)
        metadata["model_predictions_available"].update(
            {
                model: model in metadata["model_predictions_available"]
                for model in ("hrnet", "spiga", "orformer")
            }
        )
        self._append_debug_metadata(metadata)
        logger.debug(
            "[Ensemble] runtime resolver selected=%s bucket=%s fallback=%s",
            result.selected_candidate,
            result.metadata.get("runtime_bucket"),
            result.metadata.get("fallback_reason"),
        )
        return result.landmarks.astype("float32", copy=False)

    def _fuse_face(
        self,
        predictions: list[tuple[LandmarkAdapter, LandmarkPrediction]],
        errors: list[str],
        *,
        detector_bbox: tuple[float, float, float, float] | None = None,
        image_crop: np.ndarray | None = None,
        crop_to_frame_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fuse one face's adapter predictions and return frame-space points."""
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
                image_crop=image_crop,
                crop_to_frame_matrix=crop_to_frame_matrix,
            )
            if resolver_points is not None:
                return resolver_points
            logger.debug(
                "[Ensemble] runtime resolver unavailable; falling back to %s", self._strategy
            )

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

        veto_reasons: list[str] = []
        if fused.rejected_indices:
            veto_reasons.append("outlier_rejected")
        veto_reasons.extend(f"adapter_error:{error}" for error in errors)
        selected_candidate = adapters[0].config.name if len(adapters) == 1 else self._strategy
        metadata = self._base_runtime_debug(
            selected_candidate=selected_candidate,
            fallback_used=self._setup_mode == "fallback" and self._promoted is None,
            veto_reasons=veto_reasons,
            points=fused.points,
            items=items,
            adapters=adapters,
            errors=errors,
            geometry_valid=not bool(fused.rejected_indices),
            detector_bbox=detector_bbox,
        )
        metadata.update(
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
                "weight_source": (
                    "promoted_setup" if self._promoted is not None else "adapter_config"
                ),
                "active_models": tuple(adapter.config.name for adapter in adapters),
            }
        )
        self._append_debug_metadata(metadata)
        logger.debug(
            "[Ensemble] fused face strategy=%s sources=%s rejected_landmarks=%s",
            self._strategy,
            fused.sources,
            fused.rejected_landmarks,
        )
        return fused.points

    def predict_landmarks_68(
        self,
        image: np.ndarray,
        *,
        matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return fused canonical ``(68, 2)`` landmarks in original-frame pixels."""
        matrices = (
            np.eye(3, dtype="float32")[None]
            if matrix is None
            else np.asarray(matrix, dtype="float32")[None]
        )
        per_face, errors = self._collect_predictions(image[None], matrices)
        self.last_debug_metadata = []
        return self._fuse_face(
            per_face[0],
            errors,
            detector_bbox=None,
            image_crop=image,
            crop_to_frame_matrix=matrices[0],
        )

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Run adapter predictions, fuse in frame space and return normalized landmarks."""
        if batch.ndim == 4 and batch.shape[1] in (1, 3, 4) and batch.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"Ensemble aligner expects channels-last images, got shape {batch.shape}"
            )
        matrices = self._matrices_for_batch(batch.shape[0])
        logger.debug(
            "[Ensemble] processing batch shape=%s matrices=%s", batch.shape, matrices.shape
        )
        per_face, errors = self._collect_predictions(batch, matrices)
        self.last_debug_metadata = []
        output = np.empty((batch.shape[0], 68, 2), dtype="float32")
        for idx, predictions in enumerate(per_face):
            output[idx] = frame_to_normalized_crop(
                self._fuse_face(
                    predictions,
                    errors,
                    detector_bbox=self._bbox_for_face(idx),
                    image_crop=batch[idx],
                    crop_to_frame_matrix=matrices[idx],
                ),
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
