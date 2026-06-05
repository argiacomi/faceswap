#!/usr/bin/env python3
"""Shared data builders for runtime resolver scorer training/evaluation."""

from __future__ import annotations

import concurrent.futures
import csv
import functools
import heapq
import json
import logging
import math
import threading
import typing as T
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from lib.landmarks.cache.prediction_cache import DiskPredictionCache
from lib.landmarks.coordinates import roi_to_matrix
from lib.landmarks.datasets.hard_negative_mining import (
    DEFAULT_HARD_NEGATIVE_WEIGHT,
    clamp_hard_negative_weight,
)
from lib.landmarks.datasets.manifest_io import (
    LandmarkSample,
    bbox_from_truth_fallback,
    filter_canonical_68_samples,
    load_manifest,
)
from lib.landmarks.ensemble.hard_condition_taxonomy import (
    NEW_HARD_CONDITION_LABELS,
    derive_hard_condition_taxonomy,
)
from lib.landmarks.ensemble.profile_features import profile_candidate_features
from lib.landmarks.ensemble.profile_repair import (
    DEFAULT_REPAIR_COMPRESSION,
    PROFILE_REPAIR_CANDIDATE_NAME,
    build_profile_repair_landmarks,
    is_profile_repair_context,
    profile_repair_features,
    profile_repair_provenance,
)
from lib.landmarks.ensemble.runtime_features import candidate_feature_map, runtime_feature_order
from lib.landmarks.ensemble.runtime_resolver import (
    CandidateMetrics,
    CandidateRecord,
    ModelPrediction,
    RuntimeBucketResult,
    RuntimeResolverConfig,
    _bbox_diag,
    _candidate_extra_features,
    _circular_median,
    _max_landmark_consensus_px,
    _metric_for_candidate,
    _populate_consensus_geometry,
    _shape_reasons,
    _stacked_regressor_candidate,
    append_bucket_weight_candidates,
    append_region_weight_candidate,
    build_candidates,
    infer_runtime_bucket,
    resolve_runtime,
)
from lib.landmarks.ensemble.scorer_target_config import (
    DEFAULT_COLLAPSE_COST_PENALTY,
    DEFAULT_FAILURE_COST_PENALTY,
    DEFAULT_LARGE_REGRET_THRESHOLD,
    DEFAULT_NME_FAILURE_THRESHOLD,
    DEFAULT_REGRET_NORMALIZER,
)
from lib.landmarks.ensemble.stacked_regressor import RuntimeStackedLandmarkRegressor
from lib.landmarks.ensemble.strategies import canonical_strategy
from lib.landmarks.ensemble.weights import load_optional_weight_blocks, load_weights
from lib.landmarks.evaluation.nme_metrics import evaluate_prediction
from lib.landmarks.evaluation.transform_alignment_cost import (
    TransformCostV3,
    can_fit_visible_subset_transform,
    transform_cost_v3,
)

logger = logging.getLogger(__name__)

ContextProgress = T.Callable[[T.Sequence[T.Any], str], T.Iterable[T.Any]]


DEFAULT_FAILURE_THRESHOLD: float = DEFAULT_NME_FAILURE_THRESHOLD
DEFAULT_HIGH_GAP_THRESHOLD: float = DEFAULT_LARGE_REGRET_THRESHOLD
DEFAULT_OUTLIER_THRESHOLD: float = 3.5
DEFAULT_IMAGE_BACKFILL_CROP_SCALE: float = 1.6
DEFAULT_IMAGE_BACKFILL_CROP_SIZE: int = 256

# Downstream weighted alignment cost proxy components.
# The scorer rows do not yet carry true per-region NME, so these penalties use
# the closest available runtime/geometry evidence for Faceswap-critical regions.
JAW_CROP_MASK_PENALTY: float = 0.75
EYES_IDENTITY_POSE_PENALTY: float = 0.60
MOUTH_EXPRESSION_PENALTY: float = 0.45
OCCLUDED_SIDE_INFERENCE_PENALTY: float = 0.65
CATASTROPHIC_FAILURE_PENALTY: float = DEFAULT_FAILURE_COST_PENALTY
PRODUCTION_FAILURE_PENALTY: float = 1.50
DEFAULT_RESOLVER_CANDIDATES: tuple[str, ...] = (
    "fan",
    "hrnet",
    "spiga",
    "orformer",
    "plain_average",
    "static_weighted",
    "static_weighted_downweight",
    "static_weighted_hard_drop",
    "weighted_median",
)


@dataclass(frozen=True, slots=True)
class StoredRuntimeDiagnostics:
    """Runtime diagnostics already attached to a manifest sample."""

    bucket: str
    features: dict[str, T.Any]
    source: str
    selected_candidate: str
    condition: str = ""
    hard_case_tags: tuple[str, ...] = ()


def _metadata_key(sample_id: str, face_index: int | None = 0) -> tuple[str, int]:
    """Return the stable resolver sidecar lookup key."""
    return str(sample_id).strip(), int(face_index or 0)


def load_resolver_metadata(path: Path | None) -> dict[tuple[str, int], dict[str, T.Any]]:
    """Load a JSONL resolver metadata sidecar keyed by sample id and face index."""
    if path is None:
        return {}

    records: dict[tuple[str, int], dict[str, T.Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if not sample_id:
                raise ValueError(f"{path}:{line_num} missing sample_id")

            face_index = int(row.get("face_index", 0))
            key = _metadata_key(str(sample_id), face_index)
            if key in records:
                raise ValueError(f"{path}:{line_num} duplicate resolver metadata key {key}")
            records[key] = row
    return records


def runtime_bucket_from_resolver_metadata(row: dict[str, T.Any]) -> str | None:
    """Return a runtime bucket from a resolver metadata sidecar row."""
    le = row.get("landmark_ensemble")
    if not isinstance(le, dict):
        return None

    resolver = le.get("resolver")
    if not isinstance(resolver, dict):
        resolver = {}

    bucket = (
        le.get("runtime_bucket")
        or le.get("bucket")
        or resolver.get("runtime_bucket")
        or resolver.get("bucket")
    )
    return str(bucket) if bucket else None


DEFAULT_SCORER_CANDIDATES: tuple[str, ...] = DEFAULT_RESOLVER_CANDIDATES
DEFAULT_SCORER_CANDIDATE_CSV: str = ",".join(DEFAULT_SCORER_CANDIDATES)
CANDIDATE_TABLE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "dataset",
    "condition",
    "candidate",
    "nme",
    "failure",
    "is_oracle",
    "gap_vs_oracle",
    "cloud_area_ratio",
    "hull_area_ratio",
    "points_outside_expanded_bbox_fraction",
    "eye_mouth_order_valid_after_deroll",
    "roi_center_consensus_distance",
    "landmark_consensus_distance",
    "shape_plausibility_score",
    "shape_veto_reasons",
    "max_edge_length_ratio",
    "mean_shape_fit_error",
    "topology_violation_count",
    "roll",
    "yaw",
    "roll_delta_to_consensus",
    "yaw_delta_to_consensus",
    "max_disagreement_px",
    "candidate_yaw_disagreement",
    "runtime_bucket",
    "hard_case_tags",
    "runtime_bucket_source",
    "selected_candidate_missing_from_eval",
    "geometry_veto_reasons",
    "profile_repair_used",
    "profile_repair_source_candidate",
    "profile_repair_visible_side",
    "profile_repair_method",
    "profile_repair_reason",
)


@dataclass(frozen=True, slots=True)
class CandidateQualityRow:
    """One sample/candidate row for scorer training or policy evaluation."""

    sample_id: str
    dataset: str
    condition: str
    candidate_name: str
    candidate_nme: float
    oracle_nme: float
    regret_vs_oracle: float
    normalized_regret: float
    failure_label: bool
    large_regret_label: bool
    candidate_failure_or_high_gap: bool
    selection_cost: float
    is_oracle: bool
    was_selected_by_current_policy: bool
    gap_vs_oracle: float
    runtime_bucket: str
    hard_case_tags: tuple[str, ...]
    risk_route: str
    feature_values: dict[str, float]
    selected_by_current_policy: str
    selected_candidate_missing_from_eval: bool
    oracle: str
    runtime_bucket_source: str
    geometry_veto_reasons: tuple[str, ...]
    transform_cost_v3: float
    corner_delta_v3: float
    center_delta_v3: float
    scale_delta_v3: float
    roll_delta_degrees_v3: float
    fit_delta_v3: float
    transform_oracle_cost_v3: float
    transform_regret_v3: float
    transform_oracle_candidate_v3: str
    transform_oracle_gap_v3: float
    rankable_v3: bool
    hard_invalid_v3: bool
    hard_invalid_reasons_v3: tuple[str, ...]
    soft_structural_penalty_v3: float
    face_index: int = 0
    hard_negative_weight: float = 1.0

    def to_csv_row(self) -> dict[str, T.Any]:
        """Return stable CSV fields plus a JSON feature payload."""
        return {
            "sample_id": self.sample_id,
            "face_index": self.face_index,
            "dataset": self.dataset,
            "condition": self.condition,
            "candidate_name": self.candidate_name,
            "candidate_nme": self.candidate_nme,
            "oracle_nme": self.oracle_nme,
            "regret_vs_oracle": self.regret_vs_oracle,
            "normalized_regret": self.normalized_regret,
            "failure_label": int(self.failure_label),
            "large_regret_label": int(self.large_regret_label),
            "candidate_failure_or_high_gap": int(self.candidate_failure_or_high_gap),
            "selection_cost": self.selection_cost,
            "is_oracle": int(self.is_oracle),
            "was_selected_by_current_policy": int(self.was_selected_by_current_policy),
            "gap_vs_oracle": self.gap_vs_oracle,
            "runtime_bucket": self.runtime_bucket,
            "hard_case_tags": "|".join(self.hard_case_tags),
            "runtime_bucket_source": self.runtime_bucket_source,
            "risk_route": self.risk_route,
            "geometry_veto_reasons": "|".join(self.geometry_veto_reasons),
            "transform_cost_v3": self.transform_cost_v3,
            "corner_delta_v3": self.corner_delta_v3,
            "center_delta_v3": self.center_delta_v3,
            "scale_delta_v3": self.scale_delta_v3,
            "roll_delta_degrees_v3": self.roll_delta_degrees_v3,
            "fit_delta_v3": self.fit_delta_v3,
            "transform_oracle_cost_v3": self.transform_oracle_cost_v3,
            "transform_regret_v3": self.transform_regret_v3,
            "transform_oracle_candidate_v3": self.transform_oracle_candidate_v3,
            "transform_oracle_gap_v3": self.transform_oracle_gap_v3,
            "rankable_v3": int(self.rankable_v3),
            "hard_invalid_v3": int(self.hard_invalid_v3),
            "hard_invalid_reasons_v3": "|".join(self.hard_invalid_reasons_v3),
            "soft_structural_penalty_v3": self.soft_structural_penalty_v3,
            "hard_negative_weight": self.hard_negative_weight,
            "selected_by_current_policy": self.selected_by_current_policy,
            "selected_candidate_missing_from_eval": int(self.selected_candidate_missing_from_eval),
            "oracle": self.oracle,
            "features_json": json.dumps(self.feature_values, sort_keys=True),
        }


def _required_nme(context: SampleCandidateContext, candidate_name: str) -> float:
    """Return a finite candidate NME or fail before writing invalid targets."""
    try:
        nme = float(context.nme_by_candidate[candidate_name])
    except KeyError as err:
        raise ValueError(
            f"sample {context.sample_id!r} missing NME for candidate {candidate_name!r}"
        ) from err
    if not math.isfinite(nme):
        raise ValueError(
            f"sample {context.sample_id!r} has non-finite NME for candidate {candidate_name!r}: "
            f"{nme!r}"
        )
    return nme


def _catastrophic_or_visual_collapse(reasons: T.Iterable[str]) -> bool:
    """Return whether geometry diagnostics indicate collapse-style failure."""
    collapse_markers = (
        "collapse",
        "cloud_area_too_small",
        "hull_area_too_small",
        "weak_visual",
        "visual_collapse",
    )
    return any(any(marker in str(reason) for marker in collapse_markers) for reason in reasons)


_V3_HARD_INVALID_TOKENS: tuple[str, ...] = (
    "collapse",
    "cloud_area_too_small",
    "hull_area_too_small",
    "eye_mouth",
    "flip",
    "self_intersection",
    "self-intersection",
    "topology",
    "impossible",
    "fatal",
)


_V3_SOFT_SUSPECT_TOKENS: tuple[str, ...] = (
    "low_plausibility",
    "plausibility",
    "roi",
    "hull",
    "outside",
    "borderline",
    "warning",
)

# Occluded-side topology artifacts that are fatal for ordinary candidates but only a
# soft penalty for the profile repair candidate, whose occluded side is synthetic.
_V3_PROFILE_SOFT_TOPOLOGY_TOKENS: tuple[str, ...] = (
    "self_intersection",
    "self-intersection",
    "topology",
)
_V3_HARD_INVALID_TOKENS_PROFILE_SOFT: tuple[str, ...] = tuple(
    token for token in _V3_HARD_INVALID_TOKENS if token not in _V3_PROFILE_SOFT_TOPOLOGY_TOKENS
)


_HARD_BUCKETS: frozenset[str] = frozenset({"frontal", "intermediate", "no_pose"})
_PRODUCTION_VALIDATED_SOURCE = "production_validated"
_JAW_CROP_MASK_TOKENS: tuple[str, ...] = (
    "jaw",
    "mask",
    "crop",
    "bbox",
    "outside",
    "hull",
    "cloud_area",
)
_EYES_IDENTITY_POSE_TOKENS: tuple[str, ...] = (
    "eye",
    "eyes",
    "pose",
    "yaw",
    "roll",
    "identity",
)
_MOUTH_EXPRESSION_TOKENS: tuple[str, ...] = (
    "mouth",
    "lip",
    "expression",
    "mouth_nose_jaw",
)
_OCCLUDED_SIDE_INFERENCE_TOKENS: tuple[str, ...] = (
    "occlusion",
    "occluded",
    "profile",
    "large_yaw",
    "occluded_side",
    "visible_side",
    "side_conflict",
)

# Keep this aligned with runtime_resolver.DEFAULT_MAX_SHAPE_PLAUSIBILITY_SCORE
# without importing runtime_resolver here and expanding scorer-data dependencies.
DEFAULT_V3_MAX_SHAPE_PLAUSIBILITY_SCORE: float = 0.2


def _dedupe_reasons(reasons: T.Iterable[str]) -> tuple[str, ...]:
    """Return non-empty reason strings in stable order."""
    return _dedupe_stable(str(reason or "") for reason in reasons)


def _v3_hard_invalid_reasons(
    metric: CandidateMetrics, *, profile_soft_topology: bool = False
) -> tuple[str, ...]:
    """Return fatal geometry reasons that exclude a candidate from v3 ranking.

    ``profile_soft_topology`` demotes occluded-side self-intersection / topology
    artifacts to soft penalties for the profile repair candidate (whose occluded side
    is synthesized). Visible-side core failures - eye/mouth flip, fatal/impossible
    geometry, and inability to fit the visible-subset transform - stay fatal.
    """
    hard_tokens = (
        _V3_HARD_INVALID_TOKENS_PROFILE_SOFT if profile_soft_topology else _V3_HARD_INVALID_TOKENS
    )
    reasons: list[str] = []
    for reason_group in (metric.geometry_veto_reasons, metric.shape_veto_reasons):
        for reason in reason_group:
            text = str(reason)
            lowered = text.lower()
            if any(token in lowered for token in hard_tokens):
                reasons.append(text)
    if metric.eye_mouth_order_valid_after_deroll is False:
        reasons.append("eye_mouth_flip")
    if not profile_soft_topology and int(metric.topology_violation_count or 0) > 0:
        reasons.append("topology_violation")
    return _dedupe_reasons(reasons)


def _v3_soft_suspect_reasons(
    metric: CandidateMetrics,
    *,
    hard_invalid_reasons: T.Sequence[str] = (),
    profile_soft_topology: bool = False,
) -> tuple[str, ...]:
    """Return soft geometry warnings that remain finite v3 cost penalties.

    Hard-invalid reasons win the split.  A diagnostic that excludes a candidate
    from v3 ranking must not also appear as a soft-suspect penalty reason.
    """
    hard_reason_set = {
        str(reason or "").strip() for reason in hard_invalid_reasons if str(reason or "").strip()
    }
    reasons: list[str] = []
    for reason_group in (metric.geometry_veto_reasons, metric.shape_veto_reasons):
        for reason in reason_group:
            text = str(reason or "").strip()
            if not text or text in hard_reason_set:
                continue
            lowered = text.lower()
            if any(token in lowered for token in _V3_SOFT_SUSPECT_TOKENS) or (
                profile_soft_topology
                and any(token in lowered for token in _V3_PROFILE_SOFT_TOPOLOGY_TOKENS)
            ):
                reasons.append(text)
    if profile_soft_topology and int(metric.topology_violation_count or 0) > 0:
        reasons.append("topology_violation")
    if (
        metric.shape_plausibility_score is not None
        and metric.shape_plausibility_score > DEFAULT_V3_MAX_SHAPE_PLAUSIBILITY_SCORE
    ):
        reasons.append("low_plausibility")
    return _dedupe_reasons(reasons)


@dataclass(frozen=True)
class SampleCandidateContext:
    """Resolved candidates and row data for one manifest sample."""

    sample_id: str
    face_index: int
    dataset: str
    source: str
    condition: str
    candidates: tuple[CandidateRecord, ...]
    metrics: dict[str, CandidateMetrics]
    truth_landmarks: np.ndarray | None
    normalizer: float | None
    visibility: tuple[bool, ...] | None
    nme_by_candidate: dict[str, float]
    failure_by_candidate: dict[str, bool]
    runtime_bucket: str
    hard_case_tags: tuple[str, ...]
    risk_route: str
    current_policy_choice: str
    oracle: str
    model_predictions_available: dict[str, bool]
    roll_estimate: float | None
    yaw_estimate: float | None
    candidate_yaw_disagreement: float | None
    max_disagreement_px: float
    runtime_bucket_source: str
    selected_candidate_missing_from_eval: bool
    candidate_extra_features: dict[str, dict[str, float]]
    hard_negative_weight: float = 1.0
    profile_repair_provenance: dict[str, T.Any] | None = None


def parse_candidates(
    value: str | None, weights: T.Mapping[str, T.Sequence[float]]
) -> tuple[str, ...]:
    """Parse CLI candidate list with a stable default."""
    if value:
        return tuple(item.strip() for item in value.split(",") if item.strip())
    ordered = [name for name in DEFAULT_SCORER_CANDIDATES if name in weights or _is_strategy(name)]
    ordered.extend(name for name in weights if name not in ordered)
    return tuple(dict.fromkeys(ordered))


def _dedupe_stable(items: T.Iterable[str]) -> tuple[str, ...]:
    """Return unique non-empty strings in input order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _is_adaptive_candidate(name: str) -> bool:
    """Return True for candidates appended after runtime bucket inference."""
    if name == "region_weighted":
        return True
    if "@" not in name:
        return False
    base, _ = name.split("@", 1)
    return _is_strategy(base)


@functools.lru_cache(maxsize=256)
def _is_strategy(name: str) -> bool:
    try:
        canonical_strategy(name)
    except (KeyError, ValueError):
        return False
    return True


def _requested_candidate_sets(
    requested_candidates: T.Sequence[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Split requested names once so per-sample setup avoids repeated strategy lookups."""
    requested_adaptive: set[str] = set()
    single_models: set[str] = set()
    for name in requested_candidates:
        if _is_adaptive_candidate(name):
            requested_adaptive.add(name)
        elif not _is_strategy(name):
            single_models.add(name)
    return frozenset(requested_adaptive), frozenset(single_models)


def _load_truth(sample: LandmarkSample) -> np.ndarray:
    return np.load(sample.landmarks).astype(np.float32, copy=False)  # type: ignore[no-any-return]


def _runtime_metadata(sample: LandmarkSample) -> dict[str, T.Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    resolver = metadata.get("landmark_ensemble")
    return resolver if isinstance(resolver, dict) else {}


def _runtime_features_from_metadata(resolver: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    """Extract stable runtime feature diagnostics from a metadata payload."""
    features: dict[str, T.Any] = {}
    raw_features = resolver.get("runtime_bucket_features")
    if isinstance(raw_features, dict):
        features.update(raw_features)
    nested = resolver.get("resolver")
    if isinstance(nested, dict):
        nested_features = nested.get("runtime_bucket_features")
        if isinstance(nested_features, dict):
            features.update(nested_features)
    else:
        nested = {}
    for key in (
        "candidate_yaw_disagreement",
        "max_disagreement_px",
        "landmark_pose_yaw",
        "landmark_pose_roll",
        "image_geometry_yaw_signal",
        "runtime_bucket_side",
        "runtime_bucket_side_source",
        "runtime_bucket_side_confidence",
        "runtime_bucket_side_votes",
        "runtime_bucket_side_conflict",
        "runtime_bucket_geometry_side",
        "runtime_bucket_geometry_side_source",
        "runtime_bucket_severity",
        "runtime_bucket_severity_source",
        "left_eye_visual_score",
        "right_eye_visual_score",
        "eye_visibility_asymmetry",
        "nose_offset_from_face_center",
        "mouth_nose_jaw_asymmetry",
        "max_disagreement_bbox_fraction",
        "dominant_candidate_yaw",
    ):
        if key in resolver:
            features[key] = resolver[key]
        elif key in nested:
            features[key] = nested[key]
    return features


def _stored_hard_case_tags(value: T.Any) -> tuple[str, ...]:
    """Return stored hard-case tags from list/tuple/set or pipe-delimited text."""
    if value is None:
        return ()

    raw_items: T.Iterable[T.Any]
    if isinstance(value, str):
        raw_items = value.split("|")
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = (value,)

    return _dedupe_stable(str(item) for item in raw_items)


def _normalize_stored_condition_tag(value: T.Any) -> str:
    """Return stored condition as a hard-case tag only when it is hard/occlusion-like."""
    tag = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    tag = "_".join(part for part in tag.split("_") if part)
    if not tag:
        return ""
    if tag in NEW_HARD_CONDITION_LABELS:
        return tag
    if tag.endswith("_occlusion") or tag.endswith("_occluded") or "occlud" in tag:
        return tag
    return ""


def _hard_case_tags_with_stored_condition(
    *,
    stored_condition: str,
    taxonomy_tags: T.Sequence[str],
) -> tuple[str, ...]:
    """Seed stored hard/occlusion condition into derived tags when stored tags are absent."""
    condition_tag = _normalize_stored_condition_tag(stored_condition)
    if not condition_tag:
        return tuple(taxonomy_tags)
    return tuple(dict.fromkeys((condition_tag, *taxonomy_tags)))


def _stored_runtime_diagnostics(sample: LandmarkSample) -> StoredRuntimeDiagnostics | None:
    resolver = _runtime_metadata(sample)
    bucket = resolver.get("runtime_bucket") or resolver.get("bucket")
    if not bucket:
        return None
    source = str(resolver.get("runtime_bucket_source") or "stored_manifest_landmark_ensemble")
    selected_candidate = str(resolver.get("selected_candidate") or "")
    return StoredRuntimeDiagnostics(
        bucket=str(bucket),
        features=_runtime_features_from_metadata(resolver),
        source=source,
        selected_candidate=selected_candidate,
        condition=str(resolver.get("condition") or ""),
        hard_case_tags=_stored_hard_case_tags(resolver.get("hard_case_tags")),
    )


def _stored_runtime_diagnostics_from_sidecar(
    row: dict[str, T.Any],
    *,
    key: tuple[str, int],
) -> StoredRuntimeDiagnostics:
    """Return stored diagnostics from a frozen resolver metadata sidecar row."""
    bucket = runtime_bucket_from_resolver_metadata(row)
    if not bucket:
        raise RuntimeError(f"stored resolver metadata has no runtime bucket: {key}")
    resolver = row.get("landmark_ensemble")
    if not isinstance(resolver, dict):
        resolver = {}
    source = str(resolver.get("runtime_bucket_source") or "stored_manifest_landmark_ensemble")
    selected_candidate = str(resolver.get("selected_candidate") or "")
    return StoredRuntimeDiagnostics(
        bucket=bucket,
        features=_runtime_features_from_metadata(resolver),
        source=source,
        selected_candidate=selected_candidate,
        condition=str(resolver.get("condition") or row.get("condition") or ""),
        hard_case_tags=(
            _stored_hard_case_tags(resolver.get("hard_case_tags"))
            or _stored_hard_case_tags(row.get("hard_case_tags"))
        ),
    )


def ensemble_crop_roi(
    detector_bbox: T.Sequence[float],
    *,
    crop_scale: float = DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
) -> np.ndarray:
    """Return the same square ROI used by the production ensemble aligner."""
    bbox = np.asarray(detector_bbox, dtype="float32")
    if bbox.shape != (4,):
        raise ValueError(f"detector bbox must have shape (4,), got {bbox.shape}")
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    if width <= 0 or height <= 0:
        raise ValueError(f"detector bbox must have positive width/height, got {bbox.tolist()}")
    ctr_x = int(np.rint((bbox[0] + bbox[2]) * 0.5))
    ctr_y = int(np.rint((bbox[1] + bbox[3]) * 0.5))
    half = int(np.rint(max(float(width), float(height)) * float(crop_scale) * 0.5))
    return np.asarray(  # type: ignore[no-any-return]
        [ctr_x - half, ctr_y - half, ctr_x + half, ctr_y + half],
        dtype="int32",
    )


def crop_square_rgb(
    image_rgb: np.ndarray,
    roi: T.Sequence[float],
    *,
    crop_size: int = DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
) -> np.ndarray:
    """Crop a square ROI with zero padding, then resize to the aligner crop size."""
    try:
        import cv2
    except ModuleNotFoundError as err:  # pragma: no cover - environment guard
        raise RuntimeError("image-aware runtime metadata backfill requires opencv-python") from err

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB image with shape HxWx3, got {image.shape}")
    left, top, right, bottom = (int(np.rint(value)) for value in roi)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0 or width != height:
        raise ValueError(f"ROI must be positive square ltrb, got {list(roi)}")
    if left >= 0 and top >= 0 and right <= image.shape[1] and bottom <= image.shape[0]:
        crop = image[top:bottom, left:right, :3]
        if width == crop_size:
            return crop  # type: ignore[no-any-return]
        return cv2.resize(  # type: ignore[no-any-return]
            crop,
            (crop_size, crop_size),
            interpolation=cv2.INTER_LINEAR,
        )
    canvas = np.zeros((height, width, 3), dtype=image.dtype)
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, image.shape[1])
    src_bottom = min(bottom, image.shape[0])
    if src_right > src_left and src_bottom > src_top:
        dst_left = src_left - left
        dst_top = src_top - top
        canvas[
            dst_top : dst_top + (src_bottom - src_top),
            dst_left : dst_left + (src_right - src_left),
        ] = image[src_top:src_bottom, src_left:src_right, :3]
    if width == crop_size and height == crop_size:
        return canvas  # type: ignore[no-any-return]
    return cv2.resize(  # type: ignore[no-any-return]
        canvas,
        (crop_size, crop_size),
        interpolation=cv2.INTER_LINEAR,
    )


@functools.lru_cache(maxsize=128)
def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an RGB image for image-aware runtime bucket backfill."""
    try:
        import cv2
    except ModuleNotFoundError as err:  # pragma: no cover - environment guard
        raise RuntimeError("image-aware runtime metadata backfill requires opencv-python") from err

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # type: ignore[no-any-return]


def image_aware_runtime_result(
    sample: LandmarkSample,
    *,
    predictions: T.Sequence[ModelPrediction],
    config: RuntimeResolverConfig,
    detector_bbox: T.Sequence[float],
    crop_scale: float = DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    crop_size: int = DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
) -> T.Any:
    """Run the runtime resolver with the production crop and image evidence."""
    image_rgb = load_image_rgb(sample.image)

    roi_array = ensemble_crop_roi(detector_bbox, crop_scale=crop_scale)
    roi: tuple[float, float, float, float] = (
        float(roi_array[0]),
        float(roi_array[1]),
        float(roi_array[2]),
        float(roi_array[3]),
    )

    crop: np.ndarray = crop_square_rgb(
        image_rgb,
        roi,
        crop_size=crop_size,
    ).astype(np.float32, copy=False)
    crop *= np.float32(1.0 / 255.0)

    return resolve_runtime(
        predictions,
        config,
        detector_bbox=detector_bbox,
        image_crop=crop,
        crop_to_frame_matrix=roi_to_matrix(roi),
    )


def _optional_float(value: T.Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_config(
    weights: T.Mapping[str, T.Sequence[float]],
    *,
    outlier_threshold: float,
    bucket_weights: T.Mapping[str, T.Mapping[str, T.Sequence[float]]] | None = None,
    region_weights: T.Mapping[str, T.Mapping[str, float]] | None = None,
) -> RuntimeResolverConfig:
    return RuntimeResolverConfig(
        policy="roll_aware_veto",
        weights=weights,
        bucket_weights=bucket_weights,
        region_weights=region_weights,
        general_strategy="static_weighted",
        hard_case_strategy="static_weighted_downweight",
        secondary_hard_case_strategy="static_weighted_hard_drop",
        fallback_strategy="plain_average",
        outlier_threshold=outlier_threshold,
    )


def _read_predictions(
    cache: DiskPredictionCache,
    sample: LandmarkSample,
    *,
    models: T.Iterable[str],
) -> list[ModelPrediction]:
    model_set = set(models)
    available = set(cache.available_models(sample.sample_id))
    missing = sorted(model_set - available)
    if missing:
        logger.warning("sample %s missing cached models: %s", sample.sample_id, missing)
    return [
        ModelPrediction(model, cache.read(sample.sample_id, model).landmarks)
        for model in sorted(model_set & available)
    ]


def _nme(
    landmarks: np.ndarray,
    truth: np.ndarray,
    *,
    sample: LandmarkSample,
    failure_threshold: float,
) -> tuple[float, bool]:
    metrics = evaluate_prediction(
        landmarks,
        truth,
        normalizer=sample.normalizer,
        failure_threshold=failure_threshold,
        visibility=sample.visibility,
    )
    return float(metrics["nme"]), bool(metrics["failure"])


def build_sample_context(
    sample: LandmarkSample,
    *,
    cache: DiskPredictionCache,
    requested_candidates: T.Sequence[str],
    weights: T.Mapping[str, T.Sequence[float]],
    source: str = "",
    resolver_metadata: T.Mapping[tuple[str, int], dict[str, T.Any]] | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    bucket_weights: T.Mapping[str, T.Mapping[str, T.Sequence[float]]] | None = None,
    region_weights: T.Mapping[str, T.Mapping[str, float]] | None = None,
    include_adaptive_candidates: bool = True,
    allow_image_backfill: bool = False,
    image_backfill_crop_scale: float = DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    image_backfill_crop_size: int = DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    stacked_regressor: RuntimeStackedLandmarkRegressor | None = None,
    stacked_regressor_max_residual: float = 0.0,
) -> SampleCandidateContext:
    """Build candidates, diagnostics, labels, and current-policy choice for one sample."""
    requested_adaptive, single_models = _requested_candidate_sets(requested_candidates)
    model_names = tuple(dict.fromkeys((*weights.keys(), *single_models)))
    predictions = _read_predictions(cache, sample, models=model_names)
    if not predictions:
        raise FileNotFoundError(f"sample {sample.sample_id!r} has no cached model predictions")
    config = _candidate_config(
        weights,
        outlier_threshold=outlier_threshold,
        bucket_weights=bucket_weights,
        region_weights=region_weights,
    )
    all_candidates = build_candidates(predictions, config)
    by_name = {candidate.name: candidate for candidate in all_candidates}
    seed_names = [name for name in requested_candidates if name not in requested_adaptive]
    if requested_adaptive:
        seed_names.extend(
            candidate.name for candidate in all_candidates if not candidate.is_fusion
        )
    candidates = tuple(by_name[name] for name in dict.fromkeys(seed_names) if name in by_name)
    if not candidates:
        raise ValueError(f"sample {sample.sample_id!r} produced no requested candidates")
    truth = _load_truth(sample)
    reference_bbox = sample.face_bbox or bbox_from_truth_fallback(truth)
    metrics = {
        candidate.name: _metric_for_candidate(candidate, reference_bbox=reference_bbox)
        for candidate in candidates
    }
    _populate_consensus_geometry(candidates, metrics, reference_bbox=reference_bbox)
    roll_estimate = _circular_median(
        [metric.roll_degrees for metric in metrics.values() if metric.roll_degrees is not None]
    )
    yaw_estimate = _circular_median(
        [metric.yaw_degrees for metric in metrics.values() if metric.yaw_degrees is not None]
    )
    max_disagreement_px = _max_landmark_consensus_px(candidates)
    face_index = int(sample.metadata.get("face_index", sample.metadata.get("face", 0)) or 0)
    metadata_key = _metadata_key(sample.sample_id, face_index)
    stored_sidecar = resolver_metadata.get(metadata_key) if resolver_metadata is not None else None
    if stored_sidecar is not None:
        stored_runtime = _stored_runtime_diagnostics_from_sidecar(
            stored_sidecar,
            key=metadata_key,
        )
    elif source == "gt_hard":
        raise RuntimeError(
            f"GT-hard sample missing stored resolver metadata: sample_id={sample.sample_id!r} "
            f"face_index={face_index}. Refusing image-aware backfill."
        )
    else:
        stored_runtime = _stored_runtime_diagnostics(sample)  # type: ignore[assignment]
    image_backfill_result = None
    if stored_runtime is None:
        if allow_image_backfill and reference_bbox is not None:
            try:
                image_backfill_result = image_aware_runtime_result(
                    sample,
                    predictions=predictions,
                    config=config,
                    detector_bbox=reference_bbox,
                    crop_scale=image_backfill_crop_scale,
                    crop_size=image_backfill_crop_size,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as err:
                logger.warning("sample %s image-aware backfill failed: %s", sample.sample_id, err)
        if image_backfill_result is None:
            bucket_result = infer_runtime_bucket(
                image_crop=None,
                crop_to_frame_matrix=None,
                detector_bbox=reference_bbox,
                candidates=candidates,
                metrics=metrics,
                yaw_estimate=yaw_estimate,
                roll_estimate=roll_estimate,
                max_disagreement_px=max_disagreement_px,
                hard_roll_degrees=config.hard_roll_degrees,
            )
            runtime_bucket_source = "derived_no_image_evidence"
        else:
            metadata = image_backfill_result.metadata
            bucket_result = RuntimeBucketResult(
                bucket=str(metadata.get("runtime_bucket") or metadata.get("bucket") or ""),
                features=dict(metadata.get("runtime_bucket_features") or {}),
            )
            runtime_bucket_source = "image_aware_backfill"
    else:
        bucket_result = RuntimeBucketResult(
            bucket=stored_runtime.bucket,
            features=stored_runtime.features,
        )
        runtime_bucket_source = stored_runtime.source
        stored_features = stored_runtime.features
        stored_roll = _optional_float(
            stored_features.get("landmark_pose_roll", stored_features.get("roll_estimate"))
        )
        if stored_roll is not None:
            roll_estimate = stored_roll
        stored_yaw = _optional_float(
            stored_features.get("landmark_pose_yaw", stored_features.get("yaw_estimate"))
        )
        if stored_yaw is not None:
            yaw_estimate = stored_yaw
    append_adaptive = (include_adaptive_candidates or bool(requested_adaptive)) and (
        config.bucket_weights or config.region_weights
    )
    if append_adaptive:
        candidate_list = list(candidates)
        append_bucket_weight_candidates(
            candidate_list,
            config,
            bucket=bucket_result.bucket,
            metrics=metrics,
            reference_bbox=reference_bbox,
        )
        append_region_weight_candidate(
            candidate_list,
            config,
            metrics=metrics,
            reference_bbox=reference_bbox,
        )
        candidates = tuple(candidate_list)
    if requested_adaptive:
        by_requested_name = {candidate.name: candidate for candidate in candidates}
        candidates = tuple(
            by_requested_name[name] for name in requested_candidates if name in by_requested_name
        )
    candidate_names = {candidate.name for candidate in candidates}
    missing = [name for name in requested_candidates if name not in candidate_names]
    if missing:
        logger.warning("sample %s could not build candidates: %s", sample.sample_id, missing)
    if not candidates:
        raise ValueError(f"sample {sample.sample_id!r} produced no requested candidates")
    for name, metric in metrics.items():
        metric.geometry_veto_reasons = _shape_reasons(bucket_result.bucket, name, metric)
    taxonomy = derive_hard_condition_taxonomy(
        sample,
        runtime_bucket=bucket_result.bucket,
        yaw_estimate=yaw_estimate,
        roll_estimate=roll_estimate,
    )
    condition = (
        stored_runtime.condition
        if stored_runtime and stored_runtime.condition
        else taxonomy.condition
    )
    if stored_runtime and stored_runtime.hard_case_tags:
        hard_case_tags = stored_runtime.hard_case_tags
    elif stored_runtime and stored_runtime.condition:
        hard_case_tags = _hard_case_tags_with_stored_condition(
            stored_condition=stored_runtime.condition,
            taxonomy_tags=taxonomy.hard_case_tags,
        )
    else:
        hard_case_tags = taxonomy.hard_case_tags
    candidate_extra_features = _candidate_extra_features(
        candidates,
        metrics,
        reference_bbox=reference_bbox,
        condition=condition,
        runtime_bucket=bucket_result.bucket,
        runtime_bucket_source=runtime_bucket_source,
        yaw_estimate=yaw_estimate,
        roll_estimate=roll_estimate,
    )
    candidates, profile_repair_provenance_meta = _append_profile_repair_candidate(
        candidates,
        metrics,
        candidate_extra_features,
        runtime_bucket=bucket_result.bucket,
        condition=condition,
        yaw_estimate=yaw_estimate,
        roll_estimate=roll_estimate,
        reference_bbox=reference_bbox,
        truth_landmarks=truth,
        visibility=sample.visibility,
    )

    if stacked_regressor is not None:
        candidate_list = list(candidates)
        stacked_config = replace(
            config,
            use_stacked_regressor=True,
            stacked_regressor_max_residual=stacked_regressor_max_residual,
        )
        stacked_regressor_result = _stacked_regressor_candidate(
            candidate_list,
            metrics=metrics,
            config=stacked_config,
            stacked_regressor=stacked_regressor,
            predictions=predictions,
            runtime_bucket=bucket_result.bucket,
            reference_bbox=reference_bbox,
            roll_estimate=roll_estimate,
            yaw_estimate=yaw_estimate,
            candidate_yaw_disagreement=bucket_result.features.get("candidate_yaw_disagreement"),
            max_disagreement_px=max_disagreement_px,
        )
        if stacked_regressor_result is not None:
            _stacked_metadata, stacked_feature_values, stacked_name = stacked_regressor_result
            candidates = tuple(candidate_list)
            candidate_extra_features.setdefault(stacked_name, {}).update(stacked_feature_values)

    nme_by_candidate: dict[str, float] = {}
    failure_by_candidate: dict[str, bool] = {}
    for candidate in candidates:
        nme, failure = _nme(
            candidate.landmarks,
            truth,
            sample=sample,
            failure_threshold=failure_threshold,
        )
        nme_by_candidate[candidate.name] = nme
        failure_by_candidate[candidate.name] = failure
    oracle = min(nme_by_candidate, key=lambda name: nme_by_candidate[name])
    if stored_runtime is not None and stored_runtime.selected_candidate:
        current_policy_choice = stored_runtime.selected_candidate
    elif image_backfill_result is not None:
        current_policy_choice = image_backfill_result.selected_candidate
    else:
        current = resolve_runtime(
            predictions,
            config,
            detector_bbox=reference_bbox,
        )
        current_policy_choice = current.selected_candidate
    selected_candidate_missing_from_eval = current_policy_choice not in nme_by_candidate
    hard_bucket = bucket_result.bucket not in _HARD_BUCKETS
    has_vetoed = any(metric.geometry_veto_reasons for metric in metrics.values())
    risk_route = (
        "high_risk"
        if hard_bucket or max_disagreement_px > config.hard_disagreement_px or has_vetoed
        else "low_risk"
    )
    return SampleCandidateContext(
        sample_id=sample.sample_id,
        face_index=face_index,
        dataset=sample.dataset,
        source=source,
        condition=condition,
        candidates=candidates,
        metrics=metrics,
        truth_landmarks=truth,
        normalizer=sample.normalizer,
        visibility=sample.visibility,
        nme_by_candidate=nme_by_candidate,
        failure_by_candidate=failure_by_candidate,
        runtime_bucket=bucket_result.bucket,
        hard_case_tags=hard_case_tags,
        risk_route=risk_route,
        current_policy_choice=current_policy_choice,
        oracle=oracle,
        model_predictions_available=dict.fromkeys(
            (prediction.model for prediction in predictions),
            True,
        ),
        roll_estimate=roll_estimate,
        yaw_estimate=yaw_estimate,
        candidate_yaw_disagreement=bucket_result.features.get("candidate_yaw_disagreement"),
        max_disagreement_px=(
            stored_max_disagreement
            if (
                stored_max_disagreement := _optional_float(
                    bucket_result.features.get("max_disagreement_px")
                )
            )
            is not None
            else max_disagreement_px
        ),
        runtime_bucket_source=runtime_bucket_source,
        selected_candidate_missing_from_eval=selected_candidate_missing_from_eval,
        candidate_extra_features=candidate_extra_features,
        hard_negative_weight=_hard_negative_weight_from_sample(sample),
        profile_repair_provenance=profile_repair_provenance_meta,
    )


def _candidate_v3_hard_invalid(
    candidate: CandidateRecord,
    metric: CandidateMetrics,
    *,
    truth_landmarks: np.ndarray | None,
    visibility: tuple[bool, ...] | None,
) -> bool:
    """Return ``True`` when a candidate is hard-invalid under the v3 transform cost.

    Mirrors the row builder's v3 validity so the repair trigger keys off true
    v3 all-invalid groups rather than only the geometry-veto proxy.
    """
    profile_soft_topology = candidate.name == PROFILE_REPAIR_CANDIDATE_NAME
    hard_invalid_reasons = _v3_hard_invalid_reasons(
        metric, profile_soft_topology=profile_soft_topology
    )
    truth_for_v3 = truth_landmarks
    if truth_for_v3 is None:
        truth_for_v3 = candidate.landmarks
        hard_invalid_reasons = ("missing_truth_landmarks", *hard_invalid_reasons)
    cost = transform_cost_v3(
        candidate.landmarks,
        truth_for_v3,
        visibility=visibility,
        hard_invalid_reasons=hard_invalid_reasons,
        soft_suspect_reasons=_v3_soft_suspect_reasons(
            metric,
            hard_invalid_reasons=hard_invalid_reasons,
            profile_soft_topology=profile_soft_topology,
        ),
    )
    return bool(cost.hard_invalid)


def _append_profile_repair_candidate(
    candidates: tuple[CandidateRecord, ...],
    metrics: dict[str, CandidateMetrics],
    candidate_extra_features: dict[str, dict[str, float]],
    *,
    runtime_bucket: str,
    condition: str,
    yaw_estimate: float | None,
    roll_estimate: float | None,
    reference_bbox: tuple[float, float, float, float] | None,
    truth_landmarks: np.ndarray | None,
    visibility: tuple[bool, ...] | None,
) -> tuple[tuple[CandidateRecord, ...], dict[str, T.Any] | None]:
    """Append a ``profile_visible_side_repaired`` candidate for all-invalid groups.

    Only fires on profile/occlusion contexts where every existing candidate is
    geometry-vetoed (an all-invalid proxy computable before v3 cost), so normal
    and partially-valid groups are unchanged. The repaired candidate is built
    with real landmarks and run through the normal metric/shape/v3 pipeline; it
    is appended only when it passes the repair sanity gates. Returns the
    (possibly extended) candidate tuple plus string repair provenance (or
    ``None`` when nothing was appended).
    """
    if not is_profile_repair_context((runtime_bucket, condition)):
        return candidates, None
    if not candidates or any(c.name == PROFILE_REPAIR_CANDIDATE_NAME for c in candidates):
        return candidates, None
    # A repaired candidate is scored with the same visibility-gated v3 transform as every
    # other candidate. When too few visible core landmarks exist to fit that transform
    # (0/4/5/6/7 visible core), any repair would be instantly hard-invalid with
    # ``unable_to_fit_visible_subset_transform``, so skip it before candidate creation.
    if not can_fit_visible_subset_transform(visibility):
        return candidates, None
    sized = [candidate for candidate in candidates if candidate.name in metrics]
    all_vetoed = bool(sized) and all(
        bool(metrics[candidate.name].geometry_veto_reasons) for candidate in sized
    )
    v3_all_invalid = bool(sized) and all(
        _candidate_v3_hard_invalid(
            candidate,
            metrics[candidate.name],
            truth_landmarks=truth_landmarks,
            visibility=visibility,
        )
        for candidate in sized
    )
    if not (v3_all_invalid or all_vetoed):
        return candidates, None
    repair_reason = (
        "all_candidates_hard_invalid_v3" if v3_all_invalid else "all_candidates_geometry_vetoed"
    )
    built = build_profile_repair_landmarks(
        candidates,
        metrics,
        runtime_bucket=runtime_bucket,
        condition=condition,
        yaw_estimate=yaw_estimate,
        candidate_extra_features=candidate_extra_features,
    )
    if built is None:
        return candidates, None
    repaired_landmarks, source_name, visible_side = built
    record = CandidateRecord(
        name=PROFILE_REPAIR_CANDIDATE_NAME,
        landmarks=repaired_landmarks,
        is_fusion=True,
        contributing_models=(source_name,),
    )
    metric = _metric_for_candidate(record, reference_bbox=reference_bbox)
    # Post-build safety net: an orformer-style source can carry an eye/mouth flip with empty
    # veto fields, so the pre-selection filter cannot see it. Drop the repair if the built
    # candidate's own visible core is flipped rather than appending a fatally-invalid repair.
    if metric.eye_mouth_order_valid_after_deroll is False:
        return candidates, None
    metric.geometry_veto_reasons = _shape_reasons(runtime_bucket, record.name, metric)
    metrics[record.name] = metric
    source_rank = next(
        (index for index, candidate in enumerate(candidates) if candidate.name == source_name),
        0,
    )
    shape_score = float(getattr(metric, "shape_plausibility_score", 0.0) or 0.0)
    extended = (*candidates, record)
    # Give the repair candidate the same profile feature payload as its peers so
    # the profile specialist sees its visible/occluded-side geometry, plus the
    # repair-specific provenance features.
    blob = f"{condition} {runtime_bucket}".lower()
    repair_profile_features = profile_candidate_features(
        extended,
        metrics,
        diag=_bbox_diag(reference_bbox),
        side=visible_side,
        yaw_estimate=yaw_estimate,
        roll_estimate=roll_estimate,
        has_occlusion="occlusion" in blob or "occluded" in blob,
        has_single_eye_visible="single_eye" in blob,
        has_mouth_or_jaw_occluded="mouth_or_jaw" in blob or "mouth_jaw" in blob,
    ).get(record.name, {})
    candidate_extra_features[record.name] = {
        **repair_profile_features,
        **profile_repair_features(
            visible_side=visible_side,
            source_rank=source_rank,
            shape_score=shape_score,
        ),
    }
    provenance = profile_repair_provenance(
        source_candidate=source_name,
        visible_side=visible_side,
        reason=repair_reason,
        compression=DEFAULT_REPAIR_COMPRESSION,
    )
    return extended, provenance


def _hard_negative_weight_from_sample(sample: LandmarkSample) -> float:
    """Return the mined hard-negative weight stored on a manifest sample.

    The hard-negative manifest builder writes ``metadata.hard_negative_weight``;
    samples without it fall back to the neutral default so naturally-sampled
    manifests are unaffected.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    try:
        weight = float(metadata.get("hard_negative_weight", DEFAULT_HARD_NEGATIVE_WEIGHT))
    except (TypeError, ValueError):
        return float(DEFAULT_HARD_NEGATIVE_WEIGHT)
    return float(clamp_hard_negative_weight(weight))


def _proxy_text_for_downstream_cost(
    *,
    context: SampleCandidateContext,
    metric: CandidateMetrics,
) -> str:
    """Return lower-case diagnostic text used for region-penalty proxies."""
    return "|".join(
        [
            str(context.condition or ""),
            str(context.runtime_bucket or ""),
            str(context.risk_route or ""),
            *(str(tag) for tag in (context.hard_case_tags or ())),
            *(str(reason) for reason in (metric.geometry_veto_reasons or ())),
        ]
    ).lower()


def _has_any_token(text: str, tokens: T.Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def downstream_weighted_alignment_cost(
    *,
    context: SampleCandidateContext,
    candidate_name: str,
    metric: CandidateMetrics,
    candidate_nme: float,
    oracle_nme: float,
    candidate_failure: bool,
) -> float:
    """Return the training target used by the promoted scorer.

    Intended target:
        downstream_weighted_alignment_cost =
            full_face_nme_regret
            + jaw_crop_mask_penalty
            + eyes_identity_pose_penalty
            + mouth_expression_penalty
            + occluded_side_inference_penalty
            + catastrophic_failure_penalty
            + production_failure_penalty

    Closest available proxy:
        full_face_nme_regret is normalized candidate NME regret;
        region penalties are inferred from geometry veto reasons and
        hard-condition/runtime tags until true per-region training targets
        are materialized in CandidateQualityRow.
    """
    regret = max(float(candidate_nme) - float(oracle_nme), 0.0)
    full_face_nme_regret = min(regret / DEFAULT_REGRET_NORMALIZER, 1.0)
    text = _proxy_text_for_downstream_cost(context=context, metric=metric)

    jaw_crop_mask_penalty = (
        JAW_CROP_MASK_PENALTY if _has_any_token(text, _JAW_CROP_MASK_TOKENS) else 0.0
    )
    eyes_identity_pose_penalty = (
        EYES_IDENTITY_POSE_PENALTY if _has_any_token(text, _EYES_IDENTITY_POSE_TOKENS) else 0.0
    )
    mouth_expression_penalty = (
        MOUTH_EXPRESSION_PENALTY if _has_any_token(text, _MOUTH_EXPRESSION_TOKENS) else 0.0
    )
    occluded_side_inference_penalty = (
        OCCLUDED_SIDE_INFERENCE_PENALTY
        if _has_any_token(text, _OCCLUDED_SIDE_INFERENCE_TOKENS)
        else 0.0
    )

    catastrophic_failure_penalty = 0.0
    if candidate_failure:
        catastrophic_failure_penalty += CATASTROPHIC_FAILURE_PENALTY
    if _catastrophic_or_visual_collapse(metric.geometry_veto_reasons):
        catastrophic_failure_penalty += DEFAULT_COLLAPSE_COST_PENALTY

    production_failure_penalty = (
        PRODUCTION_FAILURE_PENALTY
        if str(context.source or "").strip().lower() == _PRODUCTION_VALIDATED_SOURCE
        and candidate_failure
        else 0.0
    )

    return float(
        full_face_nme_regret
        + jaw_crop_mask_penalty
        + eyes_identity_pose_penalty
        + mouth_expression_penalty
        + occluded_side_inference_penalty
        + catastrophic_failure_penalty
        + production_failure_penalty
    )


def rows_for_context(
    context: SampleCandidateContext,
    *,
    high_gap_threshold: float = DEFAULT_HIGH_GAP_THRESHOLD,
) -> list[CandidateQualityRow]:
    """Expand a sample context into one row per candidate.

    NME/proxy fields remain diagnostic.  v3 ranking should use
    transform_regret_v3 over rows where rankable_v3 is true.
    """
    rows: list[CandidateQualityRow] = []
    oracle_nme = _required_nme(context, context.oracle)

    v3_by_candidate: dict[str, TransformCostV3] = {}
    truth_landmarks = context.truth_landmarks
    for candidate in context.candidates:
        metric = context.metrics[candidate.name]
        profile_soft_topology = candidate.name == PROFILE_REPAIR_CANDIDATE_NAME
        hard_invalid_reasons = _v3_hard_invalid_reasons(
            metric, profile_soft_topology=profile_soft_topology
        )
        truth_for_v3 = truth_landmarks
        if truth_for_v3 is None:
            truth_for_v3 = candidate.landmarks
            hard_invalid_reasons = ("missing_truth_landmarks", *hard_invalid_reasons)
        v3_by_candidate[candidate.name] = transform_cost_v3(
            candidate.landmarks,
            truth_for_v3,
            visibility=context.visibility,
            hard_invalid_reasons=hard_invalid_reasons,
            soft_suspect_reasons=_v3_soft_suspect_reasons(
                metric,
                hard_invalid_reasons=hard_invalid_reasons,
                profile_soft_topology=profile_soft_topology,
            ),
        )

    rankable_v3 = {
        name: cost for name, cost in v3_by_candidate.items() if not bool(cost.hard_invalid)
    }
    if rankable_v3:
        transform_oracle_candidate_v3 = min(
            rankable_v3,
            key=lambda name: float(rankable_v3[name].total_cost),
        )
        transform_oracle_cost_v3 = float(rankable_v3[transform_oracle_candidate_v3].total_cost)
        sorted_v3_costs = heapq.nsmallest(
            2,
            (float(cost.total_cost) for cost in rankable_v3.values()),
        )
        transform_oracle_gap_v3 = (
            sorted_v3_costs[1] - sorted_v3_costs[0] if len(sorted_v3_costs) > 1 else 0.0
        )
    else:
        transform_oracle_candidate_v3 = ""
        transform_oracle_cost_v3 = 0.0
        transform_oracle_gap_v3 = 0.0

    for candidate in context.candidates:
        metric = context.metrics[candidate.name]
        candidate_nme = _required_nme(context, candidate.name)
        regret = max(candidate_nme - oracle_nme, 0.0)
        normalized_regret = min(regret / DEFAULT_REGRET_NORMALIZER, 1.0)
        candidate_failure = bool(context.failure_by_candidate[candidate.name])
        large_regret = regret > high_gap_threshold
        selection_cost = downstream_weighted_alignment_cost(
            context=context,
            candidate_name=candidate.name,
            metric=metric,
            candidate_nme=candidate_nme,
            oracle_nme=oracle_nme,
            candidate_failure=candidate_failure,
        )
        v3_cost = v3_by_candidate[candidate.name]
        rankable = bool(rankable_v3) and not bool(v3_cost.hard_invalid)
        transform_cost = float(v3_cost.total_cost) if rankable else 0.0
        transform_regret = max(transform_cost - transform_oracle_cost_v3, 0.0) if rankable else 0.0
        rows.append(
            CandidateQualityRow(
                sample_id=context.sample_id,
                face_index=context.face_index,
                dataset=context.dataset,
                condition=context.condition,
                candidate_name=candidate.name,
                candidate_nme=candidate_nme,
                oracle_nme=oracle_nme,
                regret_vs_oracle=candidate_nme - oracle_nme,
                normalized_regret=normalized_regret,
                failure_label=candidate_failure,
                large_regret_label=large_regret,
                candidate_failure_or_high_gap=bool(candidate_failure or large_regret),
                selection_cost=selection_cost,
                is_oracle=candidate.name == context.oracle,
                was_selected_by_current_policy=(candidate.name == context.current_policy_choice),
                gap_vs_oracle=candidate_nme - oracle_nme,
                runtime_bucket=context.runtime_bucket,
                hard_case_tags=context.hard_case_tags,
                risk_route=context.risk_route,
                feature_values=candidate_feature_map(
                    candidate,
                    metric,
                    runtime_bucket=context.runtime_bucket,
                    risk_route=context.risk_route,
                    model_predictions_available=context.model_predictions_available,
                    roll_estimate=context.roll_estimate,
                    yaw_estimate=context.yaw_estimate,
                    candidate_yaw_disagreement=context.candidate_yaw_disagreement,
                    max_disagreement_px=context.max_disagreement_px,
                    runtime_bucket_source=context.runtime_bucket_source,
                    hard_case_tags=context.hard_case_tags,
                    candidate_extra_features=context.candidate_extra_features,
                ),
                selected_by_current_policy=context.current_policy_choice,
                selected_candidate_missing_from_eval=(
                    context.selected_candidate_missing_from_eval
                ),
                oracle=context.oracle,
                runtime_bucket_source=context.runtime_bucket_source,
                geometry_veto_reasons=tuple(metric.geometry_veto_reasons),
                transform_cost_v3=transform_cost,
                corner_delta_v3=float(v3_cost.corner_delta),
                center_delta_v3=float(v3_cost.center_delta),
                scale_delta_v3=float(v3_cost.scale_delta),
                roll_delta_degrees_v3=float(v3_cost.roll_delta_degrees),
                fit_delta_v3=float(v3_cost.fit_delta),
                transform_oracle_cost_v3=transform_oracle_cost_v3,
                transform_regret_v3=transform_regret,
                transform_oracle_candidate_v3=transform_oracle_candidate_v3,
                transform_oracle_gap_v3=transform_oracle_gap_v3,
                rankable_v3=rankable,
                hard_invalid_v3=bool(v3_cost.hard_invalid),
                hard_invalid_reasons_v3=tuple(v3_cost.hard_invalid_reasons),
                soft_structural_penalty_v3=float(v3_cost.soft_structural_penalty),
                hard_negative_weight=context.hard_negative_weight,
            )
        )
    return rows


def candidate_table_rows_for_context(
    context: SampleCandidateContext,
) -> list[dict[str, T.Any]]:
    """Expand a sample context into canonical candidate diagnostic table rows."""
    return list(iter_candidate_table_rows_for_context(context))


def _candidate_repair_provenance_columns(
    context: SampleCandidateContext, candidate_name: str
) -> dict[str, T.Any]:
    """Return repair provenance columns, populated only for the repair candidate."""
    provenance = context.profile_repair_provenance
    if provenance is None or candidate_name != PROFILE_REPAIR_CANDIDATE_NAME:
        return {
            "profile_repair_used": "",
            "profile_repair_source_candidate": "",
            "profile_repair_visible_side": "",
            "profile_repair_method": "",
            "profile_repair_reason": "",
        }
    return {
        "profile_repair_used": int(bool(provenance.get("profile_repair_used"))),
        "profile_repair_source_candidate": provenance.get("profile_repair_source_candidate", ""),
        "profile_repair_visible_side": provenance.get("profile_repair_visible_side", ""),
        "profile_repair_method": provenance.get("profile_repair_method", ""),
        "profile_repair_reason": provenance.get("profile_repair_reason", ""),
    }


def iter_candidate_table_rows_for_context(
    context: SampleCandidateContext,
) -> T.Iterator[dict[str, T.Any]]:
    """Yield canonical candidate diagnostic table rows for one sample context."""
    oracle_nme = context.nme_by_candidate[context.oracle]
    for candidate in context.candidates:
        metric = context.metrics[candidate.name]
        nme = context.nme_by_candidate[candidate.name]
        yield {
            "sample_id": context.sample_id,
            "dataset": context.dataset,
            "condition": context.condition,
            "candidate": candidate.name,
            "nme": nme,
            "failure": int(context.failure_by_candidate[candidate.name]),
            "is_oracle": int(candidate.name == context.oracle),
            "gap_vs_oracle": nme - oracle_nme,
            "cloud_area_ratio": metric.cloud_area_ratio,
            "hull_area_ratio": metric.hull_area_ratio,
            "points_outside_expanded_bbox_fraction": (
                metric.points_outside_expanded_bbox_fraction
            ),
            "eye_mouth_order_valid_after_deroll": (
                None
                if metric.eye_mouth_order_valid_after_deroll is None
                else int(metric.eye_mouth_order_valid_after_deroll)
            ),
            "roi_center_consensus_distance": metric.roi_center_consensus_distance,
            "landmark_consensus_distance": metric.landmark_consensus_distance,
            "shape_plausibility_score": metric.shape_plausibility_score,
            "shape_veto_reasons": "|".join(metric.shape_veto_reasons),
            "max_edge_length_ratio": metric.max_edge_length_ratio,
            "mean_shape_fit_error": metric.mean_shape_fit_error,
            "topology_violation_count": metric.topology_violation_count,
            "roll": metric.roll_degrees,
            "yaw": metric.yaw_degrees,
            "roll_delta_to_consensus": (
                None
                if metric.roll_degrees is None or context.roll_estimate is None
                else abs(metric.roll_degrees - context.roll_estimate)
            ),
            "yaw_delta_to_consensus": (
                None
                if metric.yaw_degrees is None or context.yaw_estimate is None
                else abs(metric.yaw_degrees - context.yaw_estimate)
            ),
            "max_disagreement_px": context.max_disagreement_px,
            "candidate_yaw_disagreement": context.candidate_yaw_disagreement,
            "runtime_bucket": context.runtime_bucket,
            "hard_case_tags": "|".join(context.hard_case_tags),
            "runtime_bucket_source": context.runtime_bucket_source,
            "selected_candidate_missing_from_eval": int(
                context.selected_candidate_missing_from_eval
            ),
            "geometry_veto_reasons": "|".join(metric.geometry_veto_reasons),
            **_candidate_repair_provenance_columns(context, candidate.name),
        }


def iter_candidate_table_rows(
    contexts: T.Iterable[SampleCandidateContext],
) -> T.Iterator[dict[str, T.Any]]:
    """Yield canonical diagnostic table rows for all sample contexts."""
    for context in contexts:
        yield from iter_candidate_table_rows_for_context(context)


def candidate_table_rows(
    contexts: T.Iterable[SampleCandidateContext],
) -> list[dict[str, T.Any]]:
    """Return canonical diagnostic table rows for all sample contexts."""
    return list(iter_candidate_table_rows(contexts))


def iter_contexts(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    candidates: T.Sequence[str] | None = None,
    source: str = "",
    resolver_metadata: T.Mapping[tuple[str, int], dict[str, T.Any]] | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    allow_image_backfill: bool = False,
    image_backfill_crop_scale: float = DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    image_backfill_crop_size: int = DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    context_workers: int = 0,
    stacked_regressor: RuntimeStackedLandmarkRegressor | None = None,
    stacked_regressor_max_residual: float = 0.0,
    progress: ContextProgress | None = None,
) -> T.Iterator[SampleCandidateContext]:
    """Yield sample contexts for one manifest/cache pair without retaining all of them."""
    weights = load_weights(weights_path)
    bucket_weights, region_weights = load_optional_weight_blocks(weights_path)
    explicit_candidates = bool(candidates)
    include_adaptive_candidates = not explicit_candidates
    requested = tuple(candidates or parse_candidates(None, weights))
    samples = filter_canonical_68_samples(
        load_manifest(manifest_path),
        context="scorer dataset",
        progress=progress,
    )

    worker_count = max(int(context_workers or 0), 0)
    serial_cache = DiskPredictionCache(cache_dir)
    thread_state = threading.local()

    def _cache_for_worker() -> DiskPredictionCache:
        if worker_count <= 1:
            return serial_cache
        cache = getattr(thread_state, "cache", None)
        if cache is None:
            cache = DiskPredictionCache(cache_dir)
            thread_state.cache = cache
        return T.cast(DiskPredictionCache, cache)

    def _build_one(sample: LandmarkSample) -> SampleCandidateContext | None:
        try:
            return build_sample_context(
                sample,
                cache=_cache_for_worker(),
                requested_candidates=requested,
                weights=weights,
                source=source,
                resolver_metadata=resolver_metadata,
                failure_threshold=failure_threshold,
                outlier_threshold=outlier_threshold,
                bucket_weights=bucket_weights,
                region_weights=region_weights,
                include_adaptive_candidates=include_adaptive_candidates,
                allow_image_backfill=allow_image_backfill,
                image_backfill_crop_scale=image_backfill_crop_scale,
                image_backfill_crop_size=image_backfill_crop_size,
                stacked_regressor=stacked_regressor,
                stacked_regressor_max_residual=stacked_regressor_max_residual,
            )
        except (FileNotFoundError, ValueError) as err:
            logger.warning("skipping sample %s: %s", sample.sample_id, err)
            return None

    iterator = (
        progress(samples, f"Load contexts [{source or manifest_path.stem}]")
        if progress is not None
        else samples
    )
    if worker_count > 1:
        logger.info(
            "Loading scorer contexts with %d worker threads from %s",
            worker_count,
            manifest_path,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            for context in executor.map(_build_one, iterator):
                if context is not None:
                    yield context
        return

    for sample in iterator:
        context = _build_one(sample)
        if context is not None:
            yield context


def load_contexts(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    candidates: T.Sequence[str] | None = None,
    source: str = "",
    resolver_metadata: T.Mapping[tuple[str, int], dict[str, T.Any]] | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    allow_image_backfill: bool = False,
    image_backfill_crop_scale: float = DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    image_backfill_crop_size: int = DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    context_workers: int = 0,
    stacked_regressor: RuntimeStackedLandmarkRegressor | None = None,
    stacked_regressor_max_residual: float = 0.0,
    progress: ContextProgress | None = None,
) -> list[SampleCandidateContext]:
    """Load all sample contexts for one manifest/cache pair."""
    return list(
        iter_contexts(
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            weights_path=weights_path,
            candidates=candidates,
            source=source,
            resolver_metadata=resolver_metadata,
            failure_threshold=failure_threshold,
            outlier_threshold=outlier_threshold,
            allow_image_backfill=allow_image_backfill,
            image_backfill_crop_scale=image_backfill_crop_scale,
            image_backfill_crop_size=image_backfill_crop_size,
            stacked_regressor=stacked_regressor,
            stacked_regressor_max_residual=stacked_regressor_max_residual,
            progress=progress,
        )
    )


def _candidate_table_output_row(row: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    """Return a CSV/parquet-safe row with the canonical candidate table columns."""
    return {name: row.get(name) for name in CANDIDATE_TABLE_COLUMNS}


def write_candidate_table_csv(rows: T.Iterable[T.Mapping[str, T.Any]], path: Path) -> Path:
    """Write the canonical candidate diagnostic table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANDIDATE_TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(_candidate_table_output_row(row))
    return path


def write_candidate_table_parquet(rows: T.Iterable[T.Mapping[str, T.Any]], path: Path) -> Path:
    """Write the canonical candidate diagnostic table to parquet when available."""
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "parquet export requires pandas plus a parquet engine such as pyarrow; "
            "use --output-csv in this environment or install the optional parquet stack"
        ) from err
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(
        (_candidate_table_output_row(row) for row in rows),
        columns=list(CANDIDATE_TABLE_COLUMNS),
    )
    frame.to_parquet(path, index=False)
    return path


def export_candidate_table(
    *,
    manifest_path: Path,
    cache_dir: Path,
    weights_path: Path,
    candidates: T.Sequence[str] | None = None,
    failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    allow_image_backfill: bool = False,
    image_backfill_crop_scale: float = DEFAULT_IMAGE_BACKFILL_CROP_SCALE,
    image_backfill_crop_size: int = DEFAULT_IMAGE_BACKFILL_CROP_SIZE,
    progress: ContextProgress | None = None,
) -> list[dict[str, T.Any]]:
    """Build the canonical candidate diagnostic table for one manifest/cache pair."""
    return list(
        iter_candidate_table_rows(
            iter_contexts(
                manifest_path=manifest_path,
                cache_dir=cache_dir,
                weights_path=weights_path,
                candidates=candidates,
                failure_threshold=failure_threshold,
                outlier_threshold=outlier_threshold,
                allow_image_backfill=allow_image_backfill,
                image_backfill_crop_scale=image_backfill_crop_scale,
                image_backfill_crop_size=image_backfill_crop_size,
                progress=progress,
            )
        )
    )


def write_rows_csv(rows: T.Sequence[CandidateQualityRow], path: Path) -> Path:
    """Write scorer training rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_names = list(runtime_feature_order(row.feature_values for row in rows))
    fieldnames = [
        "sample_id",
        "face_index",
        "dataset",
        "condition",
        "candidate_name",
        "candidate_nme",
        "oracle_nme",
        "regret_vs_oracle",
        "normalized_regret",
        "failure_label",
        "large_regret_label",
        "candidate_failure_or_high_gap",
        "selection_cost",
        "transform_cost_v3",
        "corner_delta_v3",
        "center_delta_v3",
        "scale_delta_v3",
        "roll_delta_degrees_v3",
        "fit_delta_v3",
        "transform_oracle_cost_v3",
        "transform_regret_v3",
        "transform_oracle_candidate_v3",
        "transform_oracle_gap_v3",
        "rankable_v3",
        "hard_invalid_v3",
        "hard_invalid_reasons_v3",
        "soft_structural_penalty_v3",
        "hard_negative_weight",
        "is_oracle",
        "was_selected_by_current_policy",
        "gap_vs_oracle",
        "runtime_bucket",
        "runtime_bucket_source",
        "hard_case_tags",
        "risk_route",
        "geometry_veto_reasons",
        "selected_by_current_policy",
        "selected_candidate_missing_from_eval",
        "oracle",
        "features_json",
        *feature_names,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = row.to_csv_row()
            feature_values = row.feature_values
            for name in feature_names:
                output[name] = feature_values.get(name, 0.0)
            writer.writerow(output)
    return path


__all__ = [
    "CANDIDATE_TABLE_COLUMNS",
    "CandidateQualityRow",
    "DEFAULT_RESOLVER_CANDIDATES",
    "DEFAULT_SCORER_CANDIDATES",
    "DEFAULT_SCORER_CANDIDATE_CSV",
    "SampleCandidateContext",
    "build_sample_context",
    "candidate_table_rows",
    "candidate_table_rows_for_context",
    "export_candidate_table",
    "iter_candidate_table_rows",
    "iter_candidate_table_rows_for_context",
    "iter_contexts",
    "load_contexts",
    "parse_candidates",
    "rows_for_context",
    "write_candidate_table_csv",
    "write_candidate_table_parquet",
    "write_rows_csv",
]
