#! /usr/env/bin/python3
"""Handles the collation, weighting masking and calculation of the selected Loss functions for
training Faceswap models"""

from __future__ import annotations

import logging
import math
import typing as T
from dataclasses import dataclass, field, replace

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.model.losses import get_loss_function
from lib.utils import get_module_objects

from .extra_losses import (
    BoundaryLoss,
    IdentityLoss,
    RegionWeightedPerceptualLoss,
    occlusion_exclusion_weight,
)

if T.TYPE_CHECKING:
    from .data import BatchMeta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BRLWStats:
    """Statistics for batch-relative loss weighting."""

    weights: torch.Tensor
    min_weight: torch.Tensor
    mean_weight: torch.Tensor
    max_weight: torch.Tensor
    std_weight: torch.Tensor
    effective_batch_size: torch.Tensor


@dataclass(frozen=True)
class BatchRelativeLossWeighting:
    """Configuration for batch-relative loss weighting."""

    enabled: bool = False
    strength: float = 0.0
    min_batch_size: int = 4
    min_weight: float = 0.5
    max_weight: float = 2.0
    detach_weights: bool = True
    eps: float = 1e-8
    protected_samples: torch.Tensor | None = None
    """Boolean mask for samples that must not receive a weight above 1.0."""

    def __post_init__(self) -> None:
        for name in ("strength", "min_weight", "max_weight", "eps"):
            value = T.cast(float, getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"BRLW {name} must be finite. Got {value}")
        if self.strength < 0.0:
            raise ValueError(f"BRLW strength must be >= 0.0. Got {self.strength}")
        if self.min_batch_size < 1:
            raise ValueError(f"BRLW minimum batch size must be >= 1. Got {self.min_batch_size}")
        if self.min_weight <= 0.0:
            raise ValueError(f"BRLW minimum weight must be > 0.0. Got {self.min_weight}")
        if self.max_weight < self.min_weight:
            raise ValueError(
                "BRLW maximum weight must be >= minimum weight. "
                f"Got min={self.min_weight}, max={self.max_weight}"
            )
        if not (self.min_weight <= 1.0 <= self.max_weight):
            raise ValueError(
                "BRLW weights must allow a mean of 1.0. "
                f"Got min={self.min_weight}, max={self.max_weight}"
            )

    @property
    def active(self) -> bool:
        """``True`` when BRLW can alter a batch loss."""
        return self.enabled and self.strength > 0.0

    def with_protected_samples(
        self, protected_samples: torch.Tensor | None
    ) -> BatchRelativeLossWeighting:
        """Return a copy with per-sample overweight protection attached."""
        return replace(self, protected_samples=protected_samples)

    def _max_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Return per-sample maximum weights, including metadata safeguards."""
        max_weights = torch.full_like(weights, self.max_weight)
        if self.protected_samples is None:
            return max_weights
        protected = self.protected_samples.to(device=weights.device, dtype=torch.bool)
        if protected.shape != weights.shape:
            return max_weights
        return torch.where(
            protected, torch.minimum(max_weights, torch.ones_like(weights)), max_weights
        )

    def _normalize_clamped(self, weights: torch.Tensor) -> torch.Tensor:
        """Normalize weights to sum to batch size without exceeding configured bounds."""
        eps = max(self.eps, torch.finfo(weights.dtype).eps)
        max_weights = self._max_weights(weights)
        normalized = torch.maximum(weights, torch.full_like(weights, self.min_weight))
        normalized = torch.minimum(normalized, max_weights)
        target = torch.as_tensor(
            float(normalized.numel()), dtype=normalized.dtype, device=normalized.device
        )
        active = torch.ones_like(normalized, dtype=torch.bool)

        for _ in range(normalized.numel()):
            active_sum = normalized[active].sum()
            if not bool(active.any()) or active_sum.abs() < eps:
                break

            remaining = target - normalized[~active].sum()
            candidate = normalized[active] * (remaining / active_sum.clamp_min(eps))
            too_low = candidate < self.min_weight
            active_max = max_weights[active]
            too_high = candidate > active_max
            active_indices = active.nonzero(as_tuple=False).flatten()

            if not bool((too_low | too_high).any()):
                normalized[active_indices] = candidate
                break

            low_indices = active_indices[too_low]
            high_indices = active_indices[too_high]
            normalized[low_indices] = self.min_weight
            normalized[high_indices] = max_weights[high_indices]
            active[low_indices] = False
            active[high_indices] = False

        return normalized

    def apply(self, sample_loss: torch.Tensor) -> tuple[torch.Tensor, BRLWStats | None]:
        """Apply BRLW to a per-sample aggregate loss vector."""
        if not self.active or sample_loss.ndim == 0 or sample_loss.numel() < self.min_batch_size:
            return sample_loss.mean(), None

        work_loss = sample_loss.to(dtype=torch.float32)
        eps = max(self.eps, torch.finfo(work_loss.dtype).eps)
        relative = work_loss / work_loss.mean().clamp_min(eps)
        weights = 1.0 + self.strength * (relative - 1.0)
        weights = self._normalize_clamped(weights)
        loss_weights = weights.to(dtype=sample_loss.dtype)
        loss_weights = loss_weights.detach() if self.detach_weights else loss_weights
        loss = (sample_loss * loss_weights).mean()

        stats = BRLWStats(
            weights=weights,
            min_weight=weights.min(),
            mean_weight=weights.mean(),
            max_weight=weights.max(),
            std_weight=weights.std(unbiased=False),
            effective_batch_size=weights.sum().square() / weights.square().sum().clamp_min(eps),
        )
        return loss, stats


@dataclass
class BatchLoss:
    """Dataclass for holding Loss values for a batch of data"""

    unweighted: list[dict[str, torch.Tensor]]
    """For each side output, the unweighted loss scalars for each function for each item in the
    batch"""
    weighted: list[dict[str, torch.Tensor]]
    """For each side output, the weighted loss scalars for each function for each item in the
    batch"""
    mask: torch.Tensor | None = None
    """The loss scalar for the mask for each item in the batch if learn_mask is selected otherwise
    ``None``. Default: ``None``"""
    brlw: BatchRelativeLossWeighting = field(default_factory=BatchRelativeLossWeighting)
    """Optional batch-relative loss weighting configuration."""
    brlw_stats: BRLWStats | None = None
    """Statistics from the last BRLW reduction, if BRLW was active for this batch."""
    _total: torch.Tensor | None = field(init=False, default=None)

    @property
    def total(self) -> torch.Tensor:
        """The total single weighted loss scalar for all items in the batch for backprop"""
        if self._total is None:
            component_losses = [y for x in self.weighted for y in x.values()]
            if self.mask is not None:
                component_losses.append(self.mask)
            if self.brlw.active and component_losses:
                sample_loss = T.cast(torch.Tensor, sum(component_losses))
                total, self.brlw_stats = self.brlw.apply(sample_loss)
            else:
                total = T.cast(torch.Tensor, sum(y.mean() for y in component_losses))
                self.brlw_stats = None
            self._total = total
        return self._total

    def to_cpu(self) -> T.Self:  # type: ignore[name-defined]
        """Detaches all contained loss values and moves them to CPU

        Returns
        -------
        This object with all tensors detached and moved to CPU
        """
        self._total = None if self._total is None else self._total.detach().cpu()
        self.unweighted = [{k: v.detach().cpu() for k, v in x.items()} for x in self.unweighted]
        self.weighted = [{k: v.detach().cpu() for k, v in x.items()} for x in self.weighted]
        self.mask = None if self.mask is None else self.mask.detach().cpu()
        if self.brlw_stats is not None:
            self.brlw_stats = BRLWStats(
                weights=self.brlw_stats.weights.detach().cpu(),
                min_weight=self.brlw_stats.min_weight.detach().cpu(),
                mean_weight=self.brlw_stats.mean_weight.detach().cpu(),
                max_weight=self.brlw_stats.max_weight.detach().cpu(),
                std_weight=self.brlw_stats.std_weight.detach().cpu(),
                effective_batch_size=self.brlw_stats.effective_batch_size.detach().cpu(),
            )
        if self.brlw.protected_samples is not None:
            self.brlw = self.brlw.with_protected_samples(
                self.brlw.protected_samples.detach().cpu()
            )
        return self


class LossCollator(nn.Module):  # pylint:disable=too-many-instance-attributes
    """Compiles the chosen loss functions and calculates the values in the training loop

    Parameters
    ----------
    functions
        List of lost function names from configuration file to collate for loss calculation
    weights
        List of weights, corresponding to the the list of functions, to apply to each loss function
    color_order
        The color order that the model is training in
    use_mask
        ``True`` if loss should be masked as `penalize mask loss` has been selected
    eye_multiplier
        The amount of extra weighting to apply to the eye area
    mouth_multiplier
        The amount of extra weighting to apply to the mouth area
    smallest_output
        The smallest output from the model. Required for initializing some loss functions
    mask_loss
        The loss function to use if learn_mask is enabled. Default: ``None`` (not enabled)
    occlusion_strength
        Strength (``0.0 - 1.0``) for excluding occluded regions (using ``meta.mask_occlusion``)
        from the reconstruction loss. ``0.0`` disables occlusion exclusion. Default: ``0.0``
    boundary_loss
        Optional configured boundary-aware reconstruction loss. Default: ``None`` (disabled)
    boundary_weight
        The weight to apply to the boundary loss. Default: ``0.0``
    region_perceptual_loss
        Optional configured region-weighted perceptual loss. Default: ``None`` (disabled)
    region_perceptual_weight
        The weight to apply to the region-weighted perceptual loss. Default: ``0.0``
    identity_loss
        Optional configured frozen identity embedding loss. Default: ``None`` (disabled)
    identity_weight
        The weight to apply to the identity loss. Default: ``0.0``
    identity_start_iteration
        The iteration at which the identity loss begins to be applied. Default: ``0``
    brlw_enabled
        ``True`` to enable batch-relative loss weighting. Default: ``False``
    brlw_strength
        Maximum BRLW strength. ``None`` selects the conservative automatic value. Default:
        ``None``
    brlw_min_batch_size
        Minimum batch size before BRLW can apply. Default: ``4``
    brlw_min_weight
        Minimum per-sample BRLW weight. Default: ``0.5``
    brlw_max_weight
        Maximum per-sample BRLW weight. Default: ``2.0``
    brlw_warmup_iterations
        Number of iterations to ramp BRLW when no phase schedule is injected. ``None`` selects
        the automatic warmup. Default: ``None``
    brlw_detach_weights
        ``True`` to detach BRLW weights from gradients. Default: ``True``
    """

    _AUTO_BRLW_STRENGTH = 0.25
    _AUTO_BRLW_WARMUP_ITERATIONS = 10_000

    def __init__(
        self,
        functions: list[str],
        weights: list[float],
        color_order: T.Literal["bgr", "rgb"],
        use_mask: bool,
        eye_multiplier: float,
        mouth_multiplier: float,
        smallest_output: int,
        mask_loss: str | None = None,
        occlusion_strength: float = 0.0,
        boundary_loss: BoundaryLoss | None = None,
        boundary_weight: float = 0.0,
        region_perceptual_loss: RegionWeightedPerceptualLoss | None = None,
        region_perceptual_weight: float = 0.0,
        identity_loss: IdentityLoss | None = None,
        identity_weight: float = 0.0,
        identity_start_iteration: int = 0,
        brlw_enabled: bool = False,
        brlw_strength: float | None = None,
        brlw_min_batch_size: int = 4,
        brlw_min_weight: float = 0.5,
        brlw_max_weight: float = 2.0,
        brlw_warmup_iterations: int | None = None,
        brlw_detach_weights: bool = True,
    ) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        self._color_order: T.Literal["bgr", "rgb"] = color_order
        self._use_mask = use_mask
        self._eye_multiplier = eye_multiplier
        self._mouth_multiplier = mouth_multiplier
        self._smallest_output = smallest_output
        self._mask_loss = mask_loss
        self._functions, self._weights = self._configure_functions(functions, weights)
        self._spatial, self._non_spatial = self._get_function_types()

        self._occlusion_strength = occlusion_strength
        self._boundary_loss = boundary_loss
        self._boundary_weight = boundary_weight
        self._region_perceptual_loss = region_perceptual_loss
        self._region_perceptual_weight = region_perceptual_weight
        self._identity_loss = identity_loss
        self._identity_weight = identity_weight
        self._identity_start_iteration = identity_start_iteration
        self._iteration = 0
        self._brlw_enabled = brlw_enabled
        self._brlw_strength = brlw_strength
        self._brlw_min_batch_size = brlw_min_batch_size
        self._brlw_min_weight = brlw_min_weight
        self._brlw_max_weight = brlw_max_weight
        self._brlw_warmup_iterations = brlw_warmup_iterations
        self._brlw_detach_weights = brlw_detach_weights
        self._brlw_schedule_multiplier: float | None = None

        # First configured reconstruction loss is treated as the "primary" component, the
        # remainder as "secondary". The phase scheduler injects per-component multipliers via
        # :meth:`set_schedule`; the defaults below are exact no-ops so disabled automation and
        # un-scheduled training behave identically.
        self._primary_function: str | None = next(iter(self._functions), None)
        self._schedule: dict[str, float] = {
            "primary_loss": 1.0,
            "secondary_loss": 1.0,
            "boundary_loss": 1.0,
            "region_perceptual_loss": 1.0,
            "identity_loss": 1.0,
        }

        self._mask_loss_function = (
            None
            if mask_loss is None
            else self._functions[mask_loss]
            if mask_loss in self._functions
            else get_loss_function(mask_loss)
        )

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {"functions": list(self._functions), "weights": list(self._weights.values())}
        params |= {
            k[1:]: v
            for k, v in self.__dict__.items()
            if k
            in (
                "_color_order",
                "_use_mask",
                "_eye_multiplier",
                "_mouth_multiplier",
                "_smallest_output",
                "_mask_loss",
            )
        }
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    @property
    def batch_relative_loss_weighting(self) -> BatchRelativeLossWeighting:
        """Return the BRLW reducer for the current iteration and schedule state."""
        base_strength = (
            self._AUTO_BRLW_STRENGTH if self._brlw_strength is None else self._brlw_strength
        )
        if self._brlw_schedule_multiplier is not None:
            strength = base_strength * self._brlw_schedule_multiplier
        else:
            warmup = (
                self._AUTO_BRLW_WARMUP_ITERATIONS
                if self._brlw_warmup_iterations is None
                else self._brlw_warmup_iterations
            )
            progress = 1.0 if warmup <= 0 else min(1.0, max(0.0, self._iteration / warmup))
            strength = base_strength * progress

        return BatchRelativeLossWeighting(
            enabled=self._brlw_enabled,
            strength=strength,
            min_batch_size=self._brlw_min_batch_size,
            min_weight=self._brlw_min_weight,
            max_weight=self._brlw_max_weight,
            detach_weights=self._brlw_detach_weights,
        )

    def set_iteration(self, iteration: int) -> None:
        """Update the current training iteration.

        Used to gate losses (such as the identity loss) that only begin after a configured
        iteration.

        Parameters
        ----------
        iteration
            The current model training iteration
        """
        self._iteration = iteration

    def set_schedule(self, multipliers: T.Mapping[str, float] | None) -> None:
        """Inject scheduled per-component loss multipliers from the phase scheduler.

        The multipliers are applied on top of the user-configured loss weights, which remain
        the requested maximums. Passing ``None`` (or omitting a key) restores the no-op default
        of ``1.0`` for that component, so disabled automation is an exact behavioral no-op.

        Parameters
        ----------
        multipliers
            Mapping of loss-component name (``primary_loss``, ``secondary_loss``,
            ``boundary_loss``, ``region_perceptual_loss``, ``identity_loss``) to a non-negative
            multiplier, or ``None`` to reset all components to ``1.0``.
        """
        for name in self._schedule:
            value = 1.0 if multipliers is None else float(multipliers.get(name, 1.0))
            if value < 0.0:
                raise ValueError(f"Scheduled multiplier '{name}' must be >= 0.0. Got {value}")
            self._schedule[name] = value
        self._brlw_schedule_multiplier = (
            None if multipliers is None else float(multipliers.get("secondary_loss", 1.0))
        )

    def _function_schedule_multiplier(self, name: str) -> float:
        """Return the scheduled multiplier for a configured reconstruction loss function."""
        if name == self._primary_function:
            return self._schedule["primary_loss"]
        return self._schedule["secondary_loss"]

    @staticmethod
    def _is_brlw_protected_sample(sample: T.Any) -> bool:
        """Return whether metadata marks a sample as unsafe to overweight."""
        return getattr(sample, "has_faceqa", False) and (
            getattr(sample, "duplicate_bucket", "unknown") == "duplicate"
            or getattr(sample, "identity_outlier_bucket", "unknown") in ("outlier", "reject")
        )

    def _brlw_protected_samples(
        self,
        meta: BatchMeta,
        weighted: list[dict[str, torch.Tensor]],
        mask_loss: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return a per-sample mask for metadata rows BRLW must not overweight."""
        if meta.faceqa is None or not meta.faceqa:
            return None
        reference = mask_loss
        for output_losses in weighted:
            if output_losses:
                reference = next(iter(output_losses.values()))
                break
        if reference is None or reference.ndim == 0:
            return None

        samples = meta.faceqa[0]
        if len(samples) != reference.numel():
            return None
        protected = [self._is_brlw_protected_sample(sample) for sample in samples]
        if not any(protected):
            return None
        return torch.tensor(protected, dtype=torch.bool, device=reference.device)

    def _occlusion_weight(self, meta: BatchMeta, index: int) -> torch.Tensor | None:
        """Obtain the per-pixel occlusion-exclusion weight for an output if enabled.

        Parameters
        ----------
        meta
            The meta information for the batch
        index
            The output index for obtaining the correct meta data

        Returns
        -------
        The per-pixel weight in ``(N, 1, H, W)`` order, or ``None`` if occlusion exclusion is
        disabled or no occlusion mask is available
        """
        if self._occlusion_strength <= 0.0 or meta.mask_occlusion is None:
            return None
        return occlusion_exclusion_weight(meta.mask_occlusion[index], self._occlusion_strength)

    def _configure_functions(
        self, names: list[str], weights: list[float]
    ) -> tuple[nn.ModuleDict, dict[str, float]]:
        """Configure the selected loss functions and send to the correct device

        Parameters
        ----------
        names
            List of lost function names from configuration file to collate for loss calculation
        weights
            List of weights, corresponding to the the list of functions, to apply to each loss
            function

        Returns
        -------
        functions
            ModuleDict of configured loss functions
        weights
            dict of loss names to weight to apply

        Raises
        ------
        ValueError
            If the number of function names and loss weights do not correspond
        """
        if len(names) != len(weights):
            raise ValueError(
                f"Number of loss functions ({len(names)}) and weights "
                f"({len(weights)}) should match"
            )

        functions = nn.ModuleDict()
        weight_dict: dict[str, float] = {}
        for name, weight in zip(names, weights, strict=False):
            if name is None or name == "none" or weight <= 0.0:
                continue
            functions[name] = get_loss_function(name, self._color_order)
            weight_dict[name] = weight

        logger.debug(
            "[Loss] Configured loss functions: %s",
            {k: (functions[k].__class__.__name__, weight_dict[k]) for k in functions},
        )
        return functions, weight_dict

    def _get_function_types(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Run a small tensor through each of the selected loss functions to determine which are
        spatial or non-spatial loss functions

        Returns
        -------
        spatial
            Tuple of loss names that produce spatial output
        non_spatial
            Tuple of loss names that produce non-spatial output
        """
        size = self._smallest_output
        dummy_a = torch.rand((1, 3, size, size), dtype=torch.float32)
        dummy_b = torch.rand((1, 3, size, size), dtype=torch.float32)
        spatial: list[str] = []
        non_spatial: list[str] = []
        for name, func in self._functions.items():
            out = func(dummy_a, dummy_b)
            dims = out.ndim
            if dims not in (1, 4):
                raise RuntimeError(
                    "Loss functions should return either spatial output per item "
                    f"(N, C, H, W) (4 dims) or scalar per item (N, ) (1 dim). "
                    f"Got {dims} dims for '{name}'"
                )
            dst = spatial if dims == 4 else non_spatial
            dst.append(name)

        logger.debug("[Loss] spatial: %s, non-spatial: %s", spatial, non_spatial)
        return tuple(spatial), tuple(non_spatial)

    def _get_spatial_loss(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, meta: BatchMeta, index: int
    ) -> dict[str, torch.Tensor]:
        """Obtain the unweighted loss values for the spatial loss functions

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The batch of model predictions
        meta
            The meta information for the batch
        index
            The output index for obtaining the correct meta data for the processing output

        Returns
        -------
        The unweighted loss scalar for each loss function with masks and multipliers applied
        """
        occlusion = self._occlusion_weight(meta, index)
        retval: dict[str, torch.Tensor] = {}
        for name in self._spatial:
            loss: torch.Tensor = self._functions[name](y_true, y_pred)
            if self._use_mask and meta.mask_face is not None:
                loss *= meta.mask_face[index]
            if self._eye_multiplier > 1.0 and meta.mask_eye is not None:
                loss += loss * meta.mask_eye[index] * self._eye_multiplier
            if self._mouth_multiplier > 1.0 and meta.mask_mouth is not None:
                loss += loss * meta.mask_mouth[index] * self._mouth_multiplier
            if occlusion is not None:
                loss = loss * occlusion
            retval[name] = loss.mean(dim=tuple(range(1, loss.ndim)))
        logger.trace("[Loss] Spatial loss: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _get_masked_inputs(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, meta: BatchMeta, index: int
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[float]]:
        """For non spatial loss functions the inputs need to be masked for each supplied masks

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The batch of model predictions
        meta
            The meta information for the batch
        index
            The output index for obtaining the correct meta data for the processing output

        Returns
        -------
        inputs
            The (y_true, y_pred) inputs to the loss function for each supplied mask
        weights
            The weight to be applied for each masked input
        """
        weights = [1.0]
        assert meta.mask_face is not None
        face_mask = meta.mask_face[index]
        inputs = [(y_true * face_mask, y_pred * face_mask)]
        for m_type in ("eye", "mouth"):
            masks: list[torch.Tensor] | None = getattr(meta, f"mask_{m_type}")
            if masks is None:
                continue
            mask = masks[index]
            inputs.append((y_true * mask, y_pred * mask))
            weights.append(self._eye_multiplier if m_type == "eye" else self._mouth_multiplier)
        logger.trace(  # type: ignore[attr-defined]
            "[Loss] masked inputs: %s, weights: %s",
            [[x.shape for x in i] for i in inputs],
            weights,
        )
        return inputs, weights

    def _get_non_spatial_loss(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, meta: BatchMeta, index: int
    ) -> dict[str, torch.Tensor]:
        """Obtain the unweighted loss values for the non-spatial loss functions

        Parameters
        ----------
        y_true
            The ground truth batch of images
        y_pred
            The batch of model predictions
        meta
            The meta information for the batch
        index
            The output index for obtaining the correct meta data for the processing output

        Returns
        -------
        The unweighted loss scalar for each loss function with masks and multipliers applied
        """
        occlusion = self._occlusion_weight(meta, index)
        if occlusion is not None:
            y_true = y_true * occlusion
            y_pred = y_pred * occlusion

        retval: dict[str, torch.Tensor] = {}
        if not self._use_mask:
            inputs = [(y_true, y_pred)]
            weights = [1.0]
        else:
            inputs, weights = self._get_masked_inputs(y_true, y_pred, meta, index)

        for name in self._non_spatial:
            losses = torch.stack(
                [
                    self._functions[name](inp_true, inp_pred) * weight
                    for weight, (inp_true, inp_pred) in zip(weights, inputs, strict=False)
                ]
            )
            retval[name] = losses.sum(dim=0)

        logger.trace("[Loss] Non-spatial loss: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _get_extra_losses(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        meta: BatchMeta,
        index: int,
        is_largest: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Obtain the unweighted and weighted values for the optional advanced losses.

        Parameters
        ----------
        y_true
            The ground truth batch of images in ``(N, C, H, W)`` order
        y_pred
            The batch of model predictions in ``(N, C, H, W)`` order
        meta
            The meta information for the batch
        index
            The output index for obtaining the correct meta data
        is_largest
            ``True`` if this is the largest reconstruction output (used to gate the identity
            loss so that it only runs once per side)

        Returns
        -------
        unweighted
            The unweighted advanced loss scalars for each item in the batch
        weighted
            The weighted advanced loss scalars for each item in the batch
        """
        face_mask = None if meta.mask_face is None else meta.mask_face[index]
        eye_mask = None if meta.mask_eye is None else meta.mask_eye[index]
        mouth_mask = None if meta.mask_mouth is None else meta.mask_mouth[index]
        unweighted: dict[str, torch.Tensor] = {}
        weighted: dict[str, torch.Tensor] = {}

        if self._boundary_loss is not None and face_mask is not None:
            loss = self._boundary_loss(y_true, y_pred, face_mask)
            unweighted["boundary"] = loss
            weighted["boundary"] = loss * self._boundary_weight * self._schedule["boundary_loss"]

        if self._region_perceptual_loss is not None:
            loss = self._region_perceptual_loss(y_true, y_pred, face_mask, eye_mask, mouth_mask)
            unweighted["region_perceptual"] = loss
            weighted["region_perceptual"] = (
                loss * self._region_perceptual_weight * self._schedule["region_perceptual_loss"]
            )

        if (
            self._identity_loss is not None
            and is_largest
            and self._iteration >= self._identity_start_iteration
        ):
            loss = self._identity_loss(y_true, y_pred, face_mask)
            unweighted["identity"] = loss
            weighted["identity"] = loss * self._identity_weight * self._schedule["identity_loss"]

        return unweighted, weighted

    def forward(
        self, y_true_all: list[torch.Tensor], y_pred_all: list[torch.Tensor], meta: BatchMeta
    ) -> BatchLoss:
        """Call the loss functions, reduce to batch dimension, apply masks and weighting and obtain
        the weighted and unweighted per function values and the weighted total loss scalar

        Parameters
        ----------
        y_true_all
            The ground truth batch of images for all outputs for a side of the model
        y_pred_all
            The batch of model predictions for all outputs for a side of the model
        meta
            The meta information for the batch

        Returns
        -------
        The loss scalars for the batch
        """
        # Identify the largest reconstruction output so single-shot losses (e.g. identity)
        # run once per side on the highest-resolution face.
        recon_sizes = {
            idx: y_true.shape[1] for idx, y_true in enumerate(y_true_all) if y_true.shape[-1] != 1
        }
        largest_idx = max(recon_sizes, key=lambda i: recon_sizes[i]) if recon_sizes else -1

        all_unweighted: list[dict[str, torch.Tensor]] = []
        all_weighted: list[dict[str, torch.Tensor]] = []
        mask_loss = None
        for idx, (y_true, y_pred) in enumerate(zip(y_true_all, y_pred_all, strict=False)):
            # TODO remove once channels first
            y_true = y_true.permute(0, 3, 1, 2)
            y_pred = y_pred.permute(0, 3, 1, 2)

            if y_true.shape[1] == 1:
                assert self._mask_loss_function is not None
                mask_loss = T.cast(torch.Tensor, self._mask_loss_function(y_true, y_pred))
                mask_loss = mask_loss.mean(dim=tuple(range(1, mask_loss.ndim)))
                continue

            unweighted = self._get_spatial_loss(y_true, y_pred, meta, idx)
            unweighted |= self._get_non_spatial_loss(y_true, y_pred, meta, idx)
            weighted = {
                k: v * self._weights[k] * self._function_schedule_multiplier(k)
                for k, v in unweighted.items()
            }

            extra_unweighted, extra_weighted = self._get_extra_losses(
                y_true, y_pred, meta, idx, idx == largest_idx
            )
            unweighted |= extra_unweighted
            weighted |= extra_weighted

            all_unweighted.append(unweighted)
            all_weighted.append(weighted)

        brlw = self.batch_relative_loss_weighting.with_protected_samples(
            self._brlw_protected_samples(meta, all_weighted, mask_loss)
        )
        retval = BatchLoss(
            unweighted=all_unweighted,
            weighted=all_weighted,
            mask=mask_loss,
            brlw=brlw,
        )
        logger.trace("[Loss] %s", retval)  # type:ignore[attr-defined]
        return retval


__all__ = get_module_objects(__name__)
