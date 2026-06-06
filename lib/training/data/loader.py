#! /usr/env/bin/python3
"""Handles the loading of data for training and previews for faceswap models"""

from __future__ import annotations

import logging
import os
import typing as T

import numpy as np
import torch
from torch.utils import data as tch_data
from torch.utils.data import DataLoader

from lib.logger import parse_class_init
from lib.training.faceqa_diagnostics import FaceQAMetadataIndex
from lib.training.faceqa_sampler import (
    FaceQASamplerConfig,
    FaceQASamplerSummary,
    SamplerMode,
    compute_faceqa_sample_weights,
    normalize_dimensions,
)
from lib.utils import get_module_objects
from plugins.train import train_config as mod_cfg
from plugins.train.trainer import trainer_config as trn_cfg

from .collate import Collate, LandmarkMatcher
from .data_set import MultiDataset, PreviewSet, TrainSet, get_label

if T.TYPE_CHECKING:
    from lib.align.constants import CenteringType
    from plugins.train.trainer.base import TrainConfig

    from .collate import BatchMeta

logger = logging.getLogger(__name__)
_AUTO_FACEQA_SAMPLER_STRENGTH = 0.75


def _auto_strength(value: object) -> float:
    """Return a conservative sampler strength for ``auto`` or parse a configured value."""
    text = str(value).strip().lower()
    if text == "auto":
        return _AUTO_FACEQA_SAMPLER_STRENGTH
    try:
        strength = float(text)
    except ValueError as err:
        raise ValueError(
            f"faceqa_sampler_strength must be 'auto' or a non-negative number. Got {value!r}"
        ) from err
    if strength < 0.0:
        raise ValueError(f"faceqa_sampler_strength must be >= 0.0. Got {strength}")
    return strength


class TrainLoader:  # pylint:disable=too-many-instance-attributes
    """Generator for feeding faceswap models with multiple inputs and outputs. Gets the next items
    from each of the configured loaders and collates them for feeding into a model

    Parameters
    ----------
    input_size
        The input size to the model
    output_sizes
        The output sizes to the model (list as some models have multi-scale outputs)
    color_order
        The color order of the model
    config
        The training configuration for feeding the model
    sampler
        The sampler to use for the data loaders. Default: ``None`` (RandomSampler)
    include_faceqa_diagnostics
        ``True`` to attach FaceQA metadata to training batches for diagnostics.
    faceqa_metadata_paths
        Optional FaceQA-enriched alignments fallback paths by training side.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: tuple[int, ...],
        color_order: T.Literal["bgr", "rgb"],
        config: TrainConfig,
        sampler: None | type[tch_data.RandomSampler | tch_data.DistributedSampler] = None,
        *,
        include_faceqa_diagnostics: bool = False,
        faceqa_metadata_paths: list[str | None] | None = None,
    ) -> None:
        logger.debug(parse_class_init(locals()))
        self._learn_mask = mod_cfg.Loss.learn_mask()
        self._output_sizes = output_sizes
        self._config = config
        self._process_size = max(*self._output_sizes, input_size)
        self._landmarks: None | LandmarkMatcher = None

        self._faceqa_training_diagnostics = include_faceqa_diagnostics
        self._faceqa_metadata_paths = (
            [] if faceqa_metadata_paths is None else faceqa_metadata_paths
        )
        self._faceqa_sampler = self._faceqa_sampler_config()
        self._faceqa_sampler_multiplier = (
            0.0
            if (
                self._faceqa_sampler.mode == "faceqa_curriculum"
                and trn_cfg.Automation.training_automation() != "off"
            )
            else 1.0
        )
        self._faceqa_sampler_bucket_losses: dict[tuple[str, str, str], float] = {}
        self._faceqa_sampler_summaries: list[FaceQASamplerSummary] = []

        if config.warp and config.cache_landmarks:
            self._landmarks = LandmarkMatcher(
                config.folders,
                self._process_size,
                T.cast("CenteringType", mod_cfg.centering()),
                mod_cfg.coverage() / 100.0,
                mod_cfg.vertical_offset() / 100.0,
            )

        self._input_size = input_size
        self._color_order: T.Literal["bgr", "rgb"] = T.cast(
            T.Literal["bgr", "rgb"], color_order.lower()
        )
        self._sampler = tch_data.RandomSampler if sampler is None else sampler
        self._loader = self.get_loader()
        self._iterator = T.cast(
            T.Iterator[tuple[list[torch.Tensor], list[torch.Tensor], "BatchMeta"]],
            iter(self._loader),
        )
        self._epoch = 0

    @staticmethod
    def _faceqa_sampler_config() -> FaceQASamplerConfig:
        """Build FaceQA sampler config from trainer options."""
        return FaceQASamplerConfig(
            mode=T.cast(SamplerMode, trn_cfg.Automation.training_sampler()),
            strength=_auto_strength(trn_cfg.Automation.faceqa_sampler_strength()),
            dimensions=normalize_dimensions(trn_cfg.Automation.faceqa_sampler_dimensions()),
            downweight_duplicates=trn_cfg.Automation.faceqa_downweight_duplicates(),
            downweight_outliers=trn_cfg.Automation.faceqa_downweight_outliers(),
            min_quality=T.cast(T.Any, trn_cfg.Automation.faceqa_min_quality()),
        )

    def __iter__(self) -> T.Self:  # type: ignore[name-defined]
        """This is an iterator"""
        return self

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {
            f"{k}"[1:]: v
            for k, v in self.__dict__.items()
            if k in ("_input_size", "_output_sizes", "_color_order", "_config", "_sampler")
        }
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    @property
    def faceqa_sampler_summaries(self) -> list[FaceQASamplerSummary]:
        """Current FaceQA sampler summaries, one per side where available."""
        return self._faceqa_sampler_summaries

    def _active_faceqa_sampler_config(self) -> FaceQASamplerConfig:
        """Return the sampler config with the current phase multiplier applied."""
        if self._faceqa_sampler.mode != "faceqa_curriculum":
            return self._faceqa_sampler
        return self._faceqa_sampler.with_strength_multiplier(self._faceqa_sampler_multiplier)

    def _faceqa_include_metadata(self) -> bool:
        """Return whether TrainSet should expose FaceQA metadata."""
        return self._faceqa_training_diagnostics or self._faceqa_sampler.mode != "random"

    def _faceqa_index_for_side(self, side: str, index: int) -> FaceQAMetadataIndex | None:
        """Return optional FaceQA fallback metadata index for a side."""
        if not self._faceqa_include_metadata():
            return None
        return FaceQAMetadataIndex.from_path(
            side,
            self._faceqa_metadata_paths[index]
            if index < len(self._faceqa_metadata_paths)
            else None,
        )

    def _sample_weights(
        self, data_sets: tuple[TrainSet, ...]
    ) -> tuple[np.ndarray | None, ...] | None:
        """Return side-specific sample weights, or ``None`` for exact random fallback."""
        config = self._active_faceqa_sampler_config()
        if not config.active:
            self._faceqa_sampler_summaries = []
            return None

        weights: list[np.ndarray | None] = []
        summaries: list[FaceQASamplerSummary] = []
        for data_set in data_sets:
            samples = data_set.faceqa_metadata_for_sampling()
            side_weights, summary = compute_faceqa_sample_weights(
                data_set.side,
                samples,
                config,
                self._faceqa_sampler_bucket_losses,
            )
            weights.append(side_weights)
            summaries.append(summary)

        self._faceqa_sampler_summaries = summaries
        for summary in summaries:
            logger.info(
                "FaceQA sampler side %s: metadata=%s/%s effective_samples=%.2f top_up=%s "
                "top_down=%s",
                summary.side,
                summary.metadata_count,
                summary.total_count,
                summary.effective_sample_count,
                summary.top_upweighted,
                summary.top_downweighted,
            )
        return None if not any(weight is not None for weight in weights) else tuple(weights)

    def set_faceqa_sampler_strength_multiplier(self, multiplier: float) -> None:
        """Update curriculum sampler strength from the training phase scheduler."""
        if self._faceqa_sampler.mode != "faceqa_curriculum":
            return
        multiplier = max(0.0, float(multiplier))
        if multiplier == self._faceqa_sampler_multiplier:
            return
        self._faceqa_sampler_multiplier = multiplier
        train_set = T.cast(MultiDataset, self._loader.dataset)
        train_set.set_sample_weights(
            self._sample_weights(T.cast(tuple[TrainSet, ...], train_set.datasets))
        )

    @staticmethod
    def _bucket_loss_logs(logs: T.Mapping[str, T.Any]) -> dict[tuple[str, str, str], float]:
        """Extract FaceQA bucket EMA losses from diagnostic log keys."""
        prefix = "bucket/"
        suffix = "/ema"
        retval: dict[tuple[str, str, str], float] = {}
        for key, value in logs.items():
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            parts = key[len(prefix) : -len(suffix)].split("/")
            if len(parts) != 3:
                continue
            side, dimension, bucket = parts
            if dimension in {"duplicate", "identity_outlier"}:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score > 0.0:
                retval[(side, dimension, bucket)] = score
        return retval

    def update_faceqa_sampler_loss_logs(self, logs: T.Mapping[str, T.Any]) -> None:
        """Update curriculum weights from FaceQA per-bucket loss diagnostics."""
        if self._faceqa_sampler.mode != "faceqa_curriculum":
            return
        bucket_losses = self._bucket_loss_logs(logs)
        if bucket_losses == self._faceqa_sampler_bucket_losses:
            return
        self._faceqa_sampler_bucket_losses = bucket_losses
        train_set = T.cast(MultiDataset, self._loader.dataset)
        train_set.set_sample_weights(
            self._sample_weights(T.cast(tuple[TrainSet, ...], train_set.datasets))
        )

    def get_loader(self) -> DataLoader:
        """Obtain the dataloaders for each input/output for the model

        Returns
        -------
        The Training data loaders in side order
        """
        num_workers = trn_cfg.Loader.num_processes()
        max_proc = os.cpu_count()
        max_proc = 1 if max_proc is None else max_proc
        if num_workers > max_proc:
            logger.warning(
                "Data Loader processes set to %s but only %s processors available. Lowering to %s",
                num_workers,
                max_proc,
                max_proc - 1,
            )
            num_workers = max_proc - 1

        include_faceqa = self._faceqa_include_metadata()
        data_sets = tuple(
            TrainSet(
                get_label(i, len(self._config.folders)),
                f,
                self._process_size,
                include_faceqa=include_faceqa,
                faceqa_index=self._faceqa_index_for_side(
                    get_label(i, len(self._config.folders)), i
                ),
            )
            for i, f in enumerate(self._config.folders)
        )
        train_set = MultiDataset(
            data_sets, is_random=True, sample_weights=self._sample_weights(data_sets)
        )
        collate_fn = Collate(
            self._input_size,
            self._output_sizes,
            self._color_order,
            self._config,
            landmarks=self._landmarks,
        )
        retval = DataLoader(
            dataset=train_set,
            batch_size=self._config.batch_size,
            sampler=self._sampler(train_set),
            num_workers=num_workers,
            prefetch_factor=trn_cfg.Loader.pre_fetch(),
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        logger.debug("[TrainLoader] Set loader: %s", retval)
        return retval

    def __next__(self) -> tuple[list[torch.Tensor], list[torch.Tensor], BatchMeta]:
        """Obtain the next outputs from the loader

        Returns
        -------
        inputs
            list of len (num_inputs) tensors of shape(batch_size, H, W, C) inputs for the model
        targets
            List of len (num_outputs) of target images in shape (batch_size, num_inputs, height,
            width, 3) at all model output sizes as float32 0.0 - 1.0 range
        meta
            The meta information for the batch
        """
        try:
            inputs, targets, meta = T.cast(  # type: ignore[redundant-cast]
                tuple[list[torch.Tensor], list[torch.Tensor], "BatchMeta"],
                next(self._iterator),
            )
        except StopIteration:
            epoch = self._epoch
            logger.debug("[TrainLoader] epoch %s end", epoch)

            if isinstance(self._loader.sampler, tch_data.DistributedSampler):
                self._loader.sampler.set_epoch(epoch + 1)
            T.cast(MultiDataset, self._loader.dataset).shuffle()
            self._iterator = iter(self._loader)
            inputs, targets, meta = next(self._iterator)
            self._epoch += 1

        if self._learn_mask:  # Add the face mask as it's own target
            assert meta.mask_face is not None
            targets += [meta.mask_face[-1].permute(0, 1, 3, 4, 2)]
        logger.trace(  # type:ignore[attr-defined]
            "[TrainLoader] input_shapes: %s, target_shapes: %s, meta: %s",
            [i.shape for i in inputs],
            [t.shape for t in targets],
            meta,
        )
        return inputs, targets, meta


class PreviewLoader:
    """Generator for feeding faceswap models input data for generating preview images. Gets the
    next items from each of the configured loaders and collates them for feeding into a model

    Parameters
    ----------
    input_size
        The input size to the model
    output_sizes
        The output sizes to the model (list as some models have multi-scale outputs)
    color_order
        The color order of the model
    input_folders
        list of folders to read images from for each side being trained
    batch_size
        The number of images being displayed in the preview
    sampler
        The sampler to use for the data loaders. Default: ``None`` (RandomSampler)
    num_samples
        Set to 0 for random previews from the image folder. Set to a positive integer for this
        number of images to use for a static timelapse. Default: 0
    include_region_masks
        ``True`` to include diagnostic-only eye and mouth region masks in the target. Default:
        ``False``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        color_order: T.Literal["bgr", "rgb"],
        input_folders: list[str],
        batch_size: int,
        sampler: None | type[tch_data.RandomSampler | tch_data.SequentialSampler] = None,
        num_samples: int = 0,
        include_region_masks: bool = False,
    ) -> None:
        self._output_size = output_size
        self._input_folders = input_folders
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._include_region_masks = include_region_masks

        self._input_size = input_size
        self._color_order: T.Literal["bgr", "rgb"] = T.cast(
            T.Literal["bgr", "rgb"], color_order.lower()
        )
        self._sampler = tch_data.RandomSampler if sampler is None else sampler
        self._loader = self.get_loader()
        self._iterator = T.cast(T.Iterator[tuple[torch.Tensor, torch.Tensor]], iter(self._loader))

    def __iter__(self) -> T.Self:  # type: ignore[name-defined]
        """This is an iterator"""
        return self

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(
            f"{k[1:]}={repr(v)}"
            for k, v in self.__dict__.items()
            if k
            in (
                "_input_size",
                "_output_size",
                "_color_order",
                "_input_folders",
                "_batch_size",
                "_sampler",
                "_num_samples",
                "_include_region_masks",
            )
        )
        return f"{self.__class__.__name__}({params})"

    def get_loader(self) -> DataLoader:
        """Obtain the dataloaders for each input/output for the model

        Returns
        -------
        The Training data loaders in side order
        """
        data_sets = tuple(
            PreviewSet(
                get_label(i, len(self._input_folders)),
                f,
                self._input_size,
                self._output_size,
                self._color_order,
                num_images=self._num_samples,
                include_region_masks=self._include_region_masks,
            )
            for i, f in enumerate(self._input_folders)
        )
        preview_set = MultiDataset(data_sets, is_random=self._num_samples == 0)
        retval = DataLoader(
            dataset=preview_set,
            batch_size=self._batch_size,
            sampler=self._sampler(preview_set),
            num_workers=1,  # Previews don't need speed
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        logger.debug("[PreviewLoader] Set loader : %s", retval)
        return retval

    def _items_from_loader(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtain the next outputs from the given loader index

        Returns
        -------
        feed
            The batch of feed images for a side
        targets
            A batch of full sized, full coverage input images with mask in the 4th channel.
            If diagnostic region masks are enabled, eye and mouth masks are in channels 4 and 5.
        """
        try:
            inputs, targets = T.cast(tuple[torch.Tensor, torch.Tensor], next(self._iterator))  # type: ignore[redundant-cast]

        except StopIteration:
            logger.debug("[PreviewLoader] end")
            self._iterator = iter(self._loader)
            inputs, targets = next(self._iterator)

        logger.trace(  # type:ignore[attr-defined]
            "[PreviewLoader] input_shapes: %s, target_shape: %s",
            inputs.shape,
            targets.shape,
        )
        return inputs, targets

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Obtain the next batch of data for each side for feeding the model

        Returns
        -------
        inputs
            The inputs to the model for each side of the model. The array is returned in `(side,
            batch_size, *dims)` where `side` 0 is "A" and `side` 1 is "B" etc.
        targets
            The full sized source image with mask in 4th channel for each side of the model in
            format `(side, batch_size, *dims, 4|6) where `side` 0 is "A" and `side` 1 is "B" etc.
        """
        items = self._items_from_loader()
        inputs = items[0].swapaxes(0, 1)
        targets = items[1].swapaxes(0, 1)
        logger.debug(
            "[PreviewLoader] inputs: %s, targets: %s",
            inputs.shape,
            targets.shape,
        )
        return inputs, targets


get_module_objects(__name__)
