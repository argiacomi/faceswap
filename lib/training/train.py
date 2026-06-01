#! /usr/env/bin/python3
"""Run the training loop for a training plugin"""

from __future__ import annotations

import logging
import os
import time
import typing as T
import warnings
from copy import deepcopy

import cv2
import numpy as np
import torch

from lib.logger import format_array, parse_class_init
from lib.torch_utils import (
    accelerator_empty_cache,
    accelerator_max_memory_allocated,
    accelerator_max_memory_reserved,
    accelerator_reset_peak_memory_stats,
    accelerator_synchronize,
    get_device,
    is_accelerator_oom_error,
)
from lib.training.batch_size_finder import BatchSizeProbe, TrainingBatchSizeFinder
from lib.training.data import PreviewLoader, TrainLoader, get_label, get_sorted_images
from lib.training.faceqa_diagnostics import FaceQALossDiagnostics
from lib.training.preview import Samples
from lib.training.preview_diagnostics import PreviewDiagnostics
from lib.training.tensorboard import TorchTensorBoard
from lib.utils import FaceswapError, get_module_objects
from plugins.train import train_config as mod_cfg
from plugins.train.trainer import trainer_config as trn_cfg

from .loss import LossCollator
from .optimizer import Optimizer

if T.TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from plugins.train.trainer.base import TrainerBase

    from .loss import BatchLoss

logger = logging.getLogger(__name__)


# Suppress non-Faceswap related Keras warning about backend padding mismatches
warnings.filterwarnings(
    "ignore", message="You might experience inconsistencies", category=UserWarning
)


class Trainer:  # pylint:disable=too-many-instance-attributes
    """Handles the feeding of training images to Faceswap models, the generation of Tensorboard
    logs and the creation of sample/time-lapse preview images.

    All Trainer plugins must inherit from this class.

    Parameters
    ----------
    plugin
        The plugin that will be processing each batch
    preview
        ``True`` to generate previews
    warmup_steps
        The number of steps to warmup the learning rate for. Default: 0
    timelapse_folders
        The input folders to create timelapse images from. Default: ``None`` (no timelapse)
    timelapse_output
        The folder to output timelapse images. Default: "" (no timelapse)
    """

    def __init__(
        self,
        plugin: TrainerBase,
        preview: bool,
        warmup_steps: int = 0,
        timelapse_folders: list[str] | None = None,
        timelapse_output: str = "",
    ) -> None:
        logger.debug(parse_class_init(locals()))
        self._plugin = plugin
        self._preview = preview
        self._timelapse_folders = [] if timelapse_folders is None else timelapse_folders
        self._timelapse_output = timelapse_output

        self._device = get_device()
        self._model = plugin.model
        self._out_size = max(x[1] for x in self._model.output_shapes if x[-1] != 1)
        self._configure_model(plugin)
        self._optimizer = Optimizer(
            self._model,
            mod_cfg.Optimizer,
            mixed_precision=mod_cfg.mixed_precision(),
            warmup_steps=warmup_steps,
        )
        self._optimizer.to(self._device)

        self._train_loader = self._get_train_loader()

        self._exit_early = self._handle_lr_finder()
        if self._exit_early:
            logger.debug("[Trainer] Exiting from LR Finder")
            return

        self._preview_loader = self._get_preview_loader()
        self._timelapse_loader = self._get_timelapse_loader()

        self._model.state.add_session_batchsize(plugin.batch_size)
        self._tensorboard = self._set_tensorboard()
        self._faceqa_diagnostics = self._set_faceqa_diagnostics()
        self._faceqa_diagnostic_logs: dict[str, float] = {}
        self._preview_diagnostics = self._set_preview_diagnostics()
        self._samples = Samples(
            self._model.coverage_ratio,
            mod_cfg.Loss.learn_mask() or mod_cfg.Loss.penalized_mask_loss(),
            trn_cfg.Augmentation.mask_opacity(),
            trn_cfg.Augmentation.mask_color(),
        )

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = ", ".join(
            f"{k[1:]}={repr(v)}"
            for k, v in self.__dict__.items()
            if k in ("_plugin", "_preview", "_timelapse_folders", "_timelapse_output")
        )
        return f"{self.__class__.__name__}({params})"

    @property
    def exit_early(self) -> bool:
        """``True`` if the trainer should exit early, without performing any training steps"""
        return self._exit_early

    @property
    def batch_size(self) -> int:
        """The currently configured training batch size."""
        return int(self._plugin.config.batch_size)

    def _configure_model(self, plugin: TrainerBase):
        """Add the loss functions to the model and move to the correct device

        Parameters
        ----------
        plugin
            The plugin that is training the model
        """
        loss = LossCollator(
            functions=[
                mod_cfg.Loss.loss_function(),
                mod_cfg.Loss.loss_function_2(),
                mod_cfg.Loss.loss_function_3(),
                mod_cfg.Loss.loss_function_4(),
            ],
            weights=[
                1.0,
                mod_cfg.Loss.loss_weight_2() / 100.0,
                mod_cfg.Loss.loss_weight_3() / 100.0,
                mod_cfg.Loss.loss_weight_4() / 100.0,
            ],
            color_order=self._model.color_order,
            use_mask=mod_cfg.Loss.penalized_mask_loss(),
            eye_multiplier=mod_cfg.Loss.eye_multiplier(),
            mouth_multiplier=mod_cfg.Loss.mouth_multiplier(),
            smallest_output=min(x[1] for x in self._model.output_shapes if x[-1] != 1),
            mask_loss=(
                None if not mod_cfg.Loss.learn_mask() else mod_cfg.Loss.mask_loss_function()
            ),
        )
        plugin.register_loss(loss)
        plugin.model.model.to(self._device)

    def _faceqa_training_diagnostics_enabled(self) -> bool:
        """Return whether FaceQA training diagnostics should run for this session."""
        return (
            trn_cfg.Augmentation.faceqa_training_diagnostics()
            and not self._model.state.current_session["no_logs"]
        )

    def _faceqa_training_metadata_paths(self, num_sides: int) -> list[str | None]:
        """Return optional FaceQA metadata fallback paths by training side."""
        configured = [
            trn_cfg.Augmentation.faceqa_training_metadata_a(),
            trn_cfg.Augmentation.faceqa_training_metadata_b(),
        ]
        paths = [path.strip() or None for path in configured]
        if len(paths) < num_sides:
            paths.extend([None] * (num_sides - len(paths)))
        return paths[:num_sides]

    def _get_train_loader(self) -> TrainLoader:
        """Get the loaders for training the model

        Returns
        -------
        The loaders for feeding the model's training loop
        """
        input_sizes = [x[1] for x in self._model.input_shapes]
        assert len(set(input_sizes)) == 1, f"Multiple input sizes not supported. Got {input_sizes}"

        out_sizes = [x[1] for x in self._model.output_shapes if x[-1] != 1]
        num_sides = len(self._plugin.config.folders)
        assert len(out_sizes) % num_sides == 0, (
            f"Output count ({len(out_sizes)}) doesn't match number of inputs ({num_sides})"
        )
        split = len(out_sizes) // num_sides
        split_sizes = [out_sizes[x : x + split] for x in range(0, len(out_sizes), split)]
        assert len(set(out_sizes)) == len(set(split_sizes[0])), "Sizes for each output must match"

        retval = TrainLoader(
            input_sizes[0],
            tuple(split_sizes[0]),
            self._model.color_order,
            self._plugin.config,
            self._plugin.sampler,
            include_faceqa_diagnostics=self._faceqa_training_diagnostics_enabled(),
            faceqa_metadata_paths=self._faceqa_training_metadata_paths(num_sides),
        )
        logger.debug("[Trainer] data loader: %s", retval)
        return retval

    def _set_training_batch_size(self, batch_size: int) -> None:
        """Update the active training batch size and rebuild the train loader."""
        logger.debug(
            "[Trainer] Updating training batch size from %s to %s",
            self._plugin.config.batch_size,
            batch_size,
        )
        self._plugin.batch_size = batch_size
        self._plugin.config.batch_size = batch_size
        self._train_loader = self._get_train_loader()

    def _get_preview_loader(self) -> PreviewLoader | None:
        """Get the loader for generating previews whilst training the model

        Returns
        -------
        The loader for generating preview images during training or ``None`` if previews are
        disabled
        """
        if not self._preview:
            return None
        input_size = self._model.input_shapes[0][1]
        retval = PreviewLoader(
            input_size,
            self._out_size,
            self._model.color_order,
            self._plugin.config.folders,
            trn_cfg.Augmentation.preview_images(),
            torch.utils.data.RandomSampler,
            include_region_masks=trn_cfg.Augmentation.preview_diagnostics(),
        )
        logger.debug("[Trainer] Preview data loader: %s", retval)
        return retval

    def _get_timelapse_loader(self) -> PreviewLoader | None:
        """Get the loader for generating timelapse images whilst training the model

        Returns
        -------
        The loaders for timelapse preview images during training or ``None`` if previews are
        disabled
        """
        if not self._timelapse_folders or not self._timelapse_output:
            return None
        num_images = trn_cfg.Augmentation.preview_images()
        avail_images = min(
            len(
                [
                    fname
                    for fname in os.listdir(folder)
                    if os.path.splitext(fname)[-1].lower() == ".png"
                ]
            )
            for folder in self._timelapse_folders
        )
        num_samples = min(num_images, avail_images)
        logger.debug(
            "[Train] preview count: %s, available_images: %s, timelapse count: %s",
            num_images,
            avail_images,
            num_samples,
        )
        input_size = self._model.input_shapes[0][1]
        retval = PreviewLoader(
            input_size,
            self._out_size,
            self._model.color_order,
            self._timelapse_folders,
            trn_cfg.Augmentation.preview_images(),
            torch.utils.data.SequentialSampler,
            num_samples=num_samples,
        )
        logger.debug("[Trainer] Preview data loader: %s", retval)
        return retval

    def _handle_lr_finder(self) -> bool:
        """Handle the learning rate finder.

        If this is a new model, then find the optimal learning rate and return ``True`` if user has
        just requested the graph, otherwise return ``False`` to continue training

        If it as existing model, set the learning rate to the value found by the learning rate
        finder and return ``False`` to continue training

        Returns
        -------
        ``True`` if the learning rate finder options dictate that training should not continue
        after finding the optimal leaning rate
        """
        if not self._plugin.config.lr_finder:
            return False

        if self._model.state.lr_finder > -1:
            learning_rate = self._model.state.lr_finder
            logger.info(
                "Setting learning rate from Learning Rate Finder to %s", f"{learning_rate:.1e}"
            )
            self._optimizer.set_lr(learning_rate)
            self._model.state.update_session_config("learning_rate", learning_rate)
            return False

        if self._model.state.iterations == 0 and self._model.state.session_id == 1:
            success = self._optimizer.find_learning_rate(
                self,
                mod_cfg.lr_finder_iterations(),
                1e-10,
                1e-1,
                T.cast(
                    T.Literal["default", "aggressive", "extreme"], mod_cfg.lr_finder_strength()
                ),
                T.cast(
                    T.Literal["set", "graph_and_set", "graph_and_exit"], mod_cfg.lr_finder_mode()
                ),
            )
            return mod_cfg.lr_finder_mode() == "graph_and_exit" or not success

        logger.debug("[Trainer] No learning rate finder rate. Not setting")
        return False

    def _set_tensorboard(self) -> TorchTensorBoard | None:
        """Set up Tensorboard callback for logging loss.

        Bypassed if command line option "no-logs" has been selected.

        Returns
        -------
        Tensorboard object for the the current training session. ``None`` if Tensorboard logging is
        not selected
        """
        if self._model.state.current_session["no_logs"]:
            logger.verbose("TensorBoard logging disabled")  # type: ignore
            return None
        logger.debug("[Trainer] Enabling TensorBoard Logging")

        logger.debug("[Trainer] Setting up TensorBoard Logging")
        tensorboard = TorchTensorBoard(
            log_dir=self._get_session_log_dir(), write_graph=True, update_freq="batch"
        )
        tensorboard.set_model(self._model.model)
        logger.verbose("Enabled TensorBoard Logging")  # type: ignore
        return tensorboard

    def _get_session_log_dir(self) -> str:
        """Return the current model session log directory."""
        return os.path.join(
            str(self._model.io.model_dir),
            f"{self._model.name}_logs",
            f"session_{self._model.state.session_id}",
        )

    def _set_preview_diagnostics(self) -> PreviewDiagnostics | None:
        """Set up optional preview diagnostics metrics."""
        if not trn_cfg.Augmentation.preview_diagnostics():
            logger.debug("[Trainer] Preview diagnostics disabled")
            return None
        if self._preview_loader is None:
            logger.debug("[Trainer] Preview diagnostics requested but preview is disabled")
            return None

        jsonl_path = None
        if trn_cfg.Augmentation.preview_diagnostics_jsonl():
            jsonl_path = os.path.join(self._get_session_log_dir(), "preview_diagnostics.jsonl")

        retval = PreviewDiagnostics(
            ema_alpha=trn_cfg.Augmentation.preview_diagnostics_ema_alpha(),
            jsonl_path=jsonl_path,
        )
        logger.info("Enabled preview diagnostics")
        return retval

    def _set_faceqa_diagnostics(self) -> FaceQALossDiagnostics | None:
        """Set up optional FaceQA training loss diagnostics."""
        if not trn_cfg.Augmentation.faceqa_training_diagnostics():
            logger.debug("[Trainer] FaceQA training diagnostics disabled")
            return None
        if not self._faceqa_training_diagnostics_enabled():
            logger.debug("[Trainer] FaceQA training diagnostics disabled with no-logs")
            return None
        jsonl_path = None
        if trn_cfg.Augmentation.faceqa_training_diagnostics_jsonl():
            jsonl_path = os.path.join(
                self._get_session_log_dir(),
                "faceqa_training_diagnostics.jsonl",
            )
        logger.info("Enabled FaceQA training diagnostics")
        logger.debug("[Trainer] FaceQA training diagnostics path: %s", jsonl_path)
        return FaceQALossDiagnostics(jsonl_path=jsonl_path)

    def _snapshot_model_state(self) -> tuple[str, T.Any]:
        """Return a restorable snapshot of the current model weights."""
        model = self._model.model
        if hasattr(model, "state_dict"):
            return "state_dict", deepcopy(model.state_dict())
        if hasattr(model, "get_weights"):
            return "weights", deepcopy(model.get_weights())
        raise RuntimeError("Unable to snapshot model weights for batch-size finder")

    def _restore_model_state(self, snapshot: tuple[str, T.Any]) -> None:
        """Restore model weights from :meth:`_snapshot_model_state`."""
        mode, state = snapshot
        model = self._model.model
        if mode == "state_dict":
            model.load_state_dict(state)
            return
        model.set_weights(state)

    def _snapshot_optimizer_state(self) -> tuple[dict[str, T.Any], int, int]:
        """Return a restorable snapshot of optimizer state and counters."""
        return (
            deepcopy(self._optimizer.state_dict()),
            self._optimizer._accumulation_count,  # pylint:disable=protected-access
            self._optimizer._session_steps,  # pylint:disable=protected-access
        )

    def _restore_optimizer_state(self, snapshot: tuple[dict[str, T.Any], int, int]) -> None:
        """Restore optimizer state and counters."""
        state, accumulation_count, session_steps = snapshot
        self._optimizer.load_state_dict(state)
        self._optimizer._accumulation_count = accumulation_count  # pylint:disable=protected-access
        self._optimizer._session_steps = session_steps  # pylint:disable=protected-access

    def probe_training_batch_size(self, batch_size: int) -> BatchSizeProbe:
        """Probe one batch size through a real training step without keeping mutations."""
        self._set_training_batch_size(batch_size)
        model_state = self._snapshot_model_state()
        optimizer_state = self._snapshot_optimizer_state()
        vram_allocated = 0
        vram_reserved = 0
        try:
            accelerator_empty_cache()
            accelerator_reset_peak_memory_stats()
            for _ in range(max(1, mod_cfg.Optimizer.gradient_accumulation())):
                inputs, targets, meta = next(self._train_loader)
                self._plugin.train_batch(
                    [i.to(self._device) for i in inputs],
                    [t.to(self._device) for t in targets],
                    self._optimizer,
                    meta.to(self._device),
                )
            accelerator_synchronize()
            vram_allocated = accelerator_max_memory_allocated()
            vram_reserved = accelerator_max_memory_reserved()
            return BatchSizeProbe(
                batch_size=batch_size,
                success=True,
                vram_allocated=vram_allocated,
                vram_reserved=vram_reserved,
            )
        except RuntimeError as err:
            vram_allocated = accelerator_max_memory_allocated()
            vram_reserved = accelerator_max_memory_reserved()
            if not is_accelerator_oom_error(err):
                raise
            logger.debug(
                "[Trainer] Batch-size probe OOM for batch size %s: %s",
                batch_size,
                err,
            )
            return BatchSizeProbe(
                batch_size=batch_size,
                success=False,
                vram_allocated=vram_allocated,
                vram_reserved=vram_reserved,
                error=str(err),
            )
        finally:
            self._restore_model_state(model_state)
            self._restore_optimizer_state(optimizer_state)
            accelerator_empty_cache()

    def find_batch_size(
        self,
        max_batch_size: int,
        target_effective_batch_size: int,
        auto_apply: bool = False,
    ) -> None:
        """Find and optionally apply a safe training batch-size recommendation."""
        original_batch_size = self.batch_size
        max_available = min(
            max_batch_size,
            *(len(get_sorted_images(folder)) for folder in self._plugin.config.folders),
        )
        finder = TrainingBatchSizeFinder(
            self,
            max_batch_size=max_available,
            target_effective_batch_size=target_effective_batch_size,
        )
        recommendation = finder.find()
        self._model.state.add_training_batch_size_finder(recommendation.to_state())
        self._model.state.save()

        if auto_apply and recommendation.suggested_batch_size > 0:
            self._set_training_batch_size(recommendation.suggested_batch_size)
            self._model.state.add_session_batchsize(recommendation.suggested_batch_size)
            logger.info(
                "Applied recommended training batch size: %s",
                recommendation.suggested_batch_size,
            )
            return

        self._set_training_batch_size(original_batch_size)

    def toggle_mask(self) -> None:
        """Toggle the mask overlay on or off based on user input."""
        self._samples.toggle_mask_display()

    def train_one_batch(self) -> list[BatchLoss]:
        """Process a single batch through the model and obtain the loss

        Returns
        -------
        The collated loss values detached and moved to CPU in order (A, B, ...)
        """
        try:
            inputs, targets, meta = next(self._train_loader)
            loss = self._plugin.train_batch(
                [i.to(self._device) for i in inputs],
                [t.to(self._device) for t in targets],
                self._optimizer,
                meta.to(self._device),
            )
            retval = [x.to_cpu() for x in loss]
            self._faceqa_diagnostic_logs = (
                {}
                if self._faceqa_diagnostics is None
                else self._faceqa_diagnostics.update(
                    retval,
                    meta.faceqa,
                    self._model.iterations,
                )
            )
        except RuntimeError as err:
            if not is_accelerator_oom_error(err):
                raise
            msg = (
                "You do not have enough GPU memory available to train the selected model at "
                "the selected settings. You can try a number of things:"
                "\n1) Close any other application that is using your GPU (web browsers are "
                "particularly bad for this)."
                "\n2) Lower the batchsize (the amount of images fed into the model each "
                "iteration)."
                "\n3) Try enabling 'Mixed Precision' training."
                "\n4) Use a more lightweight model, or select the model's 'LowMem' option "
                "(in config) if it has one."
            )
            raise FaceswapError(msg) from err
        return retval

    def _log_tensorboard(self, loss: list[BatchLoss]) -> None:
        """Log current loss to Tensorboard log files

        Parameters
        ----------
        loss
            The loss scalars for the batch detached and moved to cpu in order (A, B, ...)
        """
        if not self._tensorboard:
            return
        logger.trace("[Trainer] Updating TensorBoard log: %s", loss)  # type: ignore
        logs: dict[str, float | dict[str, float]] = {
            "total": T.cast(torch.Tensor, sum(x.total for x in loss)).item()
        }
        for i, out in enumerate(loss):
            lbl = get_label(i, len(loss))
            for idx, (w, u) in enumerate(zip(out.weighted, out.unweighted, strict=False)):
                key = lbl if len(out.unweighted) == 1 else f"{lbl}_{idx}"
                weighted = {k: v.mean() for k, v in w.items()}
                unweighted = {k: v.mean() for k, v in u.items()}
                logs[f"face_{key}"] = T.cast(torch.Tensor, sum(weighted.values())).item()
                logs[f"weighted_{key}"] = {k: v.item() for k, v in weighted.items()}
                logs[f"unweighted_{key}"] = {k: v.item() for k, v in unweighted.items()}
            if out.mask is not None:
                logs[f"mask_{lbl}"] = out.mask.mean().item()
        if self._faceqa_diagnostic_logs:
            logs["faceqa_diagnostics"] = self._faceqa_diagnostic_logs
        self._tensorboard.on_train_batch_end(self._model.iterations, logs=logs)

    def _collate_and_store_loss(self, loss: list[BatchLoss]) -> np.ndarray:
        """Collate the loss into totals for each side.

        The losses are summed into a total for each side. Loss totals are added to
        :attr:`model.state._history` to track the loss drop per save iteration for backup purposes.

        If NaN protection is enabled, Checks for NaNs and raises an error if detected.

        Parameters
        ----------
        loss
            The list of loss scalars in order (A, B, ...)

        Returns
        -------
        2 ``floats`` which is the total loss for each side (eg sum of face + mask loss)

        Raises
        ------
        FaceswapError
            If a NaN is detected, a :class:`FaceswapError` will be raised
        """
        # NaN protection
        if mod_cfg.nan_protection() and not all(torch.isfinite(val.total).all() for val in loss):
            loss_str = ", ".join(
                f"Loss {get_label(i, len(loss))}: {round(x.total.item(), 6)}"
                for i, x in enumerate(loss)
            )
            msg = f"NaN Detected. {loss_str}"
            failed = ", ".join(
                f"{key}({get_label(i, len(loss))})"
                for i, out in enumerate(loss)
                for unweighted in out.unweighted
                for key, sub_loss in unweighted.items()
                if not torch.isfinite(sub_loss).all()
            )
            if failed:
                msg += f". The loss function(s) that NaN'd: {failed}"
            logger.critical(msg)
            raise FaceswapError(
                "A NaN was detected and you have NaN protection enabled. Training "
                "has been terminated."
            )

        combined_loss = np.array([x.total.item() for x in loss], dtype=np.float32)
        self._model.add_history(combined_loss)
        logger.trace(  # type: ignore[attr-defined]
            "[Trainer] original loss: %s, combined_loss: %s",
            loss,
            combined_loss,
        )
        return combined_loss  # type: ignore[no-any-return]

    def _print_loss(self, loss: np.ndarray) -> None:
        """Outputs the loss for the current iteration to the console.

        Parameters
        ----------
        The loss for each side. List should contain 2 ``floats`` side "a" in position 0 and side
        "b" in position 1.
        """
        output = ", ".join(
            [
                f"Loss {side}: {side_loss:.5f}"
                for side, side_loss in zip(("A", "B"), loss, strict=False)
            ]
        )
        timestamp = time.strftime("%H:%M:%S")
        output = f"[{timestamp}] [#{self._model.iterations:05d}] {output}"
        print(f"{output}", end="\r")

    def _get_predictions(self, feed: torch.Tensor) -> npt.NDArray[np.float32]:
        """Obtain preview predictions from the model, chunking feeds into the model's batch size

        Parameters
        ----------
        feed
            The input tensor to obtain predictions from the model in shape (num_sides, N, height,
            width, 3)

        Returns
        -------
        The predictions from the model for the given preview feed
        """
        batch_size = self._plugin.batch_size
        ndim = 4 if mod_cfg.Loss.learn_mask() else 3
        retval = np.empty(
            (feed.shape[0], feed.shape[1], self._out_size, self._out_size, ndim), dtype=np.float32
        )
        for idx in range(0, feed.shape[1], batch_size):
            feed_batch = feed[:, idx : idx + batch_size]
            feed_size = feed_batch.shape[1]
            is_padded = feed_size < batch_size

            if is_padded:
                holder = torch.empty(
                    (feed_batch.shape[0], batch_size, *feed_batch.shape[2:]), dtype=feed.dtype
                )
                logger.debug(
                    "[Trainer] Padding undersized batch of shape %s to %s",
                    feed_batch.shape,
                    holder.shape,
                )
                holder[:, :feed_size] = feed_batch
                feed_batch = holder
            with torch.inference_mode():
                out = [
                    x.cpu().numpy()
                    for x in self._model.model(list(feed_batch))
                    if x.shape[1] == self._out_size
                ]  # Filter multi-scale output
            if mod_cfg.Loss.learn_mask():  # Apply mask to alpha channel
                out = [np.concatenate(out[i : i + 2], axis=-1) for i in range(0, len(out), 2)]
            out_arr = np.stack(out, axis=0)
            if is_padded:
                out_arr = out_arr[:, :feed_size]
            retval[:, idx : idx + feed_size] = out_arr
        return retval

    def _update_viewers(
        self,  # pylint:disable=too-many-locals
        viewer: Callable[[np.ndarray, str], None] | None,
        do_timelapse: bool = False,
    ) -> None:
        """Update the preview viewer and timelapse output

        Parameters
        ----------
        viewer
            The function that will display the preview image
        do_timelapse
            ``True`` to generate a timelapse preview image
        """
        if (viewer is None or self._preview_loader is None) and not do_timelapse:
            return

        if do_timelapse:
            assert self._timelapse_loader is not None
            loader = self._timelapse_loader
        else:
            assert self._preview_loader is not None
            loader = self._preview_loader
        feed, target = next(loader)

        num_sides = feed.shape[0]
        ndim = 4 if mod_cfg.Loss.learn_mask() else 3
        predictions: npt.NDArray[np.float32] = np.empty(
            (num_sides, num_sides, target.shape[1], self._out_size, self._out_size, ndim),
            dtype=np.float32,
        )
        logger.debug(
            "[Trainer] feed: %s, target: %s, predictions_holder: %s",
            feed.shape,
            target.shape,
            predictions.shape,
        )
        for side_idx in range(num_sides):
            rolled_feed = torch.roll(feed, shifts=side_idx, dims=0)
            pred = self._get_predictions(rolled_feed)
            for input_idx in range(num_sides):
                original_idx = (input_idx - side_idx) % num_sides
                predictions[original_idx, side_idx] = pred[input_idx]

        targets = target.cpu().numpy()
        if self._model.color_order == "rgb":
            predictions[..., :3] = predictions[..., 2::-1]
            targets[..., :3] = targets[..., 2::-1]
        logger.debug(
            "[Trainer] Got preview images: predictions: %s, targets: %s",
            format_array(predictions),
            format_array(targets),
        )
        if self._preview_diagnostics is not None and not do_timelapse:
            logs = self._preview_diagnostics.update(predictions, targets, self._model.iterations)
            if self._tensorboard is not None:
                self._tensorboard.on_train_batch_end(
                    self._model.iterations, logs={"preview_diagnostics": logs}
                )

        display_targets = targets[..., :4]
        samples = self._samples.get_preview(predictions, display_targets)

        if do_timelapse:
            filename = os.path.join(self._timelapse_output, str(int(time.time())) + ".jpg")
            cv2.imwrite(filename, samples)
            logger.debug("[Trainer] Created time-lapse: '%s'", filename)
            return

        if viewer is not None:
            viewer(
                samples,
                "Training - 'S': Save Now. 'R': Refresh Preview. 'M': Toggle Mask. 'F': "
                "Toggle Screen Fit-Actual Size. 'ENTER': Save and Quit",
            )

    def train_one_step(
        self, viewer: Callable[[np.ndarray, str], None] | None, do_timelapse: bool = False
    ) -> None:
        """Running training on a batch of images for each side.

        Triggered from the training cycle in :class:`scripts.train.Train`.

        * Runs a training batch through the model.

        * Outputs the iteration's loss values to the console

        * Logs loss to Tensorboard, if logging is requested.

        * If a preview or time-lapse has been requested, then pushes sample images through the \
        model to generate the previews

        * Creates a snapshot if the total iterations trained so far meet the requested snapshot \
        criteria

        Notes
        -----
        As every iteration is called explicitly, the Parameters defined should always be ``None``
        except on save iterations.

        Parameters
        ----------
        viewer
            The function that will display the preview image
        do_timelapse
            ``True`` to generate a timelapse preview image
        """
        self._model.state.increment_iterations()
        logger.trace(  # type: ignore[attr-defined]
            "[Trainer] Training one step: (iteration: %s)",
            self._model.iterations,
        )
        do_snapshot = (
            self._plugin.config.snapshot_interval != 0
            and self._model.iterations - 1 >= self._plugin.config.snapshot_interval
            and (self._model.iterations - 1) % self._plugin.config.snapshot_interval == 0
        )
        loss = self.train_one_batch()
        self._log_tensorboard(loss)
        total_loss = self._collate_and_store_loss(loss)
        self._print_loss(total_loss)
        if do_snapshot:
            self._model.io.snapshot()
        self._update_viewers(viewer, do_timelapse)

    def _clear_tensorboard(self) -> None:
        """Stop Tensorboard logging.

        Tensorboard logging needs to be explicitly shutdown on training termination. Called from
        :class:`scripts.train.Train` when training is stopped.
        """
        if not self._tensorboard:
            return
        logger.debug("[Trainer] Ending Tensorboard Session: %s", self._tensorboard)
        self._tensorboard.on_train_end()

    def save(self, is_exit: bool = False) -> None:
        """Save the model

        Parameters
        ----------
        is_exit
            ``True`` if save has been called on model exit. Default: ``False``
        """
        self._model.io.save(self._optimizer, is_exit=is_exit)
        assert self._tensorboard is not None
        self._tensorboard.on_save()
        if is_exit:
            self._clear_tensorboard()


__all__ = get_module_objects(__name__)
