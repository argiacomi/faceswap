#!/usr/bin/env python3
"""Default configurations for trainers"""

from __future__ import annotations

import gettext
import inspect
import logging
import typing as T
from dataclasses import dataclass

from lib.config import ConfigItem, GlobalSection
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


# LOCALES
_LANG = gettext.translation(
    "plugins.train.trainer.train_config", localedir="locales", fallback=True
)
_ = _LANG.gettext


@dataclass
class Loader(GlobalSection):
    """trainer.loader section"""

    helptext = _("Data Loader Options.\nControls how training data is loaded from disk")

    num_processes = ConfigItem(
        datatype=int,
        default=4,
        group=_("data loading"),
        info=_(
            "Number of processors to use for loading and processing data from disk. 0 to just "
            "use the Main process."
        ),
        rounding=1,
        min_max=(0, 32),
    )

    pre_fetch = ConfigItem(
        datatype=int,
        default=2,
        group=_("data loading"),
        info=_(
            "The Number of items that each loader should pre-fetch and hold in RAM. Default is "
            "usually fine unless you have disk contention with variable read speeds."
        ),
        rounding=1,
        min_max=(1, 10),
    )


@dataclass
class Augmentation(GlobalSection):
    """trainer.augmentation section"""

    helptext = _(
        "Data Augmentation Options.\n"
        "WARNING: The defaults for augmentation will be fine for 99.9% of use cases. "
        "Only change them if you absolutely know what you are doing!"
    )

    preview_images = ConfigItem(
        datatype=int,
        default=14,
        group=_("evaluation"),
        info=_("Number of sample faces to display for each side in the preview when training."),
        rounding=2,
        min_max=(2, 16),
    )

    preview_diagnostics = ConfigItem(
        datatype=bool,
        default=False,
        group=_("evaluation"),
        info=_(
            "Enable lightweight numerical diagnostics for preview refreshes. When enabled, "
            "preview reconstruction and detail metrics are calculated after inference without "
            "changing training behavior."
        ),
        fixed=False,
    )

    preview_diagnostics_ema_alpha = ConfigItem(
        datatype=float,
        default=0.2,
        group=_("evaluation"),
        info=_("Exponential moving average alpha for preview diagnostics metrics."),
        rounding=2,
        min_max=(0.01, 1.0),
        fixed=False,
    )

    preview_diagnostics_jsonl = ConfigItem(
        datatype=bool,
        default=False,
        group=_("evaluation"),
        info=_(
            "Write preview diagnostics metrics as JSON lines in the model's TensorBoard log "
            "session folder."
        ),
        fixed=False,
    )

    faceqa_training_diagnostics = ConfigItem(
        datatype=bool,
        default=False,
        group=_("evaluation"),
        info=_(
            "Enable FaceQA training loss diagnostics. When enabled, per-sample losses are "
            "aggregated by FaceQA metadata buckets without changing training behavior."
        ),
        fixed=False,
    )

    faceqa_training_diagnostics_jsonl = ConfigItem(
        datatype=bool,
        default=False,
        group=_("evaluation"),
        info=_(
            "Write FaceQA training diagnostics as JSON lines in the model's TensorBoard log "
            "session folder."
        ),
        fixed=False,
    )

    faceqa_training_metadata_a = ConfigItem(
        datatype=str,
        default="",
        group=_("evaluation"),
        info=_(
            "Optional FaceQA-enriched alignments file for side A. When provided, training "
            "diagnostics can use FaceQA metadata even if the extracted PNG headers are stale."
        ),
        fixed=False,
    )

    faceqa_training_metadata_b = ConfigItem(
        datatype=str,
        default="",
        group=_("evaluation"),
        info=_(
            "Optional FaceQA-enriched alignments file for side B. When provided, training "
            "diagnostics can use FaceQA metadata even if the extracted PNG headers are stale."
        ),
        fixed=False,
    )

    mask_opacity = ConfigItem(
        datatype=int,
        default=30,
        group=_("evaluation"),
        info=_(
            "The opacity of the mask overlay in the training preview. Lower values are more "
            "transparent."
        ),
        rounding=2,
        min_max=(0, 100),
    )

    mask_color = ConfigItem(
        datatype=str,
        default="#ff0000",
        choices="colorchooser",
        group=_("evaluation"),
        info=_("The RGB hex color to use for the mask overlay in the training preview."),
    )

    zoom_amount = ConfigItem(
        datatype=int,
        default=5,
        group=_("image augmentation"),
        info=_("Percentage amount to randomly zoom each training image in and out."),
        rounding=1,
        min_max=(0, 25),
    )

    rotation_range = ConfigItem(
        datatype=int,
        default=10,
        group=_("image augmentation"),
        info=_("Percentage amount to randomly rotate each training image."),
        rounding=1,
        min_max=(0, 25),
    )

    shift_range = ConfigItem(
        datatype=int,
        default=5,
        group=_("image augmentation"),
        info=_(
            "Percentage amount to randomly shift each training image horizontally and vertically."
        ),
        rounding=1,
        min_max=(0, 25),
    )

    flip_chance = ConfigItem(
        datatype=int,
        default=50,
        group=_("image augmentation"),
        info=_(
            "Percentage chance to randomly flip each training image horizontally.\n"
            "NB: This is ignored if the 'no-flip' option is enabled"
        ),
        rounding=1,
        min_max=(0, 75),
    )

    color_lightness = ConfigItem(
        datatype=int,
        default=30,
        group=_("color augmentation"),
        info=_(
            "Percentage amount to randomly alter the lightness of each training image.\n"
            "NB: This is ignored if the 'no-augment-color' option is enabled"
        ),
        rounding=1,
        min_max=(0, 75),
    )

    color_ab = ConfigItem(
        datatype=int,
        default=8,
        group=_("color augmentation"),
        info=_(
            "Percentage amount to randomly alter the 'a' and 'b' colors of the L*a*b* color "
            "space of each training image.\nNB: This is ignored if the 'no-augment-color' "
            "option is enabled"
        ),
        rounding=1,
        min_max=(0, 50),
    )

    color_clahe_chance = ConfigItem(
        datatype=int,
        default=50,
        group=_("color augmentation"),
        info=_(
            "Percentage chance to perform Contrast Limited Adaptive Histogram Equalization on "
            "each training image.\nNB: This is ignored if the 'no-augment-color' option is "
            "enabled"
        ),
        rounding=1,
        min_max=(0, 75),
        fixed=False,
    )

    color_clahe_max_size = ConfigItem(
        datatype=int,
        default=4,
        group=_("color augmentation"),
        info=_(
            "The grid size dictates how much Contrast Limited Adaptive Histogram Equalization "
            "is performed on any training image selected for clahe. Contrast will be applied "
            "randomly with a grid-size of 0 up to the maximum. This value is a multiplier "
            "calculated from the training image size.\nNB: This is ignored if the "
            "'no-augment-color' option is enabled"
        ),
        rounding=1,
        min_max=(1, 8),
    )


@dataclass
class Automation(GlobalSection):
    """trainer.automation section"""

    helptext = _(
        "Training Automation Options.\n"
        "A set-and-forget phase scheduler that gradually ramps optional detail, perceptual, "
        "boundary and identity losses as training progresses through broad-reconstruction, "
        "detail-refinement and final-refinement phases.\n"
        "The schedule is deterministic from the current iteration only: restarting at the same "
        "iteration reproduces the same phase and multipliers. The 'off' default preserves "
        "existing behavior exactly."
    )

    training_automation = ConfigItem(
        datatype=str,
        default="off",
        choices=["off", "conservative", "balanced", "aggressive"],
        gui_radio=True,
        fixed=False,
        group=_("phase scheduling"),
        info=_(
            "Automatically ramp optional loss components through deterministic training phases. "
            "Configured loss weights are treated as the requested maximums; the scheduler only "
            "scales them down during earlier phases and never above the configured value."
            "\n\toff - Disable automation. Effective loss weights are exactly as configured."
            "\n\tconservative - Late, gentle ramps with capped detail and identity strength."
            "\n\tbalanced - Moderate ramp timing and strength. Recommended starting point."
            "\n\taggressive - Early, full-strength ramps."
        ),
    )

    training_automation_dry_run = ConfigItem(
        datatype=bool,
        default=False,
        fixed=False,
        group=_("phase scheduling"),
        info=_(
            "Preview mode. Log the scheduled phase and effective multipliers each iteration "
            "without applying them to training, so you can see what the selected automation mode "
            "would do. Ignored when 'training_automation' is 'off'."
        ),
    )


def get_defaults() -> dict[str, GlobalSection]:
    """Obtain the default values for adding to the config.ini file

    Returns
    -------
    defaults
        The option names and config items
    """
    defaults = {
        k: T.cast(GlobalSection, v)
        for k, v in globals().items()
        if inspect.isclass(v) and issubclass(v, GlobalSection) and v != GlobalSection
    }
    logger.debug("Training config. options: %s", defaults)
    return defaults


__all__ = get_module_objects(__name__)
