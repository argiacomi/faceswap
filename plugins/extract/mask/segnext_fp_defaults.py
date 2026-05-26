#!/usr/bin/env python3
"""The default options for the faceswap SegNeXt Face Parsing plugin."""

# pylint:disable=duplicate-code
from lib.config import ConfigItem

HELPTEXT = (
    "SegNeXt Face Parsing options.\n"
    "Mask ported from https://github.com/e4s2022/SegNeXt-FaceParser using the "
    "CelebAMask-HQ pre-trained checkpoints (Apache 2.0). Faceswap pins the validated "
    "19-class small checkpoint from Warlord-K and the validated MSCAN-B base checkpoint "
    "from AiArt-Gao. Produces the same per-component label space as BiSeNet-FP and is a "
    "drop-in alternative when a cleaner edge or a better hair/skin boundary is required."
)


model = ConfigItem(
    datatype=str,
    default="base",
    group="settings",
    info="The SegNeXt MSCAN backbone variant.\n"
    "\n\tsmall - MSCAN-S, 14M params, 16G FLOPs, 78.19 mIoU on CelebAMask-HQ. "
    "Fastest recommended option."
    "\n\tbase - MSCAN-B, 28M params, 35G FLOPs, 78.97 mIoU on CelebAMask-HQ. "
    "Balanced quality/runtime option."
    "\n\tlarge - MSCAN-L, 49M params, 70G FLOPs, 79.34 mIoU on CelebAMask-HQ. "
    "Best reported e4s quality at the highest VRAM and runtime cost.",
    choices=["small", "base", "large"],
    gui_radio=True,
)

batch_size = ConfigItem(
    datatype=int,
    default=4,
    group="settings",
    info="The batch size to use. SegNeXt-FP runs an MSCAN encoder followed by a "
    "Hamburger NMF decoder, so this is more VRAM-hungry than BiSeNet-FP - reduce if "
    "you encounter out-of-memory errors.",
    rounding=1,
    min_max=(1, 64),
)

cpu = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Run inference on the CPU instead of the accelerated device. SegNeXt-FP is "
    "considerably slower on CPU than BiSeNet-FP because of the NMF iterations in "
    "the decoder; only enable this to save VRAM at significant speed cost.",
)

include_ears = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Whether to include ears within the face mask.",
)

include_hair = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Whether to include hair within the face mask. When enabled the mask is "
    "stored using head centering, otherwise face centering is used.",
)

include_mouth = ConfigItem(
    datatype=bool,
    default=True,
    group="settings",
    info="Include the inner mouth region in the generated mask. Disable this to preserve "
    "the destination mouth interior, teeth, and tongue during conversion while still "
    "masking the lips/face.",
)

include_glasses = ConfigItem(
    datatype=bool,
    default=False,
    group="settings",
    info="Whether to include glasses (frames + lenses) within the face mask. Keep "
    "disabled for pure-face masks: glasses are exclusion regions, not face pixels.",
)
