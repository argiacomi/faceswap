#!/usr/bin/env python3
"""SegNeXt face-parser model components vendored for the Faceswap mask plugin.

Architecture adapted from https://github.com/e4s2022/SegNeXt-FaceParser
(itself based on https://github.com/Visual-Attention-Network/SegNeXt). Both upstream repos
are Apache-2.0 licensed.

The pre-trained CelebAMask-HQ checkpoints used by Faceswap are validated by SHA256 by the
plugin loader: ``small`` comes from ``Warlord-K/SegNext-FaceParser`` on Hugging Face and
``base`` comes from ``AiArt-Gao/FaceParsing-SegNeXt``. The plugin provides a
self-contained PyTorch implementation of the MSCAN backbone and Hamburger (LightHam)
decoder so faceswap does not need an mmseg/mmcv dependency at inference time.
"""
