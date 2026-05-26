#!/usr/bin/env python3
"""Shared output contract for all mask plugins.

All mask plugins must return a :class:`MaskPluginOutput` from ``post_process``.
``MaskPluginOutput`` is a numpy ndarray subclass whose underlying data is the
``(N, H, W)`` float32 binary mask, so existing consumers that treat the return
value as a plain array require no changes.

Schema version history
----------------------
1 (current)  binary_mask + optional per_class_probs + source_id + version
"""

from __future__ import annotations

import typing as T

import numpy as np

if T.TYPE_CHECKING:
    import numpy.typing as npt

SCHEMA_VERSION = 1


class MaskPluginOutput(np.ndarray):
    """Binary mask ndarray carrying per-plugin metadata.

    The array data itself is the ``(N, H, W)`` float32 binary mask in [0, 1].
    Extra attributes are propagated through numpy views / ufunc results via
    ``__array_finalize__``, so all existing downstream arithmetic (``masks * 255.``,
    in-place fancy-index assignment, ``.astype()``, etc.) continues to work without
    modification.

    Parameters
    ----------
    binary_mask
        ``(N, H, W)`` float32 binary mask in [0.0, 1.0].
    source_id
        Plugin ``storage_name`` - the key used in ``batch.masks`` and the
        alignments file.
    version
        Schema version. Defaults to :data:`SCHEMA_VERSION`.
    per_class_probs
        ``(N, H, W, C)`` float32 per-class probabilities, or ``None`` for
        plugins that do not expose a multi-class distribution (e.g. detection-
        based obstacle masks).  For torch-based segmentation plugins (BiSeNet-FP,
        SegNeXt-FP) the values are softmaxed class probabilities.  For
        EasyPortrait they are the joint portrait x face probabilities from the two
        ONNX heads.  ``None`` for HandYolov9c.
    """

    per_class_probs: npt.NDArray[np.float32] | None
    source_id: str
    version: int

    def __new__(
        cls,
        binary_mask: npt.NDArray[np.float32],
        source_id: str,
        version: int = SCHEMA_VERSION,
        per_class_probs: npt.NDArray[np.float32] | None = None,
    ) -> MaskPluginOutput:
        obj = np.asarray(binary_mask).view(cls)
        obj.per_class_probs = per_class_probs
        obj.source_id = source_id
        obj.version = version
        return T.cast(MaskPluginOutput, obj)

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        if obj is None:
            return
        # per_class_probs is intentionally NOT propagated to derived arrays (views, ufunc
        # results, copies). Propagating it would silently keep a potentially large float32
        # array alive inside every derived ndarray produced by downstream consumers
        # (e.g. `masks * 255.` in lib/infer/mask.py). Read per_class_probs only from the
        # direct return value of post_process().
        self.per_class_probs = None
        self.source_id = getattr(obj, "source_id", "")
        self.version = getattr(obj, "version", SCHEMA_VERSION)


def softmax_last_axis(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Numerically stable softmax along the last axis.

    Used by segmentation plugins to convert raw class logits to probabilities
    before storing them in :attr:`MaskPluginOutput.per_class_probs`.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype("float32")
