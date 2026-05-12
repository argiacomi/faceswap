#!/usr/bin/env python3
"""SPIGA facial landmarks extractor for faceswap.py

Code adapted and modified from:
https://github.com/andresprados/SPIGA
SPDX-License-Identifier: BSD-3-Clause
"""

from __future__ import annotations

import os
import typing as T
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from lib.utils import GetModelFromUrl, get_module_objects
from plugins.extract.base import ExtractPlugin

from . import spiga_defaults as cfg
from ._spiga.models.spiga import SPIGA as SPIGANetwork

_HF_URL = "https://huggingface.co/aprados/spiga/resolve/main/{filename}?download=true"

_LDM_IDS_68 = [
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    24,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    1,
    119,
    2,
    121,
    3,
    4,
    124,
    5,
    126,
    6,
    128,
    129,
    130,
    17,
    16,
    133,
    134,
    135,
    18,
    7,
    138,
    139,
    8,
    141,
    142,
    11,
    144,
    145,
    12,
    147,
    148,
    20,
    150,
    151,
    22,
    153,
    154,
    21,
    156,
    157,
    23,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
]
_LDM_IDS_98 = [
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    24,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    1,
    134,
    2,
    136,
    3,
    138,
    139,
    140,
    141,
    4,
    143,
    5,
    145,
    6,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    17,
    16,
    156,
    157,
    158,
    18,
    7,
    161,
    9,
    163,
    8,
    165,
    10,
    167,
    11,
    169,
    13,
    171,
    12,
    173,
    14,
    175,
    20,
    177,
    178,
    22,
    180,
    181,
    21,
    183,
    184,
    23,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
]


@dataclass(frozen=True)
class SPIGAModelConfig:
    """Configuration for a SPIGA checkpoint."""

    filename: str
    sha256: str
    num_landmarks: int
    num_edges: int
    ldm_ids: list[int]

    @property
    def url(self) -> str:
        """Hugging Face download URL for this model."""
        return _HF_URL.format(filename=self.filename)


_MODEL_CONFIG = {
    "300w": SPIGAModelConfig(
        filename="spiga_300wprivate.pt",
        sha256=("a92f15060c3a5e62f2095df8eb59307c7e74d7f99428dded8697b754b62eee4e"),
        num_landmarks=68,
        num_edges=13,
        ldm_ids=_LDM_IDS_68,
    ),
    "merlrav": SPIGAModelConfig(
        filename="spiga_merlrav.pt",
        sha256=("52559b750c07fcf8478c6b8f11ce79beda1c067b577ac6c6d9f2a06be73d0a18"),
        num_landmarks=68,
        num_edges=13,
        ldm_ids=_LDM_IDS_68,
    ),
    "wflw": SPIGAModelConfig(
        filename="spiga_wflw.pt",
        sha256=("e9eee56e132c269c350225059a3ccc31b278b0bf7e3d475bbd38aebba2cac5b1"),
        num_landmarks=98,
        num_edges=15,
        ldm_ids=_LDM_IDS_98,
    ),
}


class SPIGA(ExtractPlugin):
    """SPIGA face alignment plugin."""

    def __init__(self) -> None:
        super().__init__(
            input_size=256,
            batch_size=cfg.batch_size(),
            is_rgb=False,  # upstream SPIGA final tensor is BGR / 255
            dtype="float32",
            scale=(0, 1),
        )
        self._target_dist = cfg.crop_scale()
        self._focal_ratio = 1.5  # Upstream SPIGA camera focal-length ratio.
        self._model_config = _MODEL_CONFIG[cfg.model()]
        self.model: SPIGAFaceswapModel
        self.realign_centering = "legacy"

    def load_model(self) -> SPIGAFaceswapModel:
        """Load the SPIGA model.

        Returns
        -------
        The loaded SPIGA model
        """
        model_path = GetModelFromUrl(
            self._model_config.filename,
            self._model_config.url,
            self._model_config.sha256,
        ).model_path
        model3d = _load_world_shape(self._model_config.ldm_ids)
        model = T.cast(
            SPIGAFaceswapModel,
            self.load_torch_model(
                SPIGAFaceswapModel(
                    num_landmarks=self._model_config.num_landmarks,
                    num_edges=self._model_config.num_edges,
                    model3d=model3d,
                    focal_ratio=self._focal_ratio,
                ),
                model_path,
            ),
        )
        return model

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Format the ROI face detection boxes for prediction.

        Parameters
        ----------
        batch
            The batch of face detection bounding boxes as (bs, l, t, r, b)

        Returns
        -------
        The face detection bounding boxes formatted to take an image patch for prediction
        """
        heights = batch[:, 3] - batch[:, 1]
        widths = batch[:, 2] - batch[:, 0]
        ctr_x = np.rint((batch[:, 0] + batch[:, 2]) * 0.5).astype("int32")
        ctr_y = np.rint((batch[:, 1] + batch[:, 3]) * 0.5).astype("int32")
        side = np.maximum(widths, heights) * self._target_dist
        half = np.rint(side * 0.5).astype("int32")

        retval = np.empty((batch.shape[0], 4), dtype=np.int32)
        retval[:, 0] = ctr_x - half
        retval[:, 1] = ctr_y - half
        retval[:, 2] = ctr_x + half
        retval[:, 3] = ctr_y + half
        return retval

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Predict the face landmarks.

        Parameters
        ----------
        batch
            The batch to feed into the aligner

        Returns
        -------
        The predictions from the aligner
        """
        feed = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        return self.from_torch(feed)

    def post_process(self, batch: np.ndarray) -> np.ndarray:
        """Process the output from the model.

        Parameters
        ----------
        batch
            The predictions from the aligner

        Returns
        -------
        The final landmarks in 0-1 space
        """
        return batch.astype("float32", copy=False)


class SPIGAFaceswapModel(nn.Module):
    """SPIGA wrapper that binds Faceswap's single tensor input to SPIGA's full input list."""

    def __init__(
        self,
        num_landmarks: int,
        num_edges: int,
        model3d: np.ndarray,
        focal_ratio: float,
    ) -> None:
        super().__init__()
        self.spiga = SPIGANetwork(num_landmarks=num_landmarks, num_edges=num_edges)
        # Camera calibration is defined in SPIGA feature-map space, so this must match the
        # upstream network's visual feature resolution for the loaded checkpoint.
        cam_matrix = _camera_matrix(
            [0, 0, self.spiga.visual_res, self.spiga.visual_res],
            focal_ratio=focal_ratio,
        )
        self.register_buffer("model3d", torch.tensor(model3d, dtype=torch.float32))
        self.register_buffer("cam_matrix", torch.tensor(cam_matrix, dtype=torch.float32))

    def load_state_dict(
        self,
        state_dict: T.Mapping[str, T.Any],
        strict: bool = True,
        assign: bool = False,
    ) -> T.Any:
        """Load upstream SPIGA weights into the wrapped SPIGA submodule."""
        weights = dict(state_dict)
        if weights and not next(iter(weights)).startswith("spiga."):
            weights = {f"spiga.{key}": val for key, val in weights.items()}
        weights.pop("model3d", None)
        weights.pop("cam_matrix", None)
        weights["model3d"] = self.model3d
        weights["cam_matrix"] = self.cam_matrix
        return super().load_state_dict(weights, strict=strict, assign=assign)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return only landmark predictions, discarding SPIGA's head-pose output."""
        batch_size = inputs.shape[0]
        model3d = self.model3d.unsqueeze(0).expand(batch_size, -1, -1)
        cam_matrix = self.cam_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        outputs = self.spiga([inputs, model3d, cam_matrix])
        return outputs["Landmarks"][-1]


def _load_world_shape(db_landmarks: list[int]) -> np.ndarray:
    """Load SPIGA's 3D mean face in the landmark order for the configured dataset."""
    filename = os.path.join(
        os.path.dirname(__file__),
        "_spiga",
        "data",
        "models3d",
        f"mean_face_3D_{len(db_landmarks)}.txt",
    )
    posit_landmarks = np.genfromtxt(filename, delimiter="|", dtype=int, usecols=0).tolist()
    mean_face_3d = np.genfromtxt(
        filename, delimiter="|", dtype=(float, float, float), usecols=(1, 2, 3)
    ).tolist()
    world_all: list[list[float] | None] = [None] * len(mean_face_3d)
    for idx, elem in enumerate(mean_face_3d):
        pt3d = [elem[2], -elem[0], -elem[1]]
        lnd_idx = db_landmarks.index(posit_landmarks[idx])
        world_all[lnd_idx] = pt3d
    assert all(item is not None for item in world_all)
    return np.array(world_all, dtype=np.float32)


def _camera_matrix(bbox: list[int], focal_ratio: float) -> np.ndarray:
    """Return SPIGA's camera intrinsic matrix for the model feature-map space."""
    focal_length_x = bbox[2] * focal_ratio
    focal_length_y = bbox[3] * focal_ratio
    face_center = (bbox[0] + (bbox[2] * 0.5)), (bbox[1] + (bbox[3] * 0.5))
    return np.array(
        [
            [focal_length_x, 0, face_center[0]],
            [0, focal_length_y, face_center[1]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


__all__ = get_module_objects(__name__)
