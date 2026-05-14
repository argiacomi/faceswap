#!/usr/bin/env python3
"""InsightFace SCRFD face detection plugin.

PyTorch architecture adapted from InsightFace SCRFD:
https://github.com/deepinsight/insightface/tree/master/detection/scrfd

SPDX-License-Identifier: MIT
Copyright (c) InsightFace contributors
"""

from __future__ import annotations

import hashlib
import logging
import os
import typing as T
from time import perf_counter_ns
from urllib import request

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.infer.profile_events import ProfileEvent
from lib.infer.torch_nms import torch_nms
from lib.utils import PROJECT_ROOT, FaceswapError, get_module_objects
from plugins.extract.base import ExtractPlugin

from . import scrfd_defaults as cfg

if T.TYPE_CHECKING:
    import numpy.typing as npt


logger = logging.getLogger(__name__)

_MODEL_CONFIG: dict[str, dict[str, T.Any]] = {
    "10g": {
        "filename": "scrfd_10g.pth",
        "sha256": "963570df5e0ebf6bb313239d0f9f3f0c096c1ff6937e8e28e45abad4d8b1d5c7",
        "url": "https://huggingface.co/DimasMP3/myklipers-models/resolve/main/"
        "insightface/models/scrfd_10g/model.pth",
        "block": "basic",
        "stage_blocks": (3, 4, 2, 3),
        "stage_planes": (56, 88, 88, 224),
        "neck_in": (56, 88, 88, 224),
        "neck_out": 56,
        "head_convs": 3,
        "head_channels": 80,
        "head_groups": 16,
    },
    "34g": {
        "filename": "scrfd_34g.pth",
        "sha256": "a6f69956639da31c96d8985c9a0ce1f5798f42cb64909159596e7a5f544ebe00",
        "url": "https://huggingface.co/DimasMP3/myklipers-models/resolve/main/"
        "insightface/models/scrfd_34g/model.pth",
        "block": "bottleneck",
        "stage_blocks": (17, 16, 2, 8),
        "stage_planes": (56, 56, 144, 184),
        "neck_in": (224, 224, 576, 736),
        "neck_out": 128,
        "head_convs": 2,
        "head_channels": 256,
        "head_groups": 32,
    },
}


class SCRFD(ExtractPlugin):
    """InsightFace SCRFD detector for face detection."""

    def __init__(self) -> None:
        super().__init__(
            input_size=640,
            batch_size=cfg.batch_size(),
            is_rgb=False,
            dtype=np.uint8,
            scale=(0, 255),
            force_cpu=cfg.cpu(),
        )
        model_name = T.cast(T.Literal["10g", "34g"], cfg.model())
        if model_name not in _MODEL_CONFIG:
            raise FaceswapError(
                f"Unsupported SCRFD model: {model_name}. Select from {list(_MODEL_CONFIG)}."
            )
        self._model_config = _MODEL_CONFIG[model_name]
        self.model: SCRFDModel
        self._confidence = cfg.confidence() / 100
        self._nms_threshold = 0.4
        self._model_path = self._get_weights_path()
        self._feature_strides = [8, 16, 32]
        self._num_anchors = 2
        self._center_cache: dict[tuple[int, int, int], npt.NDArray[np.float32]] = {}
        self._center_cache_torch: dict[tuple[int, int, int, str], torch.Tensor] = {}
        self._postprocess_mode = T.cast(
            T.Literal["auto", "torch", "numpy"], cfg.scrfd_postprocess()
        )
        self._last_profile_events: list[ProfileEvent] = []

    def _get_weights_path(self) -> str:
        """Download InsightFace SCRFD weights, if required, and return the cached path."""
        cache_dir = os.path.join(PROJECT_ROOT, ".fs_cache")
        model_path = os.path.join(cache_dir, self._model_config["filename"])
        if os.path.exists(model_path):
            logger.debug("SCRFD model exists: %s", model_path)
            if self._validate_weights(model_path):
                return model_path
            logger.warning("SCRFD model checksum mismatch. Re-downloading: %s", model_path)
            os.remove(model_path)

        os.makedirs(cache_dir, exist_ok=True)
        url = self._model_config["url"]
        download_path = f"{model_path}.part"
        # TODO: Move SCRFD into lib.utils.GetModel when these weights are hosted in the
        # faceswap-models registry. GetModel currently only handles that release layout.
        logger.info("Downloading SCRFD model from: %s", url)
        success = False
        try:
            request.urlretrieve(url, download_path)  # noqa:S310 - checksum validated model URL
            success = True
        finally:
            if not success and os.path.exists(download_path):
                os.remove(download_path)
        if not self._validate_weights(download_path):
            os.remove(download_path)
            raise FaceswapError(f"Downloaded SCRFD model failed checksum validation: {url}")
        os.replace(download_path, model_path)
        return model_path

    def _validate_weights(self, model_path: str) -> bool:
        """Validate cached SCRFD weights against the expected SHA256 hash."""
        sha256 = hashlib.sha256()
        with open(model_path, "rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == self._model_config["sha256"]

    def load_model(self) -> SCRFDModel:
        """Load the SCRFD PyTorch model."""
        model = SCRFDModel(self._model_config)
        return T.cast(SCRFDModel, self.load_torch_model(model, self._model_path))

    def pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Compile the detection image(s) for prediction."""
        if self.device.type != "cpu":
            return batch
        return self._blob_pre_process(batch)

    def _blob_pre_process(self, batch: np.ndarray) -> np.ndarray:
        """Convert BGR uint8 detector feeds into normalized NCHW float32 blobs."""
        return cv2.dnn.blobFromImages(
            list(batch),
            scalefactor=1.0 / 128.0,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
            ddepth=cv2.CV_32F,
        )

    def process(self, batch: np.ndarray) -> np.ndarray:
        """Run model to get predictions."""
        if self._should_use_torch_path():
            return self._from_torch_raw(batch)
        if batch.ndim == 4 and batch.shape[-1] == 3:
            batch = self._blob_pre_process(batch)
        return self.from_torch(batch)

    @property
    def profile_events(self) -> tuple[ProfileEvent, ...]:
        """Return the SCRFD-specific profiling events from the last post-process call."""
        return tuple(self._last_profile_events)

    def _should_use_torch_path(self) -> bool:
        """Return whether inference should preserve Torch tensors for post-processing."""
        if self._postprocess_mode == "torch":
            return True
        if self._postprocess_mode == "numpy":
            return False
        return self.device.type != "cpu"

    def _from_torch_raw(self, batch: np.ndarray) -> np.ndarray:
        """Run the Torch model while keeping output tensors on-device."""
        if not isinstance(self.model, nn.Module):
            return self.from_torch(batch)

        with torch.inference_mode():
            feed = torch.from_numpy(batch)
            feed = (
                feed.pin_memory().to(self.device, non_blocking=True)
                if self.device.type == "cuda"
                else feed.to(self.device)
            )

            if feed.ndim == 4 and feed.shape[-1] == 3:
                # Keep detector feeds compact on the host, then normalize and reorder on-device.
                feed = feed.permute(0, 3, 1, 2)
                feed = feed[:, [2, 1, 0]].to(dtype=torch.float32)
                feed.sub_(127.5).mul_(1.0 / 128.0)
                feed = feed.contiguous(memory_format=torch.channels_last)
            else:
                if feed.dtype != torch.float32:
                    feed = feed.to(dtype=torch.float32)
                feed = feed.to(memory_format=torch.channels_last)
            output = self.model(feed)

        if isinstance(output, torch.Tensor):
            retval = np.empty((1,), dtype="object")
            retval[0] = output
            return retval

        tensors = list(output)
        return self._object_array(tensors)

    @staticmethod
    def _object_array(values: T.Sequence[T.Any]) -> np.ndarray:
        """Store arbitrary Python objects without triggering implicit NumPy conversion."""
        retval = np.empty((len(values),), dtype="object")
        for idx, value in enumerate(values):
            retval[idx] = value
        return retval

    @staticmethod
    def _distance_to_bbox(
        points: npt.NDArray[np.float32], distance: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Decode SCRFD distance predictions to bounding boxes."""
        return np.stack(
            [
                points[:, 0] - distance[:, 0],
                points[:, 1] - distance[:, 1],
                points[:, 0] + distance[:, 2],
                points[:, 1] + distance[:, 3],
            ],
            axis=-1,
        )

    @staticmethod
    def _distance_to_bbox_torch(points: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
        """Decode SCRFD distance predictions to bounding boxes on-device."""
        return torch.stack(
            [
                points[:, 0] - distance[:, 0],
                points[:, 1] - distance[:, 1],
                points[:, 0] + distance[:, 2],
                points[:, 1] + distance[:, 3],
            ],
            dim=-1,
        )

    def _anchor_centers(self, height: int, width: int, stride: int) -> npt.NDArray[np.float32]:
        """Return cached SCRFD anchor centers for the feature map shape."""
        key = (height, width, stride)
        if key not in self._center_cache:
            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype("float32")
            centers = (centers * stride).reshape((-1, 2))
            centers = np.stack([centers] * self._num_anchors, axis=1).reshape((-1, 2))
            self._center_cache[key] = centers
            return centers
        return self._center_cache[key]

    def _anchor_centers_torch(
        self, height: int, width: int, stride: int, device: torch.device
    ) -> torch.Tensor:
        """Return cached SCRFD anchor centers on the requested device."""
        key = (height, width, stride, str(device))
        if key not in self._center_cache_torch:
            centers = torch.from_numpy(self._anchor_centers(height, width, stride)).to(
                device=device,
                dtype=torch.float32,
            )
            self._center_cache_torch[key] = centers
            return centers
        return self._center_cache_torch[key]

    def _nms(self, boxes: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Perform Non-Maximum Suppression and return kept boxes."""
        x_1, y_1, x_2, y_2, scores = boxes.T
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            idx = order[0]
            keep.append(idx)
            xx_1 = np.maximum(x_1[idx], x_1[order[1:]])
            yy_1 = np.maximum(y_1[idx], y_1[order[1:]])
            xx_2 = np.minimum(x_2[idx], x_2[order[1:]])
            yy_2 = np.minimum(y_2[idx], y_2[order[1:]])
            width = np.maximum(0.0, xx_2 - xx_1 + 1)
            height = np.maximum(0.0, yy_2 - yy_1 + 1)
            intersection = width * height
            overlap = intersection / (areas[idx] + areas[order[1:]] - intersection)
            order = order[np.where(overlap <= self._nms_threshold)[0] + 1]
        return boxes[keep]

    def _record_profile_event(
        self,
        stage: str,
        start_ns: int,
        end_ns: int,
        *,
        frame_index: int,
        bytes_in: int = 0,
        bytes_out: int = 0,
        transfer_direction: str = "",
    ) -> None:
        """Record a SCRFD-specific profiling event for the last post-process call."""
        self._last_profile_events.append(
            ProfileEvent(
                stage=stage,
                plugin=self.name,
                frame_index=frame_index,
                face_index=None,
                start_ns=start_ns,
                end_ns=end_ns,
                device=self.device.type,
                bytes_in=bytes_in,
                bytes_out=bytes_out,
                transfer_direction=transfer_direction,
            )
        )

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        """Return the number of bytes held by a tensor."""
        return int(tensor.nelement() * tensor.element_size())

    def _as_torch_prediction(self, prediction: T.Any) -> torch.Tensor:
        """Normalize a prediction item into a float32 Torch tensor on the plugin device."""
        if isinstance(prediction, torch.Tensor):
            return prediction.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(prediction, device=self.device, dtype=torch.float32)

    def _torch_post_process(self, batch: np.ndarray) -> np.ndarray:
        """Process SCRFD output while keeping decode, threshold and NMS on-device."""
        final_boxes: list[np.ndarray] = []
        batch_size = batch[0].shape[0]
        self._last_profile_events = []
        for img_idx in range(batch_size):
            image_scores: list[torch.Tensor] = []
            image_boxes: list[torch.Tensor] = []
            for out_idx, stride in enumerate(self._feature_strides):
                scores = self._as_torch_prediction(batch[out_idx][img_idx]).reshape(-1)
                bbox_preds = self._as_torch_prediction(
                    batch[out_idx + len(self._feature_strides)][img_idx]
                ).reshape((-1, 4))
                bbox_preds = bbox_preds * stride
                indices_start = perf_counter_ns()
                indices = torch.nonzero(scores >= self._confidence, as_tuple=False).squeeze(1)
                indices_end = perf_counter_ns()
                self._record_profile_event(
                    "scrfd_threshold",
                    indices_start,
                    indices_end,
                    frame_index=img_idx,
                    bytes_in=self._tensor_bytes(scores),
                )
                if indices.numel() == 0:
                    continue

                height = self.input_size // stride
                width = self.input_size // stride
                centers = self._anchor_centers_torch(height, width, stride, scores.device)
                decode_start = perf_counter_ns()
                selected_scores = scores.index_select(0, indices)
                selected_boxes = self._distance_to_bbox_torch(
                    centers.index_select(0, indices),
                    bbox_preds.index_select(0, indices),
                )
                decode_end = perf_counter_ns()
                self._record_profile_event(
                    "scrfd_decode",
                    decode_start,
                    decode_end,
                    frame_index=img_idx,
                    bytes_in=(self._tensor_bytes(centers) + self._tensor_bytes(bbox_preds)),
                    bytes_out=self._tensor_bytes(selected_boxes),
                )
                image_scores.append(selected_scores)
                image_boxes.append(selected_boxes)

            if not image_scores:
                final_boxes.append(np.empty((0, 4), dtype="float32"))
                continue

            scores = torch.cat(image_scores)
            boxes = torch.cat(image_boxes)
            nms_start = perf_counter_ns()
            keep = torch_nms(boxes, scores, self._nms_threshold)
            nms_end = perf_counter_ns()
            self._record_profile_event(
                "scrfd_nms",
                nms_start,
                nms_end,
                frame_index=img_idx,
                bytes_in=(self._tensor_bytes(boxes) + self._tensor_bytes(scores)),
            )

            copy_start = perf_counter_ns()
            final = boxes.index_select(0, keep).to("cpu", dtype=torch.float32).numpy()
            copy_end = perf_counter_ns()
            self._record_profile_event(
                "scrfd_copy_back",
                copy_start,
                copy_end,
                frame_index=img_idx,
                bytes_in=self._tensor_bytes(boxes.index_select(0, keep)),
                bytes_out=int(final.nbytes),
                transfer_direction="device_to_host",
            )
            final_boxes.append(final)

        retval = np.empty(len(final_boxes), dtype=object)
        for idx, box in enumerate(final_boxes):
            retval[idx] = box
        return retval

    def post_process(self, batch: np.ndarray) -> np.ndarray:
        """Process SCRFD output to bounding boxes at model input size."""
        if self._postprocess_mode == "torch" or isinstance(batch[0][0], torch.Tensor):
            return self._torch_post_process(batch)

        self._last_profile_events = []
        final_boxes = []
        batch_size = batch[0].shape[0]
        for img_idx in range(batch_size):
            image_scores = []
            image_boxes = []
            for out_idx, stride in enumerate(self._feature_strides):
                scores = batch[out_idx][img_idx].reshape(-1).astype("float32", copy=False)
                bbox_preds = batch[out_idx + len(self._feature_strides)][img_idx]
                bbox_preds = bbox_preds.reshape((-1, 4)).astype("float32", copy=False) * stride
                height = self.input_size // stride
                width = self.input_size // stride
                centers = self._anchor_centers(height, width, stride)
                indices = np.where(scores >= self._confidence)[0]
                if indices.size == 0:
                    continue
                image_scores.append(scores[indices])
                image_boxes.append(self._distance_to_bbox(centers, bbox_preds)[indices])

            if not image_scores:
                final_boxes.append(np.empty((0, 4), dtype="float32"))
                continue

            scores = np.concatenate(image_scores)
            boxes = np.vstack(image_boxes)
            detections = np.hstack((boxes, scores[:, None])).astype("float32", copy=False)
            final_boxes.append(self._nms(detections)[:, :4])

        retval = np.empty(len(final_boxes), dtype=object)
        for idx, box in enumerate(final_boxes):
            retval[idx] = box
        return retval


class ConvModule(nn.Module):
    """Small subset of MMCV ConvModule matching SCRFD checkpoint names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        num_groups: int | None = None,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=num_groups is None,
        )
        self.gn = nn.GroupNorm(num_groups, out_channels) if num_groups is not None else None
        self.activate = activate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self.conv(inputs)
        if self.gn is not None:
            output = self.gn(output)
        if self.activate:
            output = F.relu(output, inplace=True)
        return output


class Scale(nn.Module):
    """Learnable scalar used by SCRFD bbox heads."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return inputs * self.scale


class BasicBlock(nn.Module):
    """SCRFD BasicBlock."""

    expansion = 1

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = inputs if self.downsample is None else self.downsample(inputs)
        output = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        output = self.bn2(self.conv2(output))
        return F.relu(output + identity, inplace=True)


class Bottleneck(nn.Module):
    """SCRFD Bottleneck block."""

    expansion = 4

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        identity = inputs if self.downsample is None else self.downsample(inputs)
        output = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        output = F.relu(self.bn2(self.conv2(output)), inplace=True)
        output = self.bn3(self.conv3(output))
        return F.relu(output + identity, inplace=True)


class SCRFDBackbone(nn.Module):
    """ResNetV1e backbone subset used by SCRFD 10G/34G."""

    def __init__(
        self,
        block_name: T.Literal["basic", "bottleneck"],
        stage_blocks: tuple[int, int, int, int],
        stage_planes: tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 28, 3, 2, 1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.Conv2d(28, 28, 3, 1, 1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.Conv2d(28, 56, 3, 1, 1, bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block = BasicBlock if block_name == "basic" else Bottleneck
        in_channels = 56
        self.layer1, in_channels = self._make_layer(
            block, in_channels, stage_planes[0], stage_blocks[0], stride=1
        )
        self.layer2, in_channels = self._make_layer(
            block, in_channels, stage_planes[1], stage_blocks[1], stride=2
        )
        self.layer3, in_channels = self._make_layer(
            block, in_channels, stage_planes[2], stage_blocks[2], stride=2
        )
        self.layer4, _ = self._make_layer(
            block, in_channels, stage_planes[3], stage_blocks[3], stride=2
        )

    @staticmethod
    def _make_layer(
        block: type[BasicBlock] | type[Bottleneck],
        in_channels: int,
        mid_channels: int,
        blocks: int,
        stride: int,
    ) -> tuple[nn.Sequential, int]:
        """Build one residual stage."""
        layers = [block(in_channels, mid_channels, stride)]
        out_channels = mid_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(out_channels, mid_channels, 1))
        return nn.Sequential(*layers), out_channels

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass."""
        output = self.maxpool(self.stem(inputs))
        outputs = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            output = layer(output)
            outputs.append(output)
        return tuple(outputs)


class PAFPN(nn.Module):
    """Path Aggregation FPN subset used by SCRFD."""

    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int) -> None:
        super().__init__()
        neck_inputs = in_channels[1:]
        self.lateral_convs = nn.ModuleList(
            ConvModule(channels, out_channels, 1, activate=False) for channels in neck_inputs
        )
        self.fpn_convs = nn.ModuleList(
            ConvModule(out_channels, out_channels, 3, padding=1, activate=False)
            for _ in neck_inputs
        )
        self.downsample_convs = nn.ModuleList(
            ConvModule(out_channels, out_channels, 3, stride=2, padding=1, activate=False)
            for _ in range(2)
        )
        self.pafpn_convs = nn.ModuleList(
            ConvModule(out_channels, out_channels, 3, padding=1, activate=False) for _ in range(2)
        )

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Forward pass."""
        laterals = [
            lateral_conv(inputs[idx + 1]) for idx, lateral_conv in enumerate(self.lateral_convs)
        ]
        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx], size=laterals[idx - 1].shape[2:], mode="nearest"
            )

        inter_outs = [
            conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals, strict=False)
        ]
        for idx, conv in enumerate(self.downsample_convs):
            inter_outs[idx + 1] = inter_outs[idx + 1] + conv(inter_outs[idx])

        outs = [inter_outs[0]]
        outs.extend(conv(inter_outs[idx]) for idx, conv in enumerate(self.pafpn_convs, start=1))
        return tuple(outs)


class SCRFDHead(nn.Module):
    """SCRFD detection head for inference."""

    def __init__(
        self, in_channels: int, stacked_convs: int, feat_channels: int, num_groups: int
    ) -> None:
        super().__init__()
        convs = []
        for idx in range(stacked_convs):
            channels = in_channels if idx == 0 else feat_channels
            convs.append(ConvModule(channels, feat_channels, 3, padding=1, num_groups=num_groups))
        self.cls_stride_convs = nn.ModuleDict({"0": nn.ModuleList(convs)})
        self.stride_cls = nn.ModuleDict({"0": nn.Conv2d(feat_channels, 2, 3, padding=1)})
        self.stride_reg = nn.ModuleDict({"0": nn.Conv2d(feat_channels, 8, 3, padding=1)})
        self.scales = nn.ModuleList([Scale(), Scale(), Scale()])
        self.integral = Integral()

    def forward(self, feats: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Forward pass."""
        scores = []
        bboxes = []
        for feat, scale in zip(feats, self.scales, strict=False):
            cls_feat = feat
            for conv in self.cls_stride_convs["0"]:
                cls_feat = conv(cls_feat)
            cls_score = self.stride_cls["0"](cls_feat)
            bbox_pred = scale(self.stride_reg["0"](cls_feat))
            batch_size = cls_score.shape[0]
            scores.append(cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, 1).sigmoid())
            bboxes.append(bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4))
        return scores + bboxes


class Integral(nn.Module):
    """Checkpoint-compatible SCRFD integral layer placeholder."""

    def __init__(self) -> None:
        super().__init__()
        # Registered only to satisfy strict state_dict loading. SCRFD-10G/34G use direct
        # regression at inference, so this buffer is not used in the forward pass.
        self.register_buffer("project", torch.linspace(0, 8, 9))


class SCRFDModel(nn.Module):
    """SCRFD 10G/34G model."""

    def __init__(self, model_config: dict[str, T.Any]) -> None:
        super().__init__()
        self.backbone = SCRFDBackbone(
            model_config["block"],
            model_config["stage_blocks"],
            model_config["stage_planes"],
        )
        self.neck = PAFPN(model_config["neck_in"], model_config["neck_out"])
        self.bbox_head = SCRFDHead(
            model_config["neck_out"],
            model_config["head_convs"],
            model_config["head_channels"],
            model_config["head_groups"],
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass."""
        return self.bbox_head(self.neck(self.backbone(inputs)))


__all__ = get_module_objects(__name__)
