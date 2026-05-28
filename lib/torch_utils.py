#!/usr/bin/env python
"""Common multi-backend Torch utilities"""

from __future__ import annotations

import logging
import threading
import typing as T

import numpy as np
import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


def get_device(cpu: bool = False) -> torch.device:
    """Get the correctly configured device for running Torch

    Parameters
    ----------
    cpu
        ``True`` to force running on the CPU.

    Returns
    -------
    The device that torch should use
    """
    if cpu:
        logger.debug("CPU mode selected. Returning CPU device")
        return torch.device("cpu")

    if torch.cuda.is_available():
        logger.debug("Cuda available. Returning Cuda device")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        logger.debug("MPS available. Returning MPS device context")
        return torch.device("mps")

    logger.debug("No backends available. Returning CPU device context")
    return torch.device("cpu")


AcceleratorType = T.Literal["cuda", "mps"]

# MPS exposes no peak-memory API, so we maintain peak high-water marks ourselves.
# ``reset`` snapshots the current values; subsequent ``max_*`` calls take the max of the
# snapshot and the live reading. The lock guards concurrent updates from worker threads.
_MPS_PEAK_LOCK = threading.Lock()
_mps_peak_allocated: int = 0
_mps_peak_driver: int = 0


def get_accelerator_type() -> AcceleratorType | None:
    """Return the active Torch accelerator type, if any."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def accelerator_empty_cache() -> None:
    """Empty the active accelerator's allocator cache. No-op without an accelerator."""
    accelerator = get_accelerator_type()
    if accelerator == "cuda":
        torch.cuda.empty_cache()
    elif accelerator == "mps":
        torch.mps.empty_cache()


def accelerator_synchronize() -> None:
    """Synchronize the active accelerator. No-op without an accelerator."""
    accelerator = get_accelerator_type()
    if accelerator == "cuda":
        torch.cuda.synchronize()
    elif accelerator == "mps":
        torch.mps.synchronize()


def accelerator_synchronize_in_worker() -> bool:
    """Whether worker threads should call :func:`accelerator_synchronize` directly.

    CUDA's per-thread streams make synchronizing from any thread safe. MPS shares a single
    Metal command queue, and concurrent calls to ``torch.mps.synchronize()`` from worker
    threads race against ``addScheduledHandler`` on the in-flight command buffer and trigger
    Metal assertions. Pipelines that want to synchronize on MPS must do so from the main
    thread after all workers have finished dispatching.
    """
    return get_accelerator_type() != "mps"


def accelerator_total_memory() -> int:
    """Return total memory available to the active accelerator in bytes.

    Returns ``0`` if no accelerator is available.
    """
    accelerator = get_accelerator_type()
    if accelerator == "cuda":
        return int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    if accelerator == "mps":
        # ``recommended_max_memory`` reflects the Metal driver's working-set ceiling for the
        # current process, which is the closest analogue to CUDA's total device VRAM for
        # batch-size profiling on Apple Silicon. Fall back to 75% of system RAM if running on
        # a PyTorch build that predates the API.
        if hasattr(torch.mps, "recommended_max_memory"):
            return int(torch.mps.recommended_max_memory())
        try:
            import psutil  # noqa: PLC0415
        except ImportError:
            return 0
        logger.warning(
            "torch.mps.recommended_max_memory() unavailable; estimating MPS limit as 75%% of "
            "system RAM"
        )
        return int(psutil.virtual_memory().total * 0.75)
    return 0


def accelerator_reset_peak_memory_stats() -> None:
    """Reset peak memory stats on the active accelerator.

    On MPS, captures the current allocator/driver state as the new baseline for our manual
    peak tracking (PyTorch MPS has no built-in peak API).
    """
    global _mps_peak_allocated, _mps_peak_driver
    accelerator = get_accelerator_type()
    if accelerator == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif accelerator == "mps":
        with _MPS_PEAK_LOCK:
            _mps_peak_allocated = int(torch.mps.current_allocated_memory())
            _mps_peak_driver = int(torch.mps.driver_allocated_memory())


def accelerator_max_memory_allocated() -> int:
    """Return peak allocated memory in bytes since the last reset.

    On MPS the peak is tracked manually across calls — call sites that need an accurate
    peak should poll this during the benchmark window, not just at the end. Returns ``0``
    without an accelerator.
    """
    global _mps_peak_allocated
    accelerator = get_accelerator_type()
    if accelerator == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if accelerator == "mps":
        current = int(torch.mps.current_allocated_memory())
        with _MPS_PEAK_LOCK:
            _mps_peak_allocated = max(_mps_peak_allocated, current)
            return _mps_peak_allocated
    return 0


def accelerator_max_memory_reserved() -> int:
    """Return peak reserved memory in bytes since the last reset.

    On MPS this uses ``driver_allocated_memory`` (the closest analogue to CUDA's reserved
    pool) with manual peak tracking. Returns ``0`` without an accelerator.
    """
    global _mps_peak_driver
    accelerator = get_accelerator_type()
    if accelerator == "cuda":
        return int(torch.cuda.max_memory_reserved())
    if accelerator == "mps":
        current = int(torch.mps.driver_allocated_memory())
        with _MPS_PEAK_LOCK:
            _mps_peak_driver = max(_mps_peak_driver, current)
            return _mps_peak_driver
    return 0


def is_accelerator_oom_error(exc: BaseException) -> bool:
    """Return ``True`` if *exc* indicates an accelerator out-of-memory condition.

    CUDA raises :class:`torch.cuda.OutOfMemoryError` (aliased to :class:`torch.OutOfMemoryError`
    in modern PyTorch). MPS may raise that too, but on older builds it often surfaces as a
    plain :class:`RuntimeError` whose message contains both "mps" and "out of memory" — so
    string-match as a fallback for the MPS path.
    """
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if get_accelerator_type() == "mps" and isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "mps" in msg and "out of memory" in msg
    return False


class ColorSpaceConvert(nn.Module):
    """Transforms inputs between different color spaces on the GPU. Images expected in (N,C,H,W)
    order

    Notes
    -----
    The following color space transformations are implemented:
        - rgb to lab
        - rgb to xyz
        - srgb to _rgb
        - srgb to ycxcz
        - xyz to ycxcz
        - xyz to lab
        - xyz to rgb
        - ycxcz to rgb
        - ycxcz to xyz

    Parameters
    ----------
    from_space
        One of "srgb", "rgb", "ycxcz", "xyz"
    to_space
        One of "lab", "rgb", "ycxcz", "xyz"

    Raises
    ------
    ValueError
        If the requested color space conversion is not defined
    """

    _ref_illuminant: torch.Tensor
    _inv_ref_illuminant: torch.Tensor
    _rgb_xyz_map: torch.Tensor

    def __init__(
        self,
        from_space: T.Literal["srgb", "rgb", "ycxcz", "xyz"],
        to_space: T.Literal["lab", "rgb", "ycxcz", "xyz"],
    ) -> None:
        functions = {
            "rgb_lab": self._rgb_to_lab,
            "rgb_xyz": self._rgb_to_xyz,
            "srgb_rgb": self._srgb_to_rgb,
            "srgb_ycxcz": self._srgb_to_ycxcz,
            "xyz_ycxcz": self._xyz_to_ycxcz,
            "xyz_lab": self._xyz_to_lab,
            "xyz_rgb": self._xyz_to_rgb,
            "ycxcz_rgb": self._ycxcz_to_rgb,
            "ycxcz_xyz": self._ycxcz_to_xyz,
        }
        super().__init__()
        logger.debug(parse_class_init(locals()))
        func_name = f"{from_space.lower()}_{to_space.lower()}"
        if func_name not in functions:
            raise ValueError(f"The color transform {from_space} to {to_space} is not defined.")
        self._func = functions[func_name]

        ref_illuminant = np.array(
            [[[0.950428545]], [[1.000000000]], [[1.088900371]]], dtype=np.float32
        )
        self.register_buffer("_ref_illuminant", torch.from_numpy(ref_illuminant).float())
        self.register_buffer("_inv_ref_illuminant", torch.from_numpy(1.0 / ref_illuminant).float())
        self.register_buffer("_rgb_xyz_map", self._get_rgb_xyz_map())

    @classmethod
    def _get_rgb_xyz_map(cls) -> torch.Tensor:
        """Obtain the mapping and inverse mapping for rgb to xyz color space conversion.

        Returns
        -------
        The mapping and inverse Tensors for rgb to xyz color space conversion
        """
        mapping = np.array(
            [
                [10135552 / 24577794, 8788810 / 24577794, 4435075 / 24577794],
                [2613072 / 12288897, 8788810 / 12288897, 887015 / 12288897],
                [1425312 / 73733382, 8788810 / 73733382, 70074185 / 73733382],
            ]
        )
        inverse = np.linalg.inv(mapping)
        return torch.from_numpy(np.stack([mapping, inverse], axis=0)).float()

    def _rgb_to_lab(self, image: torch.Tensor) -> torch.Tensor:
        """RGB to LAB conversion.

        Parameters
        ----------
        image
            The image tensor in RGB format

        Returns
        -------
        The image tensor in LAB format
        """
        converted = self._rgb_to_xyz(image)
        return self._xyz_to_lab(converted)

    def _rgb_xyz_rgb(self, image: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
        """RGB to XYZ or XYZ to RGB conversion.

        Notes
        -----
        The conversion in both directions is the same, but the mapping matrix for XYZ to RGB is
        the inverse of RGB to XYZ.

        References
        ----------
        https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz

        Parameters
        ----------
        mapping
            The mapping matrix to perform either the XYZ to RGB or RGB to XYZ color space
            conversion

        image
            The image tensor in RGB format

        Returns
        -------
        The image tensor in XYZ format
        """
        dim = image.shape
        image = image.reshape(dim[0], dim[1], dim[2] * dim[3])
        converted = mapping @ image
        return converted.view(dim)

    def _rgb_to_xyz(self, image: torch.Tensor) -> torch.Tensor:
        """RGB to XYZ conversion.

        Parameters
        ----------
        image
            The image tensor in RGB format

        Returns
        -------
        The image tensor in XYZ format
        """
        return self._rgb_xyz_rgb(image, self._rgb_xyz_map[0])

    @classmethod
    def _srgb_to_rgb(cls, image: torch.Tensor) -> torch.Tensor:
        """SRGB to RGB conversion.

        Notes
        -----
        RGB Image is clipped to a small epsilon to stabilize training

        Parameters
        ----------
        image
            The image tensor in SRGB format

        Returns
        -------
        The image tensor in RGB format
        """
        limit = 0.04045
        return torch.where(
            image > limit,
            ((torch.clamp(image, min=limit) + 0.055) / 1.055) ** 2.4,
            image / 12.92,
        )

    def _srgb_to_ycxcz(self, image: torch.Tensor) -> torch.Tensor:
        """SRGB to YcXcZ conversion.

        Parameters
        ----------
        image
            The image tensor in SRGB format

        Returns
        -------
        The image tensor in YcXcZ format
        """
        converted = self._srgb_to_rgb(image)
        converted = self._rgb_to_xyz(converted)
        return self._xyz_to_ycxcz(converted)

    def _xyz_to_lab(self, image: torch.Tensor) -> torch.Tensor:
        """XYZ to LAB conversion.

        Parameters
        ----------
        image
            The image tensor in XYZ format

        Returns
        -------
        The image tensor in LAB format
        """
        image = image * self._inv_ref_illuminant
        delta = 6 / 29
        delta_cube = delta**3
        factor = 1 / (3 * (delta**2))

        clamped_term = torch.clamp(image, min=delta_cube) ** (1.0 / 3.0)
        div = factor * image + (4 / 29)

        image = torch.where(image > delta_cube, clamped_term, div)
        return torch.cat(
            [
                116 * image[:, 1:2] - 16.0,
                500 * (image[:, 0:1] - image[:, 1:2]),
                200 * (image[:, 1:2] - image[:, 2:3]),
            ],
            dim=1,
        )

    def _xyz_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        """XYZ to YcXcZ conversion.

        Parameters
        ----------
        image
            The image tensor in XYZ format

        Returns
        -------
        The image tensor in RGB format
        """
        return self._rgb_xyz_rgb(image, self._rgb_xyz_map[1])

    def _xyz_to_ycxcz(self, image: torch.Tensor) -> torch.Tensor:
        """XYZ to YcXcZ conversion.

        Parameters
        ----------
        image
            The image tensor in XYZ format

        Returns
        -------
        The image tensor in YcXcZ format
        """
        image = image * self._inv_ref_illuminant
        return torch.cat(
            [
                116 * image[:, 1:2] - 16.0,
                500 * (image[:, 0:1] - image[:, 1:2]),
                200 * (image[:, 1:2] - image[:, 2:3]),
            ],
            dim=1,
        )

    def _ycxcz_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        """YcXcZ to RGB conversion.

        Parameters
        ----------
        image
            The image tensor in YcXcZ format

        Returns
        -------
        The image tensor in RGB format
        """
        converted = self._ycxcz_to_xyz(image)
        return self._xyz_to_rgb(converted)

    def _ycxcz_to_xyz(self, image: torch.Tensor) -> torch.Tensor:
        """YcXcZ to XYZ conversion.

        Parameters
        ----------
        image
            The image tensor in YcXcZ format

        Returns
        -------
        The image tensor in XYZ format
        """
        ch_y = (image[:, 0:1] + 16.0) / 116
        return (
            torch.cat(
                [ch_y + (image[:, 1:2] / 500.0), ch_y, ch_y - (image[:, 2:3] / 200.0)],
                dim=1,
            )
            * self._ref_illuminant
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Call the color-space conversion function.

        Parameters
        ----------
        image
            The image tensor in the color-space defined by :attr:`from_space`

        Returns
        -------
        The image tensor in the color-space defined by :attr:`to_space`
        """
        return self._func(image)


__all__ = get_module_objects(__name__)
