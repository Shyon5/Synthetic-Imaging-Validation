"""Pairwise intensity and structural image-similarity metrics."""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Sequence

import numpy as np
from skimage.metrics import structural_similarity

from ..utils.checks import infer_data_range, validate_pair


def mse(real: Any, synthetic: Any) -> float:
    """Return mean squared error in squared input intensity units; optimum is 0."""

    real_array, synthetic_array = validate_pair(real, synthetic)
    return float(np.mean(np.square(real_array.astype(np.float64) - synthetic_array.astype(np.float64))))


def mae(real: Any, synthetic: Any) -> float:
    """Return mean absolute error in input intensity units; optimum is 0."""

    real_array, synthetic_array = validate_pair(real, synthetic)
    return float(np.mean(np.abs(real_array.astype(np.float64) - synthetic_array.astype(np.float64))))


def rmse(real: Any, synthetic: Any) -> float:
    """Return root mean squared error in input intensity units; optimum is 0."""

    return float(math.sqrt(mse(real, synthetic)))


def nrmse(
    real: Any,
    synthetic: Any,
    *,
    normalization: Literal["range", "mean", "l2"] = "range",
) -> float:
    """Return RMSE normalized by a statistic of the real image.

    ``range`` divides by max-min, ``mean`` by the absolute mean, and ``l2`` by
    root mean square. A zero denominator yields 0 for identical inputs and
    infinity otherwise.
    """

    real_array, synthetic_array = validate_pair(real, synthetic)
    error = rmse(real_array, synthetic_array)
    reference = real_array.astype(np.float64)
    if normalization == "range":
        denominator = float(reference.max() - reference.min())
    elif normalization == "mean":
        denominator = abs(float(reference.mean()))
    elif normalization == "l2":
        denominator = float(np.sqrt(np.mean(np.square(reference))))
    else:
        raise ValueError("normalization must be 'range', 'mean', or 'l2'.")
    if denominator == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return float(error / denominator)


def psnr(real: Any, synthetic: Any, *, data_range: Optional[float] = None) -> float:
    """Return peak signal-to-noise ratio in decibels; identical inputs yield infinity.

    Pass ``data_range`` when the valid modality range is known. Otherwise it is
    inferred from the combined observed minimum and maximum.
    """

    real_array, synthetic_array = validate_pair(real, synthetic)
    error = mse(real_array, synthetic_array)
    if error == 0.0:
        return float("inf")
    resolved_range = infer_data_range(real_array, synthetic_array, data_range=data_range)
    return float(20.0 * math.log10(resolved_range) - 10.0 * math.log10(error))


def _resolve_win_size(shape: Sequence[int], channel_axis: Optional[int], win_size: Optional[int]) -> int:
    spatial_shape = list(shape)
    if channel_axis is not None:
        axis = int(channel_axis) % len(spatial_shape)
        spatial_shape.pop(axis)
    minimum = min(int(v) for v in spatial_shape)
    if win_size is None:
        candidate = min(7, minimum)
        if candidate % 2 == 0:
            candidate -= 1
    else:
        candidate = int(win_size)
    if candidate < 3 or candidate % 2 == 0 or candidate > minimum:
        raise ValueError(
            f"win_size must be an odd integer between 3 and the smallest spatial dimension ({minimum})."
        )
    return candidate


def ssim(
    real: Any,
    synthetic: Any,
    *,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    batch_axis: Optional[int] = None,
    win_size: Optional[int] = None,
    gaussian_weights: bool = True,
) -> float:
    """Return mean SSIM for a 2D image or 3D volume, optionally with channels/batches.

    SSIM is usually in [-1, 1] and is 1 for identical inputs. Set axis
    arguments explicitly; no channel or batch dimension is inferred.
    """

    real_array, synthetic_array = validate_pair(real, synthetic)
    resolved_range = infer_data_range(real_array, synthetic_array, data_range=data_range)
    if batch_axis is not None:
        axis = int(batch_axis) % real_array.ndim
        values = []
        adjusted_channel = channel_axis
        if channel_axis is not None:
            channel = int(channel_axis) % real_array.ndim
            if channel == axis:
                raise ValueError("batch_axis and channel_axis must be different.")
            adjusted_channel = channel - 1 if channel > axis else channel
        for index in range(real_array.shape[axis]):
            values.append(
                ssim(
                    np.take(real_array, index, axis=axis),
                    np.take(synthetic_array, index, axis=axis),
                    data_range=resolved_range,
                    channel_axis=adjusted_channel,
                    win_size=win_size,
                    gaussian_weights=gaussian_weights,
                )
            )
        return float(np.mean(values))

    if real_array.ndim not in (2, 3, 4):
        raise ValueError("SSIM expects a 2D/3D image plus at most one explicit channel axis.")
    resolved_window = _resolve_win_size(real_array.shape, channel_axis, win_size)
    return float(
        structural_similarity(
            real_array.astype(np.float64),
            synthetic_array.astype(np.float64),
            data_range=resolved_range,
            channel_axis=channel_axis,
            win_size=resolved_window,
            gaussian_weights=bool(gaussian_weights),
        )
    )


_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _to_torch_image(array: np.ndarray, channel_axis: Optional[int], batch_axis: Optional[int]):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when optional extra is missing
        raise ImportError("MS-SSIM requires the optional 'torch' dependencies.") from exc

    ndim = array.ndim
    if batch_axis is not None and channel_axis is not None:
        batch = int(batch_axis) % ndim
        channel = int(channel_axis) % ndim
        if batch == channel:
            raise ValueError("batch_axis and channel_axis must be different.")
        canonical = np.moveaxis(array, (batch, channel), (0, 1))
    elif batch_axis is not None:
        canonical = np.moveaxis(array, int(batch_axis) % ndim, 0)[:, None, ...]
    elif channel_axis is not None:
        canonical = np.moveaxis(array, int(channel_axis) % ndim, 0)[None, ...]
    else:
        canonical = array[None, None, ...]
    if canonical.ndim not in (4, 5):
        raise ValueError("MS-SSIM expects 2D or 3D spatial data with optional batch/channel axes.")
    return torch.as_tensor(np.ascontiguousarray(canonical), dtype=torch.float32)


def ms_ssim(
    real: Any,
    synthetic: Any,
    *,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    batch_axis: Optional[int] = None,
    max_scales: int = 5,
) -> float:
    """Return adaptive multi-scale SSIM using optional PyTorch/torchmetrics.

    The function tries the standard five-scale weights, reducing scale count
    and odd Gaussian kernel size for small 2D/3D inputs. Output is normally in
    [0, 1], with 1 optimal. Install with ``pip install .[torch]``.
    """

    try:
        from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MS-SSIM requires torch and torchmetrics; install the 'torch' extra.") from exc

    real_array, synthetic_array = validate_pair(real, synthetic)
    resolved_range = infer_data_range(real_array, synthetic_array, data_range=data_range)
    real_tensor = _to_torch_image(real_array.astype(np.float32), channel_axis, batch_axis)
    synthetic_tensor = _to_torch_image(synthetic_array.astype(np.float32), channel_axis, batch_axis)
    scales_limit = max(1, min(int(max_scales), len(_MS_SSIM_WEIGHTS)))
    errors = []
    for scales in range(scales_limit, 0, -1):
        weights = np.asarray(_MS_SSIM_WEIGHTS[:scales], dtype=np.float64)
        weights = tuple(float(v) for v in weights / weights.sum())
        for kernel in (11, 9, 7, 5, 3):
            try:
                value = multiscale_structural_similarity_index_measure(
                    real_tensor,
                    synthetic_tensor,
                    data_range=resolved_range,
                    kernel_size=kernel,
                    sigma=max((kernel - 1) / 7.0, 1e-6),
                    betas=weights,
                )
                return float(value.item())
            except (ValueError, RuntimeError, AssertionError) as exc:
                errors.append(str(exc))
    detail = errors[-1]
    raise ValueError(f"MS-SSIM could not be computed for shape {real_array.shape}: {detail}")
