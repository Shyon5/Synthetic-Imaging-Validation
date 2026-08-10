"""Pairwise intensity and structural image-similarity metrics."""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Sequence

import numpy as np
from scipy.ndimage import convolve1d
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


def _canonicalize_ms_ssim_image(
    array: np.ndarray,
    channel_axis: Optional[int],
    batch_axis: Optional[int],
) -> np.ndarray:
    """Return float32 MS-SSIM input in ``[N, C, spatial...]`` order."""

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
    return np.ascontiguousarray(canonical, dtype=np.float32)


def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """Build the float32 Gaussian kernel used by the TorchMetrics reference."""

    positions = np.arange(
        start=(1 - kernel_size) / 2,
        stop=(1 + kernel_size) / 2,
        step=1,
        dtype=np.float32,
    )
    kernel = np.exp(-np.square(positions / np.float32(sigma)) / np.float32(2.0))
    return kernel / kernel.sum()


def _gaussian_filter_spatial(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a separable Gaussian filter over spatial axes only."""

    filtered = values
    for axis in range(2, values.ndim):
        filtered = convolve1d(filtered, kernel, axis=axis, mode="mirror")
    return filtered


def _average_pool_by_two(values: np.ndarray) -> np.ndarray:
    """Match stride-two 2D/3D average pooling, dropping odd trailing samples."""

    spatial_shape = values.shape[2:]
    cropped = values[
        (slice(None), slice(None))
        + tuple(slice(0, (size // 2) * 2) for size in spatial_shape)
    ]
    pooled_shape = list(cropped.shape[:2])
    for size in cropped.shape[2:]:
        pooled_shape.extend((size // 2, 2))
    reduction_axes = tuple(range(3, 2 * cropped.ndim - 2, 2))
    return cropped.reshape(pooled_shape).mean(axis=reduction_axes, dtype=np.float32)


def _validate_ms_ssim_candidate(shape: Sequence[int], scales: int, kernel_size: int) -> None:
    """Apply the reference size checks before evaluating one adaptive candidate."""

    if shape[-1] < 2**scales or shape[-2] < 2**scales:
        raise ValueError(
            f"For {scales} scales, image height and width must be at least {2**scales}."
        )
    scale_guard = max(1, scales - 1) ** 2
    if shape[-2] // scale_guard <= kernel_size - 1:
        raise ValueError("Image height is too small for the requested scales and kernel size.")
    if shape[-1] // scale_guard <= kernel_size - 1:
        raise ValueError("Image width is too small for the requested scales and kernel size.")

    padding = (kernel_size - 1) // 2
    current_shape = tuple(int(size) for size in shape[2:])
    for _ in range(scales):
        if any(size <= padding for size in current_shape):
            raise ValueError("A spatial dimension is too small for reflection padding.")
        current_shape = tuple(size // 2 for size in current_shape)


def _numpy_ms_ssim_candidate(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    data_range: float,
    kernel_size: int,
    weights: tuple[float, ...],
) -> float:
    """Compute one fixed-scale MS-SSIM candidate with NumPy and SciPy."""

    scales = len(weights)
    _validate_ms_ssim_candidate(real.shape, scales, kernel_size)
    sigma = max((kernel_size - 1) / 7.0, 1e-6)
    kernel = _gaussian_kernel_1d(kernel_size, sigma)
    padding = (kernel_size - 1) // 2
    c1 = np.float32((0.01 * data_range) ** 2)
    c2 = np.float32((0.03 * data_range) ** 2)
    scale_values = []

    for _ in range(scales):
        real_mean = _gaussian_filter_spatial(real, kernel)
        synthetic_mean = _gaussian_filter_spatial(synthetic, kernel)
        real_mean_sq = np.square(real_mean)
        synthetic_mean_sq = np.square(synthetic_mean)
        mean_product = real_mean * synthetic_mean

        real_variance = np.maximum(
            _gaussian_filter_spatial(np.square(real), kernel) - real_mean_sq,
            np.float32(0.0),
        )
        synthetic_variance = np.maximum(
            _gaussian_filter_spatial(np.square(synthetic), kernel) - synthetic_mean_sq,
            np.float32(0.0),
        )
        covariance = _gaussian_filter_spatial(real * synthetic, kernel) - mean_product
        contrast_numerator = np.float32(2.0) * covariance + c2
        contrast_denominator = real_variance + synthetic_variance + c2
        contrast = contrast_numerator / contrast_denominator
        similarity = (
            (np.float32(2.0) * mean_product + c1) * contrast_numerator
        ) / ((real_mean_sq + synthetic_mean_sq + c1) * contrast_denominator)

        interior = (slice(None), slice(None)) + tuple(
            slice(padding, -padding) for _ in real.shape[2:]
        )
        contrast = np.maximum(contrast[interior], np.float32(0.0))
        similarity = np.maximum(similarity, np.float32(0.0))
        scale_values.append(contrast.reshape(contrast.shape[0], -1).mean(axis=-1))
        final_similarity = similarity.reshape(similarity.shape[0], -1).mean(axis=-1)

        real = _average_pool_by_two(real)
        synthetic = _average_pool_by_two(synthetic)

    scale_values[-1] = final_similarity
    beta = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    per_image = np.prod(np.stack(scale_values, axis=0) ** beta, axis=0)
    return float(np.mean(per_image))


def _torchmetrics_ms_ssim_candidate(  # pragma: no cover - optional reference backend
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    data_range: float,
    kernel_size: int,
    weights: tuple[float, ...],
) -> float:
    """Compute one candidate with the optional TorchMetrics reference backend."""

    try:
        import torch
        from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
    except ImportError as exc:
        raise ImportError(
            "The 'torchmetrics' MS-SSIM backend requires torch and torchmetrics; "
            "install the 'torch' extra."
        ) from exc

    value = multiscale_structural_similarity_index_measure(
        torch.as_tensor(real, dtype=torch.float32),
        torch.as_tensor(synthetic, dtype=torch.float32),
        gaussian_kernel=True,
        sigma=max((kernel_size - 1) / 7.0, 1e-6),
        kernel_size=kernel_size,
        reduction="elementwise_mean",
        data_range=data_range,
        k1=0.01,
        k2=0.03,
        betas=weights,
        normalize="relu",
    )
    return float(value.item())


def ms_ssim(
    real: Any,
    synthetic: Any,
    *,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    batch_axis: Optional[int] = None,
    max_scales: int = 5,
    backend: Literal["numpy", "torchmetrics"] = "numpy",
) -> float:
    """Return adaptive multi-scale SSIM for a 2D image or 3D volume.

    The function tries the standard five-scale weights, reducing scale count
    and odd Gaussian kernel size for small inputs. The default ``numpy`` backend
    uses NumPy/SciPy and requires no optional dependency. ``torchmetrics`` keeps
    the previous implementation available as an optional numerical reference.
    Results from the two backends are expected to agree within floating-point
    tolerance rather than bit for bit. Output is normally in [0, 1], with 1
    optimal.
    """

    if backend not in ("numpy", "torchmetrics"):
        raise ValueError("backend must be 'numpy' or 'torchmetrics'.")

    real_array, synthetic_array = validate_pair(real, synthetic)
    resolved_range = infer_data_range(real_array, synthetic_array, data_range=data_range)
    real_image = _canonicalize_ms_ssim_image(real_array, channel_axis, batch_axis)
    synthetic_image = _canonicalize_ms_ssim_image(synthetic_array, channel_axis, batch_axis)
    scales_limit = max(1, min(int(max_scales), len(_MS_SSIM_WEIGHTS)))
    implementation = (
        _numpy_ms_ssim_candidate if backend == "numpy" else _torchmetrics_ms_ssim_candidate
    )
    errors = []
    for scales in range(scales_limit, 0, -1):
        weights = np.asarray(_MS_SSIM_WEIGHTS[:scales], dtype=np.float64)
        weights = tuple(float(v) for v in weights / weights.sum())
        for kernel in (11, 9, 7, 5, 3):
            try:
                return implementation(
                    real_image,
                    synthetic_image,
                    data_range=resolved_range,
                    kernel_size=kernel,
                    weights=weights,
                )
            except (ValueError, RuntimeError, AssertionError) as exc:
                errors.append(str(exc))
    detail = errors[-1]
    raise ValueError(f"MS-SSIM could not be computed for shape {real_array.shape}: {detail}")
