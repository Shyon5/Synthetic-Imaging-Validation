"""Pipeline-independent spatial and morphology metrics for binary masks."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from ..utils.checks import to_numpy, validate_spacing
from .segmentation import connected_component_statistics, foreground_fraction


def _mask(values: Any, threshold: float, name: str = "mask") -> np.ndarray:
    array = to_numpy(values, name=name)
    if array.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D or 3D, got {array.shape}.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return array >= float(threshold)


def _widths(width: Any, ndim: int) -> tuple[int, ...]:
    if np.isscalar(width):
        values = (int(width),) * ndim
    else:
        values = tuple(int(v) for v in width)
    if len(values) != ndim or any(v < 0 for v in values):
        raise ValueError(f"border_width must contain {ndim} non-negative integers.")
    return values


def border_statistics(
    mask: Any,
    *,
    border_width: Any = 1,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return foreground occupancy within a fixed-width image-border region."""

    binary = _mask(mask, threshold)
    widths = _widths(border_width, binary.ndim)
    border = np.zeros(binary.shape, dtype=bool)
    for axis, width in enumerate(widths):
        width = min(width, binary.shape[axis])
        if width == 0:
            continue
        low = [slice(None)] * binary.ndim
        high = [slice(None)] * binary.ndim
        low[axis], high[axis] = slice(0, width), slice(binary.shape[axis] - width, None)
        border[tuple(low)] = True
        border[tuple(high)] = True
    active = int(binary.sum())
    border_active = int(np.count_nonzero(binary & border))
    element_name = "pixels" if binary.ndim == 2 else "voxels"
    result = {
        "spatial_dims": int(binary.ndim),
        "element_name": element_name,
        "border_foreground_elements": float(border_active),
        "border_foreground_ratio": float(border_active / active) if active else 0.0,
        "border_region_foreground_fraction": (
            float(border_active / border.sum()) if border.any() else 0.0
        ),
    }
    result[f"border_active_{element_name}"] = float(border_active)
    # Established names retained for callers while generic keys are preferred.
    result["border_active_ratio"] = result["border_foreground_ratio"]
    result["border_region_active_fraction"] = result["border_region_foreground_fraction"]
    return result


def distance_to_border_statistics(
    mask: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
) -> dict[str, float]:
    """Summarize foreground pixel/voxel distance to the nearest image border.

    Distances are measured from pixel/voxel centers in spacing units. Empty masks
    return NaN summaries because no foreground location exists.
    """

    binary = _mask(mask, threshold)
    checked_spacing = np.asarray(validate_spacing(spacing, binary.ndim), dtype=np.float64)
    coordinates = np.argwhere(binary).astype(np.float64)
    if coordinates.size == 0:
        return {"minimum": np.nan, "mean": np.nan, "median": np.nan, "p05": np.nan}
    upper = np.asarray(binary.shape, dtype=np.float64) - 1.0
    axis_distances = np.minimum(coordinates, upper - coordinates) * checked_spacing
    distances = np.min(axis_distances, axis=1)
    return {
        "minimum": float(np.min(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "p05": float(np.percentile(distances, 5.0)),
    }


def centroid_statistics(
    mask: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
) -> dict[str, Any]:
    """Return foreground centroid in index, normalized, and physical coordinates."""

    binary = _mask(mask, threshold)
    checked_spacing = np.asarray(validate_spacing(spacing, binary.ndim), dtype=np.float64)
    coordinates = np.argwhere(binary).astype(np.float64)
    if coordinates.size == 0:
        return {"centroid_index": None, "centroid_normalized": None, "centroid_physical": None}
    centroid = coordinates.mean(axis=0)
    denominator = np.maximum(np.asarray(binary.shape, dtype=np.float64) - 1.0, 1.0)
    return {
        "centroid_index": centroid.tolist(),
        "centroid_normalized": (centroid / denominator).tolist(),
        "centroid_physical": (centroid * checked_spacing).tolist(),
    }


def mask_spatial_report(
    mask: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    connectivity: Optional[int] = None,
    border_width: Any = 1,
) -> dict[str, Any]:
    """Combine pipeline-independent morphology and spatial statistics for one mask."""

    report = {
        "foreground_fraction": foreground_fraction(mask, threshold=threshold),
        "components": connected_component_statistics(
            mask, threshold=threshold, spacing=spacing, connectivity=connectivity
        ),
        "border": border_statistics(mask, border_width=border_width, threshold=threshold),
        "distance_to_border": distance_to_border_statistics(mask, threshold=threshold, spacing=spacing),
        "centroid": centroid_statistics(mask, threshold=threshold, spacing=spacing),
    }
    return report
