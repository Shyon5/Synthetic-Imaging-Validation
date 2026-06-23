"""Small, explicit intensity-normalization utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from .checks import to_numpy


def minmax_normalize(values: Any, *, output_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Linearly map finite values to ``output_range``; constants map to its lower bound."""

    array = to_numpy(values).astype(np.float64, copy=False)
    low, high = map(float, output_range)
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("output_range must be finite and increasing.")
    source_min, source_max = float(array.min()), float(array.max())
    if source_max == source_min:
        return np.full_like(array, low, dtype=np.float64)
    return (array - source_min) / (source_max - source_min) * (high - low) + low


def zscore_normalize(values: Any, *, ddof: int = 0) -> np.ndarray:
    """Return zero-mean, unit-variance values; constants map to zeros."""

    array = to_numpy(values).astype(np.float64, copy=False)
    resolved_ddof = int(ddof)
    if resolved_ddof != ddof or resolved_ddof < 0 or resolved_ddof >= array.size:
        raise ValueError("ddof must be an integer in [0, number of elements).")
    std = float(np.std(array, ddof=resolved_ddof))
    if std == 0.0:
        return np.zeros_like(array, dtype=np.float64)
    return (array - float(np.mean(array))) / std
