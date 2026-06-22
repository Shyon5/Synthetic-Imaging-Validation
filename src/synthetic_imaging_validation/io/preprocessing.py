"""Explicit preprocessing helpers used before metric calculation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..utils.checks import to_numpy, validate_pair


def binarize(values: Any, *, threshold: float = 0.5) -> np.ndarray:
    """Return a boolean mask using ``values >= threshold``."""

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return to_numpy(values, name="mask") >= float(threshold)


def prepare_pair(
    real: Any,
    synthetic: Any,
    *,
    dtype: np.dtype = np.float64,
    clip: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a pair to matching finite NumPy arrays with optional clipping."""

    real_array, synthetic_array = validate_pair(real, synthetic)
    real_array = real_array.astype(dtype, copy=False)
    synthetic_array = synthetic_array.astype(dtype, copy=False)
    if clip is not None:
        low, high = (float(clip[0]), float(clip[1]))
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("clip must be a finite (low, high) pair with high > low.")
        real_array = np.clip(real_array, low, high)
        synthetic_array = np.clip(synthetic_array, low, high)
    return real_array, synthetic_array

