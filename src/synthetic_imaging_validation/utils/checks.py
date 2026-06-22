"""Input validation shared by all metric modules."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np


def to_numpy(values: Any, *, name: str = "input", min_ndim: int = 1) -> np.ndarray:
    """Convert an array-like or PyTorch tensor to a finite NumPy array.

    The returned array shares memory when possible. Complex, object, empty,
    NaN, and infinite inputs are rejected rather than silently repaired.
    """

    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().numpy()
    try:
        array = np.asarray(values)
    except Exception as exc:
        raise TypeError(f"{name} cannot be converted to a NumPy array.") from exc
    if array.dtype.kind in {"O", "U", "S", "V"}:
        raise TypeError(f"{name} must contain numeric values, got dtype {array.dtype}.")
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must contain real values.")
    if array.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimension(s), got {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def validate_pair(real: Any, synthetic: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return two finite arrays after enforcing identical shapes."""

    real_array = to_numpy(real, name="real")
    synthetic_array = to_numpy(synthetic, name="synthetic")
    if real_array.shape != synthetic_array.shape:
        raise ValueError(f"Shape mismatch: real {real_array.shape}, synthetic {synthetic_array.shape}.")
    return real_array, synthetic_array


def validate_spacing(spacing: Optional[Iterable[float]], ndim: int) -> tuple[float, ...]:
    """Validate one positive finite spacing value per spatial array axis."""

    if spacing is None:
        return (1.0,) * int(ndim)
    try:
        values = tuple(float(v) for v in spacing)
    except Exception as exc:
        raise TypeError("spacing must be an iterable of numbers.") from exc
    if len(values) != int(ndim):
        raise ValueError(f"spacing must have {ndim} values, got {len(values)}.")
    if not np.isfinite(values).all() or any(v <= 0.0 for v in values):
        raise ValueError("spacing values must be finite and strictly positive.")
    return values


def infer_data_range(real: Any, synthetic: Any = None, *, data_range: Optional[float] = None) -> float:
    """Resolve a positive intensity range for PSNR and structural metrics."""

    if data_range is not None:
        value = float(data_range)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("data_range must be finite and strictly positive.")
        return value
    arrays = [to_numpy(real, name="real")]
    if synthetic is not None:
        arrays.append(to_numpy(synthetic, name="synthetic"))
    low = min(float(np.min(a)) for a in arrays)
    high = max(float(np.max(a)) for a in arrays)
    return high - low if high > low else 1.0

