"""Optional plotting helpers; matplotlib is imported only when requested."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .checks import to_numpy


def middle_slice(volume: Any, *, axis: int = -1) -> np.ndarray:
    """Return the central 2D slice of a 3D scalar volume."""

    array = to_numpy(volume, name="volume")
    if array.ndim != 3:
        raise ValueError(f"volume must be 3D, got {array.shape}.")
    resolved_axis = int(axis) % 3
    return np.take(array, array.shape[resolved_axis] // 2, axis=resolved_axis)


def plot_histogram_comparison(
    real: Any,
    synthetic: Any,
    *,
    bins: int = 64,
    value_range: Optional[tuple[float, float]] = None,
    ax=None,
):
    """Plot normalized real and synthetic intensity histograms on one axis."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib; install the 'viz' extra.") from exc
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(to_numpy(real).ravel(), bins=bins, range=value_range, density=True, alpha=0.5, label="Real")
    ax.hist(to_numpy(synthetic).ravel(), bins=bins, range=value_range, density=True, alpha=0.5, label="Synthetic")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    return ax

