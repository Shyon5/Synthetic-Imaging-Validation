"""Intensity-distribution and descriptive-statistics metrics."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance

from ..utils.checks import to_numpy


def histogram(
    values: Any,
    *,
    bins: int = 64,
    value_range: Optional[tuple[float, float]] = None,
    density: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a flattened intensity histogram and bin edges."""

    array = to_numpy(values).astype(np.float64, copy=False).ravel()
    if int(bins) < 2:
        raise ValueError("bins must be at least 2.")
    return np.histogram(array, bins=int(bins), range=value_range, density=bool(density))


def intensity_statistics(
    values: Any,
    *,
    percentiles: Iterable[float] = (1, 5, 25, 50, 75, 95, 99),
) -> dict[str, float]:
    """Return count, mean, standard deviation, extrema, and selected percentiles."""

    array = to_numpy(values).astype(np.float64, copy=False).ravel()
    requested = tuple(float(q) for q in percentiles)
    if any(not np.isfinite(q) or q < 0.0 or q > 100.0 for q in requested):
        raise ValueError("percentiles must lie in [0, 100].")
    result = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    for q, value in zip(requested, np.percentile(array, requested)):
        result[f"p{q:g}"] = float(value)
    return result


def wasserstein_distance(real: Any, synthetic: Any) -> float:
    """Return exact empirical 1D Wasserstein-1 distance in input intensity units."""

    real_array = to_numpy(real, name="real").astype(np.float64, copy=False).ravel()
    synthetic_array = to_numpy(synthetic, name="synthetic").astype(np.float64, copy=False).ravel()
    return float(scipy_wasserstein_distance(real_array, synthetic_array))


def _shared_probabilities(
    real: Any,
    synthetic: Any,
    bins: int,
    value_range: Optional[tuple[float, float]],
    pseudocount: float,
) -> tuple[np.ndarray, np.ndarray]:
    real_array = to_numpy(real, name="real").astype(np.float64, copy=False).ravel()
    synthetic_array = to_numpy(synthetic, name="synthetic").astype(np.float64, copy=False).ravel()
    if int(bins) < 2:
        raise ValueError("bins must be at least 2.")
    if value_range is None:
        low = min(float(real_array.min()), float(synthetic_array.min()))
        high = max(float(real_array.max()), float(synthetic_array.max()))
        if high == low:
            high = low + 1.0
        value_range = (low, high)
    smooth = float(pseudocount)
    if not np.isfinite(smooth) or smooth <= 0.0:
        raise ValueError("pseudocount must be finite and strictly positive.")
    p, edges = np.histogram(real_array, bins=int(bins), range=value_range)
    q, _ = np.histogram(synthetic_array, bins=edges)
    p = p.astype(np.float64) + smooth
    q = q.astype(np.float64) + smooth
    return p / p.sum(), q / q.sum()


def kl_divergence(
    real: Any,
    synthetic: Any,
    *,
    bins: int = 64,
    value_range: Optional[tuple[float, float]] = None,
    pseudocount: float = 1e-12,
    base: float = 2.0,
) -> float:
    """Return histogram KL(real || synthetic), in bits by default.

    KL is asymmetric and sensitive to binning. A small pseudocount makes it
    finite when a bin occurs in only one sample.
    """

    p, q = _shared_probabilities(real, synthetic, bins, value_range, pseudocount)
    if not np.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive and different from 1.")
    return float(np.sum(p * np.log(p / q)) / np.log(base))


def jensen_shannon_divergence(
    real: Any,
    synthetic: Any,
    *,
    bins: int = 64,
    value_range: Optional[tuple[float, float]] = None,
    pseudocount: float = 1e-12,
    base: float = 2.0,
) -> float:
    """Return symmetric histogram Jensen-Shannon divergence, in [0, 1] for base 2."""

    p, q = _shared_probabilities(real, synthetic, bins, value_range, pseudocount)
    if not np.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive and different from 1.")
    midpoint = 0.5 * (p + q)
    scale = np.log(base)
    return float(0.5 * np.sum(p * np.log(p / midpoint)) / scale + 0.5 * np.sum(q * np.log(q / midpoint)) / scale)


def compare_distributions(
    real: Any,
    synthetic: Any,
    *,
    bins: int = 64,
    percentiles: Iterable[float] = (5, 25, 50, 75, 95),
) -> dict[str, Any]:
    """Return a compact descriptive and distribution-distance report."""

    real_stats = intensity_statistics(real, percentiles=percentiles)
    synthetic_stats = intensity_statistics(synthetic, percentiles=percentiles)
    return {
        "real": real_stats,
        "synthetic": synthetic_stats,
        "absolute_mean_difference": abs(real_stats["mean"] - synthetic_stats["mean"]),
        "wasserstein": wasserstein_distance(real, synthetic),
        "kl_real_synthetic": kl_divergence(real, synthetic, bins=bins),
        "jensen_shannon": jensen_shannon_divergence(real, synthetic, bins=bins),
    }

