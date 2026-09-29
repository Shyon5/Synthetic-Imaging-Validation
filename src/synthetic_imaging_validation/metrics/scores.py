"""Experimental, fixed-protocol composite scores; not clinical quality ratings."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ..utils.checks import to_numpy, validate_pair
from .distribution import jensen_shannon_divergence, wasserstein_distance
from .image_similarity import mae, ms_ssim


SCORE_VERSION = "base-1"


def validate_score_range(value_range: Sequence[float]) -> tuple[float, float]:
    """Validate finite fixed intensity bounds with a positive, finite width."""
    bounds = np.asarray(value_range, dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all():
        raise ValueError("value_range must contain two finite bounds (low, high).")
    low, high = map(float, bounds)
    width = high - low
    if not np.isfinite(width) or width <= 0:
        raise ValueError("value_range must have a positive, finite width.")
    return low, high


def _bounded(values: Any, bounds: tuple[float, float], name: str) -> np.ndarray:
    array = to_numpy(values, name=name)
    # Permit representation error at an endpoint, not substantive clipping.
    eps = np.finfo(array.dtype).eps if array.dtype.kind == "f" else 0.0
    tolerance = 4 * eps * max(abs(bounds[0]), abs(bounds[1]), bounds[1] - bounds[0])
    if float(array.min()) < bounds[0] - tolerance or float(array.max()) > bounds[1] + tolerance:
        raise ValueError(f"{name} contains intensities outside value_range={bounds}; prepare inputs explicitly.")
    return array


def _report(components: dict[str, float], raw: dict[str, float], protocol: dict[str, Any], details: bool):
    checked = np.asarray(list(components.values()), dtype=np.float64)
    if not np.isfinite(checked).all():
        raise ValueError("Score components must be finite.")
    components = dict(zip(components, map(float, np.clip(checked, 0.0, 1.0))))
    value = float(100.0 * np.mean(list(components.values())))
    if details:
        return {"value": value, "components": components, "raw_metrics": raw, "protocol": protocol}
    return value


def similarity_score(
    real: Any,
    synthetic: Any,
    *,
    value_range: Sequence[float],
    channel_axis: Optional[int] = None,
    backend: str = "numpy",
    return_details: bool = False,
) -> Any:
    """Return the experimental 0--100 score for one aligned 2D/3D image pair.

    Formula: 100 * (clip(MS-SSIM, 0, 1) + clip(1 - MAE/R, 0, 1)) / 2,
    where R = high - low from the required fixed ``value_range``. Both arrays
    must lie within those bounds (up to dtype endpoint round-off). Inputs are
    never normalized or clipped. Identical inputs score 100, not necessarily
    implying clinical adequacy. No batch axis is accepted: evaluate each case
    separately and average its score. Channels may be explicit.

    Returns a float by default. ``return_details=True`` also returns normalized
    components, raw metrics and a versioned protocol mapping. PyTorch tensors
    are accepted through the standard array conversion; the default backend
    needs only NumPy/SciPy. A broad R and shared background can inflate scores.
    """
    bounds = validate_score_range(value_range)
    real_array, synthetic_array = validate_pair(_bounded(real, bounds, "real"), _bounded(synthetic, bounds, "synthetic"))
    width = bounds[1] - bounds[0]
    error = mae(real_array, synthetic_array)
    structural = ms_ssim(real_array, synthetic_array, data_range=width,
                         channel_axis=channel_axis, backend=backend)
    return _report(
        {"structure": structural, "intensity": 1.0 - error / width},
        {"mae": error, "ms_ssim": structural},
        {"version": SCORE_VERSION, "value_range": list(bounds), "weights": [0.5, 0.5],
         "component_order": ["structure", "intensity"], "backend": backend,
         "channel_axis": channel_axis, "max_scales": 5},
        return_details,
    )


def intensity_distribution_score(
    real: Any,
    synthetic: Any,
    *,
    value_range: Sequence[float],
    bins: int = 64,
    return_details: bool = False,
) -> Any:
    """Return experimental 0--100 agreement of flattened intensity distributions.

    Formula: 100 * ((1 - JS_base2) + (1 - Wasserstein1/R)) / 2. All normalized
    components are clipped to [0, 1] for numerical round-off. Histogram edges
    are fixed by ``bins`` and ``value_range``; out-of-range intensities raise
    an error rather than being dropped from the histogram. Pseudocount is
    1e-12. Endpoint dtype round-off is assigned to the first/last histogram bin.

    Input sizes may differ. No alignment is required; all dimensions are pooled,
    so this cannot detect spatial rearrangement. Cohort pooling is NOT the mean
    of per-case scores. For equal case weighting, call once per pair then take
    the mean. Mixed channels/units should be evaluated separately by the caller.
    ``return_details=True`` returns value, components, raw metrics and protocol.
    """
    bounds = validate_score_range(value_range)
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer of at least 2.")
    real_array = _bounded(real, bounds, "real")
    synthetic_array = _bounded(synthetic, bounds, "synthetic")
    # Only already-validated endpoint round-off can fall outside these bins.
    divergence = jensen_shannon_divergence(
        np.clip(real_array.astype(np.float64, copy=False), *bounds),
        np.clip(synthetic_array.astype(np.float64, copy=False), *bounds),
        bins=int(bins), value_range=bounds, base=2.0, pseudocount=1e-12,
    )
    distance = wasserstein_distance(real_array, synthetic_array)
    return _report(
        {"histogram": 1.0 - divergence, "transport": 1.0 - distance / (bounds[1] - bounds[0])},
        {"js": divergence, "wasserstein": distance},
        {"version": SCORE_VERSION, "value_range": list(bounds), "weights": [0.5, 0.5],
         "component_order": ["histogram", "transport"], "bins": int(bins),
         "js_base": 2.0, "pseudocount": 1e-12},
        return_details,
    )
