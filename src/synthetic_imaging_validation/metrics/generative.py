"""Distribution metrics for user-supplied, comparable feature embeddings."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy import linalg

from ..utils.checks import to_numpy


def _feature_pair(real_features: Any, synthetic_features: Any, *, min_samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
    real = to_numpy(real_features, name="real_features", min_ndim=2).astype(np.float64, copy=False)
    synthetic = to_numpy(synthetic_features, name="synthetic_features", min_ndim=2).astype(np.float64, copy=False)
    if real.ndim != 2 or synthetic.ndim != 2:
        raise ValueError("Feature inputs must be 2D arrays shaped [samples, features].")
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError(f"Feature dimension mismatch: {real.shape[1]} vs {synthetic.shape[1]}.")
    if real.shape[0] < min_samples or synthetic.shape[0] < min_samples:
        raise ValueError(f"At least {min_samples} samples per domain are required.")
    return real, synthetic


def frechet_distance(real_features: Any, synthetic_features: Any, *, covariance_epsilon: float = 1e-6) -> float:
    """Return Fréchet distance between two empirical Gaussian feature distributions.

    This is FID-like only when the supplied embeddings come from a fixed,
    validated image encoder. It requires at least two samples per domain and
    has optimum 0.
    """

    real, synthetic = _feature_pair(real_features, synthetic_features, min_samples=2)
    epsilon = float(covariance_epsilon)
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("covariance_epsilon must be finite and non-negative.")
    mean_real, mean_synthetic = real.mean(axis=0), synthetic.mean(axis=0)
    cov_real = np.atleast_2d(np.cov(real, rowvar=False))
    cov_synthetic = np.atleast_2d(np.cov(synthetic, rowvar=False))
    identity = np.eye(real.shape[1], dtype=np.float64)
    cov_real = cov_real + identity * epsilon
    cov_synthetic = cov_synthetic + identity * epsilon
    covariance_mean = linalg.sqrtm(cov_real @ cov_synthetic)
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(covariance_mean.imag, 0.0, atol=1e-6):
            raise ValueError("Covariance square root contains significant imaginary values.")
        covariance_mean = covariance_mean.real
    difference = mean_real - mean_synthetic
    value = difference @ difference + np.trace(cov_real + cov_synthetic - 2.0 * covariance_mean)
    return float(max(float(value), 0.0))


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.sum(x * x, axis=1, keepdims=True) + np.sum(y * y, axis=1)[None, :] - 2.0 * (x @ y.T)
    return np.maximum(result, 0.0)


def kernel_inception_distance(real_features: Any, synthetic_features: Any) -> float:
    """Return unbiased polynomial-kernel MMD (KID); estimates may be slightly negative."""

    real, synthetic = _feature_pair(real_features, synthetic_features, min_samples=2)
    dimension = float(real.shape[1])
    kernel = lambda x, y: ((x @ y.T) / dimension + 1.0) ** 3
    k_rr, k_ss, k_rs = kernel(real, real), kernel(synthetic, synthetic), kernel(real, synthetic)
    n, m = real.shape[0], synthetic.shape[0]
    return float(
        (k_rr.sum() - np.trace(k_rr)) / (n * (n - 1))
        + (k_ss.sum() - np.trace(k_ss)) / (m * (m - 1))
        - 2.0 * k_rs.mean()
    )


def _neighbor_radii(features: np.ndarray, k: int) -> np.ndarray:
    effective_k = min(max(1, int(k)), features.shape[0] - 1)
    distances = _pairwise_squared_distances(features, features)
    np.fill_diagonal(distances, np.inf)
    return np.sqrt(np.partition(distances, effective_k - 1, axis=1)[:, effective_k - 1])


def _membership(query: np.ndarray, reference: np.ndarray, radii: np.ndarray) -> float:
    return float(np.mean((_pairwise_squared_distances(query, reference) <= np.square(radii)[None, :]).any(axis=1)))


def feature_precision_recall(real_features: Any, synthetic_features: Any, *, k: int = 3) -> tuple[float, float]:
    """Return manifold precision (fidelity) and recall (coverage), each in [0, 1]."""

    real, synthetic = _feature_pair(real_features, synthetic_features, min_samples=2)
    real_radii = _neighbor_radii(real, k)
    synthetic_radii = _neighbor_radii(synthetic, k)
    return _membership(synthetic, real, real_radii), _membership(real, synthetic, synthetic_radii)


def rbf_mmd(real_features: Any, synthetic_features: Any, *, sigma: Optional[float] = None) -> float:
    """Return unbiased squared MMD with an RBF kernel.

    When ``sigma`` is omitted, the median non-zero pairwise distance across
    both domains is used. Unbiased estimates may be slightly negative.
    """

    real, synthetic = _feature_pair(real_features, synthetic_features, min_samples=2)
    if sigma is None:
        combined = np.concatenate([real, synthetic], axis=0)
        values = _pairwise_squared_distances(combined, combined)
        upper = values[np.triu_indices(values.shape[0], k=1)]
        upper = upper[upper > 0.0]
        sigma_squared = float(np.median(upper)) if upper.size else 1.0
    else:
        sigma_value = float(sigma)
        if not np.isfinite(sigma_value) or sigma_value <= 0.0:
            raise ValueError("sigma must be finite and strictly positive.")
        sigma_squared = sigma_value**2
    sigma_squared = max(sigma_squared, 1e-12)
    k_rr = np.exp(-_pairwise_squared_distances(real, real) / (2.0 * sigma_squared))
    k_ss = np.exp(-_pairwise_squared_distances(synthetic, synthetic) / (2.0 * sigma_squared))
    k_rs = np.exp(-_pairwise_squared_distances(real, synthetic) / (2.0 * sigma_squared))
    n, m = real.shape[0], synthetic.shape[0]
    return float(
        (k_rr.sum() - np.trace(k_rr)) / (n * (n - 1))
        + (k_ss.sum() - np.trace(k_ss)) / (m * (m - 1))
        - 2.0 * k_rs.mean()
    )


def _quantile_resample(values: np.ndarray, size: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, int(size))
    return np.quantile(values, quantiles)


def sliced_wasserstein_distance(
    real_features: Any,
    synthetic_features: Any,
    *,
    num_projections: int = 128,
    seed: int = 42,
) -> float:
    """Return mean Wasserstein-1 distance over random unit feature projections."""

    real, synthetic = _feature_pair(real_features, synthetic_features)
    projections = int(num_projections)
    if projections < 1:
        raise ValueError("num_projections must be at least 1.")
    rng = np.random.default_rng(int(seed))
    directions = rng.normal(size=(projections, real.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    target_size = max(real.shape[0], synthetic.shape[0])
    distances = []
    for direction in directions:
        projected_real = _quantile_resample(real @ direction, target_size)
        projected_synthetic = _quantile_resample(synthetic @ direction, target_size)
        distances.append(float(np.mean(np.abs(projected_real - projected_synthetic))))
    return float(np.mean(distances))
