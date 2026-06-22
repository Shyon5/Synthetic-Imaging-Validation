import numpy as np
import pytest

from synthetic_imaging_validation.metrics.distribution import (
    intensity_statistics,
    jensen_shannon_divergence,
    kl_divergence,
    wasserstein_distance,
)


def test_identical_distributions_have_zero_distance():
    values = np.arange(100, dtype=np.float64)
    assert wasserstein_distance(values, values) == 0.0
    assert kl_divergence(values, values) == pytest.approx(0.0)
    assert jensen_shannon_divergence(values, values) == pytest.approx(0.0)


def test_shifted_distribution_has_expected_wasserstein_distance():
    values = np.arange(100, dtype=np.float64)
    assert wasserstein_distance(values, values + 2.0) == pytest.approx(2.0)
    assert jensen_shannon_divergence(values, values + 2.0) > 0.0


def test_statistics_and_non_finite_rejection():
    stats = intensity_statistics(np.array([0.0, 1.0, 2.0]), percentiles=(50,))
    assert stats["mean"] == 1.0
    assert stats["p50"] == 1.0
    with pytest.raises(ValueError, match="NaN or infinite"):
        wasserstein_distance(np.array([0.0, np.inf]), np.array([0.0]))

