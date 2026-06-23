import numpy as np
import pytest

from synthetic_imaging_validation.metrics.distribution import (
    compare_distributions,
    histogram,
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


def test_histogram_and_combined_report():
    counts, edges = histogram([0, 0, 1, 1], bins=2, value_range=(0, 2))
    np.testing.assert_array_equal(counts, [2, 2])
    assert len(edges) == 3
    density, _ = histogram([0, 1], bins=2, value_range=(0, 2), density=True)
    assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least 2"):
        histogram([0, 1], bins=1)

    report = compare_distributions([0, 1, 2], [1, 2, 3], bins=4, percentiles=(50,))
    assert report["absolute_mean_difference"] == 1.0
    assert report["wasserstein"] == 1.0
    assert report["real"]["p50"] == 1.0


@pytest.mark.parametrize("percentiles", [(-1,), (101,), (np.nan,)])
def test_invalid_percentiles_are_rejected(percentiles):
    with pytest.raises(ValueError, match="percentiles"):
        intensity_statistics([0, 1], percentiles=percentiles)


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (kl_divergence, {"bins": 1}, "at least 2"),
        (kl_divergence, {"pseudocount": 0}, "pseudocount"),
        (kl_divergence, {"base": 1}, "base"),
        (jensen_shannon_divergence, {"base": -2}, "base"),
    ],
)
def test_histogram_divergence_parameters(function, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function([1, 1], [1, 1], **kwargs)


def test_constant_distributions_use_a_valid_shared_range():
    assert kl_divergence(np.ones(4), np.ones(4)) == pytest.approx(0.0)
    assert jensen_shannon_divergence(np.ones(4), np.ones(4)) == pytest.approx(0.0)
    assert kl_divergence([0, 1], [0, 1], value_range=(0, 2)) == pytest.approx(0.0)
