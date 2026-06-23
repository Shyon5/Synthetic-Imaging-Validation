import numpy as np
import pytest

from synthetic_imaging_validation.metrics.generative import (
    feature_precision_recall,
    frechet_distance,
    kernel_inception_distance,
    rbf_mmd,
    sliced_wasserstein_distance,
)
from synthetic_imaging_validation.metrics.spatial import (
    border_statistics,
    centroid_statistics,
    distance_to_border_statistics,
    mask_spatial_report,
)


def test_pipeline_independent_spatial_statistics():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 2] = 1
    mask[4, 4] = 1
    border = border_statistics(mask, border_width=1)
    assert border["border_active_ratio"] == 0.5
    distances = distance_to_border_statistics(mask, spacing=(2.0, 1.0))
    assert distances["minimum"] == 0.0
    centroid = centroid_statistics(mask, spacing=(2.0, 1.0))
    assert centroid["centroid_index"] == [2.0, 3.0]
    report = mask_spatial_report(mask, spacing=(2.0, 1.0), border_width=1)
    assert set(report) == {
        "foreground_fraction",
        "components",
        "border",
        "distance_to_border",
        "centroid",
    }


def test_identical_feature_sets_have_near_optimal_distances():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(32, 8))
    assert frechet_distance(features, features) == pytest.approx(0.0, abs=1e-8)
    assert kernel_inception_distance(features, features) < 0.1
    assert sliced_wasserstein_distance(features, features, num_projections=16) == pytest.approx(0.0)


def test_feature_shape_and_non_finite_errors():
    with pytest.raises(ValueError, match="dimension mismatch"):
        frechet_distance(np.ones((3, 2)), np.ones((3, 4)))
    bad = np.ones((3, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        frechet_distance(bad, np.ones((3, 2)))


def test_empty_spatial_statistics_and_border_width_validation():
    empty = np.zeros((5, 5), dtype=np.uint8)
    distances = distance_to_border_statistics(empty)
    assert all(np.isnan(value) for value in distances.values())
    assert centroid_statistics(empty) == {
        "centroid_index": None,
        "centroid_normalized": None,
        "centroid_physical": None,
    }
    assert border_statistics(empty, border_width=0)["border_region_foreground_fraction"] == 0.0
    assert border_statistics(np.ones((2, 2)), border_width=10)["border_foreground_ratio"] == 1.0
    with pytest.raises(ValueError, match="non-negative integers"):
        border_statistics(empty, border_width=(1,))
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        border_statistics(np.zeros(4))
    with pytest.raises(ValueError, match="threshold must be finite"):
        border_statistics(empty, threshold=np.nan)


def test_feature_precision_recall_and_rbf_mmd():
    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    precision, recall = feature_precision_recall(features, features, k=99)
    assert precision == 1.0
    assert recall == 1.0
    assert rbf_mmd(features, features, sigma=1.0) <= 0.0
    assert np.isfinite(rbf_mmd(features, features))
    constant = np.ones((3, 2))
    assert rbf_mmd(constant, constant) == pytest.approx(0.0)


@pytest.mark.parametrize("sigma", [0, -1, np.nan])
def test_rbf_mmd_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        rbf_mmd(np.ones((2, 2)), np.ones((2, 2)), sigma=sigma)


def test_generative_parameter_and_sample_validation():
    with pytest.raises(ValueError, match="2D arrays"):
        frechet_distance(np.ones((2, 2, 2)), np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="At least 2 samples"):
        kernel_inception_distance(np.ones((1, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="covariance_epsilon"):
        frechet_distance(np.ones((2, 2)), np.ones((2, 2)), covariance_epsilon=-1)
    with pytest.raises(ValueError, match="at least 1"):
        sliced_wasserstein_distance(np.ones((2, 2)), np.ones((2, 2)), num_projections=0)


def test_sliced_wasserstein_supports_different_sample_counts():
    real = np.arange(6, dtype=float).reshape(3, 2)
    synthetic = np.arange(10, dtype=float).reshape(5, 2)
    first = sliced_wasserstein_distance(real, synthetic, num_projections=4, seed=7)
    second = sliced_wasserstein_distance(real, synthetic, num_projections=4, seed=7)
    assert first == second
    assert first > 0.0
