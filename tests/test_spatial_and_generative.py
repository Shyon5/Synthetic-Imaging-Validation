import numpy as np
import pytest

from synthetic_imaging_validation.metrics.generative import (
    frechet_distance,
    kernel_inception_distance,
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
