import numpy as np
import pytest

from synthetic_imaging_validation.metrics.segmentation import (
    average_surface_distance,
    connected_component_statistics,
    dice,
    hausdorff_distance,
    iou,
    surface_distance_statistics,
    surface_distances,
    volume_ratio,
)


def test_identical_and_empty_masks():
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    assert dice(mask, mask) == 1.0
    assert iou(mask, mask) == 1.0
    assert volume_ratio(mask, mask) == 1.0
    assert hausdorff_distance(mask, mask) == 0.0
    assert average_surface_distance(mask, mask) == 0.0


def test_one_empty_mask_semantics():
    empty = np.zeros((8, 8), dtype=np.uint8)
    nonempty = empty.copy()
    nonempty[3, 3] = 1
    assert dice(nonempty, empty) == 0.0
    assert iou(nonempty, empty) == 0.0
    assert np.isinf(volume_ratio(nonempty, empty))
    assert np.isinf(hausdorff_distance(nonempty, empty))


def test_surface_distance_uses_spacing():
    reference = np.zeros((9, 9), dtype=np.uint8)
    predicted = np.zeros_like(reference)
    reference[4, 4] = 1
    predicted[5, 4] = 1
    assert hausdorff_distance(predicted, reference, spacing=(2.0, 1.0)) == pytest.approx(2.0)


def test_connected_components_and_shape_error():
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[5:7, 5:7, 5:7] = 1
    stats = connected_component_statistics(mask, spacing=(1.0, 1.0, 2.0), connectivity=6)
    assert stats["component_count"] == 2
    assert stats["largest_component_voxels"] == 8
    assert stats["total_volume"] == 18.0
    with pytest.raises(ValueError, match="Shape mismatch"):
        dice(np.zeros((2, 2)), np.zeros((3, 3)))


def test_mask_input_and_connectivity_validation():
    mask = np.zeros((4, 4))
    with pytest.raises(ValueError, match="threshold must be finite"):
        dice(mask, mask, threshold=np.nan)
    with pytest.raises(ValueError, match="2D or 3D"):
        connected_component_statistics(np.zeros(4))
    with pytest.raises(ValueError, match="2D connectivity"):
        connected_component_statistics(mask, connectivity=3)
    with pytest.raises(ValueError, match="3D connectivity"):
        connected_component_statistics(np.zeros((3, 3, 3)), connectivity=4)


def test_empty_component_statistics_and_zero_prediction_ratio():
    empty = np.zeros((4, 4), dtype=np.uint8)
    stats = connected_component_statistics(empty, connectivity=None)
    assert stats["component_count"] == 0
    assert stats["component_measures"] == []
    assert stats["largest_component_measure"] == 0.0
    reference = empty.copy()
    reference[1, 1] = 1
    assert volume_ratio(empty, reference) == 0.0


def test_surface_distance_arrays_and_summary_statistics():
    reference = np.zeros((7, 7), dtype=np.uint8)
    predicted = np.zeros_like(reference)
    reference[3, 3] = 1
    predicted[4, 3] = 1
    forward, backward = surface_distances(predicted, reference, spacing=(2, 1))
    np.testing.assert_allclose(forward, [2.0])
    np.testing.assert_allclose(backward, [2.0])
    summary = surface_distance_statistics(predicted, reference, spacing=(2, 1))
    assert summary == {
        "average": 2.0,
        "median": 2.0,
        "hausdorff_95": 2.0,
        "hausdorff": 2.0,
    }

    empty = np.zeros_like(reference)
    assert surface_distance_statistics(empty, empty)["hausdorff"] == 0.0
    assert all(np.isinf(value) for value in surface_distance_statistics(predicted, empty).values())
    forward, backward = surface_distances(predicted, empty)
    assert np.isinf(forward).all() and np.isinf(backward).all()


@pytest.mark.parametrize("percentile", [0, -1, 101, np.nan])
def test_hausdorff_rejects_invalid_percentiles(percentile):
    mask = np.zeros((4, 4))
    with pytest.raises(ValueError, match="percentile"):
        hausdorff_distance(mask, mask, percentile=percentile)
