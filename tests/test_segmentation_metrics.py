import numpy as np
import pytest

from synthetic_imaging_validation.metrics.segmentation import (
    average_surface_distance,
    connected_component_statistics,
    dice,
    hausdorff_distance,
    iou,
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

