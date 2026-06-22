import json

import numpy as np
import pytest

from synthetic_imaging_validation.cli.validate import main
from synthetic_imaging_validation.metrics.image_similarity import ssim
from synthetic_imaging_validation.metrics.segmentation import (
    active_voxel_fraction,
    component_area_distribution,
    component_measure_distribution,
    component_volume_distribution,
    connected_component_statistics,
    foreground_fraction,
    foreground_measure_ratio,
    volume_ratio,
)


def test_2d_multichannel_and_batch_ssim():
    image = np.linspace(0, 1, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    assert ssim(image, image, data_range=1.0, channel_axis=-1) == pytest.approx(1.0)

    batch = np.stack([image[..., 0], image[..., 1]], axis=0)
    assert ssim(batch, batch, data_range=1.0, batch_axis=0) == pytest.approx(1.0)


def test_2d_component_reports_use_area_terms():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[7:10, 7:10] = 1
    stats = connected_component_statistics(mask, spacing=(0.5, 2.0), connectivity=4)

    assert stats["spatial_dims"] == 2
    assert stats["measure_name"] == "area"
    assert stats["element_name"] == "pixels"
    assert stats["component_areas"] == [4.0, 9.0]
    assert stats["total_area"] == 13.0
    np.testing.assert_array_equal(component_area_distribution(mask, spacing=(0.5, 2.0)), [4.0, 9.0])
    np.testing.assert_array_equal(component_measure_distribution(mask, spacing=(0.5, 2.0)), [4.0, 9.0])


def test_generic_measure_names_work_for_2d_and_3d():
    mask_2d = np.zeros((8, 8), dtype=np.uint8)
    mask_2d[2:6, 2:6] = 1
    mask_3d = np.zeros((8, 8, 8), dtype=np.uint8)
    mask_3d[2:6, 2:6, 2:6] = 1

    assert foreground_fraction(mask_2d) == active_voxel_fraction(mask_2d)
    assert foreground_measure_ratio(mask_2d, mask_2d) == volume_ratio(mask_2d, mask_2d)
    assert connected_component_statistics(mask_3d)["measure_name"] == "volume"
    np.testing.assert_array_equal(component_volume_distribution(mask_3d), [64.0])

    with pytest.raises(ValueError, match="expects a 3D mask"):
        component_volume_distribution(mask_2d)
    with pytest.raises(ValueError, match="expects a 2D mask"):
        component_area_distribution(mask_3d)


def test_cli_validates_2d_npy_pair(tmp_path):
    real = np.zeros((16, 16), dtype=np.float32)
    synthetic = real.copy()
    real[4:10, 4:10] = 1.0
    synthetic[5:11, 4:10] = 1.0
    real_path = tmp_path / "real.npy"
    synthetic_path = tmp_path / "synthetic.npy"
    output_path = tmp_path / "metrics.json"
    np.save(real_path, real)
    np.save(synthetic_path, synthetic)

    assert main(
        [
            "--real",
            str(real_path),
            "--synthetic",
            str(synthetic_path),
            "--metrics",
            "mae",
            "dice",
            "foreground_fraction",
            "measure_ratio",
            "connected_components",
            "--spacing",
            "0.8",
            "0.8",
            "--output",
            str(output_path),
        ]
    ) == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["connected_components"]["real"]["measure_name"] == "area"
    assert report["foreground_fraction"]["real"] == pytest.approx(36 / 256)
    assert report["measure_ratio"] == 1.0

