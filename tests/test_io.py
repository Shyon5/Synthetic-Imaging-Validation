import numpy as np
import pytest

from synthetic_imaging_validation.io.loading import load_image, load_pair


def test_load_npy_and_pair_validation(tmp_path):
    path = tmp_path / "image.npy"
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    np.save(path, values)
    loaded = load_image(path)
    np.testing.assert_array_equal(loaded.array, values)
    real, synthetic = load_pair(path, values)
    np.testing.assert_array_equal(real.array, synthetic.array)


def test_load_nifti_preserves_spacing_and_affine(tmp_path):
    nib = pytest.importorskip("nibabel")
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), affine), path)
    loaded = load_image(path)
    assert loaded.spacing == pytest.approx((1.0, 2.0, 3.0))
    np.testing.assert_allclose(loaded.affine, affine)

    path_2d = tmp_path / "image_2d.nii.gz"
    affine_2d = np.diag([0.7, 1.2, 1.0, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((12, 10), dtype=np.float32), affine_2d), path_2d)
    loaded_2d = load_image(path_2d)
    assert loaded_2d.array.shape == (12, 10)
    assert loaded_2d.spacing == pytest.approx((0.7, 1.2))


def test_non_finite_and_shape_mismatch_are_rejected():
    with pytest.raises(ValueError, match="NaN or infinite"):
        load_image(np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_pair(np.zeros((2, 2)), np.zeros((3, 3)))
