import numpy as np
import pytest

from synthetic_imaging_validation.io.loading import ImageData, load_directory, load_image, load_pair


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


def test_npz_key_selection_and_path_errors(tmp_path):
    single = tmp_path / "single.npz"
    multiple = tmp_path / "multiple.npz"
    np.savez(single, image=np.arange(3))
    np.savez(multiple, first=np.arange(2), second=np.arange(3))
    np.testing.assert_array_equal(load_image(single).array, np.arange(3))
    np.testing.assert_array_equal(load_image(multiple, npz_key="second").array, np.arange(3))

    with pytest.raises(ValueError, match="contains 2 arrays"):
        load_image(multiple)
    with pytest.raises(KeyError, match="missing"):
        load_image(multiple, npz_key="missing")
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.npy")
    with pytest.raises(ValueError, match="Expected a file"):
        load_image(tmp_path)
    unsupported = tmp_path / "image.txt"
    unsupported.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input format"):
        load_image(unsupported)


def test_image_data_spacing_and_spatial_pair_validation():
    array = np.zeros((3, 4), dtype=np.float32)
    data = load_image(array, spacing=(0.5, 2.0))
    assert data.spacing == (0.5, 2.0)
    assert load_image(data) is data
    with pytest.raises(ValueError, match="do not provide it twice"):
        load_image(data, spacing=(1, 1))

    real = ImageData(array, spacing=(1.0, 1.0), affine=np.eye(4))
    different_spacing = ImageData(array.copy(), spacing=(1.0, 2.0), affine=np.eye(4))
    with pytest.raises(ValueError, match="Spacing mismatch"):
        load_pair(real, different_spacing)
    different_affine = ImageData(array.copy(), spacing=(1.0, 1.0), affine=np.diag([2, 1, 1, 1]))
    with pytest.raises(ValueError, match="affine mismatch"):
        load_pair(real, different_affine)

    first, second = load_pair(real, different_affine, require_spatial_match=False)
    assert first.affine is not None and second.affine is not None
    unequal = load_pair(array, np.zeros((2, 2)), require_same_shape=False)
    assert unequal[0].array.shape != unequal[1].array.shape


def test_load_directory_is_sorted_and_optionally_recursive(tmp_path):
    np.save(tmp_path / "b.npy", np.ones((2, 2)))
    np.save(tmp_path / "A.npy", np.zeros((2, 2)))
    (tmp_path / "ignore.txt").write_text("ignored", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    np.save(nested / "c.npy", np.full((2, 2), 2))

    top_level = load_directory(tmp_path)
    assert [item.path.name for item in top_level] == ["A.npy", "b.npy"]
    recursive = load_directory(tmp_path, recursive=True)
    assert [item.path.name for item in recursive] == ["A.npy", "b.npy", "c.npy"]
    with pytest.raises(NotADirectoryError):
        load_directory(tmp_path / "missing")
