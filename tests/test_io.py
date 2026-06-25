import numpy as np
import pytest

from synthetic_imaging_validation.io.loading import ImageData, is_supported_image_path, load_directory, load_image, load_pair
from synthetic_imaging_validation.io.pairing import (
    image_file_key,
    load_manifest_pairs,
    load_paired_directories,
    pair_directory_files,
)


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
    matching = ImageData(array.copy(), spacing=(1.0, 1.0), affine=np.eye(4))
    assert load_pair(real, matching)[1].affine is not None
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


def test_image_file_keys_and_supported_path_detection(tmp_path):
    nii = tmp_path / "case_001.nii.gz"
    npy = tmp_path / "case_002.npy"
    text = tmp_path / "notes.txt"
    nii.write_bytes(b"placeholder")
    np.save(npy, np.zeros((2, 2)))
    text.write_text("ignored", encoding="utf-8")

    assert image_file_key(nii) == "case_001"
    assert image_file_key(npy) == "case_002"
    assert image_file_key(text) == "notes"
    assert is_supported_image_path(npy)
    assert not is_supported_image_path(text)
    assert not is_supported_image_path(tmp_path)


def test_pair_directory_files_by_stem_and_sorted_order(tmp_path):
    real_dir = tmp_path / "real"
    synthetic_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synthetic_dir.mkdir()
    np.save(real_dir / "case_b.npy", np.full((2, 2), 2))
    np.save(real_dir / "case_a.npy", np.ones((2, 2)))
    np.save(synthetic_dir / "case_a.npy", np.ones((2, 2)))
    np.save(synthetic_dir / "case_b.npy", np.full((2, 2), 3))

    stem_pairs = pair_directory_files(real_dir, synthetic_dir)
    assert [key for key, _, _ in stem_pairs] == ["case_a", "case_b"]
    loaded = load_paired_directories(real_dir, synthetic_dir)
    assert [pair.key for pair in loaded] == ["case_a", "case_b"]
    assert loaded[1].synthetic.array.mean() == 3.0

    np.save(synthetic_dir / "zz_extra_name.npy", np.full((2, 2), 4))
    (synthetic_dir / "case_b.npy").unlink()
    sorted_pairs = pair_directory_files(real_dir, synthetic_dir, pairing="sorted")
    assert [key for key, _, _ in sorted_pairs] == ["case_a", "case_b"]
    assert sorted_pairs[1][2].name == "zz_extra_name.npy"


def test_directory_pairing_errors_are_explicit(tmp_path):
    real_dir = tmp_path / "real"
    synthetic_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synthetic_dir.mkdir()
    np.save(synthetic_dir / "case_a.npy", np.zeros((2, 2)))

    with pytest.raises(ValueError, match="No supported real files"):
        pair_directory_files(real_dir, synthetic_dir)
    (synthetic_dir / "case_a.npy").unlink()
    np.save(real_dir / "case_a.npy", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="No supported synthetic files"):
        pair_directory_files(real_dir, synthetic_dir)
    (real_dir / "case_a.npy").unlink()
    np.save(synthetic_dir / "case_a.npy", np.zeros((2, 2)))
    with pytest.raises(NotADirectoryError):
        pair_directory_files(tmp_path / "missing", synthetic_dir)
    with pytest.raises(ValueError, match="pairing must be one of"):
        pair_directory_files(real_dir, synthetic_dir, pairing="guess")

    np.save(real_dir / "case_b.npy", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Directory pairing mismatch"):
        pair_directory_files(real_dir, synthetic_dir)
    np.save(synthetic_dir / "case_c.npy", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="same number"):
        pair_directory_files(real_dir, synthetic_dir, pairing="sorted")

    np.savez(real_dir / "case_b.npz", image=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="Duplicate real key"):
        pair_directory_files(real_dir, synthetic_dir)


def test_directory_pairing_reports_one_sided_missing_keys(tmp_path):
    real_dir = tmp_path / "real"
    synthetic_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synthetic_dir.mkdir()
    np.save(synthetic_dir / "case_0.npy", np.zeros((2, 2)))
    for index in range(7):
        np.save(real_dir / f"case_{index}.npy", np.zeros((2, 2)))

    with pytest.raises(ValueError, match=r"missing synthetic.*6 total"):
        pair_directory_files(real_dir, synthetic_dir)

    for path in real_dir.glob("*.npy"):
        path.unlink()
    np.save(real_dir / "case_0.npy", np.zeros((2, 2)))
    np.save(synthetic_dir / "case_extra.npy", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="missing real for: case_extra"):
        pair_directory_files(real_dir, synthetic_dir)


def test_load_manifest_pairs_with_relative_paths_and_metadata(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "real.npy", np.ones((2, 2)))
    np.save(data_dir / "synthetic.npy", np.zeros((2, 2)))
    manifest = tmp_path / "pairs.csv"
    manifest.write_text(
        "case_id,real,synthetic,class\n"
        "case-1,data/real.npy,data/synthetic.npy,low\n",
        encoding="utf-8",
    )

    pairs = load_manifest_pairs(manifest, key_column="case_id")
    assert len(pairs) == 1
    assert pairs[0].key == "case-1"
    assert pairs[0].metadata == {"case_id": "case-1", "class": "low"}
    np.testing.assert_array_equal(pairs[0].real.array, np.ones((2, 2)))

    external_manifest = tmp_path / "external.csv"
    external_manifest.write_text("real,synthetic\nreal.npy,synthetic.npy\n", encoding="utf-8")
    inferred = load_manifest_pairs(external_manifest, base_dir=data_dir)
    assert inferred[0].key == "real"


def test_manifest_pairing_errors_include_row_context(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest_pairs(tmp_path / "missing.csv")

    empty_manifest = tmp_path / "empty.csv"
    empty_manifest.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="header row"):
        load_manifest_pairs(empty_manifest)

    missing_columns = tmp_path / "missing_columns.csv"
    missing_columns.write_text("real\nimage.npy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest_pairs(missing_columns)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.save(data_dir / "real.npy", np.zeros((2, 2)))
    np.save(data_dir / "synthetic.npy", np.zeros((2, 2)))

    empty_key = tmp_path / "empty_key.csv"
    empty_key.write_text("case_id,real,synthetic\n,data/real.npy,data/synthetic.npy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty key"):
        load_manifest_pairs(empty_key, key_column="case_id")

    blank_rows = tmp_path / "blank_rows.csv"
    blank_rows.write_text("real,synthetic\n,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        load_manifest_pairs(blank_rows)

    blank_then_valid = tmp_path / "blank_then_valid.csv"
    blank_then_valid.write_text(
        "real,synthetic\n,\ndata/real.npy,data/synthetic.npy\n",
        encoding="utf-8",
    )
    assert load_manifest_pairs(blank_then_valid)[0].key == "real"

    empty_path = tmp_path / "empty_path.csv"
    empty_path.write_text("real,synthetic\n,data/synthetic.npy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest row 2"):
        load_manifest_pairs(empty_path)

    missing_file = tmp_path / "missing_file.csv"
    missing_file.write_text("real,synthetic\ndata/missing.npy,data/synthetic.npy\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Manifest row 2"):
        load_manifest_pairs(missing_file)
