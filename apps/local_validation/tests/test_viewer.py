import nibabel as nib
import numpy as np
import pytest

from apps.local_validation.viewer import prepare_image, grids_match, extract_slice, slice_figure
from apps.local_validation.metric_labels import METRICS, metric_label, metric_help


def test_names_keep_identifiers_and_explain_direction():
    assert metric_label("ms_ssim").startswith("MS-SSIM")
    assert metric_label("similarity_score.value") == "Similarity Score / value"
    assert metric_label("custom_measure") == "Custom Measure"
    for key in METRICS:
        assert metric_help(key)
    assert "not inherently better" in metric_help("connected_components")


def test_numpy_2d_shared_window_and_unchanged_data(tmp_path):
    path = tmp_path / "image.npy"
    data = np.arange(20, dtype=float).reshape(4, 5)
    np.save(path, data)
    before = path.read_bytes()
    image = prepare_image(path)
    assert not image.anatomical
    figure = slice_figure(image, image, window=(3, 12), difference=True)
    assert len(figure.axes) == 3
    assert figure.axes[0].images[0].get_clim() == figure.axes[1].images[0].get_clim() == (3, 12)
    np.testing.assert_array_equal(figure.axes[2].images[0].get_array(), np.zeros_like(data))
    np.testing.assert_array_equal(image.array, data)
    assert path.read_bytes() == before
    assert "Array axis" in figure.axes[0].get_xlabel()
    assert figure.axes[0].images[0].origin == "upper"


def test_nifti_orientation_spacing_and_all_planes(tmp_path):
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    affine = np.diag([-2.0, -3.0, 4.0, 1.0])
    path = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    before = path.read_bytes()
    image = prepare_image(path)
    assert image.anatomical and not image.oblique
    assert image.metadata["Stored orientation"] == "LPS"
    np.testing.assert_array_equal(image.array, data[::-1, ::-1, :])
    assert nib.aff2axcodes(image.affine) == ("R", "A", "S")
    for axis, aspect in ((0, 4 / 3), (1, 4 / 2), (2, 3 / 2)):
        plane, actual, xlabel, ylabel = extract_slice(image, axis, 1)
        np.testing.assert_array_equal(plane, np.fliplr(np.take(image.array, 1, axis=axis).T))
        assert xlabel == ("A → P" if axis == 0 else "R → L")
        assert actual == pytest.approx(aspect)
        assert "→" in xlabel and "→" in ylabel
    assert path.read_bytes() == before
    assert slice_figure(image, image, index=1).axes[0].images[0].origin == "lower"


def test_permuted_axes_oblique_and_geometry_checks(tmp_path):
    data = np.ones((4, 5, 6), dtype=np.float32)
    affine = np.array([[0, 0, 2, 0], [3, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 1]], dtype=float)
    p = tmp_path / "permuted.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), p)
    image = prepare_image(p)
    assert image.array.shape == (6, 4, 5) and image.spacing == (2, 3, 4)
    affine[0, 0] = 0.5
    q = tmp_path / "oblique.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), q)
    other = prepare_image(q)
    assert other.oblique and not grids_match(image, other)
    with pytest.raises(ValueError, match="matching geometry"):
        slice_figure(image, other, difference=True)
    figure = slice_figure(image, other)
    assert len(figure.axes) == 2
    affine[0, 3] = 100
    nib.save(nib.Nifti1Image(data, affine), q)
    assert not grids_match(other, prepare_image(q))


def test_masks_threshold_equality_and_overlap_colours(tmp_path):
    r, s = tmp_path / "r.npy", tmp_path / "s.npy"
    np.save(r, [[0, 0.5], [0.0, 1.0]])
    np.save(s, [[0, 0.0], [0.5, 1.0]])
    a, b = prepare_image(r), prepare_image(s)
    figure = slice_figure(a, b, masks=True, threshold=0.5, difference=True)
    np.testing.assert_array_equal(figure.axes[2].images[0].get_array(), [[0, 1], [2, 3]])
    assert figure.axes[2].images[0].get_cmap().N == 4
    assert grids_match(a, b)


def test_channels_2d_nifti_and_bad_selections(tmp_path):
    path = tmp_path / "channels.npy"
    data = np.ones((2, 5, 6, 7))
    data[1] *= 2
    np.save(path, data)
    with pytest.raises(ValueError, match="channel axis"):
        prepare_image(path)
    image = prepare_image(path, channel_axis=0, channel=1)
    assert image.array.shape == (5, 6, 7) and image.array.min() == 2
    assert not image.anatomical
    for axis, channel in ((5, 0), (0, 2), (0, -1)):
        with pytest.raises(ValueError, match="outside"):
            prepare_image(path, axis, channel)
    four = tmp_path / "four.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 6, 7, 2)), np.eye(4)), four)
    assert prepare_image(four, -1, 1).anatomical
    with pytest.raises(ValueError, match="last axis"):
        prepare_image(four, 0, 0)
    two = tmp_path / "two.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 6)), np.eye(4)), two)
    assert not prepare_image(two).anatomical
    np.save(path, np.ones((5, 6)))
    assert not grids_match(prepare_image(two), prepare_image(path))
    assert not grids_match(image, prepare_image(path))
    with pytest.raises(ValueError, match="dimensionality"):
        slice_figure(image, prepare_image(path))


def test_invalid_window_slice_and_nonfinite_inputs(tmp_path):
    path = tmp_path / "image.npz"
    np.savez(path, data=np.zeros((3, 4, 5)))
    image = prepare_image(path)
    for axis, index in ((3, 0), (0, -1), (2, 5)):
        with pytest.raises(ValueError, match="slice"):
            extract_slice(image, axis, index)
    for bounds in ((1, 1), (2, 1), (np.nan, 1)):
        with pytest.raises(ValueError, match="bounds"):
            slice_figure(image, image, window=bounds)
    np.savez(path, data=np.full((5, 6), np.nan))
    with pytest.raises(ValueError, match="NaN"):
        prepare_image(path)
