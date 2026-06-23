import numpy as np
import pytest

from synthetic_imaging_validation.io.preprocessing import binarize, prepare_pair
from synthetic_imaging_validation.utils.checks import infer_data_range, to_numpy, validate_spacing
from synthetic_imaging_validation.utils.normalization import minmax_normalize, zscore_normalize
from synthetic_imaging_validation.utils.visualization import middle_slice, plot_histogram_comparison


class _BrokenArray:
    def __array__(self):
        raise RuntimeError("cannot convert")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (_BrokenArray(), "cannot be converted"),
        (["a", "b"], "numeric values"),
        (np.array([1 + 2j]), "real values"),
        (np.array([]), "must not be empty"),
        (np.array([np.nan]), "NaN or infinite"),
    ],
)
def test_to_numpy_rejects_unsupported_values(values, message):
    with pytest.raises((TypeError, ValueError), match=message):
        to_numpy(values)


def test_to_numpy_and_spacing_validation():
    values = np.arange(4)
    assert to_numpy(values) is values
    with pytest.raises(ValueError, match="at least 2 dimension"):
        to_numpy(values, min_ndim=2)

    assert validate_spacing(None, 2) == (1.0, 1.0)
    assert validate_spacing([0.5, 2], 2) == (0.5, 2.0)
    with pytest.raises(TypeError, match="iterable of numbers"):
        validate_spacing(object(), 2)
    with pytest.raises(ValueError, match="must have 2 values"):
        validate_spacing([1], 2)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_spacing([1, 0], 2)


def test_data_range_resolution():
    assert infer_data_range([2, 2]) == 1.0
    assert infer_data_range([0, 1], [-2, 3]) == 5.0
    assert infer_data_range([0, 1], data_range=10) == 10.0
    with pytest.raises(ValueError, match="strictly positive"):
        infer_data_range([0, 1], data_range=0)


def test_binarize_and_prepare_pair():
    values = np.array([-1.0, 0.5, 2.0])
    np.testing.assert_array_equal(binarize(values), [False, True, True])
    with pytest.raises(ValueError, match="threshold must be finite"):
        binarize(values, threshold=np.nan)

    real, synthetic = prepare_pair(values, values + 1, dtype=np.float32, clip=(0, 1))
    assert real.dtype == np.float32
    np.testing.assert_array_equal(real, [0, 0.5, 1])
    np.testing.assert_array_equal(synthetic, [0, 1, 1])
    unclipped_real, unclipped_synthetic = prepare_pair(values, values + 1)
    np.testing.assert_array_equal(unclipped_real, values)
    np.testing.assert_array_equal(unclipped_synthetic, values + 1)
    with pytest.raises(ValueError, match="finite.*high > low"):
        prepare_pair(values, values, clip=(1, 1))


def test_normalization_helpers():
    values = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(minmax_normalize(values, output_range=(-1, 1)), [-1, 0, 1])
    np.testing.assert_array_equal(minmax_normalize(np.ones(3)), np.zeros(3))
    with pytest.raises(ValueError, match="finite and increasing"):
        minmax_normalize(values, output_range=(1, 0))

    normalized = zscore_normalize(values)
    assert normalized.mean() == pytest.approx(0.0)
    assert normalized.std() == pytest.approx(1.0)
    np.testing.assert_array_equal(zscore_normalize(np.ones(3)), np.zeros(3))
    with pytest.raises(ValueError, match="ddof"):
        zscore_normalize(values, ddof=3)
    with pytest.raises(ValueError, match="ddof"):
        zscore_normalize(values, ddof=0.5)


def test_visualization_helpers():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    volume = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    np.testing.assert_array_equal(middle_slice(volume, axis=1), volume[:, 2, :])
    with pytest.raises(ValueError, match="must be 3D"):
        middle_slice(np.zeros((3, 3)))

    axis = plot_histogram_comparison([0, 1], [1, 2], bins=2)
    assert axis.get_xlabel() == "Intensity"
    assert len(axis.patches) == 4
    same_axis = plot_histogram_comparison([0, 1], [1, 2], bins=2, ax=axis)
    assert same_axis is axis
    axis.figure.clf()
