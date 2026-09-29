import numpy as np
import pytest

from synthetic_imaging_validation import intensity_distribution_score, similarity_score
from synthetic_imaging_validation.metrics import scores


@pytest.mark.parametrize("shape", [(32, 32), (16, 16, 16)])
def test_identical_scores_and_raw_metrics(shape):
    real = np.random.default_rng(4).uniform(0.2, 0.8, shape).astype(np.float32)
    original = real.copy()
    assert similarity_score(real, real, value_range=(0, 1)) == pytest.approx(100, abs=1e-4)
    assert intensity_distribution_score(real, real, value_range=(0, 1)) == pytest.approx(100)
    sim = similarity_score(real, real + 0.1, value_range=(0, 1), return_details=True)
    dist = intensity_distribution_score(real, real + 0.1, value_range=(0, 1), return_details=True)
    assert sim["value"] == pytest.approx(50 * (np.clip(sim["raw_metrics"]["ms_ssim"], 0, 1) + 0.9))
    assert dist["value"] == pytest.approx(50 * (1 - dist["raw_metrics"]["js"] + 1 - dist["raw_metrics"]["wasserstein"]))
    assert sim["protocol"]["weights"] == [0.5, 0.5]
    assert dist["protocol"]["value_range"] == [0, 1]
    assert 0 <= sim["value"] < 100 and 0 <= dist["value"] < 100
    np.testing.assert_array_equal(real, original)


def test_documented_similarity_formula_and_roundoff(monkeypatch):
    monkeypatch.setattr(scores, "ms_ssim", lambda *a, **k: 0.8)
    assert similarity_score(np.zeros((8, 8)), np.full((8, 8), 0.05), value_range=(0, 1)) == pytest.approx(87.5)
    monkeypatch.setattr(scores, "ms_ssim", lambda *a, **k: 1.00000001)
    assert similarity_score(np.zeros((8, 8)), np.zeros((8, 8)), value_range=(0, 1)) == 100
    monkeypatch.setattr(scores, "ms_ssim", lambda *a, **k: np.nan)
    with pytest.raises(ValueError, match="finite"):
        similarity_score(np.zeros((8, 8)), np.zeros((8, 8)), value_range=(0, 1))


def test_distributions_permutation_endpoints_and_unequal_sizes():
    image = np.tile(np.linspace(0, 1, 32), (32, 1))
    shuffled = image[:, ::-1]
    assert intensity_distribution_score(image, shuffled, value_range=(0, 1)) == pytest.approx(100)
    assert similarity_score(image, shuffled, value_range=(0, 1)) < 90
    assert intensity_distribution_score([0, 0], [1, 1, 1], value_range=(0, 1)) == pytest.approx(0, abs=1e-7)
    assert intensity_distribution_score([1, 1], [1, 1, 1], value_range=(0, 1)) == pytest.approx(100)
    # Float32 cannot represent 0.1 exactly: histogram endpoint round-off must
    # not silently discard all these samples.
    a = np.full(10, 0.1, dtype=np.float32)
    b = np.full(10, 0.1, dtype=np.float64)
    assert intensity_distribution_score(a, b, value_range=(0, 0.1)) == pytest.approx(100, abs=1e-5)


def test_worse_offset_reduces_scores_and_units_are_explicit():
    real = np.full((32, 32), 0.25)
    for function in (similarity_score, intensity_distribution_score):
        assert function(real, real + 0.05, value_range=(0, 1)) > function(real, real + 0.4, value_range=(0, 1))
    real = np.linspace(0, 1, 100)
    assert intensity_distribution_score(real, real ** 2, value_range=(0, 1)) == pytest.approx(
        intensity_distribution_score(real * 20, real ** 2 * 20, value_range=(0, 20)))


@pytest.mark.parametrize("bounds", [None, (), (0,), (1, 1), (1, 0), (0, np.inf), (np.nan, 1), (-1e308, 1e308)])
def test_invalid_range(bounds):
    for function in (similarity_score, intensity_distribution_score):
        with pytest.raises(ValueError, match="value_range"):
            function(np.zeros((8, 8)), np.zeros((8, 8)), value_range=bounds)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01, 1.01])
def test_nonfinite_and_out_of_range_rejected(bad):
    for function in (similarity_score, intensity_distribution_score):
        with pytest.raises(ValueError):
            function(np.full((8, 8), bad), np.zeros((8, 8)), value_range=(0, 1))
        with pytest.raises(ValueError):
            function(np.zeros((8, 8)), np.full((8, 8), bad), value_range=(0, 1))


def test_invalid_shapes_empty_bins_channels_and_tensor_conversion():
    for bins in (True, 1, 2.5):
        with pytest.raises(ValueError, match="bins"):
            intensity_distribution_score([0], [0], value_range=(0, 1), bins=bins)
    with pytest.raises(ValueError, match="Shape mismatch"):
        similarity_score(np.zeros((8, 8)), np.zeros((8, 9)), value_range=(0, 1))
    with pytest.raises(ValueError, match="empty"):
        intensity_distribution_score([], [0], value_range=(0, 1))
    channels = np.ones((3, 16, 16)) * 0.5
    assert similarity_score(channels, channels, value_range=(0, 1), channel_axis=0) == pytest.approx(100)

    class TensorLike:
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return channels

    assert similarity_score(TensorLike(), channels, value_range=(0, 1), channel_axis=0) == pytest.approx(100)


def test_grouped_score_support():
    from synthetic_imaging_validation import paired_metrics_by_class, distribution_metrics_by_class
    data = np.ones((2, 16, 16)) * 0.5
    kwargs = {name: {"value_range": (0, 1)} for name in ("similarity_score", "intensity_distribution_score")}
    report = paired_metrics_by_class(data, data, [0, 1], metrics=list(kwargs), metric_kwargs=kwargs)
    for label in ("0", "1"):
        assert report[label]["count"] == 1
        for name in kwargs:
            assert report[label]["metrics"][name]["mean"] == pytest.approx(100)
    report = distribution_metrics_by_class(data, data, [0, 0], [0, 0], metrics=["intensity_distribution_score"],
                                           metric_kwargs={"intensity_distribution_score": {"value_range": (0, 1)}})
    assert report["0"]["n_real"] == report["0"]["n_synthetic"] == 2
    assert report["0"]["metrics"]["intensity_distribution_score"] == pytest.approx(100)
