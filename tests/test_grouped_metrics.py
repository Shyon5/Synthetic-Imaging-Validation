import numpy as np
import pytest

from synthetic_imaging_validation.metrics.grouped import (
    distribution_metrics_by_class,
    paired_metrics_by_class,
)


def test_paired_metrics_are_calculated_per_sample_and_class():
    real = np.zeros((6, 8, 8), dtype=np.float32)
    synthetic = real.copy()
    synthetic[:3] += 0.1
    synthetic[3:] += 0.2
    labels = np.array(["a", "a", "a", "b", "b", "b"])

    report = paired_metrics_by_class(
        real,
        synthetic,
        labels,
        metrics=["mae", "mse"],
        classes=["a", "b", "missing"],
        class_names={"a": "class_a", "b": "class_b"},
    )

    assert report["class_a"]["count"] == 3
    assert report["class_a"]["metrics"]["mae"]["mean"] == pytest.approx(0.1)
    assert report["class_b"]["metrics"]["mse"]["mean"] == pytest.approx(0.04)
    assert report["missing"]["status"] == "insufficient_samples"
    assert report["missing"]["metrics"] == {}


def test_paired_metrics_accept_custom_callables_and_kwargs():
    real = np.zeros((2, 16, 16), dtype=np.float32)
    synthetic = real.copy()
    synthetic[:, 4:8, 4:8] = 1.0
    labels = [0, 0]

    report = paired_metrics_by_class(
        real,
        synthetic,
        labels,
        metrics={"maximum_error": lambda x, y: np.max(np.abs(x - y))},
    )
    assert report["0"]["metrics"]["maximum_error"]["mean"] == 1.0

    structural = paired_metrics_by_class(
        real,
        real,
        labels,
        metrics=["ssim", "psnr"],
        metric_kwargs={"ssim": {"data_range": 1.0}, "psnr": {"data_range": 1.0}},
    )
    assert structural["0"]["metrics"]["ssim"]["mean"] == pytest.approx(1.0)
    assert structural["0"]["metrics"]["psnr"]["positive_infinity_count"] == 2


def test_distribution_metrics_support_independent_class_counts():
    real = np.array([[0.0], [0.0], [2.0], [2.0], [2.0]])
    synthetic = np.array([[1.0], [1.0], [2.0], [2.0], [2.0], [2.0]])
    real_labels = ["a", "a", "b", "b", "b"]
    synthetic_labels = ["a", "a", "b", "b", "b", "b"]

    report = distribution_metrics_by_class(
        real,
        synthetic,
        real_labels,
        synthetic_labels,
        metrics=["wasserstein", "js"],
        min_samples=2,
    )
    assert report["a"]["n_real"] == 2
    assert report["a"]["n_synthetic"] == 2
    assert report["a"]["metrics"]["wasserstein"] == pytest.approx(1.0)
    assert report["b"]["metrics"]["wasserstein"] == 0.0


def test_feature_metrics_and_insufficient_classes_are_explicit():
    features = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [3.0, 2.0], [2.0, 3.0]]
    )
    labels = np.array(["a", "a", "a", "b", "b", "b"])
    report = distribution_metrics_by_class(
        features,
        features.copy(),
        labels,
        labels,
        metrics=["frechet", "feature_precision_recall", "sliced_wasserstein"],
        classes=["a", "b", "c"],
        min_samples=2,
        metric_kwargs={"sliced_wasserstein": {"num_projections": 8, "seed": 1}},
    )
    assert report["a"]["metrics"]["frechet"] == pytest.approx(0.0, abs=1e-8)
    assert report["a"]["metrics"]["feature_precision_recall"] == {
        "precision": 1.0,
        "recall": 1.0,
    }
    assert report["c"]["status"] == "insufficient_samples"


def test_labels_must_be_explicit_one_dimensional_categories():
    values = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="one-dimensional categorical"):
        paired_metrics_by_class(
            values,
            values,
            np.eye(3),
            metrics=["mae"],
        )
    with pytest.raises(ValueError, match="sample batch has 3"):
        paired_metrics_by_class(values, values, [0, 1], metrics=["mae"])

