import numpy as np
import pytest

from synthetic_imaging_validation.metrics.grouped import (
    _normalise_label,
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


@pytest.mark.parametrize("label", [None, np.nan, np.inf, ["unhashable"]])
def test_invalid_class_labels_are_rejected(label):
    values = np.zeros((1, 2, 2))
    labels = np.empty(1, dtype=object)
    labels[0] = label
    with pytest.raises((TypeError, ValueError), match="Class labels"):
        paired_metrics_by_class(values, values, labels, metrics="mae")


def test_grouped_configuration_errors_are_clear():
    values = np.zeros((2, 3, 3))
    labels = [0, 1]
    with pytest.raises(ValueError, match="batch_axis"):
        paired_metrics_by_class(values, values, labels, metrics="mae", batch_axis=3)
    with pytest.raises(ValueError, match="Shape mismatch"):
        paired_metrics_by_class(values, np.zeros((3, 3, 3)), labels, metrics="mae")
    with pytest.raises(ValueError, match="At least one class"):
        paired_metrics_by_class(values, values, labels, metrics="mae", classes=[])
    with pytest.raises(ValueError, match="not unique"):
        paired_metrics_by_class(
            values, values, labels, metrics="mae", class_names={0: "same", 1: "same"}
        )
    with pytest.raises(ValueError, match="empty display name"):
        paired_metrics_by_class(values, values, labels, metrics="mae", class_names={0: ""})
    with pytest.raises(ValueError, match="Unknown metrics"):
        paired_metrics_by_class(values, values, labels, metrics=["unknown"])
    with pytest.raises(ValueError, match="At least one metric"):
        paired_metrics_by_class(values, values, labels, metrics=[])
    with pytest.raises(TypeError, match="Metric callables"):
        paired_metrics_by_class(values, values, labels, metrics={"bad": 1})
    with pytest.raises(TypeError, match="must be a mapping"):
        paired_metrics_by_class(values, values, labels, metrics="mae", metric_kwargs={"mae": 1})
    with pytest.raises(ValueError, match="min_samples"):
        paired_metrics_by_class(values, values, labels, metrics="mae", min_samples=0)


def test_custom_metric_output_and_failure_context():
    values = np.zeros((2, 3, 3))
    labels = [0, 0]
    with pytest.raises(TypeError, match="must return a scalar"):
        paired_metrics_by_class(values, values, labels, metrics={"bad": lambda x, y: True})

    def failing_metric(_real, _synthetic):
        raise RuntimeError("broken metric")

    with pytest.raises(ValueError, match="failed for class 0.*sample index 0"):
        paired_metrics_by_class(values, values, labels, metrics={"broken": failing_metric})

    report = distribution_metrics_by_class(
        values,
        values,
        labels,
        labels,
        metrics={"summary": lambda real, synthetic: {"difference": np.float64(0)}},
    )
    assert report["0"]["metrics"]["summary"] == {"difference": 0.0}


def test_distribution_group_errors_include_context():
    values = np.zeros((2, 2))
    labels = [0, 0]
    with pytest.raises(ValueError, match="min_samples"):
        distribution_metrics_by_class(values, values, labels, labels, metrics="wasserstein", min_samples=0)
    with pytest.raises(ValueError, match="failed for class 0"):
        distribution_metrics_by_class(
            values,
            values,
            labels,
            labels,
            metrics={"broken": lambda real, synthetic: (_ for _ in ()).throw(RuntimeError("bad"))},
        )


def test_builtin_mask_wrappers_and_numpy_scalar_labels():
    mask = np.zeros((1, 8, 8), dtype=np.uint8)
    mask[:, 2:6, 2:6] = 1
    report = paired_metrics_by_class(
        mask,
        mask,
        np.array([1]),
        metrics=["hausdorff95", "measure_ratio"],
    )
    assert report["1"]["metrics"]["hausdorff95"]["mean"] == 0.0
    assert report["1"]["metrics"]["measure_ratio"]["mean"] == 1.0
    assert _normalise_label(np.int64(4)) == 4
    with pytest.raises(ValueError, match="At least one metric"):
        paired_metrics_by_class(mask, mask, [1], metrics={})


@pytest.mark.torch
def test_torch_class_labels_are_supported():
    torch = pytest.importorskip("torch")
    values = np.zeros((2, 3, 3))
    report = paired_metrics_by_class(values, values, torch.tensor([0, 1]), metrics="mae")
    assert set(report) == {"0", "1"}
