# Metrics by class

Class-wise evaluation is useful when a global average could hide different behavior across diagnoses, acquisition sites, severity groups, or other categorical strata. The grouping API is independent of model conditioning: labels are passed directly by the caller and are never inferred from model metadata.

Two functions cover the common cases.

## Paired metrics

Use `paired_metrics_by_class` when each synthetic sample has an aligned real target and one categorical label applies to the pair.

```python
from synthetic_imaging_validation import paired_metrics_by_class

report = paired_metrics_by_class(
    real_images,                 # [N, ...]
    synthetic_images,            # [N, ...]
    labels=["a", "a", "b", "b"],
    metrics=["mae", "ssim", "psnr"],
    classes=["a", "b", "c"],
    class_names={"a": "Group A", "b": "Group B", "c": "Group C"},
    metric_kwargs={
        "ssim": {"data_range": 1.0},
        "psnr": {"data_range": 1.0},
    },
)
```

Each metric is calculated for each pair before class aggregation. This matters for nonlinear metrics such as PSNR and prevents results from depending on batch composition.

Built-in paired metrics are:

- `mae`, `mse`, `rmse`, `nrmse`, `psnr`, `ssim`, `ms_ssim`;
- `dice`, `iou`, `hausdorff`, `hausdorff95`, `average_surface_distance`;
- `measure_ratio`, defined as synthetic foreground area/volume divided by the real value.

The output includes the observed count, finite-value summary, and counts of NaN or infinite results. Explicitly requested but absent classes are retained with `status="insufficient_samples"`.

## Distribution metrics

Use `distribution_metrics_by_class` for independent real and synthetic cohorts. The cohorts may contain different numbers of samples and therefore require separate label arrays.

```python
from synthetic_imaging_validation import distribution_metrics_by_class

report = distribution_metrics_by_class(
    real_features,
    synthetic_features,
    real_labels,
    synthetic_labels,
    metrics=["frechet", "kid", "feature_precision_recall"],
    classes=["a", "b"],
    min_samples=10,
)
```

Built-in distribution metrics are:

- intensity comparisons: `wasserstein`, `kl`, `js`;
- feature comparisons: `frechet`, `kid`, `feature_precision_recall`, `rbf_mmd`, `sliced_wasserstein`.

Intensity metrics flatten all dimensions after the sample axis. Feature metrics require subgroup arrays shaped `[samples, features]` and a shared encoder/preprocessing protocol.

## Custom metrics

Both functions accept a mapping instead of built-in names:

```python
custom = paired_metrics_by_class(
    real,
    synthetic,
    labels,
    metrics={"maximum_error": lambda x, y: abs(x - y).max()},
)
```

A paired callable receives one real and one synthetic sample and must return a scalar. A distribution callable receives the complete real and synthetic subgroup and may return either a scalar or a mapping of scalar values.

## Labels and edge cases

- Labels must be a one-dimensional categorical sequence of strings, integers, or other hashable scalar values.
- One-hot, multilabel, and continuous conditioning arrays are deliberately not interpreted. Convert them to explicit categorical labels or membership groups before calling the API.
- `classes` controls ordering and keeps expected but absent classes visible.
- `class_names` changes display keys without changing the underlying values.
- Report `count` or `n_real`/`n_synthetic` with every result. Class-wise feature metrics can be unreliable even when the global cohort is large.
- Avoid drawing conclusions from many class/metric combinations without accounting for multiplicity and subgroup imbalance.

