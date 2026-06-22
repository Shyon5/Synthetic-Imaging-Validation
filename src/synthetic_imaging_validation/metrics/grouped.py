"""Class-stratified evaluation without assumptions about a conditioning pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..utils.checks import to_numpy
from .distribution import jensen_shannon_divergence, kl_divergence, wasserstein_distance
from .generative import (
    feature_precision_recall,
    frechet_distance,
    kernel_inception_distance,
    rbf_mmd,
    sliced_wasserstein_distance,
)
from .image_similarity import mae, mse, ms_ssim, nrmse, psnr, rmse, ssim
from .segmentation import (
    average_surface_distance,
    dice,
    foreground_measure_ratio,
    hausdorff_distance,
    iou,
)

MetricCallable = Callable[..., Any]
MetricCollection = Union[str, Sequence[str], Mapping[str, MetricCallable]]


def _hd95(real: Any, synthetic: Any, **kwargs: Any) -> float:
    return hausdorff_distance(real, synthetic, percentile=95.0, **kwargs)


def _synthetic_to_real_measure_ratio(real: Any, synthetic: Any, **kwargs: Any) -> float:
    return foreground_measure_ratio(synthetic, real, **kwargs)


def _feature_pr(real: Any, synthetic: Any, **kwargs: Any) -> Dict[str, float]:
    precision, recall = feature_precision_recall(real, synthetic, **kwargs)
    return {"precision": precision, "recall": recall}


PAIRED_METRICS: Dict[str, MetricCallable] = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "nrmse": nrmse,
    "psnr": psnr,
    "ssim": ssim,
    "ms_ssim": ms_ssim,
    "dice": dice,
    "iou": iou,
    "hausdorff": hausdorff_distance,
    "hausdorff95": _hd95,
    "average_surface_distance": average_surface_distance,
    "measure_ratio": _synthetic_to_real_measure_ratio,
}


DISTRIBUTION_METRICS: Dict[str, MetricCallable] = {
    "wasserstein": wasserstein_distance,
    "kl": kl_divergence,
    "js": jensen_shannon_divergence,
    "frechet": frechet_distance,
    "kid": kernel_inception_distance,
    "feature_precision_recall": _feature_pr,
    "rbf_mmd": rbf_mmd,
    "sliced_wasserstein": sliced_wasserstein_distance,
}


def _as_batch(values: Any, *, name: str, batch_axis: int) -> np.ndarray:
    array = to_numpy(values, name=name)
    axis = int(batch_axis)
    if axis < -array.ndim or axis >= array.ndim:
        raise ValueError(f"batch_axis={batch_axis} is invalid for {name} shape {array.shape}.")
    return np.moveaxis(array, axis, 0)


def _normalise_label(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError("Class labels must not contain None.")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Class labels must not contain NaN or infinity.")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"Class labels must be scalar and hashable, got {value!r}.") from exc
    return value


def _as_labels(labels: Any, *, expected: int, name: str) -> np.ndarray:
    if hasattr(labels, "detach") and hasattr(labels, "cpu"):
        labels = labels.detach().cpu().numpy()
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional categorical label array. "
            "One-hot and multilabel arrays are not inferred automatically."
        )
    if len(array) != int(expected):
        raise ValueError(f"{name} has {len(array)} items but the sample batch has {expected}.")
    return np.asarray([_normalise_label(value) for value in array.tolist()], dtype=object)


def _ordered_classes(*label_arrays: np.ndarray, classes: Optional[Sequence[Any]]) -> list[Any]:
    values = classes if classes is not None else [value for labels in label_arrays for value in labels]
    ordered = []
    seen = set()
    for raw_value in values:
        value = _normalise_label(raw_value)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    if not ordered:
        raise ValueError("At least one class must be available or explicitly requested.")
    return ordered


def _class_keys(
    classes: Sequence[Any], class_names: Optional[Mapping[Any, str]]
) -> list[Tuple[str, Any]]:
    pairs = []
    seen_names = set()
    names = class_names or {}
    for value in classes:
        key = str(names.get(value, value))
        if not key:
            raise ValueError(f"Class {value!r} has an empty display name.")
        if key in seen_names:
            raise ValueError(f"Class display name '{key}' is not unique.")
        seen_names.add(key)
        pairs.append((key, value))
    return pairs


def _resolve_metrics(metrics: MetricCollection, registry: Mapping[str, MetricCallable]) -> Dict[str, MetricCallable]:
    if isinstance(metrics, Mapping):
        resolved = {str(name): function for name, function in metrics.items()}
        invalid = [name for name, function in resolved.items() if not callable(function)]
        if invalid:
            raise TypeError(f"Metric callables are required for: {invalid}.")
        if not resolved:
            raise ValueError("At least one metric must be requested.")
        return resolved

    names = [metrics] if isinstance(metrics, str) else list(metrics)
    if not names:
        raise ValueError("At least one metric must be requested.")
    unknown = [str(name) for name in names if str(name) not in registry]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}. Available metrics: {sorted(registry)}.")
    return {str(name): registry[str(name)] for name in names}


def _kwargs_for(metric_kwargs: Optional[Mapping[str, Mapping[str, Any]]], name: str) -> Dict[str, Any]:
    if metric_kwargs is None:
        return {}
    values = metric_kwargs.get(name, {})
    if not isinstance(values, Mapping):
        raise TypeError(f"metric_kwargs['{name}'] must be a mapping.")
    return dict(values)


def _scalar(value: Any, *, metric_name: str) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Metric '{metric_name}' must return a scalar, got {type(value).__name__}.")
    return float(value)


def _summarise(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "positive_infinity_count": int(np.isposinf(array).sum()),
        "negative_infinity_count": int(np.isneginf(array).sum()),
        "nan_count": int(np.isnan(array).sum()),
    }


def _normalise_output(value: Any, *, metric_name: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(name): _scalar(item, metric_name=f"{metric_name}.{name}")
            for name, item in value.items()
        }
    return _scalar(value, metric_name=metric_name)


def paired_metrics_by_class(
    real: Any,
    synthetic: Any,
    labels: Any,
    metrics: MetricCollection,
    *,
    classes: Optional[Sequence[Any]] = None,
    class_names: Optional[Mapping[Any, str]] = None,
    batch_axis: int = 0,
    min_samples: int = 1,
    metric_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Calculate per-sample paired metrics and summarize them by class.

    ``real`` and ``synthetic`` must contain aligned samples along ``batch_axis``.
    ``labels`` is a one-dimensional categorical array with one value per pair.
    Built-in metric names are listed in :data:`PAIRED_METRICS`; alternatively,
    pass ``{name: callable}``, where each callable receives one real sample and
    one synthetic sample. Metric-specific keyword arguments can be supplied as
    ``{metric_name: {argument: value}}``.

    Classes listed in ``classes`` are retained even when missing, making class
    coverage explicit. Metrics are calculated per sample before aggregation so
    results do not depend on batch composition. Non-finite metric outputs are
    counted and excluded from finite summary statistics.
    """

    real_batch = _as_batch(real, name="real", batch_axis=batch_axis)
    synthetic_batch = _as_batch(synthetic, name="synthetic", batch_axis=batch_axis)
    if real_batch.shape != synthetic_batch.shape:
        raise ValueError(
            f"Shape mismatch after moving batch_axis: real {real_batch.shape}, "
            f"synthetic {synthetic_batch.shape}."
        )
    label_array = _as_labels(labels, expected=len(real_batch), name="labels")
    requested_classes = _ordered_classes(label_array, classes=classes)
    functions = _resolve_metrics(metrics, PAIRED_METRICS)
    minimum = int(min_samples)
    if minimum < 1:
        raise ValueError("min_samples must be at least 1.")

    results: Dict[str, Dict[str, Any]] = {}
    for key, class_value in _class_keys(requested_classes, class_names):
        indices = np.flatnonzero(label_array == class_value)
        entry: Dict[str, Any] = {
            "class_value": class_value,
            "count": int(indices.size),
            "status": "ok" if indices.size >= minimum else "insufficient_samples",
            "metrics": {},
        }
        if indices.size >= minimum:
            for metric_name, function in functions.items():
                values = []
                kwargs = _kwargs_for(metric_kwargs, metric_name)
                for index in indices:
                    try:
                        value = function(real_batch[index], synthetic_batch[index], **kwargs)
                    except Exception as exc:
                        raise ValueError(
                            f"Metric '{metric_name}' failed for class {class_value!r}, "
                            f"sample index {int(index)}: {exc}"
                        ) from exc
                    values.append(_scalar(value, metric_name=metric_name))
                entry["metrics"][metric_name] = _summarise(values)
        results[key] = entry
    return results


def distribution_metrics_by_class(
    real: Any,
    synthetic: Any,
    real_labels: Any,
    synthetic_labels: Any,
    metrics: MetricCollection,
    *,
    classes: Optional[Sequence[Any]] = None,
    class_names: Optional[Mapping[Any, str]] = None,
    batch_axis: int = 0,
    min_samples: int = 2,
    metric_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare independent real and synthetic cohorts separately for each class.

    Real and synthetic cohorts may contain different numbers of samples and use
    separate one-dimensional label arrays. Built-in names are listed in
    :data:`DISTRIBUTION_METRICS`; custom callables receive the complete real and
    synthetic subgroup. Feature metrics require subgroup arrays shaped
    ``[samples, features]``. Intensity metrics flatten all remaining dimensions.

    A class is evaluated only when both cohorts meet ``min_samples``. The
    returned counts should always be reported because class imbalance and small
    subgroups can dominate uncertainty.
    """

    real_batch = _as_batch(real, name="real", batch_axis=batch_axis)
    synthetic_batch = _as_batch(synthetic, name="synthetic", batch_axis=batch_axis)
    real_label_array = _as_labels(real_labels, expected=len(real_batch), name="real_labels")
    synthetic_label_array = _as_labels(
        synthetic_labels, expected=len(synthetic_batch), name="synthetic_labels"
    )
    requested_classes = _ordered_classes(
        real_label_array, synthetic_label_array, classes=classes
    )
    functions = _resolve_metrics(metrics, DISTRIBUTION_METRICS)
    minimum = int(min_samples)
    if minimum < 1:
        raise ValueError("min_samples must be at least 1.")

    results: Dict[str, Dict[str, Any]] = {}
    for key, class_value in _class_keys(requested_classes, class_names):
        real_group = real_batch[real_label_array == class_value]
        synthetic_group = synthetic_batch[synthetic_label_array == class_value]
        enough = len(real_group) >= minimum and len(synthetic_group) >= minimum
        entry: Dict[str, Any] = {
            "class_value": class_value,
            "n_real": int(len(real_group)),
            "n_synthetic": int(len(synthetic_group)),
            "status": "ok" if enough else "insufficient_samples",
            "metrics": {},
        }
        if enough:
            for metric_name, function in functions.items():
                kwargs = _kwargs_for(metric_kwargs, metric_name)
                try:
                    value = function(real_group, synthetic_group, **kwargs)
                except Exception as exc:
                    raise ValueError(
                        f"Metric '{metric_name}' failed for class {class_value!r}: {exc}"
                    ) from exc
                entry["metrics"][metric_name] = _normalise_output(
                    value, metric_name=metric_name
                )
        results[key] = entry
    return results

