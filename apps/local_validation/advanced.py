"""Cohort and feature workflows using the same public metric API as Python clients."""
import csv
from pathlib import Path
import numpy as np
from synthetic_imaging_validation import load_image, distribution_metrics_by_class
from synthetic_imaging_validation.metrics.grouped import DISTRIBUTION_METRICS
from synthetic_imaging_validation.metrics.distribution import intensity_statistics
from .service import inside, catalogue

FEATURE_METRICS = ("frechet", "kid", "feature_precision_recall", "rbf_mmd", "sliced_wasserstein")
COHORT_METRICS = ("wasserstein", "js", "kl", "intensity_distribution_score")


def observed_range(pairs):
    """Scan full arrays one at a time. Do not estimate ranges from preview slices."""
    rows = []
    for pair in pairs:
        for role, path in (("real", pair.real), ("synthetic", pair.synthetic)):
            a = load_image(path).array
            rows.append({"case": pair.key, "role": role, "minimum": float(a.min()), "maximum": float(a.max())})
    if not rows:
        raise ValueError("Select at least one pair to inspect.")
    return min(r["minimum"] for r in rows), max(r["maximum"] for r in rows), rows


def read_labels(root, filename, column, expected):
    """Read one non-empty label per feature row, in the same order (no inferred join)."""
    with inside(root, filename).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Label CSV is missing column {column!r}.")
        labels = [row.get(column, "") for row in reader]
    if len(labels) != expected or any(not value or not value.strip() for value in labels):
        raise ValueError("Label CSV must have exactly one non-empty label per feature row.")
    return labels


def evaluate_features(root, real, synthetic, metrics, *, kwargs=None, labels=None):
    """Evaluate precomputed N x F matrices, not raw image tensors or an implicit encoder."""
    if not metrics or set(metrics) - set(FEATURE_METRICS):
        raise ValueError("Choose supported feature metrics.")
    a, b = load_image(inside(root, real)).array, load_image(inside(root, synthetic)).array
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Feature files must be N × F matrices with the same feature dimension.")
    # Core pairwise kernels allocate quadratic matrices; fail before allocating
    # an unexpectedly large cohort. Larger studies remain available through API.
    if set(metrics) & {"kid", "rbf_mmd", "feature_precision_recall"} and max(len(a), len(b)) > 5000:
        raise ValueError("Quadratic feature metrics in the UI are limited to 5,000 samples per domain. Use a documented subset or the API.")
    kwargs = kwargs or {}
    report = {name: DISTRIBUTION_METRICS[name](a, b, **kwargs.get(name, {})) for name in metrics}
    if labels:
        left = read_labels(root, labels[0], labels[2], len(a))
        right = read_labels(root, labels[1], labels[2], len(b))
        report["by_class"] = distribution_metrics_by_class(a, b, left, right, metrics=metrics, metric_kwargs=kwargs)
    report["sample_counts"] = {"real": len(a), "synthetic": len(b), "features": a.shape[1]}
    return report


def evaluate_cohorts(root, real, synthetic, metrics, *, bins=64, value_range=None, max_voxels=5_000_000):
    """Exact pooled-voxel comparison, with explicit memory cap and no silent subsampling.

    Larger images contribute more voxels; this is not the mean of per-case metrics.
    Arrays from each domain may have different shapes and case counts.
    """
    if not metrics or set(metrics) - set(COHORT_METRICS):
        raise ValueError("Choose supported cohort metrics.")
    domains, counts = [], []
    for folder in (real, synthetic):
        directory = inside(root, folder)
        files, _, _ = catalogue(directory)
        files = [name for name in files if len(Path(name).parts) == 1]
        if not files:
            raise ValueError("Each cohort folder must contain supported images (no recursive search).")
        arrays, size = [], 0
        for name in files:
            a = load_image(inside(root, str(directory / name))).array.ravel()
            size += a.size
            if size > max_voxels:
                raise ValueError(f"Cohort exceeds {max_voxels:,} voxels. Use a smaller explicit subset or the Python API; no automatic sampling is applied.")
            arrays.append(a)
        domains.append(np.concatenate(arrays))
        counts.append(len(files))
    a, b = domains
    if value_range is not None:
        low, high = value_range
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("Choose finite increasing intensity bounds.")
        if min(a.min(), b.min()) < low or max(a.max(), b.max()) > high:
            raise ValueError("Cohort intensities exceed the selected bounds; no values have been clipped.")
    result = {}
    for metric in metrics:
        kwargs = {} if metric == "wasserstein" else {"bins": bins, "value_range": value_range}
        result[metric] = DISTRIBUTION_METRICS[metric](a, b, **kwargs)
    result["intensity_statistics"] = {"real": intensity_statistics(a), "synthetic": intensity_statistics(b)}
    result["case_counts"] = dict(zip(("real", "synthetic"), counts))
    return result
