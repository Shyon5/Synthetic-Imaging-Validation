# Metric reference

This page defines the metrics and their outputs. For task-specific recommendations, failure modes, and sample-size guidance, see [Choosing validation metrics](metric_selection.md).

All applicable metrics can be stratified by categorical labels through the APIs described in [Metrics by class](grouped_metrics.md).

## General input contract

Metric functions accept NumPy arrays and PyTorch tensors. File paths are first loaded with `load_image` or `load_pair`. Numeric inputs must be non-empty, real-valued, and finite. Pairwise shapes must match. No function silently registers, resamples, normalizes, clips, or changes axis order.

For NIfTI data, `ImageData.spacing` follows the loaded array axes. Supply that tuple to physical-distance metrics. Use an explicit `data_range` for PSNR/SSIM whenever a modality or preprocessing protocol defines one.

## Image similarity

| Metric | Meaning and output | Inputs and limitations |
| --- | --- | --- |
| `mae` | Mean absolute error; input intensity units; 0 is optimal. | Aligned equal-shape arrays. Sensitive to intensity calibration. |
| `mse` | Mean squared error; squared intensity units; 0 is optimal. | Penalizes large errors more strongly than MAE. |
| `rmse` | Square root of MSE; input intensity units; 0 is optimal. | Same alignment requirement as MSE. |
| `nrmse` | RMSE divided by real-image range, absolute mean, or RMS; 0 is optimal. | The normalization convention must be reported. Zero reference denominator can produce infinity. |
| `psnr` | Logarithmic signal-to-error ratio in dB; larger is better; identical inputs give infinity. | Requires a scientifically meaningful `data_range`; inferred observed range is convenient but less comparable across cohorts. |
| `ssim` | Local luminance/contrast/structure similarity, usually [-1, 1]; 1 is optimal. | 2D/3D spatial input. Declare channel and batch axes. Window size must fit the smallest dimension. |
| `ms_ssim` | Multi-scale SSIM, normally [0, 1]; 1 is optimal. | Optional torch/torchmetrics dependency. Adaptively reduces scales/kernel for smaller 2D/3D inputs; report effective preprocessing and range. |

```python
from synthetic_imaging_validation.metrics.image_similarity import mae, nrmse, psnr, ssim

results = {
    "mae": mae(real, synthetic),
    "nrmse": nrmse(real, synthetic, normalization="range"),
    "psnr": psnr(real, synthetic, data_range=1.0),
    "ssim": ssim(real, synthetic, data_range=1.0),
}
```

## Intensity distributions

| Metric | Meaning and output | Inputs and limitations |
| --- | --- | --- |
| `histogram` | Counts or densities and shared bin edges. | Flattens all axes. Bin count/range strongly affect interpretation. |
| `intensity_statistics` | Count, mean, standard deviation, extrema, and configurable percentiles. | Descriptive only; does not test significance. |
| `wasserstein_distance` | Exact empirical 1D Wasserstein-1 distance; input intensity units; 0 is optimal. | Flattens voxels and ignores spatial location. Cohort sizes may differ. |
| `kl_divergence` | Histogram KL(real‖synthetic), bits by default; 0 is optimal. | Asymmetric and bin-sensitive. A documented pseudocount prevents infinite estimates. |
| `jensen_shannon_divergence` | Symmetric histogram divergence; [0, 1] in base 2; 0 is optimal. | Still depends on common binning and intensity range. |
| `compare_distributions` | Descriptive statistics plus mean difference, Wasserstein, KL, and JS. | A convenience report, not a statistical hypothesis test. |

```python
from synthetic_imaging_validation.metrics.distribution import compare_distributions

report = compare_distributions(real_cohort, synthetic_cohort, bins=128)
```

## Masks and segmentations

Masks are scalar 2D/3D arrays binarized with `value >= threshold`. Dice and IoU define two empty masks as a perfect match. Surface metrics define two empty masks as distance 0 and a one-empty comparison as infinity.

| Metric | Meaning and output | Inputs and limitations |
| --- | --- | --- |
| `dice` | Twice intersection over summed foreground; [0, 1]. | Requires voxel-wise alignment. |
| `iou` | Intersection over union; [0, 1]. | Requires voxel-wise alignment. |
| `foreground_fraction` | Foreground pixel/voxel fraction; [0, 1]. | Describes one mask and ignores spacing. `active_voxel_fraction` is a compatibility alias. |
| `foreground_measure_ratio` | Predicted/reference area ratio in 2D or volume ratio in 3D; 1 is matched. | Assumes the same grid/spacing; can be infinite if only the prediction is non-empty. `volume_ratio` is a compatibility alias. |
| `connected_component_statistics` | Count, foreground load, component sizes, physical areas/volumes, and largest-component values. | Uses 2D area keys or 3D volume keys. Choose connectivity explicitly when comparisons must follow a protocol. |
| `component_measure_distribution` | One physical area (2D) or volume (3D) per component. | `component_area_distribution` and `component_volume_distribution` enforce a specific dimension. Empty masks return an empty array. |
| `hausdorff_distance` | Maximum symmetric contour/surface distance, or HD percentile; spacing units; 0 is optimal. | Highly sensitive to outliers at percentile 100. Use HD95 when appropriate and report it. |
| `average_surface_distance` | Mean of both directed contour/surface-distance means; spacing units; 0 is optimal. | Requires aligned masks and correct spacing. |
| `surface_distance_statistics` | ASSD, symmetric median, HD95, and maximum Hausdorff. | Surface definition depends on chosen connectivity. |

```python
from synthetic_imaging_validation.metrics.segmentation import (
    connected_component_statistics,
    dice,
    foreground_measure_ratio,
    surface_distance_statistics,
)

print(dice(prediction, reference))
print(foreground_measure_ratio(prediction, reference))
print(connected_component_statistics(prediction, spacing=(0.8, 0.8)))  # 2D area
print(surface_distance_statistics(prediction, reference, spacing=(0.8, 0.8, 2.0)))
```

## Pipeline-independent spatial mask metrics

| Metric | Meaning and output | Inputs and limitations |
| --- | --- | --- |
| `border_statistics` | Foreground count/ratio in an image-border band and band occupancy. | Border width is in pixels/voxels, not physical units. Cropping policy affects the result. |
| `distance_to_border_statistics` | Minimum/mean/median/5th-percentile foreground distance to the image boundary. | Uses spacing units; empty masks return NaN because location is undefined. |
| `centroid_statistics` | Foreground centroid in index, normalized, and physical coordinates. | One global centroid can hide multi-lesion structure. Empty masks return `None`. |
| `mask_spatial_report` | Combined foreground, connected-component, border, distance, and centroid report. | Descriptive; orientation, spacing, field of view, threshold, and connectivity must be standardized. |

```python
from synthetic_imaging_validation.metrics.spatial import mask_spatial_report

report = mask_spatial_report(
    mask,
    spacing=(1.0, 1.0, 2.5),
    border_width=(4, 4, 2),
)
```

## Feature-based generative quality

All functions in `metrics.generative` accept two finite matrices shaped `[samples, features]`. The source samples may be either 2D or 3D, but inputs must come from the same fixed encoder and preprocessing protocol. No image encoder is bundled; the appropriate extractor depends on the modality and task.

| Metric | Meaning and output | Inputs and limitations |
| --- | --- | --- |
| `frechet_distance` | Distance between Gaussian fits to real/synthetic embeddings; 0 is optimal. | At least two samples/domain. Biased and unstable for small sample sizes relative to feature dimension. It is only FID-like with an appropriate fixed encoder. |
| `kernel_inception_distance` | Unbiased cubic-polynomial MMD (KID); lower is better. | At least two samples/domain. An unbiased finite-sample estimate may be slightly negative. |
| `feature_precision_recall` | Manifold precision (fidelity) and recall (coverage), each [0, 1]. | Sensitive to feature scale, sample count, and neighbor `k`; quadratic memory. |
| `rbf_mmd` | Unbiased squared RBF-MMD; lower is better. | Sensitive to kernel bandwidth; median heuristic is the default. Estimate may be slightly negative. |
| `sliced_wasserstein_distance` | Mean Wasserstein-1 over random feature projections; lower is better. | Stochastic but seeded; depends on embedding scale and projection count. |

```python
from synthetic_imaging_validation.metrics.generative import (
    feature_precision_recall,
    frechet_distance,
    sliced_wasserstein_distance,
)

ffd = frechet_distance(real_features, synthetic_features)
precision, recall = feature_precision_recall(real_features, synthetic_features, k=3)
swd = sliced_wasserstein_distance(real_features, synthetic_features, seed=42)
```
