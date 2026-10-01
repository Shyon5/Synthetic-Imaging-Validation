"""Presentation labels only: metric identifiers and numerical outputs stay stable."""

METRICS = {
    "mae": ("MAE — Mean Absolute Error", "Average absolute intensity error. Lower is better; 0 means identical values. Requires aligned pairs."),
    "mse": ("MSE — Mean Squared Error", "Average squared intensity error. Lower is better. Large errors receive more weight."),
    "rmse": ("RMSE — Root Mean Squared Error", "Error expressed in the original intensity units. Lower is better; sensitive to large errors."),
    "nrmse": ("NRMSE — Normalized RMSE", "RMSE divided by the real image's observed intensity range. Lower is better; constant references can produce undefined or infinite values."),
    "psnr": ("PSNR — Peak Signal-to-Noise Ratio", "Intensity agreement in dB. Higher is better; identical images give infinity. Set a consistent intensity range width."),
    "ssim": ("SSIM — Structural Similarity", "Local intensity, contrast and structural agreement. Higher is better; 1 is ideal. Requires corresponding, aligned anatomy."),
    "ms_ssim": ("MS-SSIM — Multi-scale Structural Similarity", "Structural agreement at several scales. Higher is better; 1 is ideal. Still requires alignment; it is not a distribution metric."),
    "wasserstein": ("Wasserstein Distance", "Difference between intensity distributions, in intensity units. Lower is better. Does not detect spatial rearrangement."),
    "js": ("Jensen–Shannon Divergence", "Symmetric histogram disagreement. Lower is better; base-2 divergence is in [0, 1]. Depends on the bins and histogram interval."),
    "kl": ("KL — Kullback–Leibler Divergence", "Directional histogram disagreement. Lower is better; depends on binning and pseudocounts. Not symmetric."),
    "dice": ("Dice Similarity", "Binary-mask overlap, from 0 to 1. Higher is better. Requires aligned masks; empty-mask conventions are metric-specific."),
    "iou": ("IoU — Intersection over Union", "Binary-mask overlap divided by the union, from 0 to 1. Higher is better; 1 means matching masks."),
    "hausdorff95": ("HD95 — 95th-percentile Hausdorff Distance", "A robust boundary-distance summary. Lower is better. Units follow spacing; an empty/non-empty pair gives infinity."),
    "average_surface_distance": ("Average Surface Distance", "Mean boundary distance. Lower is better. Uses contour pixels in 2D and surface voxels in 3D."),
    "measure_ratio": ("Foreground Area / Volume Ratio", "Synthetic foreground measure divided by real foreground measure. 1 means equal size, not necessarily overlap. Neither larger nor smaller is universally better."),
    "foreground_fraction": ("Foreground Fraction", "Fraction of pixels or voxels above the mask threshold, reported separately for real and synthetic. No universal best value."),
    "connected_components": ("Connected-component Statistics", "Number and sizes of disconnected foreground regions. Useful for fragmentation; more or fewer components is not inherently better."),
    "border": ("Image-border Statistics", "Foreground occupancy and distances near image borders. Interpretation depends on anatomy and cropping; no universal optimum."),
    "similarity": ("Similarity Score", "Experimental 0–100 summary: equal weights for MS-SSIM and 1 − MAE/range width. Higher means closer agreement, not clinical validity."),
    "intensity_distribution": ("Intensity Distribution Score", "Experimental 0–100 summary: equal weights for 1 − JS and 1 − Wasserstein/range width. Requires fixed bounds; ignores spatial arrangement."),
}

METRICS.update({
    "hausdorff": ("Maximum Hausdorff Distance", "Largest symmetric boundary distance. Sensitive to isolated foreground; spacing defines the units."),
    "intensity_statistics": ("Intensity Statistics and Percentiles", "Per-image extrema, mean, standard deviation and selected percentiles. Descriptive, not a quality ranking."),
    "histogram": ("Shared-bin Intensity Histograms", "Counts and edges for both images using common bins over their combined observed range. No spatial information."),
    "compare_distributions": ("Intensity Distribution Report", "Statistics, mean difference, Wasserstein, KL and Jensen–Shannon, using the API defaults."),
    "surface_statistics": ("Surface Distance Report", "Average, median, HD95 and maximum symmetric surface distance. Empty/non-empty pairs have infinite distance."),
    "component_measures": ("Connected-component Areas / Volumes", "Full list of component measures, in spacing units squared or cubed. Connectivity is configurable."),
    "distance_to_border": ("Distance to Image Borders", "Foreground centre distances to the nearest image border. Not anatomical boundary distances."),
    "centroid": ("Foreground Centroid", "Index, normalized and spacing-scaled centroid. Physical coordinates use index origin, not the full NIfTI world affine."),
    "mask_spatial_report": ("Mask Morphology Report", "Foreground fraction, connected components, image borders, border distances and centroid. No anatomy-specific priors."),
    "frechet": ("Fréchet Feature Distance", "Gaussian mean/covariance distance between precomputed features. Requires a common encoder; small cohorts give biased, unstable estimates."),
    "kid": ("KID — Polynomial Feature MMD", "Unbiased polynomial-kernel MMD on precomputed features. Can be negative with finite samples."),
    "feature_precision_recall": ("Feature Precision and Recall", "Neighbourhood coverage in feature space. Sensitive to sample count and the number of neighbours."),
    "rbf_mmd": ("RBF Feature MMD", "Unbiased Gaussian-kernel MMD; depends on bandwidth and may be negative."),
    "sliced_wasserstein": ("Sliced Wasserstein Feature Distance", "Mean distance over seeded random feature projections. Keep projection count and seed fixed for comparisons."),
})


def metric_label(identifier: str) -> str:
    """Format an option or nested display field without changing its identifier."""
    base, *parts = identifier.split(".")
    key = base[:-6] if base.endswith("_score") else base
    label = METRICS.get(key, (base.replace("_", " ").title(), ""))[0]
    return label + (" / " + " / ".join(p.replace("_", " ") for p in parts) if parts else "")


def metric_help(identifier: str) -> str:
    """Return the short, task-oriented explanation for a metric option."""
    return METRICS[identifier][1]
