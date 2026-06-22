"""Public metric functions grouped by validation domain."""

from .distribution import (
    compare_distributions,
    histogram,
    intensity_statistics,
    jensen_shannon_divergence,
    kl_divergence,
    wasserstein_distance,
)
from .generative import (
    feature_precision_recall,
    frechet_distance,
    kernel_inception_distance,
    rbf_mmd,
    sliced_wasserstein_distance,
)
from .grouped import (
    DISTRIBUTION_METRICS,
    PAIRED_METRICS,
    distribution_metrics_by_class,
    paired_metrics_by_class,
)
from .image_similarity import mae, mse, ms_ssim, nrmse, psnr, rmse, ssim
from .segmentation import (
    active_voxel_fraction,
    average_surface_distance,
    component_area_distribution,
    component_measure_distribution,
    component_volume_distribution,
    connected_component_statistics,
    dice,
    foreground_fraction,
    foreground_measure_ratio,
    hausdorff_distance,
    iou,
    surface_distance_statistics,
    surface_distances,
    volume_ratio,
)
from .spatial import (
    border_statistics,
    centroid_statistics,
    distance_to_border_statistics,
    mask_spatial_report,
)

__all__ = [name for name in globals() if not name.startswith("_")]
