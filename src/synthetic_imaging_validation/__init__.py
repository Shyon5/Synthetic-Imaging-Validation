"""Validation metrics for synthetic medical images and masks."""

from .io.loading import ImageData, is_supported_image_path, load_directory, load_image, load_pair
from .io.pairing import ImagePair, image_file_key, load_manifest_pairs, load_paired_directories, pair_directory_files
from .metrics.distribution import (
    compare_distributions,
    intensity_statistics,
    jensen_shannon_divergence,
    kl_divergence,
    wasserstein_distance,
)
from .metrics.generative import (
    feature_precision_recall,
    frechet_distance,
    kernel_inception_distance,
    rbf_mmd,
    sliced_wasserstein_distance,
)
from .metrics.grouped import distribution_metrics_by_class, paired_metrics_by_class
from .metrics.image_similarity import mae, mse, ms_ssim, nrmse, psnr, rmse, ssim
from .metrics.segmentation import (
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
    volume_ratio,
)

__all__ = [
    "ImageData",
    "ImagePair",
    "active_voxel_fraction",
    "average_surface_distance",
    "component_area_distribution",
    "component_measure_distribution",
    "component_volume_distribution",
    "compare_distributions",
    "connected_component_statistics",
    "dice",
    "distribution_metrics_by_class",
    "feature_precision_recall",
    "foreground_fraction",
    "foreground_measure_ratio",
    "frechet_distance",
    "hausdorff_distance",
    "intensity_statistics",
    "iou",
    "image_file_key",
    "is_supported_image_path",
    "jensen_shannon_divergence",
    "kernel_inception_distance",
    "kl_divergence",
    "load_directory",
    "load_image",
    "load_manifest_pairs",
    "load_pair",
    "load_paired_directories",
    "mae",
    "mse",
    "ms_ssim",
    "nrmse",
    "paired_metrics_by_class",
    "pair_directory_files",
    "psnr",
    "rmse",
    "rbf_mmd",
    "sliced_wasserstein_distance",
    "ssim",
    "surface_distance_statistics",
    "volume_ratio",
    "wasserstein_distance",
]

__version__ = "0.1.0"
