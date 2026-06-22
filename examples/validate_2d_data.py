"""Validate one 2D image pair and one 2D binary-mask pair."""

import numpy as np

from synthetic_imaging_validation import foreground_fraction, foreground_measure_ratio, mae, psnr, ssim
from synthetic_imaging_validation.metrics.distribution import wasserstein_distance
from synthetic_imaging_validation.metrics.segmentation import (
    connected_component_statistics,
    dice,
    hausdorff_distance,
    iou,
)

rng = np.random.default_rng(11)
real_image = rng.random((128, 128), dtype=np.float32)
synthetic_image = np.clip(real_image + rng.normal(0.0, 0.025, real_image.shape), 0.0, 1.0)

print("2D image metrics")
print("MAE:", mae(real_image, synthetic_image))
print("PSNR:", psnr(real_image, synthetic_image, data_range=1.0))
print("SSIM:", ssim(real_image, synthetic_image, data_range=1.0))
print("Wasserstein:", wasserstein_distance(real_image, synthetic_image))

reference_mask = np.zeros((128, 128), dtype=np.uint8)
synthetic_mask = np.zeros_like(reference_mask)
reference_mask[30:70, 35:75] = 1
synthetic_mask[32:72, 36:76] = 1
spacing = (0.7, 0.7)

print("\n2D mask metrics")
print("Dice:", dice(synthetic_mask, reference_mask))
print("IoU:", iou(synthetic_mask, reference_mask))
print("Area ratio:", foreground_measure_ratio(synthetic_mask, reference_mask))
print("Foreground fraction:", foreground_fraction(synthetic_mask))
print("Hausdorff:", hausdorff_distance(synthetic_mask, reference_mask, spacing=spacing))
print("Components:", connected_component_statistics(synthetic_mask, spacing=spacing))

