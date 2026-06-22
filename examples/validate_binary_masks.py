"""Compute overlap, surface, and component metrics for binary masks."""

import numpy as np

from synthetic_imaging_validation.metrics.segmentation import (
    average_surface_distance,
    connected_component_statistics,
    dice,
    hausdorff_distance,
    iou,
    volume_ratio,
)

reference = np.zeros((32, 32, 32), dtype=np.uint8)
synthetic = np.zeros_like(reference)
reference[8:20, 8:20, 8:20] = 1
synthetic[9:21, 8:20, 8:20] = 1
spacing = (1.0, 1.0, 2.5)

print("Dice:", dice(synthetic, reference))
print("IoU:", iou(synthetic, reference))
print("Volume ratio:", volume_ratio(synthetic, reference))
print("Hausdorff:", hausdorff_distance(synthetic, reference, spacing=spacing))
print("ASSD:", average_surface_distance(synthetic, reference, spacing=spacing))
print("Components:", connected_component_statistics(synthetic, spacing=spacing))

