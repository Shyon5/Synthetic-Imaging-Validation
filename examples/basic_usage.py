"""Compare two NumPy images with intensity and distribution metrics."""

import numpy as np

from synthetic_imaging_validation.metrics.distribution import compare_distributions
from synthetic_imaging_validation.metrics.image_similarity import mae, psnr, ssim

rng = np.random.default_rng(7)
real = rng.random((64, 64), dtype=np.float32)
synthetic = np.clip(real + rng.normal(0.0, 0.03, real.shape), 0.0, 1.0)

print("MAE:", mae(real, synthetic))
print("PSNR:", psnr(real, synthetic, data_range=1.0))
print("SSIM:", ssim(real, synthetic, data_range=1.0))
print("Distribution report:", compare_distributions(real, synthetic))

