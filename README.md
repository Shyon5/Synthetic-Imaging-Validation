# Synthetic Imaging Validation

[![Tests](https://github.com/Shyon5/Synthetic-Imaging-Validation/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Shyon5/Synthetic-Imaging-Validation/actions/workflows/tests.yml)

`synthetic-imaging-validation` is a small, model-agnostic Python package for comparing real and synthetic medical images. It provides pairwise image similarity, intensity-distribution analysis, mask overlap and surface metrics, spatial plausibility checks, and feature-distribution metrics.

The package does not generate images, prescribe a preprocessing pipeline, or bundle a feature encoder. Inputs and assumptions remain explicit so results can be reproduced across datasets and research groups.

## Installation

Create a Python 3.10, 3.11, or 3.12 environment and install the project locally:

```bash
python -m pip install -e .
```

For MS-SSIM support:

```bash
python -m pip install -e ".[torch]"
```

For development and tests:

```bash
python -m pip install -e ".[test]"
```

Use `.[test,torch]` to include the optional MS-SSIM tests.

Core dependencies are NumPy 1.26, SciPy, scikit-image, and nibabel. NumPy 2.x is intentionally excluded from the first release while cross-platform compatibility is established. PyTorch and torchmetrics are optional and used only by MS-SSIM. Matplotlib is optional via the `viz` extra.

## Quick start

```python
import numpy as np

from synthetic_imaging_validation import mae, psnr, ssim
from synthetic_imaging_validation.metrics.distribution import wasserstein_distance

real = np.zeros((64, 64), dtype=np.float32)
synthetic = real.copy()
synthetic[20:30, 20:30] = 0.1

print(mae(real, synthetic))
print(psnr(real, synthetic, data_range=1.0))
print(ssim(real, synthetic, data_range=1.0))
print(wasserstein_distance(real, synthetic))
```

Masks use the same array/tensor interface:

```python
from synthetic_imaging_validation.metrics.segmentation import dice, hausdorff_distance

score = dice(synthetic_mask, reference_mask, threshold=0.5)
hd95_mm = hausdorff_distance(
    synthetic_mask,
    reference_mask,
    threshold=0.5,
    spacing=(1.0, 1.0, 2.5),
    percentile=95,
)
```

## Available metrics

- Image similarity: MAE, MSE, RMSE, NRMSE, PSNR, SSIM, optional adaptive MS-SSIM.
- Distribution/statistics: histograms, mean/std/min/max/percentiles, Wasserstein-1, histogram KL divergence, histogram Jensen-Shannon divergence.
- Segmentation: Dice, IoU, foreground fraction, area/volume ratio, connected-component area/volume distributions, Hausdorff/HD95, average and summary contour/surface distances.
- Spatial mask analysis: border occupancy, distance to image borders, centroids, and combined pipeline-independent morphology reports.
- Feature-based generative quality: Fréchet feature distance, KID, manifold precision/recall, RBF-MMD, and sliced Wasserstein distance.
- Optional class-wise evaluation for paired metrics and independent real/synthetic cohorts.

The Fréchet implementation currently operates on precomputed `[samples, features]` matrices and is valid for either 2D- or 3D-derived embeddings. No image/volume encoder is bundled yet, so it is not presented as canonical Inception FID or medical 3D-FID.

See [docs/metrics.md](docs/metrics.md) for definitions, [docs/metric_selection.md](docs/metric_selection.md) for metric choice, [docs/grouped_metrics.md](docs/grouped_metrics.md) for class-wise evaluation, and [docs/dimensionality.md](docs/dimensionality.md) for the 2D/3D shape contract.

## Supported inputs and conventions

`load_image` accepts:

- NumPy arrays;
- PyTorch tensors (converted with `detach().cpu().numpy()`);
- NIfTI `.nii` and `.nii.gz` files;
- NumPy `.npy` and single-array `.npz` files.

`load_directory` loads a sorted directory without guessing subject pairing. Array shape is never silently squeezed, permuted, resampled, or reoriented. NIfTI spacing follows array-axis order, and pair loading checks shape, spacing, and affine when both inputs contain that metadata.

All metrics reject empty numeric arrays and NaN/Inf values. Pairwise metrics reject incompatible shapes. Channel and batch axes must be supplied explicitly for SSIM/MS-SSIM. Mask metrics accept scalar 2D or 3D arrays and binarize with `values >= threshold`. The same implementations serve both dimensions; connected-component reports use area terminology in 2D and volume terminology in 3D.

Images must already be registered and resampled to a common grid before voxel-wise or surface comparison. Distribution-only comparisons do not require spatial alignment.

## Command line

```bash
python -m synthetic_imaging_validation.cli.validate \
  --real path/to/real.nii.gz \
  --synthetic path/to/synthetic.nii.gz \
  --metrics psnr ssim wasserstein dice hausdorff95 \
  --data-range 1.0 \
  --output results.json
```

The installed `synthetic-imaging-validate` command is equivalent. Results may be written as JSON or long-form CSV. Use `--help` for mask thresholds, explicit spacing, border widths, and axis options.

## Examples and tests

From the repository root:

```bash
python examples/basic_usage.py
python examples/validate_2d_data.py
python examples/validate_binary_masks.py
python examples/validate_nifti_pair.py real.nii.gz synthetic.nii.gz
pytest
```

## Adding a metric

1. Place it in the module matching its scientific role under `src/synthetic_imaging_validation/metrics/`.
2. Reuse `to_numpy`, `validate_pair`, and `validate_spacing` instead of adding implicit conversions.
3. Document direction, units/range, empty-input behavior, dimensional assumptions, and required alignment.
4. Add identical-input, perturbed-input, invalid-shape, and non-finite-input tests.
5. Add CLI exposure only when the metric has unambiguous file-level inputs.

## Future Docker support

The package and CLI have no local path assumptions and write only to user-selected destinations. A future release can add a minimal image and bind-mounted input/output convention without changing metric APIs. Other useful future work includes directory pairing manifests, optional resampling, confidence intervals, and validated medical-imaging encoders.
