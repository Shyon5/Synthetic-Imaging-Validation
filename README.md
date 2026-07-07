# Synthetic Imaging Validation

[![OS Tests](https://github.com/Shyon5/Synthetic-Imaging-Validation/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Shyon5/Synthetic-Imaging-Validation/actions/workflows/tests.yml)

`Synthetic-Imaging-Validation` is a focused Python package for comparing real and synthetic medical images. It brings image similarity, intensity-distribution, segmentation, spatial, and feature-based metrics into one model-independent interface.

Validation remains separate from the code that generated the images. The package does not prescribe a preprocessing pipeline, assume a dataset layout, or bundle a feature encoder. Shapes, spacing, intensity ranges, and alignment requirements stay explicit so that the same evaluation can be reproduced on different datasets.

## Installation

Python 3.9 through 3.14 are supported. Every version is tested with the complete suite, including the optional PyTorch and plotting features, on GitHub's `ubuntu-latest`, `windows-latest`, and `macos-latest` hosted runners. The matrix therefore tracks the latest runner image for each operating system; older OS releases are not tested. Python 3.9 is included for compatibility with existing research environments, although it is end-of-life upstream and should not be preferred for new installations.

From a local checkout, install the core package with:

```bash
python -m pip install .
```

For development, use an editable install and include the test tools:

```bash
python -m pip install -e ".[test]"
```

Optional features are installed as extras:

```bash
python -m pip install ".[torch]"       # MS-SSIM
python -m pip install ".[viz]"         # plotting helpers
python -m pip install -e ".[test,torch,viz]"  # development with the complete test suite
```

### Dependencies

The base installation is deliberately small:

| Package | Supported versions | Used for |
| --- | --- | --- |
| NumPy | `>=1.26,<3.0` | Array conversion and numerical operations throughout the package |
| SciPy | `>=1.13,<2.0` | Statistical distances, connected components, surface distances, and matrix operations |
| scikit-image | `>=0.24,<1.0` | SSIM |
| nibabel | `>=5.3,<6.0` | Reading NIfTI files and their spatial metadata |

Both NumPy 1.26 and NumPy 2.x are supported. NumPy 1.26 is tested with Python 3.10–3.12; Python 3.13 and 3.14 use NumPy 2.x because NumPy 1.26 does not support those interpreters. The minimum SciPy, scikit-image, and nibabel versions were chosen from releases with NumPy 2 support.

The optional extras are:

| Extra | Packages | When it is needed |
| --- | --- | --- |
| `torch` | PyTorch `>=2.2,<3.0`, torchmetrics `>=1.3,<2.0` | MS-SSIM only |
| `viz` | Matplotlib `>=3.8,<4.0` | Histogram and slice plotting helpers |
| `test` | pytest `>=8.0,<10.0`, pytest-cov `>=5.0,<8.0` | Running the test suite and measuring coverage |

PyTorch is not required for the core metrics. PyTorch tensors are accepted when PyTorch is already available and are converted internally to NumPy. `pyproject.toml` is the source of truth for dependency constraints; `requirements.txt` mirrors the core runtime dependencies for convenience.

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

## What is included

Metrics are grouped by the kind of comparison they make:

- Image similarity: MAE, MSE, RMSE, NRMSE, PSNR, SSIM, optional adaptive MS-SSIM.
- Distribution/statistics: histograms, mean/std/min/max/percentiles, Wasserstein-1, histogram KL divergence, histogram Jensen-Shannon divergence.
- Segmentation: Dice, IoU, foreground fraction, area/volume ratio, connected-component area/volume distributions, Hausdorff/HD95, average and summary contour/surface distances.
- Spatial mask analysis: border occupancy, distance to image borders, centroids, and combined pipeline-independent morphology reports.
- Feature-based generative quality: Fréchet feature distance, KID, manifold precision/recall, RBF-MMD, and sliced Wasserstein distance.
- Optional class-wise evaluation for paired metrics and independent real/synthetic cohorts.

The Fréchet implementation works on precomputed `[samples, features]` matrices. Those features may come from 2D images or 3D volumes, but the package does not currently provide the encoder. For that reason, the result is described as a Fréchet feature distance rather than canonical Inception FID or medical 3D-FID.

For metric definitions and limitations, see [docs/metrics.md](docs/metrics.md). Practical guidance is collected in [docs/metric_selection.md](docs/metric_selection.md). Input pairing is described in [docs/data_loading.md](docs/data_loading.md), while [docs/grouped_metrics.md](docs/grouped_metrics.md) covers class-wise evaluation and [docs/dimensionality.md](docs/dimensionality.md) explains the 2D/3D shape conventions.

## Supported inputs and conventions

`load_image` accepts:

- NumPy arrays;
- PyTorch tensors (converted with `detach().cpu().numpy()`);
- NIfTI `.nii` and `.nii.gz` files;
- NumPy `.npy` and single-array `.npz` files.

For multiple files, the recommended layout is simple:

```text
validation_data/
  real/
    case_001.nii.gz
    case_002.nii.gz
  synthetic/
    case_001.nii.gz
    case_002.nii.gz
```

Directory pairing can match files by filename stem, or a CSV manifest can specify each real/synthetic pair explicitly. `load_directory` still returns files in a stable sorted order, but it does not guess subject pairing on its own. Arrays are not silently squeezed, permuted, resampled, or reoriented. NIfTI spacing follows array-axis order; when both inputs carry spatial metadata, pair loading checks shape, spacing, and affine.

Empty arrays and values containing NaN or infinity are rejected with a clear error. Pairwise metrics also require matching shapes. Channel and batch axes must be given explicitly for SSIM and MS-SSIM. Mask metrics accept scalar 2D or 3D arrays and binarize them with `values >= threshold`. The same implementation handles both dimensions, using area terminology in 2D and volume terminology in 3D.

Voxel-wise and surface metrics assume that the images have already been registered and resampled onto a common grid. Distribution-only metrics do not require spatial alignment, but real and synthetic cohorts should still share the same preprocessing protocol. See [docs/data_loading.md](docs/data_loading.md) for the expected input preparation.

## Command line

```bash
synthetic-imaging-validate --real path/to/real.nii.gz --synthetic path/to/synthetic.nii.gz --metrics psnr ssim wasserstein dice hausdorff95 --data-range 1.0 --output results.json
```

For a directory pair:

```bash
synthetic-imaging-validate --real-dir validation_data/real --synthetic-dir validation_data/synthetic --pairing stem --metrics mae ssim wasserstein dice --output results.csv
```

For a manifest:

```bash
synthetic-imaging-validate --manifest validation_data/pairs.csv --key-column case_id --metrics mae ssim wasserstein dice --output results.json
```

Use `--output-json results.json --output-csv results.csv` when you want both
formats from the same run. The older `--output` option is still available for a
single JSON or CSV file.

Add `--group-by label` when a manifest column should be used for grouped paired-metric summaries. Without `--group-by`, extra manifest columns remain metadata only.

The module form, `python -m synthetic_imaging_validation.cli.validate`, is equivalent. Results can be written as JSON, long-form CSV, or both. Run `synthetic-imaging-validate --help` to see the options for mask thresholds, spacing, border widths, array axes, directory pairing, manifest columns, and grouped summaries.

## Examples and tests

The current test suite has 100% statement and branch coverage. The full threshold is enforced by CI across every supported Python version and runner platform.

After installing the package in editable mode, run the examples from the repository root:

```bash
python examples/basic_usage.py
python examples/validate_2d_data.py
python examples/validate_binary_masks.py
python examples/validate_nifti_pair.py real.nii.gz synthetic.nii.gz
python -m pytest
```

To reproduce the coverage check used in CI:

```bash
python -m pytest --cov --cov-report=term-missing --cov-fail-under=100
```

## Contributing a metric

New metrics should fit the existing input and validation conventions:

1. Place it in the module matching its scientific role under `src/synthetic_imaging_validation/metrics/`.
2. Reuse `to_numpy`, `validate_pair`, and `validate_spacing` instead of adding implicit conversions.
3. Document direction, units/range, empty-input behavior, dimensional assumptions, and required alignment.
4. Add identical-input, perturbed-input, invalid-shape, and non-finite-input tests.
5. Add CLI exposure only when the metric has unambiguous file-level inputs.

## Future Docker support

Docker is intentionally out of scope for the first release. The package and CLI do not assume local paths and write only to destinations selected by the user, so container support can be added later without changing the metric APIs. Other likely additions are optional resampling, confidence intervals, and validated medical-imaging encoders.
