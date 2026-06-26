# Loading real and synthetic data

The package keeps data loading explicit on purpose. It will read common image
files and check that paired inputs are compatible, but it will not try to guess
your dataset structure from arbitrary folders.

## Expected preprocessing

This package validates prepared data. It does not perform registration,
resampling, reorientation, intensity normalization, clipping, denoising, mask
cleanup, or automatic threshold selection.

Before using paired spatial metrics, make sure that each real/synthetic pair:

- refers to the same case or target;
- has the same shape;
- is defined on the same voxel or pixel grid;
- uses the same orientation and field of view;
- has compatible spacing and affine metadata when using NIfTI files;
- uses comparable intensity scaling when intensity metrics are reported;
- uses masks that can be thresholded with the same rule.

For distribution-only image metrics, voxel-wise alignment is not required, but
the preprocessing protocol should still be shared across real and synthetic
cohorts. For example, intensity clipping, normalization, reconstruction scale,
and background handling should be consistent before comparing histograms or
intensity statistics.

For unpaired mask analysis, a one-to-one target is not needed. The cohort should
still use consistent orientation, spacing, field of view, connectivity, and mask
thresholding; otherwise morphology and location summaries can reflect
preprocessing differences rather than generation quality.

If a project needs registration, resampling, harmonization, or mask cleanup,
treat that as a separate preprocessing step before running this validator. A
future version may add optional preprocessing utilities, but the current package
keeps those choices outside the metric calculation.

The simplest convention is to keep real and synthetic files in two separate
directories:

```text
study_validation/
  real/
    case_001.nii.gz
    case_002.nii.gz
    case_003.nii.gz
  synthetic/
    case_001.nii.gz
    case_002.nii.gz
    case_003.nii.gz
```

With this layout, use stem-based pairing:

```bash
synthetic-imaging-validate \
  --real-dir study_validation/real \
  --synthetic-dir study_validation/synthetic \
  --pairing stem \
  --metrics mae ssim wasserstein dice \
  --output results.csv
```

`stem` means “remove the image suffix and match the remaining filename”. For
example, `case_001.nii.gz` is paired with `case_001.nii.gz`, and
`case_002.npy` is paired with `case_002.npz` because both have the key
`case_002`. If a key exists only on one side, the command stops with an error.
That is usually what you want: a wrong pairing can make a validation result look
convincing while measuring the wrong thing.

## When filenames do not match

If real and synthetic files cannot share the same names, use a CSV manifest:

```csv
case_id,real,synthetic,label
case_001,real/patient_A.nii.gz,synthetic/sample_0001.nii.gz,low
case_002,real/patient_B.nii.gz,synthetic/sample_0002.nii.gz,high
```

The repository also includes a small editable template at
[`examples/manifest_template.csv`](../examples/manifest_template.csv). It is not
meant to run as-is; copy it and replace the paths with your own files.

Then run:

```bash
synthetic-imaging-validate \
  --manifest study_validation/pairs.csv \
  --key-column case_id \
  --metrics mae ssim wasserstein dice \
  --output results.json
```

By default, manifest paths are resolved relative to the manifest file. If you
prefer paths relative to another folder, pass `--base-dir`.

The default manifest columns are `real` and `synthetic`. You can change them:

```bash
synthetic-imaging-validate \
  --manifest pairs.csv \
  --real-column reference_path \
  --synthetic-column generated_path \
  --key-column subject_id
```

Extra manifest columns are kept as metadata in the JSON output. They do not
change the result unless you explicitly request grouping. For example, a
`label` column can be summarized with:

```bash
synthetic-imaging-validate \
  --manifest study_validation/pairs.csv \
  --key-column case_id \
  --group-by label \
  --metrics mae ssim dice \
  --output results.json
```

This adds a `grouped_summary` block for the paired metrics. If `--group-by` is
not provided, `label` remains metadata only.

For directory and manifest inputs, JSON output contains a `pairs` list with one
record per real/synthetic pair and a `summary` block with finite scalar metric
means, standard deviations, minima, and maxima. When grouping is requested, JSON
also contains `grouped_summary`. CSV output uses one row per pair and metric,
followed by global and grouped summary rows.

## Pairing by sorted order

There is also a `sorted` directory mode:

```bash
synthetic-imaging-validate \
  --real-dir real \
  --synthetic-dir synthetic \
  --pairing sorted
```

This pairs the first real file alphabetically with the first synthetic file
alphabetically, and so on. Use it only for small, carefully checked folders.
For shared project work, `stem` pairing or a manifest is safer.

## Python API

The same tools are available from Python:

```python
from synthetic_imaging_validation.io import load_manifest_pairs, load_paired_directories
from synthetic_imaging_validation.metrics.image_similarity import mae

pairs = load_paired_directories("study_validation/real", "study_validation/synthetic")

for pair in pairs:
    print(pair.key, mae(pair.real.array, pair.synthetic.array))
```

For a manifest:

```python
pairs = load_manifest_pairs("study_validation/pairs.csv", key_column="case_id")
```

If you only want to inspect how directory files would be matched before loading
the arrays, use `pair_directory_files`.

## Supported file formats and checks

The supported file formats are:

- NIfTI: `.nii`, `.nii.gz`;
- NumPy arrays: `.npy`;
- NumPy archives: `.npz`.

Pair loading checks that shapes match. For NIfTI files, it also checks spacing
and affine when both inputs provide that metadata. No resampling, reorientation,
normalization, or registration is performed automatically.

That means the recommended workflow is:

1. prepare real and synthetic files on the same grid when using voxel-wise or
   surface metrics;
2. use `real/` and `synthetic/` directories with matching filenames, or write a
   small manifest CSV;
3. run the CLI and inspect the per-pair output before relying on cohort means.
