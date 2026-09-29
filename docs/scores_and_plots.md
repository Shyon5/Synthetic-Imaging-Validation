# Scores and validation plots

The package provides two experimental summary scores and optional plots of
per-case results or validation histories. Scores require only the base install.
For plotting, install Matplotlib through the existing extra:

```bash
python -m pip install -e ".[viz]"
```

## What the scores mean

Both scores range from 0 to 100, with higher values indicating closer agreement.
They are descriptive prototypes, version `base-1`, not validated measures of
clinical quality. A score of 90 does not mean that an image is 90% correct.
Always keep the underlying metrics alongside the score.

Let `R = high - low` be a fixed intensity interval width, and `clip(x)` limit
values to [0, 1]. The two components receive equal weight:

| Score | Definition | Use |
| --- | --- | --- |
| Similarity | `100 * (clip(MS-SSIM) + clip(1 - MAE/R)) / 2` | Aligned, equal-shape real/synthetic pairs |
| Intensity distribution | `100 * (clip(1 - JS) + clip(1 - Wasserstein/R)) / 2` | Agreement of intensity distributions, not spatial structure |

Similarity combines structure with absolute intensity error. MS-SSIM uses the
existing adaptive implementation, up to five scales. Keep image dimensions and
preprocessing consistent when comparing scores: small inputs can use fewer scales.

The distribution score combines histogram shape with intensity displacement.
`JS` is Jensen-Shannon **divergence**, using base-2 logarithms, not its square
root. Histograms use 64 equal-width bins by default over the specified interval,
with a pseudocount of `1e-12`. Wasserstein distance uses the original intensity
samples, not histogram bin centres. Tiny rounding or pseudocount differences
are possible even for equivalent distributions with different sample counts.
Rearranging voxels leaves this score unchanged; it cannot detect misplaced anatomy.

These components overlap to some extent. Equal weights are a transparent starting
point, not a literature-validated optimal combination. There are no calibrated
quality thresholds, feature scores, mask scores or combined overall quality score.

## Choosing an intensity interval

Specify `value_range=(low, high)` in Python or `--score-range LOW HIGH` in the CLI.
For example, use `(0, 1)` only if both images already follow that normalization.
The package does not normalize the images for you. Non-finite values and values
outside the interval raise errors; small floating-point endpoint errors are
tolerated. Histogram calculations clamp only those endpoint rounding errors.

Keep the same interval and bin count across cases, epochs and models. Do not
derive a separate range from each image. An unnecessarily wide interval makes
intensity differences look smaller. Large backgrounds can also dominate both
scores. Define any cropping or region of interest before evaluation and keep it
consistent. Do not flatten a spatial ROI before computing similarity.

Channels must share a meaningful intensity scale. Evaluate channels separately
when their units differ, as for CT and PET. The distribution API flattens its
inputs and allows different sample counts; similarity requires spatially aligned
2D/3D arrays. Array batches are not accepted by the score CLI: use paired files
or evaluate each case through the API.

## Calculate scores with the CLI

```bash
synthetic-imaging-validate --manifest pairs.csv --metrics mae ms_ssim js wasserstein --scores similarity intensity_distribution --score-range 0 1 --output-json results.json --plot-output results.png --plot-metrics similarity_score intensity_distribution_score
```

Scores are opt-in. Existing commands without `--scores` retain their behaviour.
Requested raw metrics that also contribute to a score are reused, not computed
twice. With similarity enabled, the score interval width also supplies the range
for requested PSNR, SSIM and MS-SSIM; an explicit `--data-range` must agree.
With intensity distribution enabled, requested JS and KL use the same fixed
histogram interval as the score. `--bins` controls the histogram resolution.
CLI inputs still follow the existing paired shape and spatial-grid checks.

Each score contains its `value`, normalized `components` and `raw_metrics`.
For example, the similarity output includes:

- `similarity_score.value`: the 0-100 score.
- `similarity_score.components.structure`: clipped MS-SSIM, in [0, 1].
- `similarity_score.components.intensity`: clipped `1 - MAE/R`, in [0, 1].
- `similarity_score.raw_metrics.mae` and `.ms_ssim`: original component metrics.

Distribution components are named `histogram` and `transport`. Protocol metadata
is stored separately under `score_protocol`, including the version, range and
weights. JSON retains it; cohort CSV keeps the existing scalar-row layout, so
save JSON alongside CSV when you need that metadata. PDF and LaTeX reports also
include the supplied score fields and protocol.

Cohort and label summaries use the **mean of per-case scores**, giving each case
equal weight. They are not scores calculated from pooled voxels or averaged raw
metrics. `--group-by label` and `--num-workers` work with scores as with other metrics.

## Track validation across epochs

Run validation after an epoch's images have been saved, using the same history
file and a new epoch number each time:

```bash
synthetic-imaging-validate --manifest epoch_10/pairs.csv --metrics mae ms_ssim --scores similarity intensity_distribution --score-range 0 1 --history outputs/history.json --epoch 10 --run-name model_a --plot-history outputs/history.png --plot-metrics mae similarity_score intensity_distribution_score
```

At the next checkpoint, change the manifest and `--epoch`. `--step` is an alias
for `--epoch`; the package does not depend on a training framework or trigger
validation automatically. Normal report outputs can be added to the same command.
Metrics are calculated once, then reused for reporting, history and plots.

The versioned history JSON stores one record per run and step. Each record holds
the mean of finite per-case values, the contributing counts and protocol metadata.
If every value for a metric is undefined or infinite, its history value is `null`
and its count is zero. Plotting leaves a gap instead of replacing it with zero.
For example, an infinite PSNR for an identical pair is excluded from that mean.

Histories record global means, not separate curves for every manifest label.
Grouped summaries remain available in ordinary reports. For subgroup histories,
select the subgroup explicitly in Python and use a separate run name.

The history writer protects against some common mistakes:

- Duplicate `(run, step)` entries are rejected instead of overwritten.
- Changes to recorded settings or metric fields within a run are rejected. Use
  a different run name for a different experiment.
- Steps can arrive out of order; curves are sorted numerically.
- File replacement is atomic, but concurrent writers are not supported. Use one
  writer per file, such as rank zero during distributed training.

Keep the validation cases, preprocessing, resolution and field of view fixed.
The stored settings cannot check all these conditions and are not a complete
provenance record. Review identifiers before sharing results. Output files are
written sequentially: if a later export fails, earlier files may already exist,
and retrying a recorded step will encounter the duplicate-entry check.

## Python API

Inside a validation loop, with `real`, `synthetic` and `epoch` already defined:

```python
from synthetic_imaging_validation import (
    append_history, intensity_distribution_score, plot_history, similarity_score,
)

values = {
    "similarity_score": similarity_score(real, synthetic, value_range=(0, 1)),
    "intensity_distribution_score": intensity_distribution_score(
        real, synthetic, value_range=(0, 1), bins=64,
    ),
}
append_history(
    "outputs/history.json", values, step=epoch, run="model_a",
    protocol={"score_version": "base-1", "value_range": [0, 1], "bins": 64},
)
plot_history("outputs/history.json", output="outputs/history.png")
```

For several cases, pass `{"pairs": evaluate_pairs(...)}` instead of a scalar
mapping. The history writer aggregates the per-case values. With API scalar
results, provide the experimental settings in `protocol`: they cannot be inferred
from a float. Score functions also accept `return_details=True` to return the
components, raw metrics and protocol. Nested protocol fields are not plotted.
CLI results already carry a top-level `score_protocol`, which the history writer
retains automatically.

Both score names are available to `paired_metrics_by_class`; supply their range
through `metric_kwargs`. `intensity_distribution_score` is also available to
`distribution_metrics_by_class`, which pools samples within each class. That
pooled distribution score is a different quantity from a mean of per-case scores.

## Plot existing results without recalculating

```bash
python -m synthetic_imaging_validation.cli.plot_history --history outputs/history.json --output outputs/history.svg --metrics mae similarity_score intensity_distribution_score
```

For an existing results JSON:

```python
import json
from pathlib import Path
from synthetic_imaging_validation import plot_results

results = json.loads(Path("results.json").read_text(encoding="utf-8"))
figure = plot_results(results, metrics=["mae", "similarity_score"], output="results.png")
```

Plots use separate panels for different metrics, so unlike units are not mixed
on one axis. At most 12 fields can be selected; score components and raw metrics
are omitted by default but can be requested by dotted name. A single pair is
supported, although cohort comparisons and histories are more informative.

The functions return Matplotlib figures and optionally save PNG, SVG or PDF.
They run without a graphical display and do not change the application's global
Matplotlib backend. A figure PDF needs only `viz`, not the `report` extra.
There is no smoothing or automatic confidence interval. Different runs may use
different protocols; their presence in one plot does not make them comparable.

For a self-contained example with small arrays:

```bash
python examples/validation_history.py --output-dir outputs/history_example
```

The example progressively reduces a controlled intensity offset. It illustrates
the workflow, not a clinical benchmark. Use a fresh output directory when rerunning
it because recorded steps are deliberately protected against replacement.
