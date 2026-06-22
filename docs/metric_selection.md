# Choosing validation metrics

The first question is whether each synthetic sample has a real target. This determines which metrics have a meaningful interpretation.

## Paired and aligned samples

Pairwise metrics are suitable for reconstruction, denoising, super-resolution, registered modality translation, and segmentation against ground truth. The two inputs must describe the same subject on the same grid.

For intensity images, MAE or RMSE gives an error in the input units. SSIM adds information about local contrast and structure. MS-SSIM evaluates structure at several resolutions and is useful when both fine and coarse patterns matter. It is not registration-invariant: a spatial shift can still lower the score.

PSNR is mainly useful when the intensity range is fixed and reported. NRMSE can help compare scales, but only if every experiment uses the same denominator convention.

For aligned masks, a practical set is Dice (or IoU), HD95, average surface distance, and area/volume ratio. Dice and IoU carry almost the same ranking information, so reporting both does not provide two independent results. Maximum Hausdorff distance is sensitive to isolated pixels; HD95 is often easier to interpret when small contour outliers are not the main concern.

## Unpaired generation

An arbitrary real image is not a target for an arbitrary synthetic image. MAE, PSNR, SSIM, MS-SSIM, Dice, and Hausdorff distance should not be averaged over random real/synthetic pairings.

This also applies to spatially conditioned models when generation is one-to-many. A condition can constrain location or anatomy without defining a unique correct output. MS-SSIM is more informative than single-scale SSIM only when some correspondence exists; it does not replace that correspondence.

For unpaired images, compare cohort distributions instead. Useful options include:

- per-subject intensity summaries followed by cohort-level comparison;
- Wasserstein or Jensen-Shannon distance with shared preprocessing;
- feature Fréchet distance, KID, MMD, sliced Wasserstein, or feature precision/recall when a fixed encoder is available;
- downstream-task performance when synthetic data are intended for training or augmentation.

For unpaired masks, compare foreground area/volume, component count and sizes, centroid, and distance from the image border. These quantities are general, but only after orientation, spacing, field of view, connectivity, and mask threshold have been standardized.

## Notes on individual metric families

### Intensity distributions

Wasserstein-1 retains the intensity unit and is usually easier to interpret than a histogram divergence. KL is directional and can be unstable when bins are sparse. Jensen-Shannon is symmetric and finite, but it still depends on the chosen bins and ignores spatial arrangement.

Avoid treating every voxel as an independent observation. Adjacent voxels are correlated, and pooling all voxels gives larger volumes more influence. A safer approach is to compute summaries per subject and compare those summaries across cohorts.

### Mask geometry

Area or volume ratio measures burden but says nothing about location. Component statistics reveal fragmentation but depend on connectivity and small-component filtering. A global centroid gives a coarse location and may be misleading for multi-lesion masks. Border statistics are meaningful only when cropping and anatomical coverage are consistent.

Empty masks deserve separate reporting. Dice defines two empty masks as a perfect match in this package, while a surface distance between an empty and a non-empty mask is infinite. A cohort mean can hide how often either case occurred.

### Feature metrics

Feature metrics are only as useful as their encoder. Real and synthetic samples must use the same frozen encoder and preprocessing. A score based on random weights or an encoder changed between experiments is not comparable across runs.

Fréchet distance fits a Gaussian to each feature distribution. KID and RBF-MMD compare distributions through kernels. Precision and recall separate fidelity from coverage, while sliced Wasserstein averages distances over feature projections. These metrics answer related but different questions and should not be expected to rank every model identically.

## Small cohorts

For pairwise metrics, a score is defined for one pair, but a class or cohort mean may still be uncertain. Report the number of subjects, show the per-case distribution, and use subject-level confidence intervals when possible.

Histogram and morphology comparisons also need independent subjects. Thousands of voxels from five subjects do not provide the same evidence as thousands of subjects.

Fréchet scores are particularly sensitive to sample size. With `N` samples and `D` features, the empirical covariance rank is at most `N - 1`; when `N <= D`, regularization can make the calculation run but cannot make the covariance estimate reliable. There is no universal sample-size cutoff. Report both `N` and `D`, compare results under repeated subsampling, and avoid strong conclusions from small per-class groups.

Finite-sample FID is biased, and equal sample counts do not remove model-dependent bias ([Chong and Forsyth, 2020](https://arxiv.org/abs/1911.07023)). KID uses an unbiased MMD estimator, although its variance can still be large for small cohorts ([Bińkowski et al., 2018](https://arxiv.org/abs/1801.01401)).

Feature precision/recall needs enough observations to estimate nearest-neighbor radii. Having just more than `k` samples satisfies the implementation but rarely produces a stable manifold estimate ([Kynkäänniemi et al., 2019](https://arxiv.org/abs/1904.06991)). MMD depends on kernel bandwidth, and sliced Wasserstein should be checked across projection seeds when data are limited.

Class-wise evaluation reduces the effective sample size further. Always report the real and synthetic count for each class; see [Metrics by class](grouped_metrics.md) for the grouped API.

## Typical metric sets

- **Paired reconstruction:** MAE or RMSE, SSIM or MS-SSIM, and optionally a justified feature metric. Add PSNR only with a fixed intensity range.
- **Unpaired image generation:** per-subject intensity distributions, feature Fréchet/KID/MMD or sliced Wasserstein, precision/recall, and downstream-task evaluation.
- **Segmentation against ground truth:** Dice or IoU, HD95, average surface distance, area/volume ratio, and component count.
- **Unpaired generated masks:** distributions of foreground area/volume, component count and size, centroid, and border distance.

