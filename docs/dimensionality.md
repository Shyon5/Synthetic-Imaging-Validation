# 2D and 3D input conventions

The same implementation is used for 2D and 3D when the mathematical definition is unchanged. Dimensional assumptions are stated at the API boundary.

## Shape conventions

| Data kind | Unbatched shape | Optional axes |
| --- | --- | --- |
| Scalar 2D image or mask | `[H, W]` | Batch: `[N, H, W]` |
| Multi-channel 2D image | `[C, H, W]` or `[H, W, C]` | Declare `channel_axis`; declare `batch_axis` when present. |
| Scalar 3D volume or mask | `[D, H, W]` | Batch: `[N, D, H, W]` |
| Multi-channel 3D volume | `[C, D, H, W]` or `[D, H, W, C]` | Declare `channel_axis`; declare `batch_axis` when present. |
| Feature embeddings | `[samples, features]` | The source images may be 2D or 3D; the extractor protocol must be identical across domains. |

No function guesses whether a length-3 axis is a color channel or a spatial axis. SSIM/MS-SSIM therefore require explicit channel and batch axes. Scalar mask metrics accept only `[H, W]` or `[D, H, W]` and intentionally reject batched/multi-channel masks.

## Metric support

| Metric group | 2D | 3D | Notes |
| --- | --- | --- | --- |
| MAE, MSE, RMSE, NRMSE, PSNR | Yes | Yes | All equal-shape numeric arrays; values are averaged over every element. |
| SSIM, MS-SSIM | Yes | Yes | Native spatial computation; channel/batch axes are explicit. |
| Histogram/statistical metrics | Yes | Yes | Spatial axes are flattened because location is intentionally ignored. |
| Dice, IoU, foreground fraction/ratio | Yes | Yes | Scalar masks only. Area is measured in 2D and volume in 3D. |
| Connected components | Yes | Yes | 4/8 connectivity in 2D; 6/18/26 connectivity in 3D. |
| Hausdorff and surface distances | Yes | Yes | Contour distance in 2D; surface distance in 3D. Spacing follows array-axis order. |
| Border, distance-to-border, centroid | Yes | Yes | Spacing, orientation, and field of view must be standardized across cohorts. |
| Fréchet/KID/MMD/feature PR/SWD | Yes | Yes | Operate on precomputed feature matrices, not directly on pixels/voxels. |

## Dimension-aware terminology

Use `foreground_fraction`, `foreground_measure_ratio`, and `component_measure_distribution` for code shared between 2D and 3D.

- In 2D, connected-component measures are areas and reports contain keys such as `component_areas` and `total_area`.
- In 3D, measures are volumes and reports contain `component_volumes` and `total_volume`.
- `active_voxel_fraction` and `volume_ratio` remain available as compatibility aliases, but their generic counterparts are clearer in dimension-independent code.

```python
from synthetic_imaging_validation import foreground_fraction, foreground_measure_ratio
from synthetic_imaging_validation.metrics.segmentation import connected_component_statistics

two_dimensional = connected_component_statistics(mask_2d, spacing=(0.8, 0.8), connectivity=8)
three_dimensional = connected_component_statistics(mask_3d, spacing=(0.8, 0.8, 2.5), connectivity=26)

assert two_dimensional["measure_name"] == "area"
assert three_dimensional["measure_name"] == "volume"
```

## Fréchet distance and dimensionality

`frechet_distance` is already dimension-independent because it accepts `[samples, features]`. The package currently does not bundle an Inception 2D, R3D-18, or medical 3D feature extractor. Consequently it should be described as a Fréchet feature distance unless the selected and documented extractor justifies a more specific name.
