"""Small adapters for built-in API metrics absent from the pairwise CLI."""
from synthetic_imaging_validation import load_pair
from synthetic_imaging_validation.metrics import distribution as dist, segmentation as seg, spatial

EXTRA_METRICS = ("intensity_statistics", "histogram", "compare_distributions", "surface_statistics",
                 "component_measures", "distance_to_border", "centroid", "mask_spatial_report")


def calculate_extras(pair, options):
    """Return per-image descriptors and symmetric surface summaries using core formulas."""
    selected = set(options["metrics"]) & set(EXTRA_METRICS)
    if not selected:
        return {}
    a, b = load_pair(pair.real, pair.synthetic, require_spatial_match=not options.get("allow_spatial_mismatch", False))
    spacing = options.get("spacing") or a.spacing
    common = {"threshold": options.get("threshold", 0.5), "spacing": spacing}
    connectivity = options.get("connectivity")
    bins = options.get("bins", 64)
    percentiles = options.get("percentiles", [1, 5, 25, 50, 75, 95, 99])
    result = {}
    if "surface_statistics" in selected:
        result["surface_statistics"] = seg.surface_distance_statistics(b.array, a.array, connectivity=connectivity, **common)
    if "compare_distributions" in selected:
        result["compare_distributions"] = dist.compare_distributions(a.array, b.array, bins=bins, percentiles=percentiles)
    low = min(float(a.array.min()), float(b.array.min()))
    high = max(float(a.array.max()), float(b.array.max()))
    if high == low:
        high = low + 1.0
    for name, image in (("real", a), ("synthetic", b)):
        for metric in selected - {"surface_statistics", "compare_distributions"}:
            values = image.array
            if metric == "intensity_statistics":
                value = dist.intensity_statistics(values, percentiles=percentiles)
            elif metric == "histogram":
                counts, edges = dist.histogram(values, bins=bins, value_range=(low, high))
                value = {"counts": counts.tolist(), "edges": edges.tolist()}
            elif metric == "component_measures":
                value = seg.component_measure_distribution(values, connectivity=connectivity, **common).tolist()
            elif metric == "distance_to_border":
                value = spatial.distance_to_border_statistics(values, **common)
            elif metric == "centroid":
                value = spatial.centroid_statistics(values, **common)
            else:
                widths = options.get("border_width") or [1]
                value = spatial.mask_spatial_report(values, connectivity=connectivity,
                         border_width=widths[0] if len(widths) == 1 else widths, **common)
            result.setdefault(metric, {})[name] = value
    return result
