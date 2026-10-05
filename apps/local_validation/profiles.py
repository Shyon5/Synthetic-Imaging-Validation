"""Portable evaluation settings. Profiles never restore paths or execute code."""
import json
import math

IMAGE_METRICS = ["mae", "mse", "rmse", "nrmse", "psnr", "ssim", "ms_ssim", "wasserstein", "js", "kl",
                 "intensity_statistics", "histogram", "compare_distributions"]
MASK_METRICS = ["dice", "iou", "hausdorff", "hausdorff95", "average_surface_distance", "measure_ratio",
               "foreground_fraction", "connected_components", "border", "surface_statistics",
               "component_measures", "distance_to_border", "centroid", "mask_spatial_report"]


def defaults(masks=False):
    """Return a fresh set of conservative settings for images or masks."""
    return dict(metrics=["dice", "iou", "hausdorff95"] if masks else ["mae", "rmse", "ssim", "ms_ssim", "wasserstein"],
                scores=[], score_range=[0.0, 1.0], data_range=1.0, workers=1, threshold=0.5,
                bins=64, spacing=None, channel_axis=None, batch_axis=None, border_width=[1],
                connectivity=None, percentiles=[1, 5, 25, 50, 75, 95, 99], ms_ssim_backend="numpy",
                allow_spatial_mismatch=False, plot_metrics=None, nrmse_normalization="range")


def validate_settings(options, masks=False):
    """Check portable fields without reading images or changing user files."""
    if not isinstance(options, dict):
        raise ValueError("Settings must be a JSON object.")
    result = defaults(masks)
    result.update({key: value for key, value in options.items() if key in result})
    choices = MASK_METRICS if masks else IMAGE_METRICS
    for field, allowed in (("metrics", choices), ("scores", [] if masks else ["similarity", "intensity_distribution"])):
        values = result[field]
        if not isinstance(values, list) or any(not isinstance(v, str) or v not in allowed for v in values):
            raise ValueError(f"Choose supported {field} for {'masks' if masks else 'images'}.")
        result[field] = list(dict.fromkeys(values))
    if not result["metrics"]:
        raise ValueError("Select at least one metric.")
    def number(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        return value
    for name, low, high in (("workers", 1, 16), ("bins", 2, 4096)):
        value = result[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}.")
    bounds = result["score_range"]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("Score interval needs two bounds.")
    low, high = (number(v, "Score bound") for v in bounds)
    if high <= low or not math.isfinite(high - low):
        raise ValueError("Score upper bound must exceed lower bound with a finite interval width.")
    if result["data_range"] is not None and number(result["data_range"], "Intensity range width") <= 0:
        raise ValueError("Intensity range width must be positive.")
    number(result["threshold"], "Mask threshold")
    for name in ("channel_axis", "batch_axis"):
        if result[name] is not None and type(result[name]) is not int:
            raise ValueError(f"{name} must be an integer or null.")
    if result["scores"] and result["batch_axis"] is not None:
        raise ValueError("Scores need one image per file, without a batch axis.")
    if result["connectivity"] is not None and (type(result["connectivity"]) is not int or result["connectivity"] not in (1, 2, 3)):
        raise ValueError("Connectivity must be 1, 2, 3 or null.")
    if type(result["allow_spatial_mismatch"]) is not bool:
        raise ValueError("Geometry override must be true or false.")
    for name in ("spacing", "border_width", "percentiles", "plot_metrics"):
        values = result[name]
        if values is None and name in ("spacing", "plot_metrics"):
            continue
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a non-empty list.")
        for value in values:
            if name == "plot_metrics":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Plot fields must be non-empty names.")
            elif name == "border_width":
                if type(value) is not int or value < 0:
                    raise ValueError("Border widths must be non-negative integers.")
            elif name == "spacing" and number(value, name) <= 0:
                raise ValueError("Spacing must be positive.")
            elif name == "percentiles" and not 0 <= number(value, name) <= 100:
                raise ValueError("Percentiles must be between 0 and 100.")
    if result["ms_ssim_backend"] not in ("numpy", "torchmetrics"):
        raise ValueError("Unknown MS-SSIM backend.")
    if result["nrmse_normalization"] not in ("range", "mean", "l2"):
        raise ValueError("NRMSE normalization must be range, mean or l2.")
    if "similarity" in result["scores"]:
        result["data_range"] = high - low
    return result


def export_profile(options, masks=False):
    """Serialize settings only; omit paths, labels, case IDs and PDF font paths."""
    return json.dumps({"profile_version": 1, "kind": "masks" if masks else "images",
                       "settings": validate_settings(options, masks)}, indent=2, allow_nan=False)


def import_profile(content, masks=False):
    """Read a versioned profile or a previous run's settings.json (settings only)."""
    if len(content) > 100_000:
        raise ValueError("Settings file is too large (maximum 100 KB).")
    try:
        data = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Choose a valid JSON settings file.") from exc
    if not isinstance(data, dict):
        raise ValueError("Settings must be a JSON object.")
    if "profile_version" in data:
        if type(data["profile_version"]) is not int or data["profile_version"] != 1:
            raise ValueError("Unsupported settings version.")
        if data.get("kind") != ("masks" if masks else "images"):
            raise ValueError("Switch Images / Binary masks to match this settings file first.")
        data = data.get("settings")
    elif "metrics" not in data:
        raise ValueError("Choose settings.json or an exported profile, not a results file.")
    return validate_settings(data, masks)
