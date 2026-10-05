"""Read-only checks before expensive pairwise evaluation; no metrics are run."""
import importlib.util
import numpy as np
from synthetic_imaging_validation import load_pair
from synthetic_imaging_validation.metrics.image_similarity import _resolve_win_size
from .profiles import validate_settings, MASK_METRICS


def check_inputs(pairs, options, *, masks=False, progress=None):
    """Inspect one full-resolution pair at a time and collect errors per case.

    A passed check is not a guarantee of anatomical alignment or sufficient RAM.
    File contents are read anew during evaluation, never cached by this check.
    """
    options = validate_settings(options, masks)
    if not pairs:
        raise ValueError("Select at least one pair.")
    rows = []
    selected = set(options["metrics"])
    structural = bool(selected & {"ssim", "ms_ssim"} or "similarity" in options["scores"])
    for index, pair in enumerate(pairs):
        row = {"Case": pair.key, "Status": "Ready", "Details": ""}
        warnings = []
        try:
            a, b = load_pair(pair.real, pair.synthetic,
                             require_spatial_match=not options["allow_spatial_mismatch"])
            row.update({"Shape": str(a.array.shape), "Real min": float(a.array.min()),
                        "Real max": float(a.array.max()), "Synthetic min": float(b.array.min()),
                        "Synthetic max": float(b.array.max())})
            ndim = a.array.ndim
            axes = []
            for field in ("channel_axis", "batch_axis"):
                axis = options[field]
                if axis is not None:
                    if not -ndim <= axis < ndim:
                        raise ValueError(f"{field} does not fit this array's {ndim} dimensions.")
                    axes.append(axis % ndim)
            if len(set(axes)) != len(axes):
                raise ValueError("Channel and batch axes must be different.")
            if structural:
                shape = tuple(n for i, n in enumerate(a.array.shape) if i not in axes)
                if len(shape) not in (2, 3):
                    raise ValueError("Structural metrics need 2D or 3D spatial data. Check the channel and batch axes.")
                if "ssim" in selected:
                    _resolve_win_size(shape, None, None)
                if (selected & {"psnr", "ssim", "ms_ssim"}) and options["data_range"] is None:
                    warnings.append("Set a fixed intensity range width for comparisons across runs.")
            if selected & set(MASK_METRICS):
                if ndim not in (2, 3):
                    raise ValueError("Mask metrics need a single 2D or 3D mask per file.")
                if options["connectivity"] is not None and options["connectivity"] > ndim:
                    raise ValueError("Connectivity cannot exceed the number of spatial dimensions.")
                if len(options["border_width"]) not in (1, ndim):
                    raise ValueError("Use one border width or one per spatial axis.")
                if not np.any(a.array >= options["threshold"]) or not np.any(b.array >= options["threshold"]):
                    warnings.append("An empty mask is present; some distances may be infinite.")
            if options["spacing"] is not None and len(options["spacing"]) != ndim:
                raise ValueError(f"Spacing needs {ndim} values in array-axis order.")
            if options["scores"]:
                low, high = options["score_range"]
                for role, values in (("Real", a.array), ("Synthetic", b.array)):
                    eps = np.finfo(values.dtype).eps if values.dtype.kind == "f" else 0
                    tol = 4 * eps * max(abs(low), abs(high), high - low)
                    if values.min() < low - tol or values.max() > high + tol:
                        raise ValueError(f"{role} values exceed the score interval [{low}, {high}]. Prepare inputs or choose your fixed protocol bounds; no clipping is applied.")
            if options["ms_ssim_backend"] == "torchmetrics" and ("ms_ssim" in selected or "similarity" in options["scores"]):
                if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchmetrics") is None:
                    raise ValueError("TorchMetrics is not installed. Choose the NumPy MS-SSIM backend.")
            if options["allow_spatial_mismatch"]:
                warnings.append("NIfTI geometry matching is switched off.")
            if a.spacing is None and options["spacing"] is None and masks:
                warnings.append("No spacing provided: distances use pixels/voxels, not verified millimetres.")
            if warnings:
                row.update(Status="Review", Details=" ".join(warnings))
        except (ValueError, TypeError, OSError, ImportError, KeyError) as exc:
            row.update(Status="Error", Details=str(exc))
        rows.append(row)
        if progress:
            progress(index + 1, len(pairs))
    return {"ready": not any(row["Status"] == "Error" for row in rows), "cases": rows,
            "errors": sum(row["Status"] == "Error" for row in rows)}
