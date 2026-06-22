"""Minimal command-line validation of one aligned image or mask pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..io.loading import load_pair
from ..metrics.distribution import jensen_shannon_divergence, kl_divergence, wasserstein_distance
from ..metrics.image_similarity import mae, mse, ms_ssim, nrmse, psnr, rmse, ssim
from ..metrics.segmentation import (
    active_voxel_fraction,
    average_surface_distance,
    connected_component_statistics,
    dice,
    foreground_fraction,
    foreground_measure_ratio,
    hausdorff_distance,
    iou,
    volume_ratio,
)
from ..metrics.spatial import border_statistics

DEFAULT_METRICS = ("mae", "mse", "rmse", "psnr", "ssim", "wasserstein")
SUPPORTED_METRICS = (
    "mae",
    "mse",
    "rmse",
    "nrmse",
    "psnr",
    "ssim",
    "ms_ssim",
    "wasserstein",
    "kl",
    "js",
    "dice",
    "iou",
    "hausdorff",
    "hausdorff95",
    "average_surface_distance",
    "volume_ratio",
    "active_voxel_fraction",
    "measure_ratio",
    "foreground_fraction",
    "connected_components",
    "border",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one aligned real/synthetic medical image pair."
    )
    parser.add_argument("--real", required=True, help="Real .nii[.gz], .npy, or .npz file.")
    parser.add_argument("--synthetic", required=True, help="Synthetic .nii[.gz], .npy, or .npz file.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS), choices=SUPPORTED_METRICS)
    parser.add_argument("--output", type=Path, help="Optional .json or .csv result file.")
    parser.add_argument("--data-range", type=float, help="Known intensity range for PSNR/SSIM/MS-SSIM.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold (default: 0.5).")
    parser.add_argument("--bins", type=int, default=64, help="Histogram bins for KL/JS (default: 64).")
    parser.add_argument("--spacing", nargs="+", type=float, help="Axis-order spacing when files lack it.")
    parser.add_argument("--channel-axis", type=int, help="Explicit channel axis for SSIM/MS-SSIM.")
    parser.add_argument("--batch-axis", type=int, help="Explicit batch axis for SSIM/MS-SSIM.")
    parser.add_argument("--border-width", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--allow-spatial-mismatch",
        action="store_true",
        help="Skip NIfTI spacing/affine checks. Shape checks still apply.",
    )
    return parser


def _resolve_spacing(explicit: Optional[Sequence[float]], metadata: Optional[Sequence[float]], ndim: int):
    if explicit is not None:
        if len(explicit) != ndim:
            raise ValueError(f"--spacing requires {ndim} values for this input.")
        return tuple(explicit)
    if metadata is not None:
        if len(metadata) != ndim:
            raise ValueError("Input spacing dimensionality does not match the array.")
        return tuple(metadata)
    return None


def calculate_metrics(args: argparse.Namespace) -> dict[str, Any]:
    """Calculate requested CLI metrics and return a JSON-compatible dictionary."""

    real_data, synthetic_data = load_pair(
        args.real,
        args.synthetic,
        require_spatial_match=not args.allow_spatial_mismatch,
    )
    real, synthetic = real_data.array, synthetic_data.array
    spacing = _resolve_spacing(args.spacing, real_data.spacing, real.ndim)
    results: dict[str, Any] = {}
    requested = set(args.metrics)
    simple = {"mae": mae, "mse": mse, "rmse": rmse, "nrmse": nrmse}
    for name, function in simple.items():
        if name in requested:
            results[name] = function(real, synthetic)
    if "psnr" in requested:
        results["psnr"] = psnr(real, synthetic, data_range=args.data_range)
    if "ssim" in requested:
        results["ssim"] = ssim(
            real,
            synthetic,
            data_range=args.data_range,
            channel_axis=args.channel_axis,
            batch_axis=args.batch_axis,
        )
    if "ms_ssim" in requested:
        results["ms_ssim"] = ms_ssim(
            real,
            synthetic,
            data_range=args.data_range,
            channel_axis=args.channel_axis,
            batch_axis=args.batch_axis,
        )
    if "wasserstein" in requested:
        results["wasserstein"] = wasserstein_distance(real, synthetic)
    if "kl" in requested:
        results["kl"] = kl_divergence(real, synthetic, bins=args.bins)
    if "js" in requested:
        results["js"] = jensen_shannon_divergence(real, synthetic, bins=args.bins)
    if "dice" in requested:
        results["dice"] = dice(synthetic, real, threshold=args.threshold)
    if "iou" in requested:
        results["iou"] = iou(synthetic, real, threshold=args.threshold)
    if "hausdorff" in requested:
        results["hausdorff"] = hausdorff_distance(
            synthetic, real, threshold=args.threshold, spacing=spacing
        )
    if "hausdorff95" in requested:
        results["hausdorff95"] = hausdorff_distance(
            synthetic, real, percentile=95.0, threshold=args.threshold, spacing=spacing
        )
    if "average_surface_distance" in requested:
        results["average_surface_distance"] = average_surface_distance(
            synthetic, real, threshold=args.threshold, spacing=spacing
        )
    if "volume_ratio" in requested:
        results["volume_ratio"] = volume_ratio(synthetic, real, threshold=args.threshold)
    if "measure_ratio" in requested:
        results["measure_ratio"] = foreground_measure_ratio(
            synthetic, real, threshold=args.threshold
        )
    if "active_voxel_fraction" in requested:
        results["active_voxel_fraction"] = {
            "real": active_voxel_fraction(real, threshold=args.threshold),
            "synthetic": active_voxel_fraction(synthetic, threshold=args.threshold),
        }
    if "foreground_fraction" in requested:
        results["foreground_fraction"] = {
            "real": foreground_fraction(real, threshold=args.threshold),
            "synthetic": foreground_fraction(synthetic, threshold=args.threshold),
        }
    if "connected_components" in requested:
        results["connected_components"] = {
            "real": connected_component_statistics(real, threshold=args.threshold, spacing=spacing),
            "synthetic": connected_component_statistics(synthetic, threshold=args.threshold, spacing=spacing),
        }
    if "border" in requested:
        width: Any = args.border_width[0] if len(args.border_width) == 1 else args.border_width
        results["border"] = {
            "real": border_statistics(real, threshold=args.threshold, border_width=width),
            "synthetic": border_statistics(synthetic, threshold=args.threshold, border_width=width),
        }
    return _json_safe(results)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(item, name))
    elif isinstance(value, list):
        rows.append((prefix, json.dumps(value)))
    else:
        rows.append((prefix, value))
    return rows


def _write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerows(_flatten(results))
    else:
        raise ValueError("--output must end with .json or .csv.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    args = _parser().parse_args(argv)
    try:
        results = calculate_metrics(args)
        if args.output:
            _write_results(args.output, results)
        print(json.dumps(results, indent=2, sort_keys=True))
    except (FileNotFoundError, ImportError, KeyError, TypeError, ValueError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
