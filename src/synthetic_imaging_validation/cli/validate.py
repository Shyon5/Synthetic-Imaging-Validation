"""Command-line validation of aligned image or mask pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..io.loading import load_pair
from ..io.pairing import ImagePair, image_file_key, load_manifest_pairs, load_paired_directories
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
        description="Compare aligned real/synthetic medical image pairs."
    )
    parser.add_argument("--real", help="Real .nii[.gz], .npy, or .npz file.")
    parser.add_argument("--synthetic", help="Synthetic .nii[.gz], .npy, or .npz file.")
    parser.add_argument("--real-dir", type=Path, help="Directory containing real files.")
    parser.add_argument("--synthetic-dir", type=Path, help="Directory containing synthetic files.")
    parser.add_argument("--manifest", type=Path, help="CSV manifest with real/synthetic file columns.")
    parser.add_argument(
        "--pairing",
        choices=("stem", "sorted"),
        default="stem",
        help="Directory pairing rule: filename stem match (default) or alphabetical order.",
    )
    parser.add_argument("--recursive", action="store_true", help="Include nested files in directory mode.")
    parser.add_argument("--real-column", default="real", help="Manifest column containing real image paths.")
    parser.add_argument(
        "--synthetic-column",
        default="synthetic",
        help="Manifest column containing synthetic image paths.",
    )
    parser.add_argument("--key-column", help="Optional manifest column used as the pair identifier.")
    parser.add_argument(
        "--group-by",
        help="Optional manifest metadata column used to summarize paired metrics by group.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory for relative manifest paths. Defaults to the manifest directory.",
    )
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS), choices=SUPPORTED_METRICS)
    parser.add_argument("--output", type=Path, help="Optional single .json or .csv result file.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON result file.")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV result file.")
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


def _input_mode(args: argparse.Namespace) -> str:
    modes = []
    if args.manifest is not None:
        modes.append("manifest")
    if args.real_dir is not None or args.synthetic_dir is not None:
        if args.real_dir is None or args.synthetic_dir is None:
            raise ValueError("Directory mode requires both --real-dir and --synthetic-dir.")
        modes.append("directories")
    if args.real is not None or args.synthetic is not None:
        if not args.real or not args.synthetic:
            raise ValueError("File mode requires both --real and --synthetic.")
        modes.append("files")
    if len(modes) != 1:
        raise ValueError(
            "Choose exactly one input mode: --real/--synthetic, "
            "--real-dir/--synthetic-dir, or --manifest."
        )
    return modes[0]


def _load_cli_pairs(args: argparse.Namespace) -> tuple[str, list[ImagePair]]:
    mode = _input_mode(args)
    require_spatial_match = not args.allow_spatial_mismatch
    if mode == "files":
        real_data, synthetic_data = load_pair(
            args.real,
            args.synthetic,
            require_spatial_match=require_spatial_match,
        )
        return mode, [
            ImagePair(
                key=image_file_key(args.real),
                real=real_data,
                synthetic=synthetic_data,
            )
        ]
    if mode == "directories":
        return mode, load_paired_directories(
            args.real_dir,
            args.synthetic_dir,
            pairing=args.pairing,
            recursive=args.recursive,
            require_spatial_match=require_spatial_match,
        )
    return mode, load_manifest_pairs(
        args.manifest,
        real_column=args.real_column,
        synthetic_column=args.synthetic_column,
        key_column=args.key_column,
        base_dir=args.base_dir,
        require_spatial_match=require_spatial_match,
    )


def _calculate_pair_metrics(pair: ImagePair, args: argparse.Namespace) -> dict[str, Any]:
    real, synthetic = pair.real.array, pair.synthetic.array
    spacing = _resolve_spacing(args.spacing, pair.real.spacing, real.ndim)
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
    return results


def _path_text(path: Optional[Path]) -> Optional[str]:
    return None if path is None else str(path)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _summarize_pair_metrics(metrics_by_pair: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    for metrics in metrics_by_pair:
        for name, value in _flatten(metrics):
            if _is_finite_number(value):
                buckets.setdefault(name, []).append(float(value))

    summary = {"count": len(metrics_by_pair), "metrics": {}}
    for name in sorted(buckets):
        values = np.asarray(buckets[name], dtype=np.float64)
        summary["metrics"][name] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def _summarize_grouped_pair_metrics(records: list[dict[str, Any]], group_by: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        if group_by not in metadata or str(metadata[group_by]).strip() == "":
            raise ValueError(
                f"--group-by column '{group_by}' is missing for pair '{record.get('key', '')}'."
            )
        group = str(metadata[group_by])
        groups.setdefault(group, []).append(record["metrics"])

    return {
        "column": group_by,
        "groups": {
            group: _summarize_pair_metrics(metrics)
            for group, metrics in groups.items()
        },
    }


def calculate_metrics(args: argparse.Namespace) -> dict[str, Any]:
    """Calculate requested CLI metrics and return a JSON-compatible dictionary."""

    mode, pairs = _load_cli_pairs(args)
    if args.group_by and mode != "manifest":
        raise ValueError("--group-by is available only with --manifest.")
    if mode == "files":
        return _json_safe(_calculate_pair_metrics(pairs[0], args))

    metrics_by_pair = [_calculate_pair_metrics(pair, args) for pair in pairs]
    records = []
    for pair, metrics in zip(pairs, metrics_by_pair):
        records.append(
            {
                "key": pair.key,
                "real": _path_text(pair.real.path),
                "synthetic": _path_text(pair.synthetic.path),
                "metadata": pair.metadata or {},
                "metrics": metrics,
            }
        )
    results: dict[str, Any] = {
        "pairs": records,
        "summary": _summarize_pair_metrics(metrics_by_pair),
    }
    if args.group_by:
        results["grouped_summary"] = _summarize_grouped_pair_metrics(records, args.group_by)
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


def _write_pairwise_csv(handle: Any, results: dict[str, Any]) -> None:
    writer = csv.writer(handle)
    writer.writerow(["scope", "key", "real", "synthetic", "metric", "value"])
    for record in results["pairs"]:
        for metric, value in _flatten(record["metrics"]):
            writer.writerow(
                [
                    "pair",
                    record.get("key", ""),
                    record.get("real") or "",
                    record.get("synthetic") or "",
                    metric,
                    value,
                ]
            )
    for metric, stats in results.get("summary", {}).get("metrics", {}).items():
        for stat_name, value in stats.items():
            writer.writerow(["summary", "", "", "", f"{metric}.{stat_name}", value])
    for group, summary in results.get("grouped_summary", {}).get("groups", {}).items():
        for metric, stats in summary.get("metrics", {}).items():
            for stat_name, value in stats.items():
                writer.writerow(
                    ["group", str(group), "", "", f"{metric}.{stat_name}", value]
                )


def _write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            if "pairs" in results:
                _write_pairwise_csv(handle, results)
            else:
                writer = csv.writer(handle)
                writer.writerow(["metric", "value"])
                writer.writerows(_flatten(results))
    else:
        raise ValueError("--output must end with .json or .csv.")


def _requested_outputs(args: argparse.Namespace) -> list[Path]:
    """Return all result files requested by the CLI.

    ``--output`` is kept for backward compatibility and accepts either JSON or CSV.
    ``--output-json`` and ``--output-csv`` are format-specific helpers that allow
    both files to be written from a single metric calculation.
    """

    outputs = []
    if args.output:
        outputs.append(args.output)
    if args.output_json:
        if args.output_json.suffix.lower() != ".json":
            raise ValueError("--output-json must end with .json.")
        outputs.append(args.output_json)
    if args.output_csv:
        if args.output_csv.suffix.lower() != ".csv":
            raise ValueError("--output-csv must end with .csv.")
        outputs.append(args.output_csv)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    args = _parser().parse_args(argv)
    try:
        results = calculate_metrics(args)
        for output in _requested_outputs(args):
            _write_results(output, results)
        print(json.dumps(results, indent=2, sort_keys=True))
    except (FileNotFoundError, ImportError, KeyError, TypeError, ValueError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
