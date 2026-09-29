"""Command-line validation of aligned image or mask pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..evaluation import parallel_map, resolve_num_workers
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
from ..reporting import require_pdf_support, write_report
from ..metrics.scores import similarity_score, intensity_distribution_score, validate_score_range
from ..history import append_history
from ..plotting import plot_results, plot_history, require_plot_support, validate_plot_path

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
    parser.add_argument("--scores", nargs="+", choices=("similarity", "intensity_distribution"),
                        help="Optional experimental composite scores (0-100), not clinical ratings.")
    parser.add_argument("--score-range", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        help="Required fixed intensity bounds for scores; inputs are not rescaled.")
    parser.add_argument("--plot-output", type=Path, help="Per-pair metric figure (.png/.svg/.pdf; requires viz).")
    parser.add_argument("--plot-metrics", nargs="+", help="Scalar fields to plot; score names may omit .value.")
    parser.add_argument("--history", type=Path, help="Append this evaluation to a versioned history JSON.")
    parser.add_argument("--step", "--epoch", dest="step", type=int, help="Non-negative history epoch/step.")
    parser.add_argument("--run-name", default="validation", help="History series name (default: validation).")
    parser.add_argument("--plot-history", type=Path, help="Plot updated --history to .png/.svg/.pdf (requires viz).")
    parser.add_argument("--output", type=Path, help="Optional single .json, .csv, .pdf, or .tex result file.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON result file.")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV result file.")
    parser.add_argument("--output-pdf", type=Path, help="Optional PDF report (requires the report extra).")
    parser.add_argument("--output-latex", type=Path, help="Optional standalone LaTeX (.tex) report.")
    parser.add_argument("--pdf-font", type=Path, help="Optional TrueType font for PDF labels outside the default character set.")
    parser.add_argument("--data-range", type=float, help="Known intensity range for PSNR/SSIM/MS-SSIM.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold (default: 0.5).")
    parser.add_argument("--bins", type=int, default=64, help="Histogram bins for KL/JS (default: 64).")
    parser.add_argument("--spacing", nargs="+", type=float, help="Axis-order spacing when files lack it.")
    parser.add_argument("--channel-axis", type=int, help="Explicit channel axis for SSIM/MS-SSIM.")
    parser.add_argument("--batch-axis", type=int, help="Explicit batch axis for SSIM/MS-SSIM.")
    parser.add_argument(
        "--ms-ssim-backend",
        choices=("numpy", "torchmetrics"),
        default="numpy",
        help=(
            "MS-SSIM backend: lightweight NumPy/SciPy (default) or the optional "
            "TorchMetrics reference backend."
        ),
    )
    parser.add_argument("--border-width", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--allow-spatial-mismatch",
        action="store_true",
        help="Skip NIfTI spacing/affine checks. Shape checks still apply.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show a progress bar while evaluating real/synthetic pairs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads used for pair-level metric calculation. "
            "Use 1 for sequential execution (default), or 0 to use available CPU cores."
        ),
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
    cached = {}
    data_range = args.data_range
    if args.scores:
        bounds = validate_score_range(args.score_range)
        protocols = {}
        if "similarity" in args.scores:
            data_range = bounds[1] - bounds[0]
            detail = similarity_score(real, synthetic, value_range=bounds, channel_axis=args.channel_axis,
                                      backend=args.ms_ssim_backend, return_details=True)
            protocols["similarity_score"] = detail.pop("protocol")
            cached.update(detail["raw_metrics"])
            results["similarity_score"] = detail
        if "intensity_distribution" in args.scores:
            detail = intensity_distribution_score(real, synthetic, value_range=bounds, bins=args.bins,
                                                  return_details=True)
            protocols["intensity_distribution_score"] = detail.pop("protocol")
            cached.update(detail["raw_metrics"])
            results["intensity_distribution_score"] = detail
        results["score_protocol"] = protocols
    requested = set(args.metrics)
    simple = {"mae": mae, "mse": mse, "rmse": rmse, "nrmse": nrmse}
    for name, function in simple.items():
        if name in requested:
            results[name] = cached[name] if name in cached else function(real, synthetic)
    if "psnr" in requested:
        results["psnr"] = psnr(real, synthetic, data_range=data_range)
    if "ssim" in requested:
        results["ssim"] = ssim(
            real,
            synthetic,
            data_range=data_range,
            channel_axis=args.channel_axis,
            batch_axis=args.batch_axis,
        )
    if "ms_ssim" in requested:
        results["ms_ssim"] = cached["ms_ssim"] if "ms_ssim" in cached else ms_ssim(
            real,
            synthetic,
            data_range=data_range,
            channel_axis=args.channel_axis,
            batch_axis=args.batch_axis,
            backend=args.ms_ssim_backend,
        )
    if "wasserstein" in requested:
        results["wasserstein"] = cached["wasserstein"] if "wasserstein" in cached else wasserstein_distance(real, synthetic)
    if "kl" in requested:
        histogram_range = args.score_range if args.scores and "intensity_distribution" in args.scores else None
        results["kl"] = kl_divergence(real, synthetic, bins=args.bins, value_range=histogram_range)
    if "js" in requested:
        results["js"] = cached["js"] if "js" in cached else jensen_shannon_divergence(real, synthetic, bins=args.bins)
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


def _resolve_num_workers(requested: int, num_pairs: int) -> int:
    """Resolve the effective number of workers for pair-level metric execution."""

    try:
        return resolve_num_workers(requested, num_pairs)
    except ValueError as exc:
        raise ValueError(str(exc).replace("num_workers", "--num-workers")) from exc


def _calculate_metrics_for_pairs(
    pairs: list[ImagePair],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Calculate metrics for all pairs, optionally in parallel.

    Parallel execution is done per real/synthetic pair and preserves the input
    order. Threads are used instead of processes to avoid copying large image
    arrays between workers, which is especially important on Windows.
    """

    workers = _resolve_num_workers(getattr(args, "num_workers", 1), len(pairs))
    return parallel_map(
        pairs,
        lambda pair: _calculate_pair_metrics(pair, args),
        num_workers=workers,
        show_progress=args.show_progress,
        progress_description="Validating pairs",
        progress_unit="pair",
    )


def calculate_metrics(args: argparse.Namespace) -> dict[str, Any]:
    """Calculate requested CLI metrics and return a JSON-compatible dictionary."""

    if args.scores:
        if args.score_range is None:
            raise ValueError("--scores requires --score-range LOW HIGH.")
        bounds = validate_score_range(args.score_range)
        if args.batch_axis is not None:
            raise ValueError("Scores require one case per pair; --batch-axis is not supported with --scores.")
        if "similarity" in args.scores and args.data_range is not None and not np.isclose(
            args.data_range, bounds[1] - bounds[0], rtol=1e-7, atol=0
        ):
            raise ValueError("--data-range must equal the width of --score-range for similarity scores.")
    elif args.score_range is not None:
        raise ValueError("--score-range requires --scores.")
    mode, pairs = _load_cli_pairs(args)
    if args.group_by and mode != "manifest":
        raise ValueError("--group-by is available only with --manifest.")
    if mode == "files":
        return _json_safe(_calculate_metrics_for_pairs(pairs, args)[0])

    metrics_by_pair = _calculate_metrics_for_pairs(pairs, args)
    protocols = [metrics.pop("score_protocol", None) for metrics in metrics_by_pair]
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
    if protocols[0] is not None:
        results["score_protocol"] = protocols[0]
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


def _write_results(path: Path, results: dict[str, Any], *, pdf_font: Optional[Path] = None) -> None:
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
    elif path.suffix.lower() in {".pdf", ".tex"}:
        write_report(results, path, pdf_font=pdf_font)
    else:
        raise ValueError("--output must end with .json, .csv, .pdf, or .tex.")


def _requested_outputs(args: argparse.Namespace) -> list[Path]:
    """Return all result files requested by the CLI.

    All four formats reuse a single metric calculation. JSON and CSV retain
    their existing schemas; PDF and LaTeX present the same result mapping.
    """

    outputs = []
    if args.output:
        if args.output.suffix.lower() not in {".json", ".csv", ".pdf", ".tex"}:
            raise ValueError("--output must end with .json, .csv, .pdf, or .tex.")
        outputs.append(args.output)
    if args.output_json:
        if args.output_json.suffix.lower() != ".json":
            raise ValueError("--output-json must end with .json.")
        outputs.append(args.output_json)
    if args.output_csv:
        if args.output_csv.suffix.lower() != ".csv":
            raise ValueError("--output-csv must end with .csv.")
        outputs.append(args.output_csv)
    for option, suffix in (("output_pdf", ".pdf"), ("output_latex", ".tex")):
        path = getattr(args, option, None)
        if path is not None:
            if path.suffix.lower() != suffix:
                raise ValueError(f"--{option.replace('_', '-')} must end with {suffix}.")
            outputs.append(path)
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    args = _parser().parse_args(argv)
    try:
        outputs = _requested_outputs(args)
        if (args.history is None) != (args.step is None):
            raise ValueError("--history and --step/--epoch must be supplied together.")
        if args.history is not None and (args.history.suffix.lower() != ".json" or args.step < 0):
            raise ValueError("History requires a .json path and a non-negative step.")
        if args.plot_history is not None and args.history is None:
            raise ValueError("--plot-history requires --history and --step.")
        if args.plot_metrics and args.plot_output is None and args.plot_history is None:
            raise ValueError("--plot-metrics requires --plot-output or --plot-history.")
        additional = [path for path in (args.history, args.plot_output, args.plot_history) if path is not None]
        paths = [path.resolve() for path in outputs + additional]
        if len(paths) != len(set(paths)):
            raise ValueError("Result, history and plot destinations must be distinct.")
        for path in (args.plot_output, args.plot_history):
            if path is not None:
                validate_plot_path(path)
                require_plot_support()
        if any(output.suffix.lower() == ".pdf" for output in outputs):
            require_pdf_support()
            if args.pdf_font is not None and not args.pdf_font.is_file():
                raise FileNotFoundError(f"PDF font not found: {args.pdf_font}")
        results = calculate_metrics(args)
        if args.history is not None:
            protocol = {name: getattr(args, name) for name in (
                "data_range", "bins", "threshold", "spacing", "channel_axis", "batch_axis",
                "border_width", "ms_ssim_backend", "allow_spatial_mismatch")}
            protocol["metrics"] = sorted(set(args.metrics))
            history = append_history(args.history, results, step=args.step, run=args.run_name, protocol=protocol)
            if args.plot_history is not None:
                plot_history(history, metrics=args.plot_metrics, output=args.plot_history)
        if args.plot_output is not None:
            plot_results(results, metrics=args.plot_metrics, output=args.plot_output)
        for output in outputs:
            _write_results(output, results, pdf_font=args.pdf_font)
        print(json.dumps(results, indent=2, sort_keys=True))
    except (OSError, ImportError, KeyError, TypeError, ValueError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
