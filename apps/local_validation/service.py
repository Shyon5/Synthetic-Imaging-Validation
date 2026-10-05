"""File-backed application adapter; metric formulas remain in the core package.

Only paths inside the configured input root may be read. Each run is evaluated
one pair per worker, keeping memory bounded by the worker count rather than the
cohort size. Progress callbacks run on the caller's thread, never worker threads.
"""

from __future__ import annotations

import csv
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from synthetic_imaging_validation import pair_directory_files, plot_results
from synthetic_imaging_validation.cli import validate
from synthetic_imaging_validation.io.loading import is_supported_image_path
from synthetic_imaging_validation.history import result_series


@dataclass(frozen=True)
class Pair:
    key: str
    real: Path
    synthetic: Path
    metadata: dict[str, str]


def inside(root: Path, value: str) -> Path:
    """Resolve an existing input path and reject traversal or escaping symlinks."""
    root = root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Inputs must stay inside the selected data folder.")
    if not path.exists():
        raise ValueError(f"Input not found: {value}")
    return path


def catalogue(root: Path) -> tuple[list[str], list[str], list[str]]:
    """List bounded relative file/directory choices; never follow escaping links."""
    root = root.resolve()
    images, manifests, directories = [], [], ["."]
    if not root.is_dir():
        return images, manifests, directories
    for count, path in enumerate(root.rglob("*"), 1):
        if count > 10000:
            raise ValueError("The data folder has over 10,000 entries. Mount a smaller study folder.")
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(name)
        elif is_supported_image_path(path):
            images.append(name)
        elif path.suffix.lower() == ".csv":
            manifests.append(name)
    return sorted(images), sorted(manifests), sorted(directories)


def plan_pairs(root: Path, mode: str, *, real: str = "", synthetic: str = "",
               manifest: str = "", recursive: bool = False, group_by: str = "",
               pairing: str = "stem", real_column: str = "real", synthetic_column: str = "synthetic",
               key_column: str = "case_id", base_dir: str = "") -> list[Pair]:
    """Preview pairing without loading volumes; require explicit case correspondence."""
    root = root.resolve()
    pairs = []
    if mode == "files":
        pairs = [Pair("pair", inside(root, real), inside(root, synthetic), {})]
    elif mode == "directories":
        rows = pair_directory_files(inside(root, real), inside(root, synthetic), recursive=recursive, pairing=pairing)
        pairs = [Pair(key, inside(root, str(r)), inside(root, str(s)), {}) for key, r, s in rows]
    elif mode == "manifest":
        path = inside(root, manifest)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if real_column == synthetic_column or not {real_column, synthetic_column}.issubset(fields) or len(fields) != len(set(fields)):
                raise ValueError("Manifest needs unique column names including the selected real and synthetic columns.")
            if key_column and key_column != "case_id" and key_column not in fields:
                raise ValueError("Selected case ID column is missing.")
            parent = inside(root, base_dir) if base_dir else path.parent
            for index, row in enumerate(reader, 1):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed manifest row {index}.")
                paths = []
                for column in (real_column, synthetic_column):
                    if not row[column].strip():
                        raise ValueError(f"Empty {column} path on manifest row {index}.")
                    paths.append(inside(root, str(parent / row[column].strip())))
                metadata = {k: v for k, v in row.items() if k not in {real_column, synthetic_column}}
                key = row.get(key_column, "").strip() or str(index)
                pairs.append(Pair(key, paths[0], paths[1], metadata))
    else:
        raise ValueError("Unknown input mode.")
    if not pairs:
        raise ValueError("No pairs found.")
    if len({pair.key for pair in pairs}) != len(pairs):
        raise ValueError("Case IDs must be unique.")
    for pair in pairs:
        if not all(is_supported_image_path(p) for p in (pair.real, pair.synthetic)):
            raise ValueError(f"Unsupported image file in pair {pair.key}.")
        if group_by and not pair.metadata.get(group_by, "").strip():
            raise ValueError(f"Missing group column '{group_by}' for pair {pair.key}.")
    return pairs


def make_demo(root: Path, kind: str = "images") -> Path:
    """Create small 2D phantoms, never patient images, plus a labelled manifest."""
    for name in ("real", "synthetic"):
        (root / name).mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[-1:1:64j, -1:1:64j]
    rows = []
    for i in range(4):
        if kind == "masks":
            real = (x * x + y * y < 0.4 ** 2).astype(np.float32)
            synthetic = ((x - 0.035 * i) ** 2 + y * y < (0.4 + i * 0.02) ** 2).astype(np.float32)
        else:
            real = (0.1 + 0.65 * np.exp(-5 * (x * x + y * y))).astype(np.float32)
            synthetic = real + np.float32(0.025 * i)
        name = f"case_{i + 1:02d}.npy"
        np.save(root / "real" / name, real)
        np.save(root / "synthetic" / name, synthetic)
        rows.append([f"case_{i + 1:02d}", f"real/{name}", f"synthetic/{name}", "A" if i < 2 else "B"])
    path = root / "pairs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "real", "synthetic", "label"])
        writer.writerows(rows)
    return path


def run_validation(pairs: list[Pair], options: dict, output_root: Path,
                   progress: Optional[Callable[[int, int], None]] = None) -> tuple[dict, Path]:
    """Call the existing CLI calculation API, then export a unique result bundle.

    Metric, range and geometry validation is delegated to the same core path
    used by the CLI. The adapter only supplies file paths and UI options.
    """
    metrics = options.get("metrics", [])
    if not metrics:
        raise ValueError("Select at least one metric.")
    if not pairs:
        raise ValueError("Select at least one pair.")
    workers = int(options.get("workers", 1))
    if not 1 <= workers <= 16:
        raise ValueError("Choose between 1 and 16 workers.")
    from .extra_metrics import EXTRA_METRICS, calculate_extras
    unknown = set(metrics) - set(validate.SUPPORTED_METRICS) - set(EXTRA_METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    cli_metrics = [name for name in metrics if name in validate.SUPPORTED_METRICS]
    custom_nrmse = "nrmse" in metrics and options.get("nrmse_normalization", "range") != "range"
    if custom_nrmse:
        cli_metrics.remove("nrmse")
    # A cheap placeholder makes the unchanged CLI parser usable for API-only metrics.
    common = ["--metrics", *(cli_metrics or ["mae"])]
    for key, flag in (("data_range", "--data-range"), ("threshold", "--threshold"),
                      ("bins", "--bins"), ("channel_axis", "--channel-axis"),
                      ("batch_axis", "--batch-axis"), ("ms_ssim_backend", "--ms-ssim-backend")):
        if options.get(key) is not None:
            common += [flag, str(options[key])]
    if options.get("spacing"):
        common += ["--spacing", *map(str, options["spacing"])]
    if options.get("border_width"):
        common += ["--border-width", *map(str, options["border_width"])]
    if options.get("allow_spatial_mismatch"):
        common += ["--allow-spatial-mismatch"]
    if options.get("scores"):
        common += ["--scores", *options["scores"], "--score-range", *map(str, options["score_range"])]
    # Parse before creating workers: argument errors must not trigger work.
    validate._parser().parse_args(common)

    def calculate(pair: Pair) -> dict:
        args = validate._parser().parse_args(["--real", str(pair.real), "--synthetic", str(pair.synthetic), *common])
        try:
            result = validate.calculate_metrics(args)
            if not cli_metrics:
                result.pop("mae", None)
            if custom_nrmse:
                from synthetic_imaging_validation import load_pair, nrmse
                a, b = load_pair(pair.real, pair.synthetic,
                                 require_spatial_match=not options.get("allow_spatial_mismatch", False))
                result["nrmse"] = nrmse(a.array, b.array, normalization=options["nrmse_normalization"])
            result.update(calculate_extras(pair, options))
            return result
        except (ValueError, OSError, TypeError, ImportError) as exc:
            raise ValueError(f"Pair '{pair.key}': {exc}") from exc

    values = []
    if progress:
        progress(0, len(pairs))
    with ThreadPoolExecutor(max_workers=min(workers, len(pairs))) as executor:
        # Keep only a bounded window of active work. This avoids executor.map's
        # eager submission retaining the entire cohort during slow early cases.
        for start in range(0, len(pairs), workers):
            for result in executor.map(calculate, pairs[start:start + workers]):
                values.append(result)
                if progress:
                    progress(len(values), len(pairs))
    protocol = values[0].get("score_protocol")
    records = []
    for pair, value in zip(pairs, values):
        value.pop("score_protocol", None)
        records.append({"key": pair.key, "real": str(pair.real), "synthetic": str(pair.synthetic),
                        "metadata": pair.metadata, "metrics": value})
    report = {"pairs": records, "summary": validate._summarize_pair_metrics(values)}
    if protocol is not None:
        report["score_protocol"] = protocol
    if options.get("group_by"):
        report["grouped_summary"] = validate._summarize_grouped_pair_metrics(records, options["group_by"])
    return save_report(report, options, output_root)


def save_report(report: dict, options: dict, output_root: Path) -> tuple[dict, Path]:
    """Export an immutable run folder; optional figures contain scalar fields only."""
    report = validate._json_safe(report)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output_root / f"validation_{stamp}_{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    for suffix in ("json", "csv", "pdf", "tex"):
        validate._write_results(destination / f"results.{suffix}", report, pdf_font=options.get("pdf_font"))
    (destination / "settings.json").write_text(json.dumps(options, indent=2) + "\n", encoding="utf-8")
    _, scalar_rows = result_series(report)
    available = sorted({key for row in scalar_rows for key in row})
    selected = options.get("plot_metrics") or options.get("metrics") or available
    fields = [name for name in available if any(name == item or name.startswith(item + ".") for item in selected)
              and ".components." not in name and ".raw_metrics." not in name][:6]
    if fields:
        figure = plot_results(report, metrics=fields, output=destination / "metrics.png")
        for extension in ("svg", "pdf"):
            figure.savefig(destination / f"metrics.{extension}")
        figure.clear()
    with ZipFile(destination / "results_bundle.zip", "w", ZIP_DEFLATED) as bundle:
        for path in sorted(destination.iterdir()):
            if path.suffix != ".zip":
                bundle.write(path, path.name)
    return report, destination


def record_run_history(report, options, output_root, *, filename="history.json", run="validation", step=0):
    """Append a completed result with its saved metric settings, not current controls."""
    from synthetic_imaging_validation import append_history
    from .profiles import defaults
    if not filename.strip() or Path(filename).suffix.lower() != ".json":
        raise ValueError("Choose a .json history filename.")
    path = (output_root / filename).resolve()
    if not path.is_relative_to(output_root.resolve()):
        raise ValueError("History must stay inside the results folder.")
    if not run.strip():
        raise ValueError("Enter a run name.")
    # Leave paths, display choices and worker count out of the numerical protocol.
    fields = set(defaults()) - {"workers", "plot_metrics"}
    protocol = {key: value for key, value in options.items() if key in fields}
    append_history(path, report, step=step, run=run.strip(), protocol={"evaluation_settings": protocol})
    return path
