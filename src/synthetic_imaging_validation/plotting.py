"""Optional, headless plots of saved results and validation histories."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from .history import load_history, result_series


def require_plot_support():
    """Load Matplotlib's non-interactive canvas without changing global backends."""
    try:
        return importlib.import_module("matplotlib.backends.backend_agg")
    except ImportError as exc:
        raise ImportError('Plotting requires Matplotlib; install the "viz" extra.') from exc


def validate_plot_path(path: Union[str, Path]) -> Path:
    """Accept explicit PNG, SVG or PDF figure destinations."""
    path = Path(path)
    if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("Plot path must end with .png, .svg or .pdf.")
    return path


def _select(available: set[str], metrics: Optional[Sequence[str]]) -> list[str]:
    if metrics is None:
        names = sorted(name for name in available if ".components." not in name and ".raw_metrics." not in name)
    else:
        names = [metrics] if isinstance(metrics, str) else list(metrics)
        names = [name + ".value" if name not in available and name + ".value" in available else name for name in names]
    if not names or len(names) > 12 or len(names) != len(set(names)):
        raise ValueError("Select between 1 and 12 distinct scalar metrics for a plot.")
    missing = set(names) - available
    if missing:
        raise ValueError(f"Unknown plot metrics: {sorted(missing)}. Available: {sorted(available)}")
    return names


def _figure(names: list[str], title: str):
    backend = require_plot_support()
    from matplotlib.figure import Figure

    columns = min(2, len(names))
    rows = (len(names) + columns - 1) // columns
    figure = Figure(figsize=(6 * columns, 3.5 * rows + 0.7), layout="constrained")
    backend.FigureCanvasAgg(figure)
    axes = [figure.add_subplot(rows, columns, index + 1) for index in range(len(names))]
    figure.suptitle(title + "\nNon-finite or missing values are gaps, not zero.", fontsize=12)
    for ax, name in zip(axes, names):
        ax.set_title(name, fontsize=10, parse_math=False)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
        if name in {"similarity_score", "intensity_distribution_score",
                    "similarity_score.value", "intensity_distribution_score.value"}:
            ax.set_ylim(-2, 102)
            ax.set_ylabel("Experimental score (0-100)")
    return figure, axes


def _save(figure: Any, output: Optional[Union[str, Path]]) -> None:
    if output is not None:
        path = validate_plot_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format=path.suffix[1:].lower(), dpi=150)


def plot_results(
    results: dict[str, Any], *, metrics: Optional[Sequence[str]] = None,
    output: Optional[Union[str, Path]] = None,
):
    """Plot per-pair values in separate panels, never mixing metric units.

    Accepts a CLI result or ``{'pairs': evaluate_pairs(...)}``. One pair is
    supported, although cohort comparisons are more informative. At most 12
    fields may be plotted; explicitly choose fields for larger result sets.
    Score names can omit the ``.value`` suffix. Returns a Matplotlib Figure;
    optionally saves PNG/SVG/PDF. Does not open a GUI or recalculate metrics.
    """
    labels, rows = result_series(results)
    names = _select({name for row in rows for name in row}, metrics)
    figure, axes = _figure(names, "Validation results by pair")
    for ax, name in zip(axes, names):
        values = [np.nan if row.get(name) is None else row[name] for row in rows]
        ax.plot(range(len(rows)), values, linestyle="none", marker="o")
        ticks = np.unique(np.linspace(0, len(rows) - 1, min(12, len(rows)), dtype=int))
        ax.set_xticks(ticks, [labels[index] for index in ticks], rotation=35, ha="right")
        ax.set_xlabel("Pair")
    _save(figure, output)
    return figure


def plot_history(
    history: Union[str, Path, dict[str, Any]], *, metrics: Optional[Sequence[str]] = None,
    output: Optional[Union[str, Path]] = None,
):
    """Plot one panel per metric and one curve per run, ordered by step.

    Accepts an ``append_history`` JSON path or mapping. No smoothing, confidence
    intervals or interpolation across missing values are added. Different runs
    may have different protocols: the caller must judge comparability. Returns
    a headless Matplotlib Figure and optionally saves PNG/SVG/PDF.
    """
    records = load_history(history)["records"]
    names = _select({name for record in records for name in record["metrics"]}, metrics)
    figure, axes = _figure(names, "Validation history")
    runs = sorted({record["run"] for record in records})
    for ax, name in zip(axes, names):
        for run in runs:
            ordered = sorted((record for record in records if record["run"] == run), key=lambda item: item["step"])
            values = [record["metrics"].get(name) for record in ordered]
            ax.plot([record["step"] for record in ordered],
                    [np.nan if value is None else value for value in values], marker="o", label=run)
        ax.set_xlabel("Epoch / step")
        ax.legend()
    _save(figure, output)
    return figure
