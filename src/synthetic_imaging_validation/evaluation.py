"""High-level helpers for evaluating real/synthetic image pairs."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, TypeVar

from tqdm.auto import tqdm

from .io.pairing import ImagePair

T = TypeVar("T")
R = TypeVar("R")


def resolve_num_workers(requested: int, num_items: int) -> int:
    """Resolve the effective worker count for item-level parallel evaluation.

    Parameters
    ----------
    requested:
        User-requested number of workers. Use ``1`` for sequential execution
        and ``0`` to use the available CPU cores.
    num_items:
        Number of items that will be evaluated.

    Returns
    -------
    int
        A worker count between 1 and ``num_items``. Empty inputs return 1.
    """

    if requested < 0:
        raise ValueError("num_workers must be greater than or equal to 0.")
    if num_items <= 1:
        return 1
    if requested == 0:
        requested = os.cpu_count() or 1
    return max(1, min(requested, num_items))


def parallel_map(
    items: Sequence[T],
    function: Callable[[T], R],
    *,
    num_workers: int = 1,
    show_progress: bool = False,
    progress_description: str = "Evaluating items",
    progress_unit: str = "item",
) -> list[R]:
    """Apply a function to items sequentially or with a thread pool.

    The output order always matches the input order. Threads are used instead of
    processes to avoid copying large arrays between workers, which keeps the
    helper portable across Windows, Linux, and macOS.
    """

    workers = resolve_num_workers(num_workers, len(items))
    if workers == 1:
        iterator = tqdm(items, desc=progress_description, unit=progress_unit) if show_progress else items
        return [function(item) for item in iterator]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(function, items)
        if show_progress:
            results = tqdm(
                results,
                total=len(items),
                desc=f"{progress_description} ({workers} workers)",
                unit=progress_unit,
            )
        return list(results)


def _path_text(path: Optional[Path]) -> Optional[str]:
    return None if path is None else str(path)


def evaluate_pairs(
    pairs: Sequence[ImagePair],
    metric_function: Callable[[ImagePair], Mapping[str, Any]],
    *,
    num_workers: int = 1,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a custom metric function on real/synthetic pairs.

    Parameters
    ----------
    pairs:
        Sequence of ``ImagePair`` objects, typically returned by
        ``load_paired_directories`` or ``load_manifest_pairs``.
    metric_function:
        Callable receiving one ``ImagePair`` and returning a mapping of metric
        names to values.
    num_workers:
        Number of worker threads used across pairs. Use ``1`` for sequential
        execution and ``0`` to use available CPU cores.
    show_progress:
        If ``True``, show a tqdm progress bar.

    Returns
    -------
    list of dict
        One record per pair with ``key``, optional paths, metadata, and the
        metric mapping returned by ``metric_function``.

    Notes
    -----
    This helper does not choose metrics for you. It is intended for notebooks
    and project scripts where the metric set is written directly in Python.
    """

    def evaluate_one(pair: ImagePair) -> dict[str, Any]:
        metrics = metric_function(pair)
        if not isinstance(metrics, Mapping):
            raise TypeError("metric_function must return a mapping of metric names to values.")
        return {
            "key": pair.key,
            "real": _path_text(pair.real.path),
            "synthetic": _path_text(pair.synthetic.path),
            "metadata": pair.metadata or {},
            "metrics": dict(metrics),
        }

    return parallel_map(
        pairs,
        evaluate_one,
        num_workers=num_workers,
        show_progress=show_progress,
        progress_description="Evaluating pairs",
        progress_unit="pair",
    )
