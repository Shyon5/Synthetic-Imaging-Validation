"""Small, versioned validation histories independent of any training framework."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def _scalar_values(mapping: dict[str, Any], prefix: str = "") -> dict[str, Optional[float]]:
    """Extract scalar leaves; retain undefined numbers as gaps, never zeros."""
    if not isinstance(mapping, dict):
        raise ValueError("Metric results must be dictionaries.")
    values = {}
    for key, item in mapping.items():
        if key in {"score_protocol", "protocol"}:
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            additions = _scalar_values(item, name)
        elif item is None or isinstance(item, str) and item in {"Infinity", "-Infinity", "NaN"}:
            additions = {name: None}
        elif isinstance(item, (int, float, np.number)) and not isinstance(item, (bool, np.bool_)):
            value = float(item)
            additions = {name: value if np.isfinite(value) else None}
        else:
            # Non-scalar leaves contribute no values. An explicit empty mapping
            # also avoids Python 3.9's untraceable optimized continue branch.
            additions = {}
        if values.keys() & additions.keys():
            raise ValueError("Ambiguous dotted metric names in results.")
        values.update(additions)
    return values


def result_series(results: dict[str, Any]) -> tuple[list[str], list[dict[str, Optional[float]]]]:
    """Return case labels and scalar metric rows from CLI/API evaluation results.

    Accepts a single metric mapping or ``{'pairs': evaluate_pairs(...)}``.
    Lists, descriptive strings and protocol metadata are not plotted. Nested
    scalar fields use dotted names, e.g. ``similarity_score.value``.
    """
    if not isinstance(results, dict):
        raise ValueError("results must be a dictionary.")
    if "pairs" not in results:
        return ["pair"], [_scalar_values(results)]
    if not isinstance(results["pairs"], list) or not results["pairs"]:
        raise ValueError("pairs must be a non-empty list of result records.")
    labels, rows = [], []
    for index, pair in enumerate(results["pairs"], start=1):
        if not isinstance(pair, dict) or "metrics" not in pair:
            raise ValueError("Each pair record must contain a metrics dictionary.")
        labels.append(str(pair.get("key", index)))
        rows.append(_scalar_values(pair["metrics"]))
    return labels, rows


def _step(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("step must be a non-negative integer.")
    return int(value)


def load_history(source: Union[str, Path, dict[str, Any]]) -> dict[str, Any]:
    """Read and validate a history JSON file (or an in-memory history mapping)."""
    history = json.loads(Path(source).read_text(encoding="utf-8")) if not isinstance(source, dict) else source
    if not isinstance(history, dict) or history.get("schema_version") != 1 or not isinstance(history.get("records"), list):
        raise ValueError("Unsupported history; expected schema_version=1 and a records list.")
    seen = set()
    for record in history["records"]:
        if not isinstance(record, dict):
            raise ValueError("Invalid history record.")
        step = _step(record.get("step"))
        run = record.get("run")
        if not isinstance(run, str) or not run.strip():
            raise ValueError("History run must be a non-empty string.")
        key = (run, step)
        if key in seen:
            raise ValueError(f"Duplicate history entry for run={run!r}, step={step}.")
        seen.add(key)
        metrics, counts = record.get("metrics"), record.get("counts")
        if not isinstance(metrics, dict) or not metrics or not isinstance(counts, dict) or metrics.keys() != counts.keys():
            raise ValueError("History metrics and counts must have matching, non-empty fields.")
        for name, value in metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("History metric names must be non-empty strings.")
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)):
                raise ValueError("History values must be finite numbers or null.")
            count = counts[name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0 or (value is None) != (count == 0):
                raise ValueError("History counts must agree with finite or missing values.")
        if not isinstance(record.get("protocol"), dict):
            raise ValueError("History records require a protocol dictionary.")
    return history


def append_history(
    path: Union[str, Path], results: dict[str, Any], *, step: int,
    run: str = "validation", protocol: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Record an evaluation at an epoch/step and return the updated history.

    For paired results, records the mean of finite per-pair values with equal
    case weighting, plus each metric's contributing count. All-nonfinite metrics
    remain null. No metric is recalculated. A plain scalar mapping is supported
    for training-loop API use. Score protocol metadata is retained automatically
    when supplied as ``results['score_protocol']``; put preprocessing and other
    experimental settings in ``protocol``.

    Duplicate (run, step), changed metric sets or changed protocols within a run
    are rejected. Use a new run name for a different experiment. Steps need not
    arrive in order. File replacement is atomic; use ONE writer per history file
    (e.g. rank zero in distributed training). Concurrent writers are not supported.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("History path must end with .json.")
    step = _step(step)
    if not isinstance(run, str) or not run.strip():
        raise ValueError("run must be a non-empty string.")
    if protocol is not None and not isinstance(protocol, dict):
        raise ValueError("protocol must be a dictionary.")
    _, rows = result_series(results)
    names = sorted({name for row in rows for name in row})
    metrics, counts = {}, {}
    for name in names:
        values = [row[name] for row in rows if row.get(name) is not None]
        metrics[name] = float(np.mean(values)) if values else None
        counts[name] = len(values)
    record = {"run": run, "step": step, "metrics": metrics, "counts": counts,
              "protocol": {"scores": results.get("score_protocol", {}), "evaluation": protocol or {}}}
    history = load_history(path) if path.exists() else {"schema_version": 1, "records": []}
    for previous in history["records"]:
        if previous["run"] == run:
            if previous["protocol"] != record["protocol"] or previous["metrics"].keys() != metrics.keys():
                raise ValueError("Metric set or protocol changed within a run; use a new run name.")
    history["records"].append(record)
    load_history(history)
    payload = json.dumps(history, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return history
