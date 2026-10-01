import copy
import importlib
import json
import runpy
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from synthetic_imaging_validation import append_history, load_history, plot_history, plot_results
from synthetic_imaging_validation import history, plotting
from synthetic_imaging_validation.cli import plot_history as plot_cli


@pytest.mark.parametrize("ignored", [
    True, False, np.bool_(True), np.bool_(False), "case", "", [1], (1,),
    np.array([1]), np.array(1),
])
def test_history_ignores_nonmetric_leaves(ignored):
    # Ignored leaves must neither become numbers nor reuse the preceding value.
    assert history._scalar_values({"ignored": ignored}) == {}
    assert history._scalar_values({
        "before": 1, "ignored": ignored,
        "nested": {"ignored": ignored, "valid": np.float64(2)}, "after": 3,
    }) == {"before": 1.0, "nested.valid": 2.0, "after": 3.0}


def test_history_means_counts_gaps_and_order(tmp_path):
    assert history._scalar_values({"score": {"value": 90, "protocol": {"bins": 64}}}) == {"score.value": 90}
    path = tmp_path / "nested" / "history.json"
    result = {"pairs": [
        {"key": "one", "metrics": {"mae": 1, "psnr": "Infinity", "nan": np.nan, "ok": True, "sizes": [1]}},
        {"metrics": {"mae": 3, "psnr": 10, "nan": None, "description": "case"}},
    ], "score_protocol": {"version": "test"}}
    h = append_history(path, result, step=np.int64(2), protocol={"range": [0, 1]})
    record = h["records"][0]
    assert record["metrics"] == {"mae": 2, "psnr": 10, "nan": None}
    assert record["counts"] == {"mae": 2, "psnr": 1, "nan": 0}
    h = append_history(path, result, step=1, protocol={"range": [0, 1]})
    assert [row["step"] for row in h["records"]] == [2, 1]
    assert load_history(path) == h
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Duplicate"):
        append_history(path, result, step=1, protocol={"range": [0, 1]})
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="protocol changed"):
        append_history(path, result, step=3, protocol={"range": [0, 2]})
    with pytest.raises(ValueError, match="Metric set"):
        append_history(path, {"mae": 1}, step=3)
    assert len(append_history(path, {"mae": 2}, step=1, run="new-model")["records"]) == 3


@pytest.mark.parametrize("step", [-1, True, 1.5, None])
def test_invalid_step(tmp_path, step):
    with pytest.raises(ValueError, match="step"):
        append_history(tmp_path / "history.json", {"mae": 0}, step=step)


def test_history_invalid_arguments(tmp_path):
    assert history._scalar_values({"mae": 1, "score_protocol": {"bins": 64}}) == {"mae": 1}
    with pytest.raises(ValueError, match=".json"):
        append_history(tmp_path / "history.csv", {}, step=1)
    for run in ("", None):
        with pytest.raises(ValueError, match="run"):
            append_history(tmp_path / "history.json", {"mae": 0}, step=1, run=run)
    with pytest.raises(ValueError, match="protocol"):
        append_history(tmp_path / "history.json", {"mae": 0}, step=1, protocol=[])
    with pytest.raises(ValueError, match="non-empty"):
        append_history(tmp_path / "history.json", {}, step=1)
    with pytest.raises(ValueError, match="Ambiguous"):
        history._scalar_values({"a.b": 1, "a": {"b": 2}})
    for result in ([], {"pairs": []}, {"pairs": 1}, {"pairs": [1]}, {"pairs": [{"metrics": []}]}):
        with pytest.raises(ValueError):
            history.result_series(result)


def test_invalid_saved_histories(tmp_path):
    good = append_history(tmp_path / "history.json", {"mae": 0}, step=0)
    variants = [[], {}, {"schema_version": 99, "records": []}, {"schema_version": 1, "records": [1]}]
    for field, value in (("run", ""), ("metrics", {}), ("counts", {}), ("protocol", None)):
        bad = copy.deepcopy(good)
        bad["records"][0][field] = value
        variants.append(bad)
    for value in (True, "text", float("inf")):
        bad = copy.deepcopy(good)
        bad["records"][0]["metrics"]["mae"] = value
        variants.append(bad)
    for value in (-1, True, 0):
        bad = copy.deepcopy(good)
        bad["records"][0]["counts"]["mae"] = value
        variants.append(bad)
    bad = copy.deepcopy(good)
    bad["records"][0]["metrics"] = {"": 0}
    bad["records"][0]["counts"] = {"": 1}
    variants.append(bad)
    for bad in variants:
        if isinstance(bad, dict):
            with pytest.raises(ValueError):
                load_history(bad)
        else:
            path = tmp_path / "bad.json"
            path.write_text(json.dumps(bad))
            with pytest.raises(ValueError):
                load_history(path)


def test_atomic_history_failure_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    append_history(path, {"mae": 1}, step=1)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(history.os, "replace", fail)
    with pytest.raises(OSError):
        append_history(path, {"mae": 2}, step=2)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(history.tempfile, "NamedTemporaryFile", fail)
    with pytest.raises(OSError):
        append_history(path, {"mae": 2}, step=2)


def test_plots_have_separate_panels_sorted_steps_and_gaps(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "history.json"
    result = {"mae": 0.2, "psnr": "Infinity", "similarity_score": {"value": 87.5, "components": {"structure": 0.8}}}
    append_history(path, result, step=2)
    result["mae"], result["psnr"] = 0.4, 10
    append_history(path, result, step=1)
    append_history(path, result, step=1, run="other")
    figure = plot_history(path, metrics=["mae", "psnr", "similarity_score"], output=tmp_path / "plots" / "history.png")
    assert len(figure.axes) == 3
    assert len(figure.axes[0].lines) == 2
    np.testing.assert_array_equal(figure.axes[0].lines[1].get_xdata(), [1, 2])
    np.testing.assert_allclose(figure.axes[0].lines[1].get_ydata(), [0.4, 0.2])
    assert np.isnan(figure.axes[1].lines[1].get_ydata()[1])
    assert figure.axes[2].get_ylim() == (-2, 102)
    assert (tmp_path / "plots" / "history.png").read_bytes().startswith(b"\x89PNG")
    plot_history(path)  # automatic selection skips component internals
    pair_plot = plot_results({"pairs": [{"key": "A", "metrics": result}, {"key": "B", "metrics": {"mae": 0.3}}]},
                             metrics="mae", output=tmp_path / "pairs.svg")
    assert [tick.get_text() for tick in pair_plot.axes[0].get_xticklabels()] == ["A", "B"]
    plot_results(result, metrics=["psnr", "similarity_score"], output=tmp_path / "pair.pdf")
    assert (tmp_path / "pair.pdf").read_bytes().startswith(b"%PDF")
    scalar_plot = plot_results({"similarity_score": 90})
    assert scalar_plot.axes[0].get_ylim() == (-2, 102)


def test_selection_and_missing_matplotlib(tmp_path, monkeypatch):
    for metrics in ([], ["mae", "mae"], [str(i) for i in range(13)]):
        with pytest.raises(ValueError, match="Select"):
            plotting._select({"mae"}, metrics)
    with pytest.raises(ValueError, match="Unknown"):
        plotting._select({"mae"}, ["mse"])
    with pytest.raises(ValueError, match="Plot path"):
        plotting.validate_plot_path("plot.txt")
    with pytest.raises(ValueError):
        plot_history({"schema_version": 1, "records": []})
    def missing(name):
        raise ImportError("missing")
    monkeypatch.setattr(plotting.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="viz"):
        plot_results({"mae": 1})
    # History persistence remains available with no plotting dependency.
    append_history(tmp_path / "history.json", {"mae": 0}, step=1)


def test_plot_cli_and_module_entrypoint(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    path = tmp_path / "history.json"
    append_history(path, {"mae": 0}, step=0)
    arguments = ["--history", str(path), "--output", str(tmp_path / "plot.png"), "--metrics", "mae"]
    assert plot_cli.main(arguments) == 0
    with pytest.raises(SystemExit):
        plot_cli.main(["--history", "missing.json", "--output", str(tmp_path / "plot.png")])
    monkeypatch.setattr(sys, "argv", ["plot_history"] + arguments)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as error:
            runpy.run_module("synthetic_imaging_validation.cli.plot_history", run_name="__main__")
    assert error.value.code == 0


def test_runnable_history_example(tmp_path, monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    script = Path(__file__).resolve().parents[1] / "examples" / "validation_history.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--output-dir", str(tmp_path)])
    runpy.run_path(str(script), run_name="__main__")
    records = load_history(tmp_path / "history.json")["records"]
    assert [row["step"] for row in records] == [1, 2, 3, 4, 5]
    for name in ("similarity_score.value", "intensity_distribution_score.value"):
        values = [row["metrics"][name] for row in records]
        assert all(a < b for a, b in zip(values, values[1:]))
    assert (tmp_path / "scores.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (tmp_path / "last_step.svg").read_text(encoding="utf-8")
    assert "Step 5:" in capsys.readouterr().out
