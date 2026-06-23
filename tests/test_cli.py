import csv
import json

import numpy as np
import pytest

from synthetic_imaging_validation.cli.validate import (
    SUPPORTED_METRICS,
    _json_safe,
    _parser,
    _resolve_spacing,
    _write_results,
    calculate_metrics,
    main,
)


@pytest.mark.torch
def test_calculate_metrics_exercises_every_cli_metric(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchmetrics")
    image = np.zeros((128, 128), dtype=np.float32)
    image[32:96, 32:96] = 1.0
    real_path = tmp_path / "real.npy"
    synthetic_path = tmp_path / "synthetic.npy"
    np.save(real_path, image)
    np.save(synthetic_path, image.copy())
    args = _parser().parse_args(
        [
            "--real",
            str(real_path),
            "--synthetic",
            str(synthetic_path),
            "--metrics",
            *SUPPORTED_METRICS,
            "--data-range",
            "1",
            "--spacing",
            "0.5",
            "2",
            "--border-width",
            "2",
            "3",
        ]
    )
    report = calculate_metrics(args)
    assert set(report) == set(SUPPORTED_METRICS)
    assert report["psnr"] == "Infinity"
    assert report["dice"] == 1.0
    assert report["connected_components"]["real"]["component_count"] == 1


def test_spacing_and_json_conversion_helpers():
    assert _resolve_spacing([1, 2], None, 2) == (1, 2)
    assert _resolve_spacing(None, [3, 4], 2) == (3, 4)
    assert _resolve_spacing(None, None, 2) is None
    with pytest.raises(ValueError, match="requires 2 values"):
        _resolve_spacing([1], None, 2)
    with pytest.raises(ValueError, match="dimensionality"):
        _resolve_spacing(None, [1], 2)

    converted = _json_safe(
        {"items": (np.int64(2), np.float64(np.inf), -np.inf, np.nan), "nested": [1]}
    )
    assert converted == {"items": [2, "Infinity", "-Infinity", None], "nested": [1]}


def test_result_writers_support_json_and_csv(tmp_path):
    results = {"mae": 0.5, "nested": {"values": [1, 2], "score": 1.0}}
    json_path = tmp_path / "nested" / "results.json"
    csv_path = tmp_path / "results.csv"
    _write_results(json_path, results)
    _write_results(csv_path, results)
    assert json.loads(json_path.read_text(encoding="utf-8")) == results
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert ["nested.values", "[1, 2]"] in rows
    assert ["nested.score", "1.0"] in rows
    with pytest.raises(ValueError, match="must end with"):
        _write_results(tmp_path / "results.txt", results)


def test_main_prints_results_and_reports_user_errors(tmp_path, capsys):
    path = tmp_path / "image.npy"
    np.save(path, np.zeros((8, 8), dtype=np.float32))
    assert main(["--real", str(path), "--synthetic", str(path), "--metrics", "mae"]) == 0
    assert json.loads(capsys.readouterr().out) == {"mae": 0.0}
    with pytest.raises(SystemExit) as exc_info:
        main(["--real", str(tmp_path / "missing.npy"), "--synthetic", str(path)])
    assert exc_info.value.code == 2
