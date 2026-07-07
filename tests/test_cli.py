import csv
import json
import runpy
import sys
import warnings

import numpy as np
import pytest

from synthetic_imaging_validation.cli.validate import (
    SUPPORTED_METRICS,
    _input_mode,
    _json_safe,
    _load_cli_pairs,
    _parser,
    _requested_outputs,
    _resolve_spacing,
    _summarize_pair_metrics,
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


def test_cli_input_mode_validation(tmp_path):
    parser = _parser()
    real = tmp_path / "real.npy"
    synthetic = tmp_path / "synthetic.npy"
    np.save(real, np.zeros((2, 2)))
    np.save(synthetic, np.zeros((2, 2)))

    assert _input_mode(parser.parse_args(["--real", str(real), "--synthetic", str(synthetic)])) == "files"
    with pytest.raises(ValueError, match="File mode requires both"):
        _input_mode(parser.parse_args(["--real", str(real)]))
    with pytest.raises(ValueError, match="Directory mode requires both"):
        _input_mode(parser.parse_args(["--real-dir", str(tmp_path)]))
    with pytest.raises(ValueError, match="Choose exactly one"):
        _input_mode(parser.parse_args([]))
    with pytest.raises(ValueError, match="Choose exactly one"):
        _input_mode(
            parser.parse_args(
                [
                    "--real",
                    str(real),
                    "--synthetic",
                    str(synthetic),
                    "--manifest",
                    str(tmp_path / "pairs.csv"),
                ]
            )
        )


def test_calculate_metrics_from_paired_directories_and_manifest(tmp_path):
    real_dir = tmp_path / "real"
    synthetic_dir = tmp_path / "synthetic"
    real_dir.mkdir()
    synthetic_dir.mkdir()
    np.save(real_dir / "case_a.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(synthetic_dir / "case_a.npy", np.ones((2, 2), dtype=np.float32))
    np.save(real_dir / "case_b.npy", np.ones((2, 2), dtype=np.float32))
    np.save(synthetic_dir / "case_b.npy", np.ones((2, 2), dtype=np.float32))

    directory_args = _parser().parse_args(
        [
            "--real-dir",
            str(real_dir),
            "--synthetic-dir",
            str(synthetic_dir),
            "--metrics",
            "mae",
            "mse",
        ]
    )
    mode, pairs = _load_cli_pairs(directory_args)
    assert mode == "directories"
    assert [pair.key for pair in pairs] == ["case_a", "case_b"]
    report = calculate_metrics(directory_args)
    assert report["summary"]["count"] == 2
    assert report["summary"]["metrics"]["mae"]["mean"] == 0.5
    assert report["pairs"][0]["metrics"] == {"mae": 1.0, "mse": 1.0}

    manifest = tmp_path / "pairs.csv"
    manifest.write_text(
        "case_id,real,synthetic,label\n"
        f"first,{real_dir / 'case_a.npy'},{synthetic_dir / 'case_a.npy'},low\n"
        f"second,{real_dir / 'case_b.npy'},{synthetic_dir / 'case_b.npy'},high\n",
        encoding="utf-8",
    )
    manifest_args = _parser().parse_args(
        ["--manifest", str(manifest), "--key-column", "case_id", "--metrics", "mae"]
    )
    manifest_report = calculate_metrics(manifest_args)
    assert manifest_report["pairs"][0]["key"] == "first"
    assert manifest_report["pairs"][0]["metadata"] == {"case_id": "first", "label": "low"}
    assert "grouped_summary" not in manifest_report

    grouped_args = _parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--key-column",
            "case_id",
            "--group-by",
            "label",
            "--metrics",
            "mae",
        ]
    )
    grouped_report = calculate_metrics(grouped_args)
    assert grouped_report["grouped_summary"]["column"] == "label"
    assert grouped_report["grouped_summary"]["groups"]["low"]["metrics"]["mae"]["mean"] == 1.0
    assert grouped_report["grouped_summary"]["groups"]["high"]["metrics"]["mae"]["mean"] == 0.0

    bad_group_args = _parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--key-column",
            "case_id",
            "--group-by",
            "missing",
            "--metrics",
            "mae",
        ]
    )
    with pytest.raises(ValueError, match="--group-by column 'missing' is missing"):
        calculate_metrics(bad_group_args)

    directory_group_args = _parser().parse_args(
        [
            "--real-dir",
            str(real_dir),
            "--synthetic-dir",
            str(synthetic_dir),
            "--group-by",
            "label",
        ]
    )
    with pytest.raises(ValueError, match="only with --manifest"):
        calculate_metrics(directory_group_args)


def test_metric_summary_skips_non_finite_and_non_scalar_values():
    summary = _summarize_pair_metrics(
        [
            {"mae": np.float32(1.0), "psnr": np.inf, "ok": True, "nested": {"dice": 0.5}},
            {"mae": 3.0, "psnr": np.inf, "nested": {"dice": 1.0}, "items": [1, 2]},
        ]
    )
    assert summary["count"] == 2
    assert summary["metrics"]["mae"]["mean"] == 2.0
    assert summary["metrics"]["nested.dice"]["max"] == 1.0
    assert "psnr" not in summary["metrics"]
    assert "items" not in summary["metrics"]


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


def test_requested_outputs_support_single_and_dual_output_modes(tmp_path):
    parser = _parser()
    real = tmp_path / "real.npy"
    synthetic = tmp_path / "synthetic.npy"
    np.save(real, np.zeros((2, 2)))
    np.save(synthetic, np.zeros((2, 2)))

    args = parser.parse_args(
        [
            "--real",
            str(real),
            "--synthetic",
            str(synthetic),
            "--output",
            str(tmp_path / "legacy.json"),
            "--output-json",
            str(tmp_path / "results.json"),
            "--output-csv",
            str(tmp_path / "results.csv"),
        ]
    )
    assert _requested_outputs(args) == [
        tmp_path / "legacy.json",
        tmp_path / "results.json",
        tmp_path / "results.csv",
    ]

    bad_json_args = parser.parse_args(
        [
            "--real",
            str(real),
            "--synthetic",
            str(synthetic),
            "--output-json",
            str(tmp_path / "results.csv"),
        ]
    )
    with pytest.raises(ValueError, match="--output-json"):
        _requested_outputs(bad_json_args)

    bad_csv_args = parser.parse_args(
        [
            "--real",
            str(real),
            "--synthetic",
            str(synthetic),
            "--output-csv",
            str(tmp_path / "results.json"),
        ]
    )
    with pytest.raises(ValueError, match="--output-csv"):
        _requested_outputs(bad_csv_args)


def test_pairwise_result_writer_supports_csv(tmp_path):
    results = {
        "pairs": [
            {
                "key": "case_a",
                "real": "real/case_a.npy",
                "synthetic": "synthetic/case_a.npy",
                "metrics": {"mae": 0.5, "nested": {"dice": 1.0}},
            }
        ],
        "summary": {"metrics": {"mae": {"count": 1, "mean": 0.5}}},
        "grouped_summary": {
            "column": "label",
            "groups": {"low": {"metrics": {"mae": {"count": 1, "mean": 0.5}}}},
        },
    }
    csv_path = tmp_path / "pairwise.csv"
    _write_results(csv_path, results)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["scope", "key", "real", "synthetic", "metric", "value"]
    assert ["pair", "case_a", "real/case_a.npy", "synthetic/case_a.npy", "nested.dice", "1.0"] in rows
    assert ["summary", "", "", "", "mae.mean", "0.5"] in rows
    assert ["group", "low", "", "", "mae.mean", "0.5"] in rows


def test_main_prints_results_and_reports_user_errors(tmp_path, capsys):
    path = tmp_path / "image.npy"
    np.save(path, np.zeros((8, 8), dtype=np.float32))
    assert main(["--real", str(path), "--synthetic", str(path), "--metrics", "mae"]) == 0
    assert json.loads(capsys.readouterr().out) == {"mae": 0.0}

    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    assert (
        main(
            [
                "--real",
                str(path),
                "--synthetic",
                str(path),
                "--metrics",
                "mae",
                "--output-json",
                str(json_path),
                "--output-csv",
                str(csv_path),
            ]
        )
        == 0
    )
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"mae": 0.0}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert ["mae", "0.0"] in list(csv.reader(handle))
    capsys.readouterr()

    with pytest.raises(SystemExit) as exc_info:
        main(["--real", str(tmp_path / "missing.npy"), "--synthetic", str(path)])
    assert exc_info.value.code == 2


def test_module_entrypoint(tmp_path, monkeypatch):
    path = tmp_path / "image.npy"
    np.save(path, np.zeros((8, 8), dtype=np.float32))
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate", "--real", str(path), "--synthetic", str(path), "--metrics", "mae"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("synthetic_imaging_validation.cli.validate", run_name="__main__")
    assert exc_info.value.code == 0
