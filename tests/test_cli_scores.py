import csv
import json

import numpy as np
import pytest

from synthetic_imaging_validation.cli import validate
from synthetic_imaging_validation import load_history


@pytest.fixture
def manifest(tmp_path):
    real = np.full((16, 16), 0.3)
    np.save(tmp_path / "real.npy", real)
    np.save(tmp_path / "synthetic.npy", real + 0.1)
    path = tmp_path / "pairs.csv"
    path.write_text("case_id,real,synthetic,label\nsame,real.npy,real.npy,a\nchanged,real.npy,synthetic.npy,b\n")
    return path


def test_scores_grouping_parallel_raw_reuse_and_all_outputs(tmp_path, manifest, monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    pytest.importorskip("reportlab")
    base = ["--manifest", str(manifest), "--key-column", "case_id", "--group-by", "label",
            "--metrics", "mae", "ms_ssim", "wasserstein", "js", "kl", "psnr", "ssim",
            "--scores", "similarity", "intensity_distribution", "--score-range", "0", "1"]
    original = validate.calculate_metrics(validate._parser().parse_args(base))

    def duplicate(*args, **kwargs):
        pytest.fail("Raw metrics shared with scores should not be recalculated")

    monkeypatch.setattr(validate, "mae", duplicate)
    monkeypatch.setattr(validate, "ms_ssim", duplicate)
    monkeypatch.setattr(validate, "wasserstein_distance", duplicate)
    monkeypatch.setattr(validate, "jensen_shannon_divergence", duplicate)
    assert validate.calculate_metrics(validate._parser().parse_args(base + ["--num-workers", "2"])) == original
    for record in original["pairs"]:
        values = record["metrics"]
        assert "score_protocol" not in values
        assert values["similarity_score"]["raw_metrics"]["mae"] == values["mae"]
        assert values["intensity_distribution_score"]["raw_metrics"]["js"] == values["js"]
    scores = [p["metrics"]["similarity_score"]["value"] for p in original["pairs"]]
    assert original["summary"]["metrics"]["similarity_score.value"]["mean"] == pytest.approx(np.mean(scores))
    assert original["grouped_summary"]["groups"]["a"]["metrics"]["similarity_score.value"]["mean"] == pytest.approx(100)
    assert original["score_protocol"]["intensity_distribution_score"]["bins"] == 64

    outputs = []
    for kind, suffix in (("json", "json"), ("csv", "csv"), ("pdf", "pdf"), ("latex", "tex")):
        outputs.extend(["--output-" + kind, str(tmp_path / ("results." + suffix))])
    history_path = tmp_path / "history.json"
    assert validate.main(base + outputs + ["--history", str(history_path), "--epoch", "10",
        "--plot-history", str(tmp_path / "history.png"), "--plot-output", str(tmp_path / "pairs.svg"),
        "--plot-metrics", "similarity_score", "intensity_distribution_score"]) == 0
    assert json.loads(capsys.readouterr().out) == original
    h = load_history(history_path)
    assert h["records"][0]["metrics"]["similarity_score.value"] == pytest.approx(np.mean(scores))
    assert h["records"][0]["protocol"]["scores"] == original["score_protocol"]
    assert h["records"][0]["counts"]["similarity_score.value"] == 2
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["metric"] == "similarity_score.value.mean" and row["scope"] == "group" for row in rows)
    assert (tmp_path / "history.png").is_file()


def test_single_pair_scores_and_single_selected_category(manifest):
    args = ["--real", str(manifest.parent / "real.npy"), "--synthetic", str(manifest.parent / "synthetic.npy"),
            "--metrics", "mae", "ms_ssim", "js", "kl"]
    for category in ("similarity", "intensity_distribution"):
        result = validate.calculate_metrics(validate._parser().parse_args(args + ["--scores", category, "--score-range", "0", "1"]))
        assert 0 <= result[category + "_score"]["value"] < 100
        assert category + "_score" in result["score_protocol"]


@pytest.mark.parametrize("options", [
    ["--scores", "similarity"],
    ["--scores", "similarity", "--score-range", "1", "0"],
    ["--scores", "similarity", "--score-range", "0", "1", "--data-range", "2"],
    ["--scores", "similarity", "--score-range", "0", "1", "--batch-axis", "0"],
    ["--score-range", "0", "1"],
    ["--history", "history.json"], ["--step", "0"],
    ["--history", "history.csv", "--step", "0"],
    ["--history", "history.json", "--step", "-1"],
    ["--plot-history", "plot.png"], ["--plot-metrics", "mae"],
    ["--plot-output", "plot.txt"],
    ["--history", "results.json", "--step", "0", "--output-json", "results.json"],
])
def test_invalid_cli_options(manifest, options):
    with pytest.raises(SystemExit) as error:
        validate.main(["--manifest", str(manifest), "--metrics", "mae"] + options)
    assert error.value.code == 2


def test_cli_history_without_plot_and_plot_without_history(tmp_path, manifest):
    pytest.importorskip("matplotlib")
    assert validate.main(["--manifest", str(manifest), "--metrics", "mae",
                          "--history", str(tmp_path / "history.json"), "--step", "1"]) == 0
    assert validate.main(["--manifest", str(manifest), "--metrics", "mae",
                          "--plot-output", str(tmp_path / "plot.pdf")]) == 0
