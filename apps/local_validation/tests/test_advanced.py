"""Numerical parity and boundary checks for additional local-app workflows."""
import json
import numpy as np
import pytest
from apps.local_validation.advanced import evaluate_features, evaluate_cohorts, observed_range, read_labels, FEATURE_METRICS
from apps.local_validation.extra_metrics import EXTRA_METRICS, calculate_extras
from apps.local_validation.service import make_demo, plan_pairs, run_validation, save_report
from apps.local_validation.workspace import select_folder, compose_environment
from synthetic_imaging_validation import append_history, load_history, plot_history
from synthetic_imaging_validation.metrics.grouped import DISTRIBUTION_METRICS
from synthetic_imaging_validation.metrics import distribution, segmentation, spatial


def test_custom_columns_sorted_and_path_base(tmp_path):
    make_demo(tmp_path)
    (tmp_path / "custom.csv").write_text("id,a,b,label\nx,real/case_01.npy,synthetic/case_01.npy,A\n", encoding="utf-8")
    pairs = plan_pairs(tmp_path, "manifest", manifest="custom.csv", real_column="a", synthetic_column="b", key_column="id", base_dir=".")
    assert pairs[0].key == "x" and pairs[0].metadata["label"] == "A"
    sorted_pairs = plan_pairs(tmp_path, "directories", real="real", synthetic="synthetic", pairing="sorted")
    assert len(sorted_pairs) == 4
    with pytest.raises(ValueError, match="column"):
        plan_pairs(tmp_path, "manifest", manifest="custom.csv", real_column="a", synthetic_column="a")
    with pytest.raises(ValueError, match="inside"):
        plan_pairs(tmp_path, "manifest", manifest="custom.csv", real_column="a", synthetic_column="b", base_dir="..")


def test_bounds_and_score_protocol_on_ct_scale(tmp_path):
    make_demo(tmp_path / "data")
    pairs = plan_pairs(tmp_path / "data", "directories", real="real", synthetic="synthetic")[:1]
    a = np.linspace(-1000, 2000, 64*64).reshape(64, 64)
    np.save(pairs[0].real, a)
    np.save(pairs[0].synthetic, a)
    low, high, rows = observed_range(pairs)
    assert (low, high) == (-1000, 2000) and len(rows) == 2
    # Successful raw metrics do not establish that the default score interval
    # [0, 1] is valid. Even identical CT-scale inputs must declare their bounds.
    raw, _ = run_validation(pairs, {"metrics": ["mae", "wasserstein", "ssim"], "data_range": 3000}, tmp_path / "raw")
    assert raw["pairs"][0]["metrics"]["mae"] == 0
    assert raw["pairs"][0]["metrics"]["wasserstein"] == 0
    assert raw["pairs"][0]["metrics"]["ssim"] == pytest.approx(1)
    with pytest.raises(ValueError, match="outside"):
        run_validation(pairs, {"metrics": ["mae"], "scores": ["similarity"], "score_range": [0, 1]}, tmp_path / "out")
    report, _ = run_validation(pairs, {"metrics": ["mae"], "scores": ["similarity", "intensity_distribution"], "score_range": [low, high]}, tmp_path / "out")
    scores = report["pairs"][0]["metrics"]
    assert scores["similarity_score"]["value"] == pytest.approx(100)
    assert scores["intensity_distribution_score"]["value"] == pytest.approx(100)
    assert np.array_equal(np.load(pairs[0].real), a)


def test_extra_image_and_mask_metrics_match_api(tmp_path):
    make_demo(tmp_path / "data", "masks")
    pair = plan_pairs(tmp_path / "data", "directories", real="real", synthetic="synthetic")[2]
    a, b = np.load(pair.real), np.load(pair.synthetic)
    options = {"metrics": list(EXTRA_METRICS), "spacing": [2, 3], "connectivity": 1, "bins": 8}
    values = calculate_extras(pair, options)
    assert values["intensity_statistics"]["real"] == distribution.intensity_statistics(a)
    assert values["surface_statistics"] == segmentation.surface_distance_statistics(b, a, spacing=[2, 3], connectivity=1)
    assert values["component_measures"]["real"] == segmentation.component_measure_distribution(a, spacing=[2, 3], connectivity=1).tolist()
    assert values["centroid"]["real"] == spatial.centroid_statistics(a, spacing=[2, 3])
    assert values["distance_to_border"]["real"] == spatial.distance_to_border_statistics(a, spacing=[2, 3])
    assert sum(values["histogram"]["real"]["counts"]) == a.size
    report, output = run_validation([pair], options, tmp_path / "results")
    assert set(report["pairs"][0]["metrics"]) == set(EXTRA_METRICS)
    assert (output / "results.pdf").is_file()
    assert "mae" not in report["pairs"][0]["metrics"]
    with pytest.raises(ValueError, match="Unknown"):
        run_validation([pair], {"metrics": ["not_a_metric"]}, tmp_path / "results")


def test_vector_only_export_and_history_roundtrip(tmp_path):
    report, path = save_report({"histogram": {"counts": [1, 2], "edges": [0, 1, 2]}}, {}, tmp_path)
    assert not (path / "metrics.png").exists()
    assert json.loads((path / "results.json").read_text()) == report
    report, path = save_report({"mae": 0.2, "ssim": 0.8}, {}, tmp_path)
    history = tmp_path / "history.json"
    append_history(history, report, step=1, protocol={"range": [0, 1]})
    append_history(history, {"mae": 0.1, "ssim": 0.9}, step=2, protocol={"range": [0, 1]})
    assert len(load_history(history)["records"]) == 2
    figure = plot_history(history, output=tmp_path / "history.svg")
    figure.clear()
    assert (tmp_path / "history.svg").is_file()
    with pytest.raises(ValueError):
        append_history(history, report, step=2, protocol={"range": [0, 1]})


def test_feature_metrics_and_labels_match_public_api(tmp_path):
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(12, 3)), rng.normal(size=(14, 3))
    np.save(tmp_path / "real.npy", a)
    np.save(tmp_path / "synth.npy", b)
    (tmp_path / "r.csv").write_text("label\n" + "A\n"*6 + "B\n"*6, encoding="utf-8")
    (tmp_path / "s.csv").write_text("label\n" + "A\n"*7 + "B\n"*7, encoding="utf-8")
    result = evaluate_features(tmp_path, "real.npy", "synth.npy", FEATURE_METRICS, labels=("r.csv", "s.csv", "label"))
    for metric in FEATURE_METRICS:
        assert result[metric] == pytest.approx(DISTRIBUTION_METRICS[metric](a, b))
    assert result["by_class"]["A"]["n_real"] == 6
    assert result["by_class"]["B"]["status"] == "ok"
    saved, destination = save_report(result, {"metrics": list(FEATURE_METRICS)}, tmp_path / "output")
    assert (destination / "results.pdf").is_file()
    with pytest.raises(ValueError, match="Label CSV"):
        read_labels(tmp_path, "r.csv", "label", 4)
    np.save(tmp_path / "wrong.npy", np.zeros((3, 4, 5)))
    with pytest.raises(ValueError, match="matrices"):
        evaluate_features(tmp_path, "wrong.npy", "synth.npy", ["frechet"])
    np.save(tmp_path / "large.npy", np.zeros((5001, 2)))
    with pytest.raises(ValueError, match="5,000"):
        evaluate_features(tmp_path, "large.npy", "large.npy", ["kid"])


def test_unpaired_cohorts_counts_and_bounds(tmp_path):
    for folder in ("real", "synthetic"):
        (tmp_path / folder).mkdir()
    a, b = np.array([[1., 2.], [3., 4.]]), np.array([[1., 2., 3.]])
    np.save(tmp_path / "real/a.npy", a)
    np.save(tmp_path / "synthetic/b.npy", b)
    np.save(tmp_path / "synthetic/c.npy", b)
    report = evaluate_cohorts(tmp_path, "real", "synthetic", ["wasserstein", "js"])
    assert report["wasserstein"] == pytest.approx(distribution.wasserstein_distance(a, np.concatenate([b, b])))
    assert report["case_counts"] == {"real": 1, "synthetic": 2}
    with pytest.raises(ValueError, match="exceed"):
        evaluate_cohorts(tmp_path, "real", "synthetic", ["js"], value_range=[0, 1])
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_cohorts(tmp_path, "real", "synthetic", ["js"], max_voxels=2)
    result = evaluate_cohorts(tmp_path, "real", "synthetic", ["intensity_distribution_score"], value_range=[0, 5])
    assert 0 <= result["intensity_distribution_score"] <= 100


def test_workspace_boundaries_and_compose_quoting(tmp_path):
    assert select_folder(tmp_path, "nested", confined=True) == tmp_path / "nested"
    with pytest.raises(ValueError, match="mount"):
        select_folder(tmp_path, "..", confined=True)
    with pytest.raises(ValueError, match="exist"):
        select_folder(tmp_path, "missing", confined=True, must_exist=True)
    assert "SIV_DATA_DIR='C:/data folder/$study'" in compose_environment("C:\\data folder\\$study", "C:/results")
    with pytest.raises(ValueError):
        compose_environment("a\nBAD=1", "out")
