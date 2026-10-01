"""Optional application tests, separate from the base package coverage suite."""

import csv
import json
import threading
from pathlib import Path
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest

from apps.local_validation.service import Pair, catalogue, inside, make_demo, plan_pairs, run_validation
from synthetic_imaging_validation.cli import validate


def test_demo_all_modes_and_labels(tmp_path):
    root = tmp_path / "inputs"
    make_demo(root)
    images, manifests, folders = catalogue(root)
    assert len(images) == 8 and manifests == ["pairs.csv"] and "real" in folders
    manifest = plan_pairs(root, "manifest", manifest="pairs.csv", group_by="label")
    directory = plan_pairs(root, "directories", real="real", synthetic="synthetic")
    assert len(manifest) == len(directory) == 4
    assert [p.key for p in manifest] == [p.key for p in directory]
    assert plan_pairs(root, "files", real=images[0], synthetic=images[1])[0].real.is_file()
    assert catalogue(tmp_path / "missing") == ([], [], ["."])


def test_reject_paths_and_bad_manifests(tmp_path):
    root = tmp_path / "inputs"
    make_demo(root)
    with pytest.raises(ValueError, match="inside"):
        inside(root, "../secret.npy")
    with pytest.raises(ValueError, match="not found"):
        inside(root, "missing.npy")
    with pytest.raises(ValueError, match="Unknown"):
        plan_pairs(root, "bad")
    with pytest.raises(ValueError, match="Unsupported"):
        plan_pairs(root, "files", real="pairs.csv", synthetic="pairs.csv")
    with pytest.raises(ValueError, match="Missing group"):
        plan_pairs(root, "manifest", manifest="pairs.csv", group_by="missing")
    invalid = [
        ("other,synthetic\na,b\n", "unique column"),
        ("real,synthetic\n", "No pairs"),
        ("real,synthetic\na\n", "Malformed"),
        ("real,synthetic\na,b,c\n", "Malformed"),
        ("real,synthetic\n,synthetic/case_01.npy\n", "Empty"),
        ("real,synthetic,case_id\nreal/case_01.npy,synthetic/case_01.npy,x\nreal/case_02.npy,synthetic/case_02.npy,x\n", "unique"),
        ("real,synthetic\n../../escape.npy,synthetic/case_01.npy\n", "inside"),
    ]
    for content, message in invalid:
        (root / "bad.csv").write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            plan_pairs(root, "manifest", manifest="bad.csv")


def test_export_parity_parallel_and_unique_outputs(tmp_path):
    root = tmp_path / "inputs"
    make_demo(root)
    pairs = plan_pairs(root, "manifest", manifest="pairs.csv", group_by="label")
    options = {"metrics": ["mae", "ms_ssim", "js"], "data_range": 1,
               "scores": ["similarity", "intensity_distribution"], "score_range": [0, 1],
               "workers": 2, "group_by": "label", "plot_metrics": ["similarity_score"]}
    calls = []
    report, output = run_validation(pairs, options, tmp_path / "results",
        lambda done, total: calls.append((done, total, threading.get_ident())))
    assert [(d, n) for d, n, _ in calls] == [(i, 4) for i in range(5)]
    assert all(thread == threading.get_ident() for _, _, thread in calls)
    cli = validate.calculate_metrics(validate._parser().parse_args([
        "--manifest", str(root / "pairs.csv"), "--key-column", "case_id", "--group-by", "label",
        "--metrics", "mae", "ms_ssim", "js", "--scores", "similarity", "intensity_distribution", "--score-range", "0", "1", "--data-range", "1",
    ]))
    assert report == cli
    assert json.loads((output / "results.json").read_text()) == report
    assert (output / "results.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "metrics.png").read_bytes().startswith(b"\x89PNG")
    with (output / "results.csv").open(newline="") as handle:
        assert any(row["scope"] == "group" for row in csv.DictReader(handle))
    with ZipFile(output / "results_bundle.zip") as bundle:
        assert set(bundle.namelist()) == {"results.json", "results.csv", "results.pdf", "results.tex", "settings.json", "metrics.png", "metrics.svg", "metrics.pdf"}
    options["workers"] = 1
    repeat, new_output = run_validation(pairs, options, tmp_path / "results")
    assert repeat == report and new_output != output


def test_masks_nested_plots_and_nifti_geometry(tmp_path):
    make_demo(tmp_path / "masks", "masks")
    pairs = plan_pairs(tmp_path / "masks", "directories", real="real", synthetic="synthetic")
    report, _ = run_validation(pairs, {"metrics": ["connected_components", "foreground_fraction"], "threshold": 0.5}, tmp_path / "results")
    assert "connected_components.real.component_count" in report["summary"]["metrics"]
    r, s = tmp_path / "real.nii.gz", tmp_path / "synthetic.nii.gz"
    array = np.ones((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(array, np.eye(4)), r)
    nib.save(nib.Nifti1Image(array, np.diag([2, 1, 1, 1])), s)
    with pytest.raises(ValueError, match="Pair 'geometry'"):
        run_validation([Pair("geometry", r, s, {})], {"metrics": ["mae"]}, tmp_path / "results")


def test_invalid_options_and_data(tmp_path):
    make_demo(tmp_path / "inputs")
    pairs = plan_pairs(tmp_path / "inputs", "directories", real="real", synthetic="synthetic")
    for options, message in (({}, "metric"), ({"metrics": ["mae"], "workers": 0}, "workers"),
                              ({"metrics": ["mae"], "workers": 17}, "workers")):
        with pytest.raises(ValueError, match=message):
            run_validation(pairs, options, tmp_path / "results")
    with pytest.raises(ValueError, match="pair"):
        run_validation([], {"metrics": ["mae"]}, tmp_path / "results")
    with pytest.raises(ValueError, match="outside"):
        run_validation(pairs, {"metrics": ["mae"], "scores": ["similarity"], "score_range": [0, 0.2]}, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_symlink_escape(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    outside = tmp_path / "outside.npy"
    np.save(outside, [1.0])
    try:
        (root / "escape.npy").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires additional Windows privileges")
    assert catalogue(root)[0] == []
    with pytest.raises(ValueError, match="inside"):
        inside(root, "escape.npy")


def test_nifti_success_and_nan_failure(tmp_path):
    real = np.linspace(0, 1, 16 ** 3, dtype=np.float32).reshape(16, 16, 16)
    for name in ("real.nii.gz", "synthetic.nii.gz"):
        nib.save(nib.Nifti1Image(real, np.diag([1, 2, 3, 1])), tmp_path / name)
    pairs = plan_pairs(tmp_path, "files", real="real.nii.gz", synthetic="synthetic.nii.gz")
    report, _ = run_validation(pairs, {"metrics": ["mae", "ssim", "ms_ssim"], "data_range": 1}, tmp_path / "results")
    assert report["summary"]["metrics"]["mae"]["mean"] == 0
    assert report["summary"]["metrics"]["ms_ssim"]["mean"] == pytest.approx(1)
    np.save(tmp_path / "invalid.npy", [np.nan])
    pairs = plan_pairs(tmp_path, "files", real="invalid.npy", synthetic="invalid.npy")
    with pytest.raises(ValueError, match="NaN"):
        run_validation(pairs, {"metrics": ["mae"]}, tmp_path / "results")
