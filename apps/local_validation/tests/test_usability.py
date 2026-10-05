"""Profiles and preflight checks must not change data or numerical definitions."""
import json
import numpy as np
import nibabel as nib
import pytest
from apps.local_validation.profiles import defaults, validate_settings, export_profile, import_profile
from apps.local_validation.preflight import check_inputs
from apps.local_validation.service import Pair, make_demo, plan_pairs, run_validation, record_run_history
from synthetic_imaging_validation import nrmse, load_history


@pytest.mark.parametrize("masks", [False, True])
def test_profile_roundtrip_and_old_settings_strip_paths(masks):
    options = defaults(masks)
    options.update(workers=2, pdf_font="private/font.ttf", real="private/patient.nii", group_by="diagnosis")
    encoded = export_profile(options, masks)
    assert "private" not in encoded and "diagnosis" not in encoded
    loaded = import_profile(encoded, masks)
    assert loaded == validate_settings(options, masks)
    assert import_profile(json.dumps(options), masks) == loaded
    with pytest.raises(ValueError, match="Switch"):
        import_profile(encoded, not masks)


@pytest.mark.parametrize("change", [
    {"workers": True}, {"workers": 0}, {"workers": 17}, {"bins": 1},
    {"score_range": [0, 0]}, {"score_range": [0, float("inf")]}, {"score_range": None},
    {"data_range": -1}, {"metrics": []}, {"metrics": ["invalid"]}, {"metrics": "mae"},
    {"scores": ["invalid"]}, {"scores": ["similarity"], "batch_axis": 0},
    {"spacing": [-1]}, {"spacing": "1"}, {"percentiles": [101]},
    {"channel_axis": True}, {"batch_axis": 1.2}, {"connectivity": True},
    {"border_width": [-1]}, {"allow_spatial_mismatch": "false"},
    {"ms_ssim_backend": "other"}, {"nrmse_normalization": "other"}, {"plot_metrics": [None]},
])
def test_profile_rejects_invalid_options(change):
    with pytest.raises(ValueError):
        validate_settings(dict(defaults(), **change))


@pytest.mark.parametrize("content", ["[]", "{", "{}", "x" * 100_001,
    '{"profile_version":2}', '{"profile_version":true}',
    '{"profile_version":1,"kind":"images","settings":null}'],
    ids=["array", "syntax", "empty", "oversize", "version", "bool_version", "null_settings"])
def test_bad_profile(content):
    with pytest.raises(ValueError):
        import_profile(content)


def test_checks_read_all_pairs_without_changing_them(tmp_path):
    make_demo(tmp_path)
    pairs = plan_pairs(tmp_path, "manifest", manifest="pairs.csv", group_by="label")
    before = pairs[0].real.read_bytes()
    calls = []
    report = check_inputs(pairs, dict(defaults(), scores=["similarity"]), progress=lambda *args: calls.append(args))
    assert report["ready"] and len(report["cases"]) == 4
    assert calls == [(i, 4) for i in range(1, 5)]
    assert pairs[0].real.read_bytes() == before
    np.save(pairs[0].synthetic, np.ones((64, 64)) * 2)
    np.save(pairs[1].synthetic, np.zeros((8, 8)))
    bad = np.zeros((64, 64)); bad[0, 0] = np.inf
    np.save(pairs[2].synthetic, bad)
    report = check_inputs(pairs, dict(defaults(), scores=["similarity"]))
    assert not report["ready"] and report["errors"] == 3
    assert report["cases"][3]["Status"] == "Ready"
    assert "score interval" in report["cases"][0]["Details"]


def test_geometry_override_and_mask_notes(tmp_path):
    a, b = tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4)), a)
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), np.diag([2., 1., 1., 1.])), b)
    pair = Pair("case", a, b, {})
    assert not check_inputs([pair], defaults(True), masks=True)["ready"]
    report = check_inputs([pair], dict(defaults(True), allow_spatial_mismatch=True), masks=True)
    assert report["ready"] and report["cases"][0]["Status"] == "Review"
    assert "empty mask" in report["cases"][0]["Details"]
    assert "switched off" in report["cases"][0]["Details"]


@pytest.mark.parametrize("change", [{"channel_axis": 8}, {"channel_axis": 0, "batch_axis": 0},
                                    {"spacing": [1]}, {"channel_axis": 0}])
def test_axes_and_spacing_errors(tmp_path, change):
    make_demo(tmp_path)
    pair = plan_pairs(tmp_path, "manifest", manifest="pairs.csv")[0]
    report = check_inputs([pair], dict(defaults(), **change))
    assert report["errors"] == 1


@pytest.mark.parametrize("normalization", ["range", "mean", "l2"])
def test_nrmse_api_parity_and_history(tmp_path, normalization):
    make_demo(tmp_path / "data")
    pairs = plan_pairs(tmp_path / "data", "manifest", manifest="pairs.csv")[:2]
    options = dict(defaults(), metrics=["nrmse"], nrmse_normalization=normalization)
    root = tmp_path / "results"
    report, destination = run_validation(pairs, options, root)
    for pair, row in zip(pairs, report["pairs"]):
        assert row["metrics"]["nrmse"] == nrmse(np.load(pair.real), np.load(pair.synthetic), normalization=normalization)
    assert json.loads((destination / "settings.json").read_text())["nrmse_normalization"] == normalization
    for step in (1, 2):
        path = record_run_history(report, options, root, step=step)
    assert len(load_history(path)["records"]) == 2
    with pytest.raises(ValueError, match="protocol changed"):
        record_run_history(report, dict(options, bins=128), root, step=3)
    with pytest.raises(ValueError, match="inside"):
        record_run_history(report, options, root, filename="../outside.json")
    with pytest.raises(ValueError, match="filename"):
        record_run_history(report, options, root, filename="report.csv")
    with pytest.raises(ValueError, match="run name"):
        record_run_history(report, options, root, run=" ")
