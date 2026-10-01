"""Export tests exercise real PDF parsing as well as dependency-free LaTeX."""

import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from synthetic_imaging_validation import write_report
from synthetic_imaging_validation import reporting
from synthetic_imaging_validation.cli import validate


@pytest.fixture
def cohort():
    return {
        "pairs": [{"key": "case_01", "real": "real/a.npy", "synthetic": "synth/a.npy",
                   "metadata": {"label": "Group A&B"},
                   "metrics": {"mae": 0.123456789012345, "psnr": "Infinity",
                               "components": {"sizes": [10, 20]}}}],
        "summary": {"count": 1, "metrics": {"mae": {"count": 1, "mean": 0.123456789012345, "std": 0.0}}},
        "grouped_summary": {"column": "label", "groups": {"Group A&B": {"count": 1, "metrics": {"mae": {"mean": 0.123456789012345}}}}},
        "protocol": "prepared data",
    }


def pdf_reader(path):
    pytest.importorskip("reportlab")
    return pytest.importorskip("pypdf").PdfReader(path)


def test_latex_preserves_values_and_escapes_text(tmp_path, cohort):
    original = copy.deepcopy(cohort)
    path = write_report(cohort, tmp_path / "nested" / "report.TEX", title=r"Study 50% & {x}_# $^~\ end")
    text = path.read_text(encoding="utf-8")
    assert r"\documentclass" in text and r"\end{document}" in text
    assert r"Group A\&B" in text
    assert "Grouped summary" in text and "Cohort summary" in text
    assert "Additional information" in text and "prepared data" in text
    assert "123456789012345" in text and "Infinity" in text
    assert "sizes[0]" in text and "sizes[1]" in text
    for escaped in (r"\%", r"\{", r"\}", r"\_", r"\#", r"\$", r"\textasciicircum{}", r"\textasciitilde{}", r"\textbackslash{}"):
        assert escaped in text
    assert cohort == original


def test_normalization_and_empty_reports(tmp_path):
    values = {"scalar": np.float32(0.5), "array": np.array([1, 2]), "tuple": (True, None),
              "nan": np.nan, "negative": -np.inf, "positive": np.inf, "empty": {},
              "text": "café\nsecond\rline"}
    text = write_report(values, tmp_path / "report.tex").read_text(encoding="utf-8")
    assert "null" in text and "-Infinity" in text and "true" in text
    assert r"café\newline{}secondline" in text
    assert reporting._rows({}) == [("", "{}")]
    assert reporting._sections({"pairs": []}) == [("Results", {"pairs": []})]
    assert reporting._sections({"pairs": [{"metrics": {}}]})[0][0] == "Pair 1: 1"
    write_report({}, tmp_path / "empty.tex")


@pytest.mark.parametrize("value,error", [({1: 2}, TypeError), ({"x": object()}, TypeError),
                                         ({"pairs": 3}, ValueError), ({"pairs": [1]}, ValueError),
                                         ([], TypeError)])
def test_invalid_reports(value, error, tmp_path):
    with pytest.raises(error):
        write_report(value, tmp_path / "result.tex")


def test_invalid_paths_and_titles(tmp_path):
    with pytest.raises(ValueError, match="must end with"):
        write_report({}, tmp_path / "result.csv")
    for title in (None, " "):
        with pytest.raises(ValueError, match="title"):
            write_report({}, tmp_path / "result.tex", title=title)


def test_pdf_multipage_and_grouped_results(tmp_path, cohort):
    pytest.importorskip("reportlab")
    original = copy.deepcopy(cohort)
    cohort["pairs"] = [dict(cohort["pairs"][0], key=f"case_{index}") for index in range(35)]
    cohort["pairs"][-1]["metadata"] = {"label": "café <study> & group"}
    path = write_report(cohort, tmp_path / "nested" / "report.pdf")
    reader = pdf_reader(path)
    assert len(reader.pages) > 2
    text = "\n".join(page.extract_text() for page in reader.pages)
    for expected in ("Cohort summary", "Grouped summary", "Group A&B", "case_34", "café <study> & group", "0.123456789012345", "Infinity", "sizes[1]", "Additional information"):
        assert expected in text
    assert text.count("Field") >= len(reader.pages)
    # Single-pair API and array-valued metrics use the same renderer.
    write_report({"values": np.arange(300), "missing": np.nan}, tmp_path / "array.pdf")
    assert original["pairs"][0]["key"] == "case_01"


def test_pdf_font_and_long_rows(tmp_path):
    reportlab = pytest.importorskip("reportlab")
    font = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    path = write_report({"note": "long text " * 1800}, tmp_path / "long.pdf", pdf_font=font)
    assert len(pdf_reader(path).pages) > 1
    with pytest.raises(FileNotFoundError, match="font"):
        write_report({}, tmp_path / "bad.pdf", pdf_font=tmp_path / "missing.ttf")
    path.write_bytes(b"previous report")
    with pytest.raises(ValueError, match="font lacks characters"):
        write_report({"label": "\U0001f680"}, path)
    assert path.read_bytes() == b"previous report"


def test_latex_without_reportlab_and_pdf_dependency_error(tmp_path, monkeypatch):
    original_import = importlib.import_module

    def without_reportlab(name, *args, **kwargs):
        if name == "reportlab":
            raise ImportError("Not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(reporting.importlib, "import_module", without_reportlab)
    write_report({"dice": 1}, tmp_path / "result.tex")
    with pytest.raises(ImportError, match=r"\[report\]"):
        write_report({}, tmp_path / "result.pdf")


def test_cli_all_outputs_calculates_once(tmp_path, monkeypatch, capsys, cohort):
    pytest.importorskip("reportlab")
    calls = []

    def calculate(args):
        calls.append(args)
        return cohort

    monkeypatch.setattr(validate, "calculate_metrics", calculate)
    args = []
    for option, extension in (("json", "json"), ("csv", "csv"), ("pdf", "pdf"), ("latex", "tex")):
        args.extend([f"--output-{option}", str(tmp_path / f"results.{extension}")])
    assert validate.main(args) == 0
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out) == cohort
    assert json.loads((tmp_path / "results.json").read_text()) == cohort
    assert "summary" in (tmp_path / "results.csv").read_text()
    assert "Grouped summary" in (tmp_path / "results.tex").read_text()
    assert pdf_reader(tmp_path / "results.pdf").pages
    validate._write_results(tmp_path / "legacy.tex", {"mae": 0})


@pytest.mark.parametrize("option", ["--output", "--output-pdf", "--output-latex"])
def test_cli_invalid_extensions_before_calculation(tmp_path, monkeypatch, option):
    def forbidden(args):
        pytest.fail("Metric calculation should not run for an invalid export request")

    monkeypatch.setattr(validate, "calculate_metrics", forbidden)
    with pytest.raises(SystemExit):
        validate.main([option, str(tmp_path / "result.wrong")])


def test_cli_missing_pdf_dependency_or_font_before_calculation(tmp_path, monkeypatch):
    def unavailable():
        raise ImportError("PDF unavailable")

    def forbidden(args):
        pytest.fail("Metric calculation should not run")

    monkeypatch.setattr(validate, "calculate_metrics", forbidden)
    monkeypatch.setattr(validate, "require_pdf_support", unavailable)
    with pytest.raises(SystemExit):
        validate.main(["--output", str(tmp_path / "report.pdf")])
    monkeypatch.setattr(validate, "require_pdf_support", lambda: None)
    with pytest.raises(SystemExit):
        validate.main(["--output-pdf", str(tmp_path / "report.pdf"), "--pdf-font", "missing-font.ttf"])


def test_manifest_end_to_end_all_formats(tmp_path, capsys):
    pytest.importorskip("reportlab")
    np.save(tmp_path / "real.npy", np.zeros((8, 8)))
    np.save(tmp_path / "same.npy", np.zeros((8, 8)))
    np.save(tmp_path / "different.npy", np.ones((8, 8)))
    manifest = tmp_path / "pairs.csv"
    manifest.write_text("case_id,real,synthetic,label\nsame,real.npy,same.npy,a\nchanged,real.npy,different.npy,b\n")
    args = ["--manifest", str(manifest), "--key-column", "case_id", "--group-by", "label",
            "--metrics", "mae", "psnr", "--data-range", "1", "--num-workers", "2"]
    for option, suffix in (("json", "json"), ("csv", "csv"), ("pdf", "pdf"), ("latex", "tex")):
        args.extend(["--output-" + option, str(tmp_path / ("results." + suffix))])
    assert validate.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["pairs"][0]["metrics"]["psnr"] == "Infinity"
    assert result["summary"]["metrics"]["mae"]["mean"] == 0.5
    assert result["summary"]["metrics"]["psnr"]["count"] == 1
    assert result["grouped_summary"]["groups"]["b"]["metrics"]["mae"]["mean"] == 1
    text = "\n".join(p.extract_text() for p in pdf_reader(tmp_path / "results.pdf").pages)
    assert "Infinity" in text and "Grouped summary" in text and "changed" in text


def test_grouped_api_and_evaluate_pairs_reports(tmp_path):
    from synthetic_imaging_validation import evaluate_pairs, paired_metrics_by_class
    from synthetic_imaging_validation.io import ImagePair, load_image

    real = np.zeros((2, 8, 8))
    report = paired_metrics_by_class(real, real, labels=[0, 1], metrics=["mae"], classes=[0, 1, 2])
    text = write_report(report, tmp_path / "classes.tex").read_text()
    assert r"insufficient\_\allowbreak{}samples" in text
    records = evaluate_pairs([ImagePair("case_a", load_image(real[0]), load_image(real[0]))],
                             lambda pair: {"mae": 0.0})
    write_report({"pairs": records}, tmp_path / "api.tex")


def test_published_deliverable():
    root = Path(__file__).resolve().parents[1]
    reader = pytest.importorskip("pypdf").PdfReader(root / "docs" / "deliverables" / "validation_guide.pdf")
    text = "\n".join(page.extract_text() for page in reader.pages)
    for phrase in ("MS-SSIM", "Hausdorff", "Frechet", "LaTeX", "--output-pdf", "NaN", "metric_selection"):
        assert phrase in text
    assert reader.outline


def test_export_reports_example(tmp_path):
    pytest.importorskip("reportlab")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "examples" / "export_reports.py")],
                   check=True, cwd=tmp_path, capture_output=True)
    assert (tmp_path / "outputs" / "array_comparison.pdf").is_file()
    assert (tmp_path / "outputs" / "array_comparison.tex").is_file()
