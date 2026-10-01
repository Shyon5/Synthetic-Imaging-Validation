# PDF and LaTeX reports

JSON and CSV remain the working formats for analysis. PDF is a reading copy
for meetings or deliverables; LaTeX is an editable document for a report.
All four can be written from the same calculation.

## Installation

LaTeX export is included in the base installation. PDF export uses the optional
ReportLab dependency:

```bash
python -m pip install ".[report]"
```

Neither pandas nor a TeX installation is needed to generate a PDF. ReportLab
is imported only when PDF output is requested. If it is missing, the CLI stops
before evaluating the images and explains how to install it.

## Command line

From the repository root, a manifest run can write all formats at once:

```bash
synthetic-imaging-validate --manifest validation_data/pairs.csv --key-column case_id --group-by label --metrics mae rmse --output-json results.json --output-csv results.csv --output-pdf results.pdf --output-latex results.tex
```

The same options work for a single file pair or two paired directories.
`--output results.pdf` and `--output results.tex` are also accepted when only
one output is needed. Existing JSON and CSV schemas are unchanged. Standard
output still contains JSON; report generation does not change progress display
or parallel metric execution.

Parent directories are created automatically. Existing output files are
replaced, so use a new directory or filename to keep earlier runs. Outputs are
written in sequence: if a later export fails, earlier successful files may
already exist. A rendering error does not truncate an existing PDF/LaTeX file.

## Python API and previously saved results

`write_report` accepts a plain metric dictionary, a CLI cohort result, or a
class-wise API report. It does not calculate metrics or add missing summaries:

```python
from synthetic_imaging_validation import mae, write_report

scores = {"mae": mae(real, synthetic)}
write_report(scores, "results.pdf", title="Validation study")
write_report(scores, "results.tex", title="Validation study")
```

There is no need to re-run an evaluation to convert an existing JSON output:

```python
import json
from pathlib import Path
from synthetic_imaging_validation import write_report

results = json.loads(Path("results.json").read_text(encoding="utf-8"))
write_report(results, "results.pdf")
write_report(results, "results.tex")
```

For `evaluate_pairs`, wrap its returned records in `{"pairs": records}`.
The writer will include those records but will not invent a cohort summary.

## What appears in the report

CLI cohort reports contain the global summary, the grouped summary when
requested, and each pair's metric values, identifiers, paths, and metadata.
Other API mappings are presented as supplied. Nested dictionaries use dotted
field names; arrays and lists use indexed rows, such as
`components.component_measures[0]`. Empty containers stay visible.

PDF tables repeat their headers across pages. No numerical rounding is applied.
NumPy scalars and arrays are supported. Infinity is printed explicitly and NaN
is represented as `null`, following the existing CLI JSON convention. Reports
do not distinguish an originally missing value from an undefined NaN once both
have become null. Boolean, text, and descriptive fields remain visible too.

For CLI summaries, a metric's `count` includes finite scalar values only, and
`std` is the population standard deviation. An infinite PSNR for an identical
pair, for example, is present in the pair's results but not in its cohort mean.
Read the count alongside the mean. Grouped API reports also retain their own
missing-value counts and insufficient-sample status where supplied.

No plots, significance tests, confidence intervals, or automatic interpretation
are added. PDF/LaTeX are presentation formats, not a replacement for structured
JSON when preserving an analysis for later processing. Very large component
lists or cohorts may produce long documents; export a deliberately selected
mapping through the API if only a summary is needed.

### Labels, fonts, and safe text

Text is escaped in both formats, including LaTeX characters such as `%`, `&`,
and `_`. It is not interpreted as embedded commands or HTML.

The default PDF font is Bitstream Vera, bundled with ReportLab. It covers
Western European text. Unsupported characters raise a clear error instead of
silently becoming empty squares. For other alphabets, supply an appropriate
TrueType font that you are permitted to use:

```bash
synthetic-imaging-validate --manifest pairs.csv --metrics mae --output-pdf results.pdf --pdf-font fonts/report.ttf
```

In Python, pass `pdf_font="fonts/report.ttf"`. Font coverage alone does not
guarantee correct shaping of complex scripts; visually check those reports.

The LaTeX file is UTF-8 and uses `longtable`, `array`, `geometry`, and
`fontspec`. Compile with XeLaTeX or LuaLaTeX, not plain pdfLaTeX:

```bash
xelatex results.tex
```

Compilation is not performed by the package and requires a separate TeX
installation (or an online editor). Change the generated `\setmainfont`
setting if the default Latin Modern font does not cover the labels. Extremely
long free-text values may need manual layout adjustments in LaTeX.

### Sharing results

Review identifiers, filenames, paths, and manifest metadata before sharing.
The reports are not anonymised. Record the evaluation protocol separately:
the current result schema does not capture every CLI argument, package version,
or preprocessing decision.

## Partner guide

The repository includes a narrative guide covering input preparation, metric
selection, usage, interpretation, and limitations:

- [PDF deliverable](deliverables/validation_guide.pdf)
- [Editable Markdown source](validation_guide.md)

The PDF is maintained separately from the Markdown source and is not rebuilt
automatically. The repository does not include a guide-conversion tool.
This is separate from exporting validation results: the CLI and API can still
generate PDF reports with the `report` extra installed, as described above.
