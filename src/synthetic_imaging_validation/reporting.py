"""Presentation-only PDF and LaTeX exports; metric values are never recalculated."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
from pathlib import Path
from typing import Any, Optional, Union
from xml.sax.saxutils import escape

import numpy as np


REPORT_NOTE = (
    "Values are exported as supplied, without recalculation or rounding. "
    "For CLI cohort summaries, count is the number of finite scalar values; "
    "std is the population standard deviation. Non-finite values are excluded "
    "from those summaries, not from per-pair results. Infinity and -Infinity "
    "remain visible; null denotes a missing or undefined value. "
    "Scores alone do not establish clinical validity."
)


def require_pdf_support() -> Any:
    """Load the optional PDF dependency, or explain how to install it."""
    try:
        return importlib.import_module("reportlab")
    except ImportError as exc:
        raise ImportError(
            'PDF export requires ReportLab. From the checkout, run '
            'python -m pip install ".[report]".'
        ) from exc


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Report dictionary keys must be strings.")
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported report value: {type(value).__name__}.")


def _rows(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict) and value:
        rows = []
        for key, item in value.items():
            rows.extend(_rows(item, f"{prefix}.{key}" if prefix else key))
        return rows
    if isinstance(value, list) and value:
        return [row for index, item in enumerate(value)
                for row in _rows(item, f"{prefix}[{index}]")]
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return [(prefix, text)]


def _sections(results: dict[str, Any]) -> list[tuple[str, Any]]:
    """Give CLI cohorts readable sections; keep arbitrary API mappings intact."""
    if "pairs" not in results:
        return [("Results", results)]
    pairs = results["pairs"]
    if not isinstance(pairs, list) or any(not isinstance(pair, dict) for pair in pairs):
        raise ValueError("A report's 'pairs' field must be a list of dictionaries.")
    sections = [("Cohort summary", results["summary"])] if "summary" in results else []
    if "grouped_summary" in results:
        sections.append(("Grouped summary", results["grouped_summary"]))
    sections.extend((f"Pair {index}: {pair.get('key', index)}", pair)
                    for index, pair in enumerate(pairs, start=1))
    remaining = {key: value for key, value in results.items()
                 if key not in {"pairs", "summary", "grouped_summary"}}
    if remaining:
        sections.append(("Additional information", remaining))
    return sections or [("Results", {"pairs": []})]


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_\allowbreak{}", "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}", "\n": r"\newline{}", "\r": "",
        ".": r".\allowbreak{}", "/": r"/\allowbreak{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _latex_report(sections: list[tuple[str, Any]], title: str) -> str:
    lines = [
        "% Compile with xelatex or lualatex; no shell escape is needed.",
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[margin=20mm]{geometry}",
        r"\usepackage{fontspec,longtable,array}",
        r"\setmainfont{Latin Modern Roman}",
        r"\setlength{\emergencystretch}{3em}",
        r"\begin{document}",
        r"\section*{" + _latex_escape(title) + "}",
        _latex_escape(REPORT_NOTE),
    ]
    for heading, data in sections:
        lines.extend([
            r"\subsection*{" + _latex_escape(heading) + "}",
            r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.53\linewidth}"
            r">{\raggedright\arraybackslash}p{0.39\linewidth}}",
            r"\hline Field & Value \\ \hline \endfirsthead",
            r"\hline Field & Value \\ \hline \endhead",
        ])
        lines.extend(_latex_escape(key) + " & " + _latex_escape(value) + r" \\"
                     for key, value in _rows(data))
        lines.extend([r"\hline", r"\end{longtable}"])
    return "\n".join(lines + [r"\end{document}", ""])


def _pdf_report(sections: list[tuple[str, Any]], title: str, font_path: Optional[Union[str, Path]]) -> bytes:
    reportlab = require_pdf_support()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

    path = Path(font_path) if font_path is not None else Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    if not path.is_file():
        raise FileNotFoundError(f"PDF font not found: {path}")
    font_name = "Validation-" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    font = TTFont(font_name, str(path))
    pdfmetrics.registerFont(font)
    body = ParagraphStyle("Body", fontName=font_name, fontSize=8, leading=11, spaceAfter=6)
    heading_style = ParagraphStyle("Heading", parent=body, fontSize=12, leading=16, spaceBefore=14, keepWithNext=True)
    title_style = ParagraphStyle("Title", parent=body, fontSize=19, leading=24, spaceAfter=16)

    def paragraph(text: str, style: Any = body) -> Any:
        unsupported = sorted({char for char in text if not char.isspace() and ord(char) not in font.face.charToGlyph})
        if unsupported:
            raise ValueError(
                "PDF font lacks characters " + repr("".join(unsupported))
                + ". Supply a suitable TrueType font with pdf_font / --pdf-font."
            )
        return Paragraph(escape(text).replace("\n", "<br/>"), style)

    stream = io.BytesIO()
    document = SimpleDocTemplate(stream, pagesize=A4, rightMargin=40, leftMargin=40,
                                 topMargin=42, bottomMargin=42, title=title, author="Synthetic Imaging Validation")
    story = [paragraph(title, title_style), paragraph(REPORT_NOTE)]
    for heading, data in sections:
        story.append(paragraph(heading, heading_style))
        rows = [[paragraph("Field"), paragraph("Value")]]
        rows.extend([paragraph(key), paragraph(value)] for key, value in _rows(data))
        table = LongTable(rows, colWidths=[document.width * 0.55, document.width * 0.45],
                          repeatRows=1, splitInRow=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4edf3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d9df")),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([table, Spacer(1, 6)])

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.drawString(40, 24, "Synthetic Imaging Validation")
        canvas.drawRightString(A4[0] - 40, 24, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return stream.getvalue()


def write_report(
    results: dict[str, Any],
    path: Union[str, Path],
    *,
    title: str = "Synthetic imaging validation results",
    pdf_font: Optional[Union[str, Path]] = None,
) -> Path:
    """Export a result mapping to PDF or standalone UTF-8 LaTeX (.tex).

    Accepts a metric dictionary, a CLI cohort report, or a grouped API report.
    Nested dictionaries become dotted field names; lists use indexed fields.
    No metric, summary or significance test is calculated by this function.
    NumPy scalars/arrays are supported. Infinity is displayed explicitly and
    NaN becomes null, matching CLI JSON conventions. Values are not rounded.

    PDF requires the ``report`` extra. The bundled font covers Western European
    text; provide ``pdf_font`` (a TrueType file) for other supported glyphs.
    Unsupported glyphs raise an error instead of silently losing label text.
    LaTeX export needs no extra Python package; compilation requires an external
    XeLaTeX/LuaLaTeX installation and a font covering the document's characters.
    User text is escaped, never interpreted as markup. Parent directories are
    created. Existing destinations are replaced only after rendering succeeds.
    Returns the destination Path. The input mapping is not modified.
    """
    path = Path(path)
    if path.suffix.lower() not in {".pdf", ".tex"}:
        raise ValueError("Report path must end with .pdf or .tex.")
    if not isinstance(results, dict):
        raise TypeError("results must be a dictionary.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Report title must be a non-empty string.")
    sections = _sections(_normalize(results))
    content = (_pdf_report(sections, title, pdf_font) if path.suffix.lower() == ".pdf"
               else _latex_report(sections, title).encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path
