"""Build the partner guide PDF from its Markdown source (requires the report extra).

This small renderer intentionally supports the guide's headings, paragraphs,
hyphen bullets, inline code, and fenced code blocks, not arbitrary Markdown.
Run from any directory; defaults are resolved relative to this script.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate


ROOT = Path(__file__).resolve().parents[1]


def inline(text: str) -> str:
    """Escape all text, then render inline code from the trusted guide source."""
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escape(text))


class GuideDocument(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"Title", "Section"}:
            title = flowable.getPlainText()
            key = f"heading-{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=0)


def build(source: Path, output: Path) -> None:
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8)
    title = ParagraphStyle("Title", parent=body, fontSize=27, leading=32, textColor=colors.HexColor("#173a50"), spaceAfter=20)
    section = ParagraphStyle("Section", parent=body, fontSize=18, leading=23, spaceAfter=14, keepWithNext=True)
    subheading = ParagraphStyle("Subheading", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=15, spaceBefore=8, keepWithNext=True)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=13, firstLineIndent=-10, spaceAfter=7)
    code = ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=7.5, leading=10,
                          backColor=colors.HexColor("#eff4f7"), borderPadding=8, spaceBefore=6, spaceAfter=14)
    story = []
    paragraph = []
    code_lines = []
    in_code = False

    def flush():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), body))
            paragraph.clear()

    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            flush()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), code, maxLineLength=110))
                code_lines.clear()
            in_code = not in_code
        elif in_code:
            code_lines.append(line)
        elif line.startswith("#"):
            flush()
            if re.match(r"## \d+\.", line):
                story.append(PageBreak())
            level, text = line.split(" ", 1)
            style = title if level == "#" else section if level == "##" else subheading
            story.append(Paragraph(inline(text), style))
        elif line.startswith("- "):
            flush()
            story.append(Paragraph("- " + inline(line[2:]), bullet))
        elif not line.strip():
            flush()
        else:
            paragraph.append(line.strip())
    flush()
    if in_code:
        raise ValueError("Unclosed code block in guide source.")

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#25a59a"))
        canvas.setLineWidth(2)
        canvas.line(44, A4[1] - 29, A4[0] - 44, A4[1] - 29)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#456070"))
        canvas.drawString(44, 25, "Synthetic Imaging Validation | Partner guide | September 2026")
        canvas.drawRightString(A4[0] - 44, 25, str(doc.page))
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    document = GuideDocument(str(output), pagesize=A4, leftMargin=44, rightMargin=44,
                             topMargin=48, bottomMargin=44,
                             title="Synthetic Imaging Validation - Practical guide for project partners",
                             author="Synthetic Imaging Validation", invariant=1)
    document.build(story, onFirstPage=page, onLaterPages=page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "docs" / "validation_guide.md")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "deliverables" / "validation_guide.pdf")
    args = parser.parse_args()
    build(args.source, args.output)
    print(f"Created {args.output}")
