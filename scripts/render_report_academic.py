"""Render PROJECT_REPORT.md as an academic-style .docx.

Matches the thesis conventions: Times New Roman throughout, all-black text
(headings included), 12pt body, formal figure captions numbered Figure N.
Figures are the only colour in the document. Produces reports/PROJECT_REPORT.docx.
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]
MD = REPO / "reports" / "PROJECT_REPORT.md"
OUT = REPO / "reports" / "PROJECT_REPORT.docx"
FIG_DIRS = [REPO / "reports", REPO / "reports" / "figures"]

BLACK = RGBColor(0, 0, 0)
FONT = "Times New Roman"


def _set_font(run, size=12, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = BLACK
    #force the east-asian + ascii font slots so Word honours TNR everywhere
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), FONT)
    rfonts.set(qn("w:hAnsi"), FONT)
    rfonts.set(qn("w:cs"), FONT)


def _find_fig(name):
    base = name.split("/")[-1]
    for d in FIG_DIRS:
        p = d / base
        if p.exists():
            return p
    return None


def _add_runs_with_bold(para, text, size=12):
    #render **bold** inline spans; everything else plain
    parts = re.split(r"(\*\*.+?\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            _set_font(para.add_run(part[2:-2]), size=size, bold=True)
        else:
            _set_font(para.add_run(part), size=size)


def main():
    md = MD.read_text().splitlines()
    doc = Document()

    #base style -> TNR 12 black
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal.font.color.rgb = BLACK

    fig_no = 0
    i = 0
    while i < len(md):
        line = md[i].rstrip()
        i += 1
        if not line.strip():
            continue

        #image line: ![caption](path)
        m = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if m:
            caption, path = m.group(1), m.group(2)
            fig = _find_fig(path)
            if fig:
                fig_no += 1
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(fig), width=Inches(5.8))
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _set_font(cap.add_run(f"Figure {fig_no}. "), size=10, bold=True, italic=True)
                _set_font(cap.add_run(caption), size=10, italic=True)
            continue

        #headings
        if line.startswith("### "):
            p = doc.add_heading(level=3)
            _set_font(p.add_run(line[4:]), size=12, bold=True)
        elif line.startswith("## "):
            p = doc.add_heading(level=2)
            _set_font(p.add_run(line[3:]), size=14, bold=True)
        elif line.startswith("# "):
            p = doc.add_heading(level=1)
            _set_font(p.add_run(line[2:]), size=16, bold=True)
        #bullet
        elif line.strip().startswith(("- ", "* ")):
            p = doc.add_paragraph(style="List Bullet")
            _add_runs_with_bold(p, line.strip()[2:])
        #table row (markdown) -> keep as monospace-ish plain line, skip separators
        elif line.strip().startswith("|"):
            if set(line.strip()) <= set("|-: "):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            p = doc.add_paragraph()
            _add_runs_with_bold(p, "   ".join(cells))
        else:
            p = doc.add_paragraph()
            _add_runs_with_bold(p, line)

    #enforce black + TNR on every heading run (headings default to a theme colour)
    for para in doc.paragraphs:
        for run in para.runs:
            run.font.color.rgb = BLACK
            run.font.name = FONT

    doc.save(str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes), {fig_no} figures embedded")


if __name__ == "__main__":
    main()
