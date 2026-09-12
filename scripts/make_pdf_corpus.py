"""Render a markdown policy source into a PDF corpus document.

The project requires the corpus to span at least two source formats. We keep the
PDF's *source* in markdown (reviewable, diffable) and generate the PDF
deterministically, so the repository never contains an unreproducible binary.

    python scripts/make_pdf_corpus.py

Input : corpus/_source_travel-expense-policy.md   (git-tracked, ignored by the
                                                   ingester via the `_source_`
                                                   filename prefix)
Output: corpus/travel-expense-policy.pdf
"""
from __future__ import annotations

import pathlib
import re
import sys

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"

JOBS = [("_source_travel-expense-policy.md", "travel-expense-policy.pdf")]

# Core PDF fonts are latin-1 only; normalise the typographic characters we use.
_SUBS = {
    "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
    "\u2265": ">=", "\u2264": "<=", "\u00d7": "x",
}


def _ascii(text: str) -> str:
    for src, dst in _SUBS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def _block(pdf: FPDF, height: float, text: str, indent: float = 0.0) -> None:
    """Write a full-width text block and return the cursor to the left margin.

    fpdf2's ``multi_cell`` defaults to ``new_x=RIGHT``, which leaves the cursor at
    the right margin; a following ``multi_cell(w=0, ...)`` then has zero width and
    raises. Always reset x explicitly.
    """
    pdf.set_x(pdf.l_margin + indent)
    pdf.multi_cell(0, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    body = raw
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            for line in raw[3:end].strip().splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip()
            body = raw[end + 4 :]
    return meta, body.lstrip("\n")


class PolicyPDF(FPDF):
    def __init__(self, title: str, doc_id: str) -> None:
        super().__init__(format="A4", unit="mm")
        self.doc_title = title
        self.doc_id = doc_id
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 18, 20)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110)
        self.cell(0, 6, _ascii(f"{self.doc_title}  ({self.doc_id})"), align="L")
        self.ln(8)
        self.set_text_color(0)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110)
        self.cell(0, 6, f"Helios Systems, Inc. - Internal - Page {self.page_no()}", align="C")
        self.set_text_color(0)


def _render(pdf: PolicyPDF, body: str) -> None:
    in_code = False
    for raw_line in body.splitlines():
        line = _ascii(raw_line.rstrip())

        if line.startswith("```"):
            in_code = not in_code
            continue

        if not line.strip():
            pdf.ln(3)
            continue

        if in_code:
            pdf.set_font("Courier", "", 9)
            _block(pdf, 4.5, line)
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, text = len(m.group(1)), m.group(2)
            sizes = {1: 17, 2: 13, 3: 11, 4: 10}
            pdf.ln(4 if level > 1 else 2)
            pdf.set_font("Helvetica", "B", sizes[level])
            _block(pdf, 7 if level == 1 else 6, text)
            pdf.ln(1.5)
            pdf.set_font("Helvetica", "", 10)
            continue

        # horizontal rule
        if re.fullmatch(r"-{3,}", line.strip()):
            pdf.ln(2)
            y = pdf.get_y()
            pdf.set_draw_color(180)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.set_draw_color(0)
            pdf.ln(3)
            continue

        # markdown table -> aligned plain text (skip the |---|---| separator row)
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            pdf.set_font("Courier", "", 8.5)
            _block(pdf, 4.5, "  ".join(c.ljust(22)[:22] for c in cells))
            pdf.set_font("Helvetica", "", 10)
            continue

        # bullets
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m:
            indent = 4 + len(m.group(1))
            pdf.set_font("Helvetica", "", 10)
            _block(pdf, 5, f"- {_strip_md(m.group(2))}", indent)
            continue

        # numbered list
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            indent = 4 + len(m.group(1))
            pdf.set_font("Helvetica", "", 10)
            _block(pdf, 5, f"{m.group(2)}. {_strip_md(m.group(3))}", indent)
            continue

        pdf.set_font("Helvetica", "", 10)
        _block(pdf, 5, _strip_md(line))


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


def build(src_name: str, out_name: str) -> pathlib.Path:
    src = CORPUS / src_name
    meta, body = _split_front_matter(src.read_text(encoding="utf-8"))

    title = meta.get("title", src.stem)
    doc_id = meta.get("doc_id", src.stem.upper())

    pdf = PolicyPDF(title, doc_id)
    pdf.set_title(title)
    pdf.set_author("Helios Systems, Inc.")
    pdf.set_subject(doc_id)
    pdf.set_keywords(
        " ".join(f"{k}={v}" for k, v in meta.items() if k in {"doc_id", "version", "owner"})
    )
    pdf.add_page()

    # Cover block: keeps doc_id/version/effective_date inside the *extractable
    # text* so the ingester derives identical metadata from the PDF as from md.
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(90)
    pdf.cell(0, 5, "HELIOS SYSTEMS, INC. - INTERNAL POLICY")
    pdf.ln(7)
    pdf.set_text_color(0)
    pdf.set_font("Courier", "", 9)
    for key in ("doc_id", "title", "version", "effective_date", "owner", "applies_to", "related"):
        if key in meta:
            _block(pdf, 4.5, _ascii(f"{key}: {meta[key]}"))
    pdf.ln(4)

    _render(pdf, body)

    out = CORPUS / out_name
    pdf.output(str(out))
    return out


def main() -> int:
    for src_name, out_name in JOBS:
        if not (CORPUS / src_name).is_file():
            print(f"ERROR: missing source {src_name}", file=sys.stderr)
            return 1
        out = build(src_name, out_name)
        print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
