"""Parse the policy corpus into one canonical representation.

The corpus deliberately spans four source formats (markdown, HTML, plain text,
PDF). Rather than writing a separate chunker per format, every parser converts
its input into **canonical markdown** -- ATX headings plus plain paragraphs --
so that exactly one heading-aware chunking implementation runs over all of them.
That keeps chunk boundaries comparable across formats, which matters because
retrieval quality is otherwise format-dependent.

Each document also carries structured metadata (``doc_id``, ``title``,
``version``, ``effective_date``, ``owner``) extracted from the source, which is
what makes precise citation possible downstream.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re

SUPPORTED_SUFFIXES = {".md", ".html", ".htm", ".txt", ".pdf"}

#: Files beginning with this prefix are build inputs (e.g. the markdown source
#: the PDF is generated from) and must not be ingested twice.
SOURCE_PREFIX = "_source_"

_META_KEYS = ("doc_id", "title", "version", "effective_date", "owner", "applies_to", "related")


@dataclasses.dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    version: str
    effective_date: str
    owner: str
    applies_to: str
    related: str
    source_file: str
    source_format: str
    markdown: str

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "doc_title": self.title,
            "version": self.version,
            "effective_date": self.effective_date,
            "owner": self.owner,
            "applies_to": self.applies_to,
            "related": self.related,
            "source_file": self.source_file,
            "source_format": self.source_format,
        }


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _normalise_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _meta_from_kv_lines(text: str) -> dict[str, str]:
    """Pull ``key: value`` lines (front matter, PDF cover block, TXT header)."""
    meta: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([a-z_]+)\s*:\s*(.+?)\s*$", line)
        if m and m.group(1) in _META_KEYS:
            meta.setdefault(m.group(1), m.group(2))
    return meta


def _build(meta: dict[str, str], path: pathlib.Path, markdown: str) -> Document:
    return Document(
        doc_id=meta.get("doc_id") or path.stem.upper(),
        title=meta.get("title") or path.stem.replace("-", " ").title(),
        version=meta.get("version", ""),
        effective_date=meta.get("effective_date", ""),
        owner=meta.get("owner", ""),
        applies_to=meta.get("applies_to", ""),
        related=meta.get("related", ""),
        source_file=path.name,
        source_format=path.suffix.lstrip(".").lower(),
        markdown=_normalise_ws(markdown),
    )


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def parse_markdown(path: pathlib.Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = raw

    if raw.lstrip().startswith("---"):
        start = raw.index("---")
        end = raw.find("\n---", start + 3)
        if end != -1:
            meta = _meta_from_kv_lines(raw[start + 3 : end])
            body = raw[end + 4 :]

    # Drop the boilerplate trailer that follows a horizontal rule at the very end.
    body = re.sub(r"\n---\n+\*This policy is a statement.*$", "\n", body, flags=re.S)
    return _build(meta, path, body)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def parse_html(path: pathlib.Path) -> Document:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")

    meta: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or "").strip()
        if name in _META_KEYS:
            meta[name] = (tag.get("content") or "").strip()
    if "title" not in meta and soup.title and soup.title.string:
        meta["title"] = soup.title.string.strip()

    body = soup.body or soup
    lines: list[str] = []

    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "table", "pre"]):
        # Skip nodes nested inside a node we will render as a whole (li, tr).
        if el.find_parent(["ul", "ol", "table", "pre"]):
            continue

        name = el.name
        if name in ("h1", "h2", "h3", "h4"):
            lines.append(f"\n{'#' * int(name[1])} {el.get_text(' ', strip=True)}\n")
        elif name == "p":
            text = el.get_text(" ", strip=True)
            if text:
                lines.append(text + "\n")
        elif name in ("ul", "ol"):
            for i, li in enumerate(el.find_all("li", recursive=False), start=1):
                bullet = f"{i}." if name == "ol" else "-"
                lines.append(f"{bullet} {li.get_text(' ', strip=True)}")
            lines.append("")
        elif name == "table":
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        elif name == "pre":
            lines.append("```\n" + el.get_text("\n", strip=True) + "\n```\n")

    return _build(meta, path, "\n".join(lines))


# --------------------------------------------------------------------------- #
# plain text
# --------------------------------------------------------------------------- #
_TXT_RULE = re.compile(r"^={10,}$")


def parse_text(path: pathlib.Path) -> Document:
    raw = _normalise_ws(path.read_text(encoding="utf-8"))
    meta = _meta_from_kv_lines(raw[:1500])

    lines = raw.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # A title fenced by ==== rules above and below is a top-level section.
        if (
            _TXT_RULE.match(line.strip())
            and i + 2 < len(lines)
            and _TXT_RULE.match(lines[i + 2].strip())
        ):
            out.append(f"\n## {lines[i + 1].strip().title()}\n")
            i += 3
            continue

        # "4.1 STIPEND" style sub-headings.
        m = re.match(r"^(\d+\.\d+)\s+([A-Z][A-Z0-9 ,'\-/&]{2,})$", line.strip())
        if m:
            out.append(f"\n### {m.group(1)} {m.group(2).title()}\n")
            i += 1
            continue

        # Drop the key: value header block; it is captured as metadata.
        if re.match(r"^\s*(doc_id|title|version|effective_date|owner|applies_to|related)\s*:", line):
            i += 1
            continue

        out.append(line)
        i += 1

    markdown = "\n".join(out)
    # The first two lines are the company/document banner; promote to an H1.
    markdown = re.sub(r"^HELIOS SYSTEMS, INC\.\n(.+)$", r"# \1", markdown, count=1, flags=re.M)
    return _build(meta, path, markdown)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def parse_pdf(path: pathlib.Path) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    raw = _normalise_ws("\n".join(pages))

    meta = _meta_from_kv_lines(raw[:1200])

    out: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()

        if re.match(r"^Helios Systems, Inc\. - Internal - Page \d+$", stripped):
            continue
        if stripped == "HELIOS SYSTEMS, INC. - INTERNAL POLICY":
            continue
        if re.match(
            r"^\s*(doc_id|title|version|effective_date|owner|applies_to|related)\s*:", stripped
        ):
            continue
        # Repeated running header, e.g. "Travel and Expense Policy  (POL-EXP-001)"
        if re.search(r"\(POL-[A-Z]+-\d+\)$", stripped):
            continue

        m = re.match(r"^(\d+)\.\s+([A-Z].*)$", stripped)
        if m:
            out.append(f"\n## {m.group(1)}. {m.group(2)}\n")
            continue
        m = re.match(r"^(\d+\.\d+)\s+([A-Z].*)$", stripped)
        if m:
            out.append(f"\n### {m.group(1)} {m.group(2)}\n")
            continue

        out.append(line)

    markdown = "\n".join(out)
    if meta.get("title"):
        markdown = f"# {meta['title']}\n\n" + markdown
    return _build(meta, path, markdown)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
_PARSERS = {
    ".md": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
    ".txt": parse_text,
    ".pdf": parse_pdf,
}


def parse_file(path: pathlib.Path) -> Document:
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"unsupported source format: {path.suffix}")
    return parser(path)


def load_corpus(corpus_dir: pathlib.Path) -> list[Document]:
    """Parse every supported file in ``corpus_dir``, sorted for determinism."""
    paths = sorted(
        p
        for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith(SOURCE_PREFIX)
    )
    docs = [parse_file(p) for p in paths]

    seen: dict[str, str] = {}
    for doc in docs:
        if doc.doc_id in seen:
            raise ValueError(
                f"duplicate doc_id {doc.doc_id!r} in {doc.source_file} and {seen[doc.doc_id]}"
            )
        seen[doc.doc_id] = doc.source_file
    return docs
