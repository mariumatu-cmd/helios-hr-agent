"""Heading-aware chunking.

**Why heading-aware rather than fixed token windows.** These documents are
numbered policies whose retrievable unit of meaning is the numbered clause
("§3.1 Notice Requirements"), not an arbitrary span. Splitting on headings gives
three properties a fixed window cannot:

1. a chunk rarely straddles two unrelated rules, so retrieved evidence is
   self-contained and the model is less likely to blend two clauses together;
2. every chunk inherits an exact, human-verifiable citation anchor
   (``POL-PTO-001 §3.1``), which is what makes citation accuracy measurable; and
3. tables -- which carry most of the hard numbers in this corpus -- stay intact
   inside their own section.

Fixed windows are still used *within* a section that exceeds the model's useful
context, with sentence-aligned overlap so a rule split across a boundary is
recoverable from either side.

The embedded text is prefixed with the document title and heading breadcrumb.
Many sections across these policies are lexically near-identical ("## 11.
Contacts" appears in most documents); without the breadcrumb their embeddings
collapse together and retrieval returns the wrong document's section.

Chunking is a pure function of the parsed document, so the index is
deterministic and reproducible.
"""
from __future__ import annotations

import re

from rag.ingest.parse import Document

# Targets are expressed in estimated tokens. bge-small-en-v1.5 truncates at 512
# tokens, so TARGET sits well inside that with room for the breadcrumb prefix.
TARGET_TOKENS = 320
MAX_TOKENS = 420
OVERLAP_TOKENS = 60
MIN_TOKENS = 45

_HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
_SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")
_SENTENCE_END = re.compile(r"(?<=[.:;!?])\s+")


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~1.3 tokens per whitespace word).

    Deliberately avoids a tokenizer dependency: the estimate only needs to be
    monotonic and roughly calibrated to keep chunks inside the embedding
    model's window, and being dependency-free keeps the free-tier image small.
    """
    words = len(text.split())
    return int(words * 1.3) + 1


def _split_heading(text: str) -> tuple[str | None, str]:
    m = _SECTION_NUMBER.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    return None, text.strip()


def _citation(doc_id: str, number: str | None, title: str) -> str:
    return f"{doc_id} \u00a7{number} {title}" if number else f"{doc_id} \u2014 {title}"


def _split_oversized(text: str) -> list[str]:
    """Split one long section into sentence-aligned windows with overlap."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if estimate_tokens(para) <= TARGET_TOKENS:
            units.append(para)
        else:
            units.extend(s.strip() for s in _SENTENCE_END.split(para) if s.strip())

    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > TARGET_TOKENS:
            windows.append("\n\n".join(current))
            # carry trailing units back as overlap
            carry: list[str] = []
            carry_tokens = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(prev)
                if carry_tokens + prev_tokens > OVERLAP_TOKENS:
                    break
                carry.insert(0, prev)
                carry_tokens += prev_tokens
            current, current_tokens = list(carry), carry_tokens
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        windows.append("\n\n".join(current))
    return windows or [text]


def _sections(markdown: str) -> list[dict]:
    """Flatten canonical markdown into leaf sections carrying a heading path."""
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body and not stack:
            return
        if not body:
            return
        number, title = _split_heading(stack[-1][1]) if stack else (None, "Overview")
        sections.append(
            {
                "heading_path": [h for _, h in stack],
                "section_number": number,
                "section_title": title,
                "text": body,
            }
        )

    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            buffer.append(line)
    flush()
    return sections


def _merge_tiny(sections: list[dict]) -> list[dict]:
    """Fold a section that is too small to retrieve on its own into its neighbour.

    A two-line section ("See POL-X for details.") is noise in the index: it
    matches weakly and displaces a real answer. Merging keeps the parent
    heading's citation, which stays accurate because the merged text is still
    from that part of the document.
    """
    merged: list[dict] = []
    for section in sections:
        if (
            merged
            and estimate_tokens(section["text"]) < MIN_TOKENS
            and merged[-1]["heading_path"][:1] == section["heading_path"][:1]
            and estimate_tokens(merged[-1]["text"]) + estimate_tokens(section["text"]) <= MAX_TOKENS
        ):
            previous = merged[-1]
            previous["text"] = (
                f"{previous['text']}\n\n"
                f"{section['section_number'] or ''} {section['section_title']}\n"
                f"{section['text']}"
            ).strip()
            continue
        merged.append(dict(section))
    return merged


def chunk_document(doc: Document) -> list[dict]:
    """Chunk one parsed document into embeddable records with citation metadata."""
    records: list[dict] = []
    base = doc.metadata

    for section in _merge_tiny(_sections(doc.markdown)):
        pieces = (
            [section["text"]]
            if estimate_tokens(section["text"]) <= MAX_TOKENS
            else _split_oversized(section["text"])
        )
        total = len(pieces)

        for part_index, piece in enumerate(pieces):
            breadcrumb = " > ".join(section["heading_path"]) or doc.title
            citation = _citation(
                doc.doc_id, section["section_number"], section["section_title"]
            )
            if total > 1:
                citation = f"{citation} (part {part_index + 1}/{total})"

            records.append(
                {
                    **base,
                    "heading_path": section["heading_path"],
                    "section": section["section_title"],
                    "section_number": section["section_number"],
                    "part": part_index + 1,
                    "part_count": total,
                    "text": piece,
                    "embed_text": f"{doc.title} > {breadcrumb}\n\n{piece}",
                    "token_estimate": estimate_tokens(piece),
                    "citation": citation,
                }
            )

    for ordinal, record in enumerate(records):
        record["ordinal"] = ordinal
    return records


def chunk_corpus(docs: list[Document]) -> list[dict]:
    """Chunk every document and assign stable, deterministic global chunk ids."""
    all_records: list[dict] = []
    for doc in docs:
        all_records.extend(chunk_document(doc))

    for index, record in enumerate(all_records):
        record["id"] = f"c{index:05d}"
    return all_records
