"""Ingestion: every source format parses, and chunks stay within bounds.

The corpus is deliberately stored in four formats (md, html, txt, pdf) because
the project requires handling more than one. These tests are what stop that
being a claim rather than a fact.
"""
from __future__ import annotations

import pytest

from config import settings
from rag.ingest.chunk import MAX_TOKENS, chunk_corpus, chunk_document
from rag.ingest.parse import load_corpus


def test_all_four_formats_are_present(corpus):
    assert {d.source_format for d in corpus} == {"md", "html", "txt", "pdf"}


def test_every_document_has_complete_metadata(corpus):
    for doc in corpus:
        assert doc.doc_id, f"{doc.source_file} has no doc_id"
        assert doc.title, f"{doc.doc_id} has no title"
        assert doc.markdown.strip(), f"{doc.doc_id} parsed to empty text"


def test_doc_ids_are_unique(corpus):
    ids = [d.doc_id for d in corpus]
    assert len(ids) == len(set(ids))


def test_pdf_metadata_matches_its_markdown_source(corpus):
    """The PDF is generated from a markdown source; the extracted front matter
    must survive the round trip, or the PDF path silently loses citations."""
    pdf = next(d for d in corpus if d.source_format == "pdf")
    assert pdf.doc_id == "POL-EXP-001"
    assert pdf.title
    assert "per diem" in pdf.markdown.lower()


def test_chunking_is_deterministic(corpus):
    assert [c["id"] for c in chunk_corpus(corpus)] == [c["id"] for c in chunk_corpus(corpus)]


def test_no_chunk_exceeds_the_token_ceiling(corpus):
    oversized = [
        (c["id"], c["token_estimate"])
        for c in chunk_corpus(corpus)
        if c["token_estimate"] > MAX_TOKENS
    ]
    assert not oversized, f"chunks over {MAX_TOKENS} tokens: {oversized}"


def test_every_chunk_carries_a_usable_citation(corpus):
    for chunk in chunk_corpus(corpus):
        assert chunk["citation"].startswith(chunk["doc_id"])
        assert len(chunk["citation"]) > len(chunk["doc_id"]) + 2


def test_embed_text_is_breadcrumbed(corpus):
    """Section headings repeat across documents ("## 11. Contacts"). Without the
    document title prefixed into the embedded text those chunks collapse onto
    each other and retrieval returns the wrong policy."""
    for chunk in chunk_corpus(corpus)[:40]:
        assert chunk["embed_text"].startswith(chunk["doc_title"])


def test_chunks_are_not_empty(corpus):
    for chunk in chunk_corpus(corpus):
        assert chunk["text"].strip()


@pytest.mark.parametrize("doc_id", ["POL-PTO-001", "POL-INTL-001", "POL-EXP-001"])
def test_key_documents_produce_several_sections(corpus, doc_id):
    doc = next(d for d in corpus if d.doc_id == doc_id)
    chunks = chunk_document(doc)
    assert len(chunks) >= 4
    assert len({c["section"] for c in chunks}) >= 3


def test_source_prefixed_files_are_excluded():
    """`_source_travel-expense-policy.md` is the PDF's input; ingesting it too
    would duplicate POL-EXP-001 and double-count it in retrieval."""
    names = {d.source_file for d in load_corpus(settings.corpus_dir)}
    assert not any(n.startswith("_source_") for n in names)
