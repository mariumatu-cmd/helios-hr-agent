"""Retrieval: the right document comes back, and out-of-corpus questions don't.

These assert on the *document* retrieved, not on exact scores, so they stay
meaningful if the index is rebuilt.
"""
from __future__ import annotations

import pytest

from rag import retrieve


@pytest.mark.parametrize(
    "query,expected_doc",
    [
        ("how much notice do I need to request three days off", "POL-PTO-001"),
        ("can I work from another country for a few weeks", "POL-INTL-001"),
        ("what is the daily meal per diem when travelling", "POL-EXP-001"),
        ("how long is paid parental leave", "POL-LEAVE-002"),
        ("password and multi-factor requirements", "POL-SEC-001"),
        ("returning my laptop when I leave", "POL-EQUIP-001"),
        ("which holidays does the company observe", "POL-HOL-001"),
        ("switching to a fully remote arrangement", "POL-REMOTE-001"),
    ],
)
def test_expected_document_is_retrieved(query, expected_doc):
    result = retrieve.search(query, k=5)
    assert result.grounded
    assert expected_doc in [h.doc_id for h in result.hits], (
        f"{expected_doc} not in {[h.doc_id for h in result.hits]}"
    )


@pytest.mark.parametrize(
    "query",
    [
        "what is the capital of Portugal",
        "write a python function to reverse a linked list",
        "which stock should I buy this quarter",
        "how do I treat a sprained ankle",
    ],
)
def test_out_of_corpus_queries_are_not_grounded(query):
    result = retrieve.search(query, k=5)
    assert not result.grounded
    assert result.reason


def test_document_filter_restricts_results():
    result = retrieve.search("notice period", k=5, doc_id="POL-INTL-001")
    assert {h.doc_id for h in result.hits} == {"POL-INTL-001"}


def test_unknown_document_filter_is_a_clean_failure():
    result = retrieve.search("anything", k=3, doc_id="POL-NOPE-999")
    assert not result.grounded
    assert "unknown doc_id" in result.reason


def test_hybrid_uses_both_retrievers():
    """If either arm stopped contributing, fusion would silently degrade to a
    single retriever while still appearing to work."""
    result = retrieve.search("POL-INTL-001 rolling 30-day limit", k=8)
    assert any(h.dense_rank for h in result.hits)
    assert any(h.bm25_rank for h in result.hits)


def test_bm25_finds_rare_identifiers():
    """The case dense retrieval is weakest at: a rare exact token."""
    result = retrieve.search("H-1B", k=8)
    assert any("POL-INTL-001" == h.doc_id for h in result.hits)


def test_get_section_by_number():
    hits = retrieve.get_section("POL-INTL-001", "2")
    assert hits
    assert all(h.doc_id == "POL-INTL-001" for h in hits)


def test_get_section_by_heading_text():
    hits = retrieve.get_section("POL-PTO-001", "Notice")
    assert hits
    assert "notice" in " ".join(h.section.lower() for h in hits)


def test_get_section_unknown_returns_empty():
    assert retrieve.get_section("POL-PTO-001", "Interstellar Travel") == []


def test_every_hit_carries_a_citation():
    for hit in retrieve.search("accrual cap", k=5).hits:
        assert hit.citation.startswith(hit.doc_id)


def test_empty_query_is_rejected():
    result = retrieve.search("   ", k=3)
    assert not result.grounded
    assert not result.hits


def test_document_catalogue_covers_the_corpus():
    documents = retrieve.list_documents()
    assert len(documents) == 12
    assert all(d["chunks"] > 0 and d["sections"] for d in documents)
