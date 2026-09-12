"""Minimal BM25 (Okapi) implementation.

Kept in-repo rather than taking a dependency: the implementation is small, it
removes a package from the free-tier image, and -- more importantly -- the
tokenizer must be *identical* at build time and query time. Owning both ends
makes that guarantee explicit rather than implicit.

BM25 is paired with dense retrieval because the two fail differently on this
corpus. Dense retrieval handles paraphrase ("can I take time off" -> "PTO
request"); BM25 handles the exact identifiers that dominate these documents
(``POL-INTL-001``, ``30-day``, ``H-1B``, ``§3.1``), where an embedding of a rare
token is unreliable. Reciprocal Rank Fusion combines them in ``rag/retrieve.py``.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# Deliberately small: policy text is short and domain-specific, so aggressive
# stop-word removal costs more than it saves. Only true function words go.
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for from by with
    as is are was were be been being do does did doing have has had having it its they them their
    you your we our i he she his her not no""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, keep hyphenated identifiers intact."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def build(corpus_texts: list[str]) -> dict:
    """Build a BM25 index over already-tokenizable raw texts."""
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_len: list[int] = []

    for doc_index, text in enumerate(corpus_texts):
        tokens = tokenize(text)
        doc_len.append(len(tokens))
        for term, freq in Counter(tokens).items():
            postings[term].append((doc_index, freq))

    n_docs = len(corpus_texts)
    avgdl = (sum(doc_len) / n_docs) if n_docs else 0.0
    idf = {
        term: math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
        for term, plist in postings.items()
    }

    return {
        "postings": {t: p for t, p in postings.items()},
        "idf": idf,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "k1": K1,
        "b": B,
        "n_docs": n_docs,
    }


def score(index: dict, query: str, allowed: set[int] | None = None) -> list[tuple[int, float]]:
    """Score documents against a query, highest first."""
    postings = index["postings"]
    idf = index["idf"]
    doc_len = index["doc_len"]
    avgdl = index["avgdl"] or 1.0
    k1 = index["k1"]
    b = index["b"]

    scores: dict[int, float] = defaultdict(float)
    for term in set(tokenize(query)):
        plist = postings.get(term)
        if not plist:
            continue
        weight = idf[term]
        for doc_index, freq in plist:
            if allowed is not None and doc_index not in allowed:
                continue
            denominator = freq + k1 * (1 - b + b * doc_len[doc_index] / avgdl)
            scores[doc_index] += weight * (freq * (k1 + 1)) / denominator

    return sorted(scores.items(), key=lambda kv: -kv[1])
