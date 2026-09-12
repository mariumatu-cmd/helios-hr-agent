"""Diagnose which signals separate in-corpus from out-of-corpus questions.

Prints, for every graded positive and every known negative:
  dense   best cosine similarity
  bm25    top BM25 score
  oov     content terms absent from the corpus vocabulary
  cov     fraction of content terms present in the vocabulary

Exists to choose the abstention rule from evidence rather than intuition.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.cases import CASES  # noqa: E402
from evaluation.run_retrieval_eval import OUT_OF_CORPUS  # noqa: E402
from rag import retrieve  # noqa: E402
from rag.ingest import bm25  # noqa: E402

# Interrogative and generic-HR scaffolding that carries no topical information.
# Present in almost every question, in and out of corpus alike, so including it
# in a coverage ratio only dilutes the signal.
GENERIC = frozenset(
    """what when where who how why which does do can may should must many much
    is are am will would could there here helios company policy employee employees
    my me get take need want tell explain give list show about like right now
    exactly still also please thanks""".split()
)


def profile(question: str) -> dict:
    index = retrieve._Index.get()
    result = retrieve.search(question, k=6)
    sparse = bm25.score(index.bm25, question)
    top_bm25 = sparse[0][1] if sparse else 0.0

    terms = [t for t in set(bm25.tokenize(question)) if t not in GENERIC and len(t) > 2]
    postings = index.bm25["postings"]
    oov = sorted(t for t in terms if t not in postings)
    known = [t for t in terms if t in postings]
    coverage = len(known) / len(terms) if terms else 1.0

    # The highest IDF among terms that ARE in the corpus: how specific is the
    # best thing this question actually matched?
    idf = index.bm25["idf"]
    max_known_idf = max((idf[t] for t in known), default=0.0)

    return {
        "dense": result.best_similarity,
        "bm25": top_bm25,
        "oov": oov,
        "coverage": coverage,
        "max_known_idf": max_known_idf,
        "terms": terms,
    }


def main() -> int:
    positives = [(c.id, c.question) for c in CASES if c.expected_citations]
    negatives = [(f"N{i:02d}", q) for i, q in enumerate(OUT_OF_CORPUS, start=1)]

    rows = []
    for label, group in (("POS", positives), ("NEG", negatives)):
        for case_id, question in group:
            p = profile(question)
            rows.append((label, case_id, question, p))

    print(f"{'':<4}{'id':<5}{'dense':>7}{'bm25':>8}{'cov':>7}{'oov':>5}  question")
    for label, case_id, question, p in sorted(rows, key=lambda r: (r[0], -r[3]["bm25"])):
        print(
            f"{label:<4}{case_id:<5}{p['dense']:>7.3f}{p['bm25']:>8.2f}"
            f"{p['coverage']:>7.2f}{len(p['oov']):>5}  {question[:52]}"
        )
        if p["oov"]:
            print(f"{'':<29}oov: {', '.join(p['oov'])}")

    print()
    for signal in ("dense", "bm25", "coverage"):
        pos = sorted(p[signal] for label, _, _, p in rows if label == "POS")
        neg = sorted(p[signal] for label, _, _, p in rows if label == "NEG")
        overlap = "SEPARABLE" if min(pos) > max(neg) else "overlapping"
        print(
            f"{signal:<10} positives [{min(pos):.3f}, {max(pos):.3f}]  "
            f"negatives [{min(neg):.3f}, {max(neg):.3f}]  -> {overlap}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
