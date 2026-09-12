"""Test a two-store lexical abstention rule against the single-signal baselines.

The diagnostic in `diagnose_abstention.py` showed that no single score --
cosine similarity, BM25, or raw term coverage -- separates in-corpus questions
from out-of-corpus ones on this corpus. But the *identity* of the unmatched
terms does separate them:

    out of corpus   sabbatical, tuition, pet, stock, revenue, rooftop
                    -> topic nouns; the system holds nothing about them
    in corpus       maya, rodriguez, weber, brazil, wants, starting, blackouts
                    -> entity names, which live in the HR database rather than
                       the policy corpus, and inflections of words that do occur

That suggests the rule the architecture actually implies: a question is
unanswerable when it contains a content term that is unknown to *both* knowledge
stores -- the policy corpus and the structured HR data -- after normalising for
inflection. This script measures whether that rule holds, so the decision is
made on evidence rather than on how plausible the story sounds.
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
from rag.vocabulary import unknown_terms  # noqa: E402


def main() -> int:
    positives = [(c.id, c.question) for c in CASES if c.expected_citations]
    negatives = [(f"N{i:02d}", q) for i, q in enumerate(OUT_OF_CORPUS, start=1)]

    print(f"{'':<4}{'id':<5}{'dense':>7}{'unk':>5}  question / unknown terms")
    counts = {"POS": [], "NEG": []}
    for label, group in (("POS", positives), ("NEG", negatives)):
        for case_id, question in group:
            result = retrieve.search(question, k=6)
            unknown = unknown_terms(question)
            counts[label].append(len(unknown))
            print(
                f"{label:<4}{case_id:<5}{result.best_similarity:>7.3f}"
                f"{len(unknown):>5}  {question[:50]}"
            )
            if unknown:
                print(f"{'':<21}{', '.join(unknown)}")

    print()
    pos_flagged = sum(1 for c in counts["POS"] if c > 0)
    neg_flagged = sum(1 for c in counts["NEG"] if c > 0)
    print(f"positives flagged as unanswerable : {pos_flagged}/{len(counts['POS'])}  "
          f"(these would be false refusals)")
    print(f"negatives flagged as unanswerable : {neg_flagged}/{len(counts['NEG'])}  "
          f"(these are correct refusals)")

    if pos_flagged == 0 and neg_flagged == len(counts["NEG"]):
        print("\nPERFECT SEPARATION on this sample.")
    elif pos_flagged == 0:
        print("\nNo false refusals; abstention recall is partial.")
    else:
        print("\nRule causes false refusals; not safe to ship as a hard gate.")
    return 0


if __name__ == "__main__":
    _ = bm25  # imported for symmetry with the diagnostic; vocabulary owns tokenizing
    raise SystemExit(main())
