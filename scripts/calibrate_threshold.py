"""Calibrate the out-of-corpus grounding threshold.

Run:  python scripts/calibrate_threshold.py

`rag/retrieve.MIN_DENSE_SIMILARITY` decides when the agent must refuse instead of
grounding an answer in the nearest-but-irrelevant policy. Picking that number by
intuition would be exactly the kind of unvalidated magic constant this project
is supposed to avoid, so it is measured: run a set of questions the corpus does
answer and a set it demonstrably does not, then report the separation between
the two best-similarity distributions.

The printed recommendation is the midpoint of the margin, rounded down; the
report in evaluation/report.md cites these numbers.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.retrieve import MIN_DENSE_SIMILARITY, search  # noqa: E402

IN_CORPUS = [
    "How much PTO do I accrue after five years?",
    "Can I work from another country for a few weeks?",
    "What is the notice period for a three day vacation?",
    "Do part-time employees get short-term disability?",
    "What happens to my laptop when I leave the company?",
    "How long is paid parental leave?",
    "What is the per diem for meals while travelling?",
    "When is the holiday shutdown?",
    "Who approves a switch to fully remote work?",
    "What are the password requirements?",
    "How do promotions get decided?",
    "What should I do in my first week at the company?",
    "Is there a blackout period for time off in customer support?",
    "Can I expense a business class flight?",
    "What counts as a reportable security incident?",
]

OUT_OF_CORPUS = [
    "What is the capital of Portugal?",
    "How do I rotate a Kubernetes service account token?",
    "Write me a Python function that sorts a linked list.",
    "What were the company's Q3 revenue figures?",
    "Which stock should I buy this quarter?",
    "How do I treat a sprained ankle?",
    "What is the airspeed velocity of an unladen swallow?",
    "Summarise the plot of Hamlet.",
    "What is our AWS root account password?",
    "How many employees does Microsoft have?",
]


def main() -> int:
    rows: list[tuple[str, float, str, bool]] = []
    for query in IN_CORPUS:
        result = search(query, k=3)
        rows.append(("in ", result.best_similarity, result.hits[0].doc_id if result.hits else "-", True))
    for query in OUT_OF_CORPUS:
        result = search(query, k=3)
        rows.append(("out", result.best_similarity, result.hits[0].doc_id if result.hits else "-", False))

    in_scores = sorted(s for label, s, _, ok in rows if ok for label, s in [(label, s)])
    out_scores = sorted(s for label, s, _, ok in rows if not ok for label, s in [(label, s)])

    queries = IN_CORPUS + OUT_OF_CORPUS
    print(f"{'kind':<5}{'best_sim':>10}  {'top doc':<16} query")
    for (label, score, doc, _), query in zip(rows, queries, strict=True):
        print(f"{label:<5}{score:>10.4f}  {doc:<16} {query[:58]}")

    lo_in, hi_out = min(in_scores), max(out_scores)
    print()
    print(f"in-corpus  min={lo_in:.4f}  median={in_scores[len(in_scores)//2]:.4f}  max={max(in_scores):.4f}")
    print(f"out-corpus min={min(out_scores):.4f}  median={out_scores[len(out_scores)//2]:.4f}  max={hi_out:.4f}")
    print(f"margin     {lo_in - hi_out:+.4f}")

    if lo_in > hi_out:
        recommended = round((lo_in + hi_out) / 2, 2)
        print(f"separable  -> recommended MIN_DENSE_SIMILARITY = {recommended}")
    else:
        recommended = None
        print("NOT separable at the document level; overlapping band "
              f"[{hi_out:.4f}, {lo_in:.4f}] -- threshold alone cannot gate these.")

    fp = sum(1 for (_, s, _, ok) in rows if not ok and s >= MIN_DENSE_SIMILARITY)
    fn = sum(1 for (_, s, _, ok) in rows if ok and s < MIN_DENSE_SIMILARITY)
    print(f"current    MIN_DENSE_SIMILARITY={MIN_DENSE_SIMILARITY} -> "
          f"{fp} false accepts, {fn} false refusals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
