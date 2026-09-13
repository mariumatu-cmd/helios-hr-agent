"""Re-calibrate the dense similarity floor now that a lexical gate exists.

    python scripts/recalibrate_threshold.py

The original calibration (scripts/calibrate_threshold.py) set the floor at 0.65
because, on that negative set, similarity alone separated in-corpus from
out-of-corpus questions. A harder negative set showed it does not: see
evaluation/results/abstention_analysis.md. Topical abstention now comes from
rag/vocabulary.py, which leaves this threshold doing a different and much
narrower job -- rejecting input that is not a question about anything at all.

This script measures where that floor belongs, over three populations:

    real      genuine questions, including terse and statement-phrased ones
    topical   plausible questions about policies that do not exist
    nonsense  strings with no informational content

The floor should sit below every `real` score (no false refusals) and, ideally,
above the `nonsense` scores. It is explicitly *not* expected to separate
`topical` -- that is the lexical gate's job, and pretending otherwise is what
produced the original mistake.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import retrieve  # noqa: E402
from rag.vocabulary import unknown_terms  # noqa: E402

REAL = [
    "How much PTO does a full-time employee with three years of service accrue?",
    "How many company holidays does Helios observe?",
    "What is the maximum number of consecutive days I may work from another country?",
    "If my laptop is stolen while travelling, who do I notify and how quickly?",
    "What is the per-night hotel cap for domestic travel?",
    "Maya Rodriguez wants to work from Portugal for 42 days starting 5 October 2026.",
    "Jonas Weber wants three days of PTO starting 21 September 2026.",
    "Tomas Silva asks to work remotely from Brazil for a month.",
    "Marcus Doyle wants to switch to fully remote.",
    "Is Sofia Marino covered by short-term disability?",
    "blackout periods",
    "parental leave",
    "equipment stipend",
    "can I carry over unused PTO",
    "who approves an exception",
    "H-1B",
]

TOPICAL = [
    "What does the Helios pet insurance policy cover?",
    "How do I enrol in the Helios company car scheme?",
    "How many paid volunteer days does Helios provide each year?",
    "What was Helios's Q3 revenue?",
    "Which Helios office has a rooftop swimming pool?",
    "What is the sabbatical policy after ten years of service?",
    "How many shares are in the employee stock purchase plan?",
]

NONSENSE = [
    "asdkjhasd kjahsdkjh asdkjh",
    "qwertyuiop zxcvbnm",
    "?????",
    "the the the the the",
    "42",
    "blorptastic zyzzyva flimflarn",
    "aaaaaaaaaaaa",
]


def scores(population: list[str]) -> list[tuple[float, str, bool]]:
    out = []
    for question in population:
        # Bypass the gates: the raw similarity is what is being calibrated.
        result = retrieve.search(question, k=6)
        out.append((result.best_similarity, question, not unknown_terms(question)))
    return sorted(out, key=lambda row: row[0])


def main() -> int:
    populations = {"real": REAL, "topical": TOPICAL, "nonsense": NONSENSE}
    measured = {name: scores(items) for name, items in populations.items()}

    for name, rows in measured.items():
        print(f"--- {name} ---")
        for score, question, lexically_ok in rows:
            gate = "lex:pass" if lexically_ok else "lex:REFUSE"
            print(f"  {score:.4f}  {gate:<11} {question[:56]}")
        values = [s for s, _, _ in rows]
        print(f"  min {min(values):.4f}  median {values[len(values) // 2]:.4f}  "
              f"max {max(values):.4f}")
        print()

    real_min = min(s for s, _, _ in measured["real"])
    nonsense_max = max(s for s, _, _ in measured["nonsense"])

    print(f"lowest genuine question : {real_min:.4f}")
    print(f"highest nonsense string : {nonsense_max:.4f}")

    if nonsense_max < real_min:
        floor = round((nonsense_max + real_min) / 2, 2)
        print(f"margin                  : {real_min - nonsense_max:+.4f}")
        print(f"\nRECOMMENDED MIN_DENSE_SIMILARITY = {floor}")
    else:
        print("\nNo separating floor exists; the similarity gate cannot be "
              "calibrated against nonsense either, and should be removed rather "
              "than tuned.")

    topical_caught_lexically = sum(1 for _, _, ok in measured["topical"] if not ok)
    print(
        f"\nlexical gate catches {topical_caught_lexically}/{len(measured['topical'])} "
        f"topical non-questions, which the similarity floor is no longer asked to do."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
