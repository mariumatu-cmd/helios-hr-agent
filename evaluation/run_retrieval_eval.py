"""Retrieval evaluation and ablations. Runs without an LLM API key.

    python -m evaluation.run_retrieval_eval

Two things are measured here that the end-to-end suite cannot isolate:

1. **Retrieval quality**, as recall@k and MRR over the cases that name a
   governing document. If the right passage is never retrieved, no amount of
   prompting will produce a grounded answer -- so this is measured separately
   from the agent, where a retrieval miss and a reasoning miss look identical.

2. **Ablations.** Each configuration is a claim the architecture makes; an
   ablation is what turns the claim into evidence.

     hybrid          dense + BM25 fused with RRF          (shipped)
     dense_only      dense vectors alone                  (is BM25 earning its place?)
     bm25_only       lexical alone                        (is the embedder earning its place?)
     no_threshold    hybrid without the 0.65 dense floor  (does the refusal gate work?)

Results are written to evaluation/results/retrieval_eval.json and a markdown
table suitable for pasting into the report is printed.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import apply_seeds, settings  # noqa: E402
from evaluation.cases import CASES  # noqa: E402
from rag import retrieve  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"

# Questions that should retrieve nothing: there is no such policy. Used to
# measure the false-accept rate, which is the number the similarity floor exists
# to control. Distinct from the calibration set in scripts/calibrate_threshold.py
# so that the threshold is not being evaluated on the data it was tuned on.
OUT_OF_CORPUS = [
    "What does the Helios pet insurance policy cover?",
    "How do I enrol in the Helios company car scheme?",
    "How many paid volunteer days does Helios provide each year?",
    "What was Helios's Q3 revenue?",
    "Which Helios office has a rooftop swimming pool?",
    "What is the sabbatical policy after ten years of service?",
    "How many shares are in the employee stock purchase plan?",
    "What is the bereavement travel allowance for a second cousin?",
]


def graded_queries() -> list[tuple[str, str, list[str]]]:
    """(case_id, question, expected_doc_ids) for every case with a known target."""
    return [
        (case.id, case.question, list(case.expected_citations))
        for case in CASES
        if case.expected_citations
    ]


def docs_in_order(result) -> list[str]:
    """Retrieved document ids, best first, de-duplicated."""
    seen: list[str] = []
    for hit in result.hits:
        if hit.doc_id not in seen:
            seen.append(hit.doc_id)
    return seen


def evaluate(mode: str, threshold: float, k: int, lexical_gate: bool = True) -> dict:
    """Run every graded query plus the out-of-corpus set under one configuration."""
    original_threshold = retrieve.MIN_DENSE_SIMILARITY
    retrieve.MIN_DENSE_SIMILARITY = threshold

    def grounded_of(result) -> bool:
        # Disabling the lexical gate means recomputing groundedness from the
        # similarity floor alone, rather than maintaining a second code path
        # inside `search()` that production would never execute.
        if lexical_gate:
            return result.grounded
        return bool(result.hits) and result.best_similarity >= threshold

    try:
        hits_at_1 = 0
        hits_at_k = 0
        reciprocal_ranks: list[float] = []
        empty = 0
        per_query: list[dict] = []

        graded = graded_queries()
        for case_id, question, expected in graded:
            result = retrieve.search(question, k=k, mode=mode)
            ranked = docs_in_order(result)
            if not ranked:
                empty += 1

            rank = next(
                (i + 1 for i, doc in enumerate(ranked) if doc in expected),
                None,
            )
            hits_at_1 += int(bool(ranked) and ranked[0] in expected)
            hits_at_k += int(rank is not None)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            per_query.append({
                "case": case_id,
                "expected": expected,
                "retrieved": ranked[:5],
                "rank": rank,
                "grounded": grounded_of(result),
                "best_similarity": round(result.best_similarity, 4),
            })

        # A false accept is an out-of-corpus question the system judged
        # *grounded*. `grounded`, not `hits`, is the signal the MCP tool acts on
        # -- retrieval always returns its best guesses, and the threshold is what
        # decides whether those guesses are allowed to become an answer.
        false_accepts = 0
        negatives: list[dict] = []
        for question in OUT_OF_CORPUS:
            result = retrieve.search(question, k=k, mode=mode)
            false_accepts += int(grounded_of(result))
            negatives.append({
                "question": question,
                "grounded": grounded_of(result),
                "best_similarity": round(result.best_similarity, 4),
            })

        # Groundedness recall on the positives: the mirror-image error. A
        # threshold that never false-accepts by refusing everything is useless,
        # so both directions are reported.
        grounded_positives = sum(1 for q in per_query if q["grounded"])

        n = len(graded)
        return {
            "mode": mode,
            "threshold": threshold,
            "k": k,
            "graded_queries": n,
            "recall_at_1": round(hits_at_1 / n, 4),
            f"recall_at_{k}": round(hits_at_k / n, 4),
            "mrr": round(sum(reciprocal_ranks) / n, 4),
            "empty_results_on_graded": empty,
            "grounded_on_graded": grounded_positives,
            "false_refusal_rate": round(1 - grounded_positives / n, 4),
            "out_of_corpus_queries": len(OUT_OF_CORPUS),
            "false_accepts": false_accepts,
            "false_accept_rate": round(false_accepts / len(OUT_OF_CORPUS), 4),
            "per_query": per_query,
            "negatives": negatives,
        }
    finally:
        retrieve.MIN_DENSE_SIMILARITY = original_threshold


CONFIGURATIONS = [
    # (label, retrieval mode, dense threshold, lexical gate)
    #
    # The first four vary retrieval; the last three vary abstention. Read
    # together they answer two separate questions: which signals find the right
    # passage, and which signals know when there is no right passage.
    ("hybrid + both gates (shipped)", "hybrid", retrieve.MIN_DENSE_SIMILARITY, True),
    ("dense only", "dense", retrieve.MIN_DENSE_SIMILARITY, True),
    ("bm25 only", "bm25", retrieve.MIN_DENSE_SIMILARITY, True),
    ("hybrid, similarity gate only", "hybrid", retrieve.MIN_DENSE_SIMILARITY, False),
    ("hybrid, lexical gate only", "hybrid", 0.0, True),
    ("hybrid, no gates", "hybrid", 0.0, False),
]


def main() -> int:
    apply_seeds()
    k = settings.retrieval_k

    info = retrieve.index_info()
    print(f"index   : {info['chunks']} chunks from {info['documents']} documents")
    print(f"model   : {info['model']}")
    print(f"k       : {k}")
    print(f"graded  : {len(graded_queries())} queries with a known target document")
    print(f"negative: {len(OUT_OF_CORPUS)} out-of-corpus queries")
    print()

    results = []
    for label, mode, threshold, lexical in CONFIGURATIONS:
        print(f"running {label} ...", flush=True)
        result = evaluate(mode, threshold, k, lexical_gate=lexical)
        result["label"] = label
        result["lexical_gate"] = lexical
        results.append(result)

    header = (
        f"| configuration | recall@1 | recall@{k} | MRR | false accepts | false refusals |\n"
        f"|---|---|---|---|---|---|"
    )
    rows = [
        f"| {r['label']} | {r['recall_at_1']:.2f} | {r[f'recall_at_{k}']:.2f} | "
        f"{r['mrr']:.3f} | {r['false_accepts']}/{r['out_of_corpus_queries']} | "
        f"{r['graded_queries'] - r['grounded_on_graded']}/{r['graded_queries']} |"
        for r in results
    ]
    table = "\n".join([header, *rows])
    print()
    print(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "retrieval_eval.json"
    path.write_text(
        json.dumps(
            {"index": info, "k": k, "configurations": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "retrieval_eval.md").write_text(
        f"# Retrieval ablations\n\n"
        f"Index: {info['chunks']} chunks / {info['documents']} documents, "
        f"model `{info['model']}`, k={k}.\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    print(f"\nwrote {path.relative_to(ROOT)} and retrieval_eval.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
