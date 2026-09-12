"""Run the evaluation suite end to end against the live agent.

    python -m evaluation.run_eval                 # all 26 cases
    python -m evaluation.run_eval --category refusal
    python -m evaluation.run_eval --case C01 --case S01
    python -m evaluation.run_eval --repeat 3      # variance across runs

Requires an LLM provider (GROQ_API_KEY or GEMINI_API_KEY). Results are written
to evaluation/results/eval_<timestamp>.json and a human-readable summary is
printed. The agent runs at temperature 0, and `--repeat` exists because that
still is not fully deterministic through a hosted API -- reporting the spread is
more honest than reporting one run as if it were the value.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import llm  # noqa: E402
from agent.mcp_client import MCPToolClient  # noqa: E402
from agent.orchestrator import run_agent  # noqa: E402
from config import apply_seeds, settings  # noqa: E402
from evaluation.cases import CASES, Case  # noqa: E402
from evaluation.score import CaseScore, aggregate, score_case  # noqa: E402
from mcp_server import data  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"


def select(args: argparse.Namespace) -> list[Case]:
    cases = CASES
    if args.category:
        cases = [c for c in cases if c.category in args.category]
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.id in wanted]
    if not cases:
        raise SystemExit("no cases matched the given filters")
    return cases


async def run_once(cases: list[Case], client: MCPToolClient) -> list[CaseScore]:
    scores: list[CaseScore] = []
    for position, case in enumerate(cases, start=1):
        # Every case starts from the same world. Without this, S01's ticket
        # would still exist when L04 lists tickets, and a passing case would
        # depend on the order the suite happened to run in.
        data.reset_writes()

        print(f"  [{position:>2}/{len(cases)}] {case.id} ({case.category}) ... ", end="", flush=True)
        try:
            trace = await run_agent(case.question, client)
        except Exception as exc:  # noqa: BLE001
            print(f"EXCEPTION {type(exc).__name__}")

            class _Failed:
                answer = ""
                tools_used: list[str] = []
                citations: list[str] = []
                steps: list = []
                total_ms = 0.0
                error = f"{type(exc).__name__}: {exc}"

            scores.append(score_case(case, _Failed()))
            continue

        score = score_case(case, trace)
        scores.append(score)
        verdict = "PASS" if score.passed else "FAIL"
        print(f"{verdict}  {score.score:.2f}  ({score.latency_ms / 1000:.1f}s)")
        for failure in score.failures:
            print(f"        - {failure}")
    return scores


def print_summary(summary: dict) -> None:
    print()
    print("=" * 72)
    print(f"cases {summary['cases']}   passed {summary['passed']}   "
          f"pass rate {summary['pass_rate']:.0%}   mean score {summary['mean_score']:.3f}")
    print("-" * 72)

    print("by dimension")
    for name, stats in summary["dimensions"].items():
        if stats["n"]:
            print(f"  {name:<16} {stats['mean']:.3f}   (n={stats['n']})")

    print("by category")
    for name, stats in sorted(summary["by_category"].items()):
        print(f"  {name:<16} {stats['mean_score']:.3f}   "
              f"pass {stats['passed']}/{stats['cases']}")

    print("by difficulty")
    for name in ("easy", "medium", "hard"):
        stats = summary["by_difficulty"].get(name)
        if stats:
            print(f"  {name:<16} {stats['mean_score']:.3f}   "
                  f"pass {stats['passed']}/{stats['cases']}")

    latency = summary["latency_ms"]
    print(f"latency (s)        mean {latency['mean'] / 1000:.1f}   "
          f"median {latency['median'] / 1000:.1f}   "
          f"p90 {latency['p90'] / 1000:.1f}   max {latency['max'] / 1000:.1f}")
    print(f"mean steps         {summary['mean_steps']:.2f}")
    print("=" * 72)


async def main_async(args: argparse.Namespace) -> int:
    apply_seeds()

    providers = llm.available_providers()
    if not providers:
        print(
            "No LLM provider configured. Set GROQ_API_KEY or GEMINI_API_KEY "
            "(see .env.example) and run again.\n"
            "The retrieval evaluation and the ablations do not need a key:\n"
            "    python -m evaluation.run_retrieval_eval",
            file=sys.stderr,
        )
        return 2

    cases = select(args)
    print(f"provider  : {settings.llm_provider} (configured: {', '.join(providers)})")
    print(f"model     : {llm.active_model()}")
    print(f"as_of_date: {settings.as_of_date}")
    print(f"cases     : {len(cases)}   repeats: {args.repeat}")
    print()

    client = MCPToolClient()
    await client.connect()
    print(f"MCP       : {client.server_name} v{client.server_version}, "
          f"{len(client.tools)} tools over {client.transport}")

    runs: list[list[CaseScore]] = []
    try:
        for repeat in range(1, args.repeat + 1):
            if args.repeat > 1:
                print(f"\n--- run {repeat}/{args.repeat} ---")
            runs.append(await run_once(cases, client))
    finally:
        await client.aclose()

    summaries = [aggregate(run) for run in runs]
    print_summary(summaries[0] if args.repeat == 1 else aggregate(
        [s for run in runs for s in run]
    ))

    if args.repeat > 1:
        means = [s["mean_score"] for s in summaries]
        print(f"across {args.repeat} runs: mean score "
              f"{statistics.mean(means):.3f} +/- {statistics.pstdev(means):.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = RESULTS_DIR / f"eval_{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": stamp,
                "provider": settings.llm_provider,
                "model": llm.active_model(),
                "as_of_date": settings.as_of_date,
                "repeats": args.repeat,
                "summary": aggregate([s for run in runs for s in run]),
                "per_run_summary": summaries,
                "cases": [[s.to_dict() for s in run] for run in runs],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {path.relative_to(ROOT)}")

    overall = aggregate([s for run in runs for s in run])
    return 0 if overall["pass_rate"] >= args.min_pass_rate else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append", help="filter by category (repeatable)")
    parser.add_argument("--case", action="append", help="run specific case ids (repeatable)")
    parser.add_argument("--repeat", type=int, default=1, help="run the suite N times")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="exit non-zero below this pass rate (for use as a gate)",
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
