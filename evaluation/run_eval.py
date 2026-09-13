"""Run the evaluation suite end to end against the live agent.

    python -m evaluation.run_eval                 # all 30 cases
    python -m evaluation.run_eval --category refusal
    python -m evaluation.run_eval --case C01 --case S01
    python -m evaluation.run_eval --repeat 3      # variance across runs
    python -m evaluation.run_eval --resume        # finish a quota-interrupted run

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
CHECKPOINT = RESULTS_DIR / "checkpoint.jsonl"


class QuotaExhausted(RuntimeError):
    """The provider refused on quota grounds, so the run cannot continue.

    Raised rather than scored. A 429 says nothing about whether the agent would
    have answered correctly, and recording it as a failed case would silently
    turn a billing limit into an evaluation result -- which is exactly what
    happened on the first full run, where seven cases "failed" in 0.2s each
    because the daily token cap had been reached.
    """


def _is_quota_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        getattr(exc, "status_code", None) == 429
        or "rate_limit" in text
        or "rate limit" in text
        or "quota" in text
        or "resource_exhausted" in text
    )


def load_checkpoint() -> dict[str, CaseScore]:
    """Case scores already completed in an interrupted run."""
    if not CHECKPOINT.exists():
        return {}
    done: dict[str, CaseScore] = {}
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        done[record["case_id"]] = CaseScore(**record)
    return done


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


async def run_once(
    cases: list[Case],
    client: MCPToolClient,
    checkpoint: bool = False,
    done: dict[str, CaseScore] | None = None,
) -> list[CaseScore]:
    done = done or {}
    scores: list[CaseScore] = []
    for position, case in enumerate(cases, start=1):
        if case.id in done:
            prior = done[case.id]
            verdict = "PASS" if prior.passed else "FAIL"
            print(f"  [{position:>2}/{len(cases)}] {case.id} ({case.category}) ... "
                  f"{verdict}  {prior.score:.2f}  (from checkpoint)")
            scores.append(prior)
            continue

        # Every case starts from the same world. Without this, S01's ticket
        # would still exist when L04 lists tickets, and a passing case would
        # depend on the order the suite happened to run in.
        data.reset_writes()

        print(f"  [{position:>2}/{len(cases)}] {case.id} ({case.category}) ... ", end="", flush=True)
        try:
            trace = await run_agent(case.question, client)
        except Exception as exc:  # noqa: BLE001
            if _is_quota_error(exc):
                print("QUOTA EXHAUSTED")
                raise QuotaExhausted(str(exc)) from exc
            print(f"EXCEPTION {type(exc).__name__}")

            class _Failed:
                answer = ""
                tools_used: list[str] = []
                citations: list[str] = []
                steps: list = []
                total_ms = 0.0
                error = f"{type(exc).__name__}: {exc}"

            score = score_case(case, _Failed())
            scores.append(score)
            if checkpoint:
                _append_checkpoint(score)
            continue

        score = score_case(case, trace)
        scores.append(score)
        if checkpoint:
            _append_checkpoint(score)
        verdict = "PASS" if score.passed else "FAIL"
        print(f"{verdict}  {score.score:.2f}  ({score.latency_ms / 1000:.1f}s)")
        for failure in score.failures:
            print(f"        - {failure}")
    return scores


def _append_checkpoint(score: CaseScore) -> None:
    """Persist one case immediately, so an interruption costs one case, not all.

    A full suite costs more tokens than the free daily allowance, so a run that
    cannot be resumed can never complete at all.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(score.to_dict(), ensure_ascii=False) + "\n")


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
    service = summary["service_latency_ms"]
    throttle = summary["throttle_ms"]
    print(f"latency   wall (s)  mean {latency['mean'] / 1000:.1f}   "
          f"p50 {latency['p50'] / 1000:.1f}   "
          f"p95 {latency['p95'] / 1000:.1f}   max {latency['max'] / 1000:.1f}")
    print(f"        service (s) mean {service['mean'] / 1000:.1f}   "
          f"p50 {service['p50'] / 1000:.1f}   "
          f"p95 {service['p95'] / 1000:.1f}   max {service['max'] / 1000:.1f}")
    if throttle["total"] > 0:
        print(f"        rate-limit waiting removed above: "
              f"{throttle['total'] / 1000:.0f}s total across "
              f"{throttle['cases_throttled']}/{summary['cases']} cases")
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
    done: dict[str, CaseScore] = {}
    if args.resume:
        if args.repeat > 1:
            raise SystemExit("--resume applies to a single run; drop --repeat")
        done = load_checkpoint()
        if done:
            print(f"resuming  : {len(done)} case(s) already scored in "
                  f"{CHECKPOINT.relative_to(ROOT)}")
    elif CHECKPOINT.exists():
        # A stale checkpoint silently mixed with a fresh run would be worse than
        # no checkpoint at all.
        CHECKPOINT.unlink()

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
    exhausted = False
    try:
        for repeat in range(1, args.repeat + 1):
            if args.repeat > 1:
                print(f"\n--- run {repeat}/{args.repeat} ---")
            try:
                runs.append(await run_once(
                    cases, client, checkpoint=args.checkpoint, done=done,
                ))
            except QuotaExhausted as exc:
                exhausted = True
                completed = len(load_checkpoint()) if args.checkpoint else 0
                print(
                    f"\nProvider quota exhausted after {completed}/{len(cases)} cases.\n"
                    f"  {str(exc)[:200]}\n"
                    "Cases not run are NOT scored as failures. Re-run with --resume "
                    "once the quota window rolls over to complete the suite:\n"
                    "    python -m evaluation.run_eval --resume",
                    file=sys.stderr,
                )
                break
    finally:
        await client.aclose()

    if exhausted:
        return 3

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
    parser.add_argument(
        "--no-checkpoint",
        dest="checkpoint",
        action="store_false",
        help="do not persist per-case results as the suite runs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse scores already in the checkpoint and run only the rest",
    )
    args = parser.parse_args()
    if args.repeat > 1:
        # Repeats measure run-to-run variance, so every repeat must actually
        # execute; a checkpoint would let run 2 replay run 1's answers.
        args.checkpoint = False
    if args.resume and not args.checkpoint:
        raise SystemExit("--resume cannot be combined with --no-checkpoint")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
