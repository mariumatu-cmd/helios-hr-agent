"""Explicitly opt-in verification of the deployed service.

python scripts/verify_deployed_tasks.py --allow-live --task pto --confirm-mock-actions
python scripts/verify_deployed_tasks.py --allow-live --task international --case R03 --case L01

Runs the two demo tasks offered in the UI (or those named by --task) and any
evaluation cases named by --case, scoring every answer with the current scorer.
Consumes LLM quota, so it stops at the first quota, provider or HTTP error
rather than spending more. The optional confirmation creates an in-memory mock
ticket.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from http.cookiejar import CookieJar
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.citations import identities  # noqa: E402
from agent.demo_tasks import DEMO_WORKFLOWS  # noqa: E402
from evaluation.cases import CASES, by_id  # noqa: E402
from evaluation.score import SCORER_VERSION, score_case  # noqa: E402

# Agent errors that describe a bad answer rather than an unhealthy provider.
# They are scored as failures; any other error stops the run, because further
# requests would only spend quota.
ANSWER_ERRORS = (
    "Final answer failed evidence validation",
    "MalformedToolCall",
    "Context budget exceeded",
)


def restore_trace(value):
    from types import SimpleNamespace

    if isinstance(value, list):
        return [restore_trace(v) for v in value]
    if isinstance(value, dict):
        # Tool payloads remain JSON dictionaries; trace objects use attributes.
        return SimpleNamespace(**{
            k: restore_trace(v) if k in ("steps", "tool_calls") else v
            for k, v in value.items()
        })
    return value


def assess(task_id: str, payload: dict) -> list[str]:
    score = score_case(by_id("C01" if task_id == "international" else "C02"), restore_trace(payload))
    failures = list(score.failures)
    calls = [
        c for s in payload.get("steps", []) for c in s.get("tool_calls", [])
        if not c.get("is_error")
    ]
    tools = {c["name"] for c in calls}
    if not tools & {"search_policy_documents", "get_policy_section"}:
        failures.append("No successful MCP policy retrieval")
    if len(payload.get("steps", [])) < 2 or len(tools) < 2:
        failures.append("Not a multi-step, multi-tool workflow")
    used = identities(payload.get("answer", ""))
    passages = set().union(*(
        identities(s["citation"]) for s in payload.get("sources", []) if s.get("snippet")
    ))
    if not used or not used & passages:
        failures.append("No inline citation backed by a returned passage")
    if task_id == "international" and not all(
        any(c.startswith(doc) for c in used) for doc in ("POL-INTL-001", "POL-REMOTE-001")
    ):
        failures.append("Expected citations to both international and remote-work policies")
    if task_id == "pto" and not payload.get("pending_actions"):
        failures.append("Missing ticket preview")
    return failures


def metrics(trace: dict) -> dict:
    """The latency and quota figures the documentation quotes for each run."""
    return {
        "agent_seconds": round(trace.get("total_ms", 0) / 1000, 1),
        "throttle_seconds": round(trace.get("throttle_ms", 0) / 1000, 1),
        "api_calls": trace.get("api_calls", 0),
        "tokens": trace.get("prompt_tokens", 0) + trace.get("completion_tokens", 0),
        "model": trace.get("model", ""),
        "fell_back": bool(trace.get("fell_back")),
    }


def describe(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            detail = json.load(exc).get("detail", "")
        except (ValueError, AttributeError):
            detail = ""
        return f"HTTP {exc.code} {detail}".strip()
    return f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--confirm-mock-actions", action="store_true")
    parser.add_argument(
        "--task", action="append", default=[], choices=[t["id"] for t in DEMO_WORKFLOWS],
        help="demo task to run (repeatable); with neither --task nor --case, both run",
    )
    parser.add_argument(
        "--case", action="append", default=[], choices=[c.id for c in CASES],
        help="evaluation case to run after the demo tasks (repeatable)",
    )
    parser.add_argument("--base-url", default="https://helios-hr-assistant-wz3c.onrender.com")
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("Add --allow-live to authorize quota-consuming model calls.")
    opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def call(path, body=None):
        request = Request(
            args.base_url.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "X-Demo-Code": os.environ.get("DEMO_ACCESS_CODE", "")},
        )
        with opener.open(request, timeout=600) as response:
            return json.load(response)

    runs = [
        (task["id"], task["question"], True) for task in DEMO_WORKFLOWS
        if task["id"] in args.task or not (args.task or args.case)
    ] + [(case_id, by_id(case_id).question, False) for case_id in args.case]
    build_sha = call("/health").get("build_sha")
    results, stop = [], ""
    for run_id, question, is_task in runs:
        started = time.monotonic()
        trace, actions, failures = {}, [], []
        try:
            trace = call("/chat", {"message": question, "fresh": True})
            failures += (
                assess(run_id, trace) if is_task
                else score_case(by_id(run_id), restore_trace(trace)).failures
            )
            error = trace.get("error", "")
            if error and not error.startswith(ANSWER_ERRORS):
                stop = error
            if is_task and args.confirm_mock_actions and not failures:
                for pending in trace.get("pending_actions", []):
                    result = call(f"/actions/{pending['id']}/confirm", {})
                    actions.append(result)
                    if (
                        result.get("status") != "completed" or result.get("api_calls") != 0
                        or not result.get("result", {}).get("ticket_id")
                    ):
                        failures.append("Confirmed mock ticket was not verified")
        except (HTTPError, URLError, TimeoutError) as exc:
            stop = describe(exc)
            failures.append(f"Request failed: {stop}")
        results.append({
            "task": run_id, "kind": "demo task" if is_task else "evaluation case",
            "prompt": question, "trace": trace, "confirmation_results": actions,
            "failures": failures, "passed": not failures,
            "wall_seconds": round(time.monotonic() - started, 2), "metrics": metrics(trace),
        })
        m = results[-1]["metrics"]
        print(
            f"{run_id:<13} {'FAIL' if failures else 'PASS'}  "
            f"wall {results[-1]['wall_seconds']:.0f}s  waited {m['throttle_seconds']:.0f}s  "
            f"calls {m['api_calls']}  tokens {m['tokens']}  {m['model']}"
            + (" (fallback)" if m["fell_back"] else "")
        )
        for failure in failures:
            print("    -", failure)
        if stop:
            print("Stopped early so that no more quota is spent:", stop)
            break
    walls = [r["wall_seconds"] for r in results]
    summary = {
        "runs": len(results), "passed": sum(r["passed"] for r in results),
        "wall_seconds_median": round(statistics.median(walls), 1),
        "wall_seconds_max": max(walls),
        "throttle_seconds": round(sum(r["metrics"]["throttle_seconds"] for r in results), 1),
        "api_calls": sum(r["metrics"]["api_calls"] for r in results),
        "tokens": sum(r["metrics"]["tokens"] for r in results),
        "stopped_early": stop or None,
    }
    print(
        f"{summary['passed']}/{summary['runs']} passed; wall median "
        f"{summary['wall_seconds_median']}s, max {summary['wall_seconds_max']}s; "
        f"{summary['api_calls']} model calls, {summary['tokens']} tokens"
    )
    path = ROOT / "evidence" / f"deployed-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps({
        "base_url": args.base_url, "build_sha": build_sha,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scorer_version": SCORER_VERSION, "summary": summary, "tasks": results,
    }, indent=2), encoding="utf-8")
    print("saved", path.relative_to(ROOT))
    return int(bool(stop) or any(not r["passed"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
