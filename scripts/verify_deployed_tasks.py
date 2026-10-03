"""Explicitly opt-in verification of the same two tasks offered in the UI.

python scripts/verify_deployed_tasks.py --allow-live --task pto --confirm-mock-actions
Consumes LLM quota. The optional confirmation creates an in-memory mock ticket.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from http.cookiejar import CookieJar
from urllib.request import HTTPCookieProcessor, Request, build_opener

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.citations import identities  # noqa: E402
from agent.demo_tasks import DEMO_WORKFLOWS  # noqa: E402
from evaluation.cases import by_id  # noqa: E402
from evaluation.score import SCORER_VERSION, score_case  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--confirm-mock-actions", action="store_true")
    parser.add_argument("--task", choices=["international", "pto"])
    parser.add_argument("--base-url", default="https://helios-hr-assistant-wz3c.onrender.com")
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("Add --allow-live to authorize quota-consuming model calls.")
    opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def post(path, body):
        request = Request(
            args.base_url.rstrip("/") + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "X-Demo-Code": os.environ.get("DEMO_ACCESS_CODE", "")},
        )
        with opener.open(request, timeout=600) as response:
            return json.load(response)

    results = []
    for task in DEMO_WORKFLOWS:
        if args.task and task["id"] != args.task:
            continue
        started = time.monotonic()
        trace = post("/chat", {"message": task["question"], "fresh": True})
        failures = assess(task["id"], trace)
        actions = []
        if args.confirm_mock_actions and not failures:
            for pending in trace.get("pending_actions", []):
                result = post(f"/actions/{pending['id']}/confirm", {})
                actions.append(result)
                if (
                    result.get("status") != "completed" or result.get("api_calls") != 0
                    or not result.get("result", {}).get("ticket_id")
                ):
                    failures.append("Confirmed mock ticket was not verified")
        results.append({
            "task": task["id"], "prompt": task["question"], "trace": trace,
            "confirmation_results": actions, "failures": failures,
            "passed": not failures, "wall_seconds": round(time.monotonic() - started, 2),
        })
        print(task["id"], "FAIL: " + "; ".join(failures) if failures else "PASS")
    path = ROOT / "evidence" / f"deployed-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps({
        "base_url": args.base_url, "scorer_version": SCORER_VERSION, "tasks": results,
    }, indent=2), encoding="utf-8")
    return int(any(not r["passed"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
