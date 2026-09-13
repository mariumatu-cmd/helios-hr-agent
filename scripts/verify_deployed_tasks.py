"""Prove the two required end-to-end agentic tasks complete on the LIVE deploy.

The rubric asks for at least two end-to-end agentic tasks in the deployed demo,
each requiring multi-step reasoning, RAG retrieval and structured/mock-data tool
use. This hits the deployed URL -- not a local process -- and records what the
grader would see: the tool-call trace, the citations, and the latency.

Writes evidence/deployed-tasks.json for the design doc to quote.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

BASE = "https://helios-hr-assistant-wz3c.onrender.com"

# Task design note: each prompt deliberately spans three capabilities that no
# single tool covers -- structured mock data (balances, usage history),
# retrieved policy prose (conditions the rule engine does not encode), and a
# concrete action (ticket or email). That is what forces genuine multi-step
# reasoning rather than one composite tool call.
TASKS = [
    (
        "international-remote-work",
        # Maya Rodriguez (E-1041) has 12 approved days inside the rolling
        # 12-month window and 12 more deliberately outside it, so a correct
        # answer must exclude the latter. 14 days keeps her within the 30-day
        # limit, so the request should pass and the ticket should be filed.
        "I'm Maya Rodriguez. I'd like to work remotely from Portugal from "
        "2026-10-05 to 2026-10-18. How many international days have I already "
        "used this year, and does this request stay within the limit? If it "
        "does, please file the ticket for me, and tell me what conditions the "
        "policy places on working from an EU country.",
    ),
    (
        "parental-leave-and-pto",
        "I'm David Okafor and my partner is due in March. How much paid "
        "parental leave am I entitled to, what is my current PTO balance, and "
        "does taking parental leave affect that balance? Cite the policy, then "
        "draft the email I should send my manager.",
    ),
]


def post(path: str, payload: dict, timeout: int = 600) -> tuple[int, dict, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            code = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        code = exc.code
    return code, json.loads(raw), (time.perf_counter() - started) * 1000


def main() -> int:
    # Each task costs several minutes and a slice of the free-tier quota, so
    # allow re-running just one while iterating.
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    tasks = [t for t in TASKS if not only or only in t[0]]
    if not tasks:
        print(f"no task matches {only!r}; known: {[t[0] for t in TASKS]}")
        return 2

    results = []
    failures = 0

    for name, message in tasks:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        code, data, wall_ms = post("/chat", {"message": message})

        # `Trace.to_dict()` is `asdict(trace)`, so the trace *is* the response
        # body -- there is no nested "trace" key.
        steps = data.get("steps") or []
        tools = data.get("tools_used") or []
        citations = data.get("citations") or []
        answer = data.get("answer") or ""

        ok = (
            code == 200
            and len(steps) >= 3
            and len(set(tools)) >= 2
            and len(citations) >= 1
            and bool(answer.strip())
            and not data.get("error")
            # A run that exhausts the step budget returns an apology, not an
            # answer. It would otherwise satisfy every count above, so assert
            # on it explicitly.
            and not data.get("truncated")
        )
        failures += 0 if ok else 1

        print(f"http            : {code}")
        print(f"steps           : {len(steps)}")
        print(f"distinct tools  : {len(set(tools))}  {sorted(set(tools))}")
        print(f"citations       : {len(citations)}")
        print(f"wall clock      : {wall_ms / 1000:.1f}s")
        print(f"service / thrtl : {data.get('total_service_ms')} / {data.get('throttle_ms')}")
        print(f"provider/model  : {data.get('provider')}/{data.get('model')}")
        print(f"fell_back       : {data.get('fell_back')}")
        print(f"peak ctx tokens : {data.get('peak_context_tokens')}")
        print(f"elided results  : {data.get('elided_results')}")
        print(f"truncated       : {data.get('truncated')}")
        print(f"error           : {data.get('error') or 'none'}")
        print(f"VERDICT         : {'PASS' if ok else 'FAIL'}")
        print(f"\n--- answer (first 600 chars) ---\n{answer[:600]}")

        results.append(
            {
                "task": name,
                "prompt": message,
                "http_status": code,
                "steps": len(steps),
                "tools_used": tools,
                "distinct_tools": sorted(set(tools)),
                "citations": citations,
                "wall_clock_ms": round(wall_ms),
                "total_service_ms": data.get("total_service_ms"),
                "throttle_ms": data.get("throttle_ms"),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "fell_back": data.get("fell_back"),
                "peak_context_tokens": data.get("peak_context_tokens"),
                "elided_results": data.get("elided_results"),
                "truncated": data.get("truncated"),
                "error": data.get("error"),
                # Per-step detail makes a stalled run diagnosable: a repeated
                # tool call is the signature of context compaction discarding a
                # result the agent still needed.
                "step_detail": [
                    {
                        "index": s.get("index"),
                        "kind": s.get("kind"),
                        "tools": [
                            c.get("name") for c in (s.get("tool_calls") or [])
                        ],
                        "context_tokens": s.get("context_tokens"),
                        "elided_results": s.get("elided_results"),
                    }
                    for s in steps
                ],
                "passed": ok,
                "answer": answer,
            }
        )

    out = pathlib.Path("evidence")
    out.mkdir(exist_ok=True)
    # Merge into any existing evidence so re-running a single task does not
    # discard the other task's result.
    path = out / "deployed-tasks.json"
    existing: dict[str, dict] = {}
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        existing = {t["task"]: t for t in prior.get("tasks", [])}
    for r in results:
        existing[r["task"]] = r

    path.write_text(
        json.dumps(
            {
                "base_url": BASE,
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tasks": [existing[t[0]] for t in TASKS if t[0] in existing],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {path}")
    print(f"\n{len(tasks) - failures}/{len(tasks)} deployed tasks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
