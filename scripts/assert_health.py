"""Assert a /health payload describes a serviceable instance.

    python scripts/assert_health.py health.json [--require-llm]

Used against the deployed container in CI and against the live Render service
after a deploy. It is deliberately separate from `scripts/healthcheck.py`:
that one boots the app in-process, this one judges a payload fetched over the
network from an instance it did not start.

The LLM provider is optional by default, because CI holds no API key -- an
unconfigured provider is a configuration state, not a build failure. Pass
`--require-llm` when checking a real deployment, where it *is* a failure.
"""
from __future__ import annotations

import json
import pathlib
import sys

EXPECTED_TOOLS = 12


def assess(health: dict, require_llm: bool) -> list[str]:
    problems: list[str] = []

    mcp = health.get("mcp", {})
    if not mcp.get("connected"):
        problems.append(f"MCP not connected: {mcp.get('error')}")
    elif mcp.get("tool_count") != EXPECTED_TOOLS:
        problems.append(f"expected {EXPECTED_TOOLS} MCP tools, found {mcp.get('tool_count')}")

    index = health.get("rag_index", {})
    if not index.get("ok"):
        problems.append(f"RAG index not loadable: {index}")
    elif not index.get("chunks"):
        problems.append(f"RAG index reports no chunks: {index}")

    llm = health.get("llm", {})
    if require_llm and not llm.get("providers_configured"):
        problems.append("no LLM provider configured")

    return problems


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    require_llm = "--require-llm" in argv[1:]
    if not args:
        print(__doc__)
        return 2

    health = json.loads(pathlib.Path(args[0]).read_text(encoding="utf-8"))
    print(json.dumps(health, indent=2))

    problems = assess(health, require_llm)
    if problems:
        print("\nHEALTH ASSERTION FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    if not health.get("llm", {}).get("providers_configured"):
        print("\nNOTE: no LLM provider configured; expected in CI, not in production.")
    print("\nHEALTH ASSERTION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
