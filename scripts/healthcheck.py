"""Boot the app in-process and assert every dependency reports healthy.

    python scripts/healthcheck.py

CI runs this after the unit tests: it is the difference between "the modules
import" and "the service actually starts, opens an MCP session, loads the index
and can answer /health". The LLM provider is exempt, because CI has no API key
and a missing key is a configuration state, not a build failure.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def main() -> int:
    with TestClient(app) as client:
        health = client.get("/health").json()
        print(json.dumps(health, indent=2))

        problems: list[str] = []
        if not health["mcp"]["connected"]:
            problems.append(f"MCP not connected: {health['mcp'].get('error')}")
        if health["mcp"]["tool_count"] < 12:
            problems.append(f"expected 12 MCP tools, found {health['mcp']['tool_count']}")
        if not health["rag_index"]["ok"]:
            problems.append(f"RAG index not loadable: {health['rag_index']}")
        if client.get("/").status_code != 200:
            problems.append("index page did not render")
        if client.get("/tools").status_code != 200:
            problems.append("/tools did not respond")

        if not health["llm"]["providers_configured"]:
            print("\nNOTE: no LLM provider configured; that is expected in CI.")

        if problems:
            print("\nHEALTHCHECK FAILED")
            for problem in problems:
                print(f"  - {problem}")
            return 1

    print("\nHEALTHCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
