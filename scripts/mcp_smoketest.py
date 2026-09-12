"""End-to-end MCP smoke test over a real stdio transport.

    python scripts/mcp_smoketest.py

This spawns `mcp_server/hr_server.py` as a *separate process* and talks to it
with the official MCP client, so it proves the tools are genuinely exposed over
the protocol -- handshake, tool discovery, typed arguments, error payloads --
rather than merely imported and called as Python functions. CI runs this.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from config import settings  # noqa: E402

EXPECTED_TOOLS = {
    "search_policy_documents",
    "get_policy_section",
    "list_policy_documents",
    "lookup_employee_profile",
    "list_employees",
    "check_pto_balance",
    "lookup_benefits_status",
    "check_international_work_usage",
    "check_policy_compliance",
    "create_hr_ticket",
    "draft_hr_email",
    "list_hr_tickets",
}

failures: list[str] = []


def expect(condition: bool, message: str) -> None:
    print(("  PASS  " if condition else "  FAIL  ") + message)
    if not condition:
        failures.append(message)


def payload(result) -> dict:
    """Unwrap the first text content block as JSON."""
    return json.loads(result.content[0].text)


async def main() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(settings.mcp_server_script)],
        env=None,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"connected to {info.server_info.name} v{info.server_info.version}\n")

            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            print(f"discovered {len(names)} tools: {', '.join(sorted(names))}\n")
            expect(EXPECTED_TOOLS <= names, f"all {len(EXPECTED_TOOLS)} expected tools advertised")
            expect(
                all(t.description and t.input_schema for t in listed.tools),
                "every tool advertises a description and an input schema",
            )

            read_only = {
                t.name for t in listed.tools
                if t.annotations and t.annotations.read_only_hint
            }
            expect(
                {"create_hr_ticket", "draft_hr_email"}.isdisjoint(read_only),
                "write tools are not annotated read-only",
            )
            expect(
                "search_policy_documents" in read_only,
                "search tool is annotated read-only",
            )

            search = payload(await session.call_tool(
                "search_policy_documents",
                {"query": "how much notice do I need for three days off", "k": 3},
            ))
            expect(search["grounded"] is True, "policy search is grounded")
            expect(
                any(h["doc_id"] == "POL-PTO-001" for h in search["hits"]),
                f"top hits include the PTO policy (got {[h['doc_id'] for h in search['hits']]})",
            )
            print(f"         top citation: {search['hits'][0]['citation']}")

            refusal = payload(await session.call_tool(
                "search_policy_documents",
                {"query": "what is the capital of Portugal", "k": 3},
            ))
            expect(refusal["grounded"] is False, "out-of-corpus query is flagged ungrounded")
            expect("instruction" in refusal, "ungrounded result carries a refusal instruction")

            compliance = payload(await session.call_tool(
                "check_policy_compliance",
                {
                    "request_type": "international_remote_work",
                    "employee": "Maya Rodriguez",
                    "country": "Portugal",
                    "start_date": "2026-10-05",
                    "end_date": "2026-11-15",
                },
            ))
            expect(compliance["compliant"] is False, "42-day Portugal request is non-compliant")
            expect(
                compliance["usage"]["days_used"] == 12,
                f"rolling window counts 12 days (got {compliance['usage']['days_used']})",
            )

            preview = payload(await session.call_tool(
                "create_hr_ticket",
                {
                    "employee": "E-1041",
                    "category": "international_remote_work",
                    "subject": "Portugal request",
                    "body": "Six weeks from Lisbon.",
                },
            ))
            expect(
                preview.get("requires_confirmation") is True,
                "unconfirmed write returns a preview instead of acting",
            )

            created = payload(await session.call_tool(
                "create_hr_ticket",
                {
                    "employee": "E-1041",
                    "category": "international_remote_work",
                    "subject": "Portugal request",
                    "body": "Six weeks from Lisbon.",
                    "confirmed": True,
                },
            ))
            expect(created.get("persisted") is False, "confirmed write is in-memory only")
            expect(
                created.get("assigned_team") == "Global Mobility",
                "ticket routes to Global Mobility",
            )

            bad = payload(await session.call_tool(
                "lookup_employee_profile", {"employee": "Nobody McNotreal"}
            ))
            expect("error" in bad and bad.get("hint"), "unknown employee returns a recoverable error")

            bad_args = payload(await session.call_tool(
                "check_policy_compliance", {"request_type": "pto", "employee": "E-1041"}
            ))
            expect("error" in bad_args, "missing required arguments return a structured error")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
