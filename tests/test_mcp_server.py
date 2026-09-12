"""MCP integration over a real stdio transport.

These spawn `mcp_server/hr_server.py` as a separate process and speak the
protocol to it. Testing the tool functions by importing them would pass even if
the server failed to expose them over MCP, which is the failure this suite
exists to catch.

Each test opens and closes its own session through `mcp_session()` rather than
sharing one via a fixture. That is deliberate: the MCP client's transport is
built on anyio cancel scopes, which must be entered and exited in the *same*
task. pytest-asyncio runs fixture setup and teardown in different tasks, so a
session-yielding fixture tears down with "Attempted to exit cancel scope in a
different task" -- or, with a mismatched loop scope, hangs silently. An explicit
`async with` inside the test body keeps entry and exit in one task. The cost is
one server process per test, which is a few seconds well spent for tests that
genuinely exercise the protocol.
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import settings

pytestmark = pytest.mark.asyncio

EXPECTED_TOOLS = {
    "search_policy_documents", "get_policy_section", "list_policy_documents",
    "lookup_employee_profile", "list_employees", "check_pto_balance",
    "lookup_benefits_status", "check_international_work_usage",
    "check_policy_compliance", "create_hr_ticket", "draft_hr_email", "list_hr_tickets",
}


@asynccontextmanager
async def mcp_session():
    params = StdioServerParameters(
        command=sys.executable, args=[str(settings.mcp_server_script)], env=None
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            yield client


def payload(result):
    return json.loads(result.content[0].text)


async def test_handshake_identifies_the_server():
    async with mcp_session() as session:
        listed = await session.list_tools()
    assert listed.tools


async def test_all_tools_are_advertised():
    async with mcp_session() as session:
        names = {t.name for t in (await session.list_tools()).tools}
    assert EXPECTED_TOOLS <= names, f"missing: {EXPECTED_TOOLS - names}"


async def test_every_tool_is_self_describing():
    """The agent discovers tools at runtime, so a tool without a description or
    an input schema is effectively invisible to the model even though it exists."""
    async with mcp_session() as session:
        tools = (await session.list_tools()).tools
    for tool in tools:
        assert tool.description, f"{tool.name} has no description"
        assert tool.input_schema.get("type") == "object", f"{tool.name} has no object schema"


async def test_write_tools_are_annotated_as_not_read_only():
    async with mcp_session() as session:
        by_name = {t.name: t for t in (await session.list_tools()).tools}
    for name in ("create_hr_ticket", "draft_hr_email"):
        assert by_name[name].annotations.read_only_hint is False
    assert by_name["search_policy_documents"].annotations.read_only_hint is True


async def test_search_returns_grounded_citations():
    async with mcp_session() as session:
        result = payload(await session.call_tool(
            "search_policy_documents", {"query": "notice required for three days off", "k": 3}
        ))
    assert result["grounded"]
    assert any(h["doc_id"] == "POL-PTO-001" for h in result["hits"])
    assert all(h["citation"] for h in result["hits"])


async def test_out_of_corpus_search_instructs_refusal():
    async with mcp_session() as session:
        result = payload(await session.call_tool(
            "search_policy_documents", {"query": "what is the capital of Portugal", "k": 3}
        ))
    assert result["grounded"] is False
    assert "instruction" in result


async def test_compliance_tool_returns_findings_and_citations():
    async with mcp_session() as session:
        result = payload(await session.call_tool("check_policy_compliance", {
            "request_type": "international_remote_work",
            "employee": "Maya Rodriguez",
            "country": "Portugal",
            "start_date": "2026-10-05",
            "end_date": "2026-11-15",
        }))
    assert result["compliant"] is False
    assert result["usage"]["days_used"] == 12
    assert result["citations"]


async def test_unknown_employee_returns_a_recoverable_error():
    """Errors come back as values so the agent can correct itself, rather than
    the protocol raising and the whole run aborting."""
    async with mcp_session() as session:
        result = payload(await session.call_tool(
            "lookup_employee_profile", {"employee": "Nobody McNotreal"}
        ))
    assert "error" in result and result["hint"]


async def test_missing_arguments_return_a_structured_error():
    async with mcp_session() as session:
        result = payload(await session.call_tool(
            "check_policy_compliance", {"request_type": "pto", "employee": "E-1041"}
        ))
    assert "error" in result


async def test_write_requires_confirmation():
    async with mcp_session() as session:
        result = payload(await session.call_tool("create_hr_ticket", {
            "employee": "E-1041", "category": "pto", "subject": "s", "body": "b",
        }))
    assert result["requires_confirmation"] is True
    assert "preview" in result


async def test_confirmed_write_is_in_memory_only():
    async with mcp_session() as session:
        result = payload(await session.call_tool("create_hr_ticket", {
            "employee": "E-1041", "category": "pto", "subject": "s", "body": "b",
            "confirmed": True,
        }))
    assert result["persisted"] is False
    assert result["ticket_id"].startswith("HR-")


async def test_section_fetch_returns_verbatim_text():
    async with mcp_session() as session:
        result = payload(await session.call_tool(
            "get_policy_section", {"doc_id": "POL-INTL-001", "section": "2"}
        ))
    assert result["passages"]
    assert all(p["doc_id"] == "POL-INTL-001" for p in result["passages"])
