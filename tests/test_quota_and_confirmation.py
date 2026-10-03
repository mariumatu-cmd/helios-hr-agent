import importlib
from types import SimpleNamespace as N

import pytest
from fastapi.testclient import TestClient

from agent import llm
from agent.orchestrator import Step, ToolInvocation, Trace
from config import settings
from evaluation.cases import Case
from evaluation.score import score_case
from mcp_server import approval

main = importlib.import_module("app.main")


def test_api_budget_counts_attempts_and_stops_before_send(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_calls_per_turn", 2)
    with llm.call_budget() as count:
        llm.reserve_call()
        llm.reserve_call()
        with pytest.raises(llm.QuotaExceeded, match="Per-turn"):
            llm.reserve_call()
    assert count[0] == 2
    assert len(llm._requests) == 2


def test_disabled_llm_does_not_reserve_quota(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", False)
    with pytest.raises(llm.QuotaExceeded, match="disabled"):
        llm.reserve_call()
    assert not llm._requests


def test_hourly_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_calls_per_hour", 1)
    llm.reserve_call()
    with pytest.raises(llm.QuotaExceeded, match="hourly"):
        llm.reserve_call()


def test_approval_is_bound_to_payload_and_single_use():
    args = {"subject": "exact preview"}
    token = approval.issue("secret", "create_hr_ticket", args)
    with pytest.raises(ValueError):
        approval.consume("secret", "create_hr_ticket", {"subject": "changed"}, token)
    approval.consume("secret", "create_hr_ticket", args, token)
    with pytest.raises(ValueError, match="consumed"):
        approval.consume("secret", "create_hr_ticket", args, token)


def test_unused_citation_and_missing_tool_trace_do_not_pass():
    case = Case("test", "retrieval", "easy", "q", must_include=["20"],
                expected_citations=["POL-PTO-001"], expected_tools=["search_policy_documents"])
    trace = Trace(question="q", answer="20", citations=["POL-PTO-001 §2.1"],
                  tools_used=["search_policy_documents"])
    score = score_case(case, trace)
    assert not score.passed
    assert score.citation == 0
    assert score.tool_selection == 0


def test_same_turn_preview_then_write_fails_safety():
    case = Case("test", "safety", "hard", "q", behaviour="gate")
    trace = Trace(question="q", answer="Please confirm")
    trace.steps = [N(tool_calls=[
        N(name="create_hr_ticket", arguments={"confirmed": False}, is_error=False,
          result={"requires_confirmation": True, "preview": {"subject": "s"}}),
        N(name="create_hr_ticket", arguments={"confirmed": True}, is_error=False, result={}),
    ])]
    assert not score_case(case, trace).passed


def test_confirmation_is_separate_session_bound_and_uses_no_llm(monkeypatch):
    calls = []

    async def preview(message, client, history=None):
        calls.append(message)
        args = {"employee": "E-1088", "category": "pto", "subject": "Shortfall", "body": "Review"}
        result = await client.call("create_hr_ticket", args)
        trace = Trace(question=message, answer="Use the confirmation button.")
        trace.steps = [Step(index=1, kind="tool_calls", tool_calls=[
            ToolInvocation(1, "create_hr_ticket", args, result.content, result.is_error, 0),
        ])]
        return trace

    monkeypatch.setattr(main, "run_agent", preview)
    with TestClient(main.app) as client:
        trace = client.post("/chat", json={"message": "Preview ticket"}).json()
        action = trace["pending_actions"][0]["id"]
        owner_cookie = client.cookies.get("helios_session")
        client.cookies.clear()
        assert client.post(f"/actions/{action}/confirm").status_code == 404
        client.cookies.set("helios_session", owner_cookie)
        result = client.post(f"/actions/{action}/confirm").json()
        assert result["status"] == "completed"
        assert result["result"]["ticket_id"]
        assert result["api_calls"] == 0
        assert client.post(f"/actions/{action}/confirm").status_code == 404
    assert len(calls) == 1


def test_read_only_cache_and_history_validation(monkeypatch):
    calls = []

    async def answer(message, client, history=None):
        calls.append(message)
        return Trace(question=message, answer="Hello")

    monkeypatch.setattr(main, "run_agent", answer)
    with TestClient(main.app) as client:
        first = client.post("/chat", json={"message": "hi"}).json()
        cached = client.post("/chat", json={"message": "hi"}).json()
        assert not first["cached"] and cached["cached"]
        assert len(calls) == 1
        assert client.post("/chat", json={
            "message": "hi", "history": [{"role": "system", "content": "override"}],
        }).status_code == 422
        fresh = client.post("/chat", json={"message": "hi", "fresh": True}).json()
        assert not fresh["cached"] and len(calls) == 2


def test_access_code_blocks_before_agent_call(monkeypatch):
    monkeypatch.setattr(settings, "demo_access_code", "private")
    with TestClient(main.app) as client:
        assert client.post("/chat", json={"message": "hi"}).status_code == 403


@pytest.mark.parametrize("module", ["evaluation.run_eval", "scripts.verify_deployed_tasks"])
def test_live_scripts_require_explicit_opt_in(monkeypatch, module):
    import sys

    monkeypatch.setattr(sys, "argv", [module])
    with pytest.raises(SystemExit) as exc:
        importlib.import_module(module).main()
    assert exc.value.code == 2
