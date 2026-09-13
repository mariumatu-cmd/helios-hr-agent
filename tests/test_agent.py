"""Agent orchestration loop, driven by a scripted model.

The loop is tested without a real LLM on purpose. A live model would make these
tests slow, non-deterministic, dependent on a secret, and unable to reproduce
the failure paths that matter most -- a malformed tool call, a tool error, a run
that never terminates. The scripted model lets each of those be asserted
exactly, and means CI needs no API key.

Answer *quality* is not tested here; that is what `evaluation/` measures against
the real provider.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent import orchestrator
from agent.mcp_client import MCPToolClient, ToolCallResult

pytestmark = pytest.mark.asyncio


# --- scripted model --------------------------------------------------------
@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeResponse:
    provider: str = "fake"
    model: str = "scripted"
    message: Any = None
    tool_calls: list = field(default_factory=list)
    text: str | None = None
    latency_ms: float = 1.0
    fell_back: bool = False
    service_ms: float = 1.0
    throttle_ms: float = 0.0
    attempts: int = 1


def tool_step(calls: list[tuple[str, dict]], text: str = "") -> FakeResponse:
    return FakeResponse(
        tool_calls=[
            FakeToolCall(id=f"call_{i}", function=FakeFunction(name, json.dumps(args)))
            for i, (name, args) in enumerate(calls)
        ],
        text=text,
    )


def final_step(text: str) -> FakeResponse:
    return FakeResponse(text=text)


@pytest.fixture
def scripted(monkeypatch):
    """Queue responses; each llm.chat call pops the next one."""
    def install(responses: list[FakeResponse]):
        queue = list(responses)
        seen: list[list[dict]] = []

        def fake_chat(messages, tools=None, **kwargs):
            seen.append(list(messages))
            if not queue:
                return final_step("ran out of scripted responses")
            return queue.pop(0)

        monkeypatch.setattr(orchestrator.llm, "chat", fake_chat)
        return seen

    return install


# --- fake MCP client -------------------------------------------------------
class FakeClient(MCPToolClient):
    """Stands in for a connected MCP session, recording what was called."""

    def __init__(self, results: dict[str, Any] | None = None):
        super().__init__()
        self.session = object()
        self.calls: list[tuple[str, dict]] = []
        self.results = results or {}
        self.tools = [
            type("T", (), {
                "name": name,
                "description": f"{name} description",
                "input_schema": {"type": "object", "properties": {}},
                "annotations": None,
            })()
            for name in ("search_policy_documents", "check_policy_compliance", "create_hr_ticket")
        ]

    async def call(self, name: str, arguments: dict) -> ToolCallResult:
        self.calls.append((name, arguments))
        content = self.results.get(name, {"ok": True})
        return ToolCallResult(
            name=name,
            arguments=arguments,
            content=content,
            is_error=isinstance(content, dict) and "error" in content,
            latency_ms=1.0,
        )


# --- tests -----------------------------------------------------------------
async def test_single_step_answer_needs_no_tools(scripted):
    scripted([final_step("Hello.")])
    trace = await orchestrator.run_agent("hi", FakeClient())
    assert trace.answer == "Hello."
    assert trace.tools_used == []
    assert trace.steps[-1].kind == "final"


async def test_tool_call_then_answer(scripted):
    scripted([
        tool_step([("search_policy_documents", {"query": "pto notice"})]),
        final_step("You need 10 business days (POL-PTO-001 §3.1)."),
    ])
    client = FakeClient({"search_policy_documents": {
        "grounded": True,
        "hits": [{"citation": "POL-PTO-001 §3.1 Notice Requirements"}],
    }})

    trace = await orchestrator.run_agent("how much notice?", client)

    assert client.calls == [("search_policy_documents", {"query": "pto notice"})]
    assert trace.tools_used == ["search_policy_documents"]
    assert "POL-PTO-001 §3.1 Notice Requirements" in trace.citations
    assert trace.grounded is True


async def test_parallel_tool_calls_in_one_step(scripted):
    scripted([
        tool_step([
            ("search_policy_documents", {"query": "a"}),
            ("check_policy_compliance", {"request_type": "pto", "employee": "E-1041"}),
        ]),
        final_step("done"),
    ])
    client = FakeClient()
    trace = await orchestrator.run_agent("q", client)

    assert len(client.calls) == 2
    assert len(trace.steps[0].tool_calls) == 2


async def test_tool_error_does_not_abort_the_run(scripted):
    """The model must get the error back as an observation and be able to
    recover, rather than the whole request failing."""
    scripted([
        tool_step([("check_policy_compliance", {"employee": "Nobody"})]),
        tool_step([("check_policy_compliance", {"employee": "E-1041"})]),
        final_step("Recovered."),
    ])
    client = FakeClient({"check_policy_compliance": {"error": "no employee", "hint": "use an id"}})

    trace = await orchestrator.run_agent("q", client)

    assert trace.answer == "Recovered."
    assert trace.steps[0].tool_calls[0].is_error is True
    assert len(client.calls) == 2


async def test_unknown_tool_is_reported_not_raised():
    client = MCPToolClient()
    client.session = object()
    client.tools = []
    result = await client.call("nonexistent_tool", {})
    assert result.is_error
    assert "unknown tool" in result.content["error"]


async def test_malformed_tool_arguments_are_handled(scripted):
    bad = FakeResponse(tool_calls=[
        FakeToolCall(id="c0", function=FakeFunction("search_policy_documents", "{not json"))
    ])
    scripted([bad, final_step("recovered")])

    trace = await orchestrator.run_agent("q", FakeClient())

    assert trace.steps[0].tool_calls[0].is_error is True
    assert trace.answer == "recovered"


async def test_step_limit_produces_an_honest_answer(scripted):
    """A model that loops must not produce a confident-sounding final answer."""
    scripted([tool_step([("search_policy_documents", {"query": "x"})]) for _ in range(10)])

    trace = await orchestrator.run_agent("q", FakeClient(), max_steps=3)

    assert trace.truncated
    assert "could not finish" in trace.answer
    assert len(trace.steps) == 4  # 3 tool steps + the truncation marker


async def test_ungrounded_search_is_recorded(scripted):
    scripted([
        tool_step([("search_policy_documents", {"query": "capital of Portugal"})]),
        final_step("That is not covered by HR policy."),
    ])
    client = FakeClient({"search_policy_documents": {"grounded": False, "hits": []}})

    trace = await orchestrator.run_agent("q", client)

    assert trace.grounded is False


async def test_missing_provider_degrades_gracefully(monkeypatch):
    def no_provider(*args, **kwargs):
        raise orchestrator.llm.NoProviderConfigured("no key")

    monkeypatch.setattr(orchestrator.llm, "chat", no_provider)
    trace = await orchestrator.run_agent("q", FakeClient())

    assert trace.error == "no key"
    assert "No language model is configured" in trace.answer


async def test_provider_exception_degrades_gracefully(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(orchestrator.llm, "chat", boom)
    trace = await orchestrator.run_agent("q", FakeClient())

    assert "upstream 503" in trace.error
    assert trace.answer
    assert trace.steps[-1].kind == "error"


async def test_provider_fallback_is_surfaced_in_the_trace(scripted):
    scripted([FakeResponse(text="ok", fell_back=True, provider="gemini")])
    trace = await orchestrator.run_agent("q", FakeClient())
    assert trace.fell_back is True
    assert trace.provider == "gemini"


async def test_history_is_passed_to_the_model(scripted):
    seen = scripted([final_step("ok")])
    await orchestrator.run_agent(
        "and then?", FakeClient(),
        history=[{"role": "user", "content": "earlier"},
                 {"role": "assistant", "content": "reply"}],
    )
    roles = [m["role"] for m in seen[0]]
    assert roles == ["system", "user", "assistant", "user"]


async def test_system_prompt_pins_the_as_of_date(scripted):
    seen = scripted([final_step("ok")])
    await orchestrator.run_agent("q", FakeClient())
    assert "2026-09-12" in seen[0][0]["content"]


async def test_tool_schemas_come_from_discovery():
    """The tool list handed to the model is derived from the MCP catalogue, not
    hardcoded -- that is what makes this a real MCP integration."""
    client = FakeClient()
    specs = client.openai_tools()
    assert {s["function"]["name"] for s in specs} == {t.name for t in client.tools}
    assert all(s["function"]["description"] for s in specs)


async def test_trace_serialises_to_json(scripted):
    scripted([
        tool_step([("search_policy_documents", {"query": "x"})]),
        final_step("done"),
    ])
    trace = await orchestrator.run_agent("q", FakeClient())
    assert json.loads(json.dumps(trace.to_dict()))["answer"] == "done"
