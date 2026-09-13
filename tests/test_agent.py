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


async def test_trace_reports_context_pressure(scripted):
    """The trace has to distinguish a run that fitted from one that only fitted
    after losing detail, or the evaluation silently compares the two."""
    scripted([
        tool_step([("search_policy_documents", {"query": "x"})]),
        final_step("done"),
    ])
    trace = await orchestrator.run_agent("q", FakeClient())
    assert trace.peak_context_tokens > 0
    assert trace.elided_results == 0
    assert all(step.context_tokens > 0 for step in trace.steps)


# --- recovering from a provider-rejected tool call --------------------------
#
# Groq validates tool-call arguments server-side and returns 400
# `tool_use_failed` when the model's own output does not match the schema it
# was given. Observed twice in one evaluation run, both on the fallback models
# the free tier pushes work onto under load: once as `"confirmed": "False"`
# written as a string, once as a call cut off mid-argument by the token cap.
#
# Before these tests the exception escaped and killed the run, so two safety
# cases scored 0.00 for a reason that had nothing to do with safety.
def _malformed(message: str = "expected boolean, but got string") -> Exception:
    return orchestrator.llm.MalformedToolCall(message, failed_generation="<tool_call>...")


async def test_a_rejected_tool_call_is_corrected_not_fatal(monkeypatch):
    calls: list[list[dict]] = []

    def chat(messages, tools=None, **kwargs):
        calls.append(list(messages))
        if len(calls) == 1:
            raise _malformed()
        return final_step("Recovered.")

    monkeypatch.setattr(orchestrator.llm, "chat", chat)
    trace = await orchestrator.run_agent("file a ticket", FakeClient())

    assert trace.answer == "Recovered."
    assert not trace.error
    assert trace.malformed_tool_calls == 1


async def test_the_correction_tells_the_model_what_was_wrong(monkeypatch):
    """A bare 'that was invalid' is not actionable.

    The two observed causes need opposite corrections -- retype the argument,
    or shorten it -- so the provider's own message is passed through.
    """
    calls: list[list[dict]] = []

    def chat(messages, tools=None, **kwargs):
        calls.append(list(messages))
        if len(calls) == 1:
            raise _malformed()
        return final_step("ok")

    monkeypatch.setattr(orchestrator.llm, "chat", chat)
    await orchestrator.run_agent("q", FakeClient())

    correction = calls[1][-1]
    assert correction["role"] == "user"
    assert "expected boolean, but got string" in correction["content"]
    assert "true and false" in correction["content"]


async def test_a_persistently_malformed_model_stops_rather_than_loops(monkeypatch):
    """Correction is bounded: every attempt costs a request against a daily quota."""
    attempts = {"n": 0}

    def chat(messages, tools=None, **kwargs):
        attempts["n"] += 1
        raise _malformed()

    monkeypatch.setattr(orchestrator.llm, "chat", chat)
    trace = await orchestrator.run_agent("q", FakeClient())

    assert attempts["n"] == orchestrator.MAX_MALFORMED_TOOL_CALLS + 1
    assert "malformed" in trace.error.lower()
    assert trace.steps[-1].kind == "error"
    assert not trace.truncated


async def test_a_corrected_run_is_visible_in_the_trace(monkeypatch):
    """A run that needed correcting is not the same as a clean one."""
    calls = {"n": 0}

    def chat(messages, tools=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _malformed()
        return final_step("done")

    monkeypatch.setattr(orchestrator.llm, "chat", chat)
    trace = await orchestrator.run_agent("q", FakeClient())

    assert json.loads(json.dumps(trace.to_dict()))["malformed_tool_calls"] == 1
    assert any(s.kind == "error" and "rejected" in s.summary for s in trace.steps)


# --- the trace is operational, not chain-of-thought -------------------------
# The brief asks for a visible trace of selected tools, arguments, outputs and
# sources, and explicitly not for chain-of-thought. The step used to record the
# model's free-text preamble to its own tool call, which is exactly that. These
# pin the replacement: the step describes the decision, and the narration has
# no path into the trace at all.

@pytest.mark.asyncio
async def test_model_narration_never_reaches_the_trace(scripted):
    narration = "Let me think. First I should figure out whether she is eligible, then check dates."
    scripted([
        tool_step([("search_policy_documents", {"query": "pto"})], text=narration),
        final_step("15 days (POL-PTO-001 SS2.1)."),
    ])
    trace = await orchestrator.run_agent("How much PTO?", FakeClient())

    serialised = json.dumps(trace.to_dict())
    assert narration not in serialised
    assert "Let me think" not in serialised


@pytest.mark.asyncio
async def test_step_summary_names_the_tools_it_selected(scripted):
    scripted([
        tool_step([("search_policy_documents", {"query": "pto"})], text="ignored narration"),
        final_step("done"),
    ])
    trace = await orchestrator.run_agent("q", FakeClient())

    step = next(s for s in trace.steps if s.kind == "tool_calls")
    assert step.summary == "selected 1 tool: search_policy_documents"


@pytest.mark.asyncio
async def test_parallel_selection_is_described_as_parallel(scripted):
    scripted([
        tool_step([
            ("search_policy_documents", {"query": "pto"}),
            ("check_policy_compliance", {"employee_id": "E-1042"}),
        ]),
        final_step("done"),
    ])
    trace = await orchestrator.run_agent("q", FakeClient())

    step = next(s for s in trace.steps if s.kind == "tool_calls")
    assert step.summary == (
        "selected 2 tools in parallel: search_policy_documents, check_policy_compliance"
    )


@pytest.mark.asyncio
async def test_final_step_reports_how_many_sources_the_answer_rests_on(scripted):
    """The answer's basis is part of the operational trace, per the brief."""
    scripted([
        tool_step([("search_policy_documents", {"query": "pto"})]),
        final_step("15 days."),
    ])
    client = FakeClient(results={
        "search_policy_documents": {
            "results": [
                {"citation": "POL-PTO-001 SS2.1"},
                {"citation": "POL-HOL-001 SS3"},
            ]
        }
    })
    trace = await orchestrator.run_agent("q", client)

    step = next(s for s in trace.steps if s.kind == "final")
    assert step.summary == "answer synthesised from 2 cited sources"
