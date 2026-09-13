"""The agent's context budget.

Groq's free tier meters tokens-per-minute, and that bucket covers the whole
request -- prompt and completion together. A prompt larger than the bucket can
therefore never succeed: retrying waits for a refill that is already big
enough, and rotating to another model relocates the same failure. Because every
agent step resends the full tool manifest plus every prior tool result, an
unbounded multi-step run grows past the limit and stalls permanently. That is
exactly what the deployed service did before this budget existed.

These tests pin the only thing that actually fixes it: bounding the prompt.
"""
from __future__ import annotations

import json

from agent import orchestrator


def _tool_message(name: str, size: int, citation: str) -> dict:
    body = json.dumps({
        "hits": [{"text": "x" * size, "citation": citation}],
        "grounded": True,
    })
    return {"role": "tool", "tool_call_id": "c1", "name": name, "content": body}


def test_a_small_request_is_left_untouched():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    fitted, tokens, elided = orchestrator.fit_context(messages, None, budget=5000)
    assert fitted is messages
    assert elided == 0
    assert tokens < 5000


def test_oldest_results_are_compacted_first():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        _tool_message("search_policy_documents", 8000, "POL-PTO-001 S2"),
        _tool_message("search_policy_documents", 8000, "POL-REM-002 S4"),
        _tool_message("check_policy_compliance", 800, "POL-REM-002 S5"),
    ]
    before = sum(orchestrator.estimate_tokens(m) for m in messages)
    fitted, tokens, elided = orchestrator.fit_context(messages, None, budget=1000)

    assert before > 1000, "fixture must actually exceed the budget"
    assert tokens <= 1000
    assert elided >= 2
    # The newest result is what the current step is reasoning about, so it
    # survives whenever compacting the older ones is enough on its own.
    assert fitted[-1]["content"] == messages[-1]["content"]


def test_the_newest_result_is_compacted_when_nothing_else_is_enough():
    """A single search can blow the budget by itself. Protecting the newest
    result would then guarantee a request that can never succeed, so the
    budget wins over recency as a last resort."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        _tool_message("search_policy_documents", 20_000, "POL-PTO-001 S2"),
    ]
    fitted, tokens, elided = orchestrator.fit_context(messages, None, budget=500)
    assert tokens <= 500
    assert elided == 1
    assert len(fitted[-1]["content"]) < len(messages[-1]["content"])


def test_citations_survive_compaction():
    """A long run must not become less grounded than a short one: compaction
    drops passage text but keeps the citation strings the answer has to quote."""
    messages = [
        {"role": "system", "content": "sys"},
        _tool_message("search_policy_documents", 8000, "POL-PTO-001 S2"),
        {"role": "user", "content": "q"},
    ]
    fitted, _, elided = orchestrator.fit_context(messages, None, budget=300)
    assert elided == 1
    assert "POL-PTO-001 S2" in fitted[1]["content"]


def test_the_callers_transcript_is_never_rewritten():
    """The message list is reused across steps, so compaction must only shrink
    the copy that gets sent -- not destroy the real transcript."""
    original = _tool_message("search_policy_documents", 8000, "POL-PTO-001 S2")
    messages = [{"role": "system", "content": "sys"}, original, {"role": "user", "content": "q"}]
    snapshot = original["content"]
    orchestrator.fit_context(messages, None, budget=300)
    assert original["content"] == snapshot


def test_only_tool_results_are_touched():
    """System, user and assistant turns are load-bearing: the first two define
    the task, and a provider rejects a tool result whose assistant turn was
    altered or dropped."""
    messages = [
        {"role": "system", "content": "s" * 8000},
        {"role": "assistant", "content": "a" * 8000, "tool_calls": [{"id": "c1"}]},
        _tool_message("search_policy_documents", 8000, "POL-PTO-001 S2"),
        {"role": "user", "content": "q"},
    ]
    fitted, _, _ = orchestrator.fit_context(messages, None, budget=100)
    assert fitted[0]["content"] == messages[0]["content"]
    assert fitted[1]["content"] == messages[1]["content"]
    assert fitted[1]["tool_calls"] == messages[1]["tool_calls"]


def test_the_tool_manifest_counts_against_the_budget():
    """The manifest is resent on every step, so a budget that ignored it would
    understate every request by the one component that never shrinks."""
    messages = [{"role": "user", "content": "q"}]
    tools = [{"function": {"name": "t", "description": "d" * 4000}}]
    _, without, _ = orchestrator.fit_context(messages, None, budget=10_000)
    _, with_tools, _ = orchestrator.fit_context(messages, tools, budget=10_000)
    assert with_tools > without + 1000
