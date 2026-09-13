"""The agent loop: plan, call tools over MCP, observe, answer.

This is a hand-written orchestration loop rather than a framework agent. The
reason is inspectability -- every step, every tool argument, every tool result
and every provider fallback is captured in a `Trace` that the web UI renders and
the evaluation harness scores. A framework would hide exactly the behaviour this
project is meant to demonstrate and measure.

Control flow per step:

    1. Send the conversation plus the MCP-discovered tool schemas to the LLM.
    2. If it returns tool calls, execute them (in parallel when it asked for
       several), append the results, and loop.
    3. If it returns text, that is the final answer.
    4. Stop at `AGENT_MAX_STEPS` and say so honestly rather than pretending to
       have finished.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from agent import llm
from agent.mcp_client import MCPToolClient, ToolCallResult
from agent.prompts import build_system_prompt
from config import settings

log = logging.getLogger(__name__)


# A token estimate good enough to budget against, without pulling in a
# provider-specific tokeniser that would only be correct for one of the two
# providers anyway. 3.5 chars/token deliberately *under*-states density, which
# over-states the token count -- the safe direction to be wrong in when the
# penalty for exceeding the budget is a request that can never succeed.
CHARS_PER_TOKEN = 3.5

# How much of an elided tool result to keep on the first pass. Enough to
# preserve the citation strings and the leading facts the model already reasoned
# over, without retaining the full passage text that drove the growth.
ELIDED_RESULT_CHARS = 400

# Second-pass floor. Stubs are not free: a long run accumulates enough of them
# that the stubs alone can exceed the budget, which was observed in practice.
# At this tier only the citations survive -- they are what the answer must
# quote, and the model can re-call the tool for anything else.
MINIMAL_RESULT_CHARS = 0


def estimate_tokens(payload: Any) -> int:
    """Approximate the token cost of anything JSON-serialisable."""
    try:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(payload)
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _compact_tool_content(content: str, keep: int = ELIDED_RESULT_CHARS) -> str:
    """Shrink one tool result, keeping its citations and its leading facts.

    Citations are preserved verbatim because the answer is required to cite what
    the system actually retrieved; dropping them during compaction would make a
    long run silently less grounded than a short one.

    The note deliberately tells the model *not* to repeat the call. An earlier
    version invited it to "re-call the tool for the full text", which turned
    compaction into a loop: the elided result prompted a repeat call, the repeat
    re-inflated the context, that forced another elision, and the run burned its
    whole step budget re-fetching what it already had. Compaction has to read as
    a settled summary, not as a retry instruction.
    """
    citations: list[str] = []
    try:
        _extract_citations(json.loads(content), citations)
    except (json.JSONDecodeError, TypeError):
        pass

    head = content[:keep]
    note = (
        f"[Earlier output from this tool, condensed to fit the context budget; "
        f"{len(content) - len(head)} characters omitted."
    )
    if citations:
        note += " Citations: " + "; ".join(citations[:12]) + "."
    note += (
        " This summary is the record of that call -- do not repeat it. If the"
        " detail you need is not here, continue with what you have or say so."
        "]"
    )
    return f"{head}\n\n{note}" if head else note


def fit_context(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    budget: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Bound a request to `budget` tokens by compacting the oldest tool results.

    Only `role == "tool"` messages are touched. The system prompt, the user's
    question and the assistant turns that carry `tool_calls` are left intact --
    the first two because they define the task, the third because the provider
    rejects a tool result whose matching assistant turn is missing or altered.

    Compaction runs oldest-first so the budget is spent on the results the
    current step is actually reasoning about, and in escalating tiers: first
    truncate old results, then strip them to citations only, and only then
    touch the newest result. A request over the provider's per-request limit
    cannot succeed at any retry count or on any model, so losing detail always
    beats failing outright.

    Returns the (possibly rewritten) messages, the estimated token count, and
    how many results were elided. The count is put into the trace so the
    evaluation can tell a compacted run from an uncompacted one rather than
    silently comparing the two.
    """
    overhead = estimate_tokens(tools) if tools else 0
    total = overhead + sum(estimate_tokens(m) for m in messages)
    if total <= budget:
        return messages, total, 0

    fitted = [dict(m) for m in messages]
    touched: set[int] = set()

    def compact(index: int, message: dict[str, Any], keep: int) -> None:
        nonlocal total
        if message.get("role") != "tool":
            return
        content = message.get("content") or ""
        if len(content) <= max(keep, 1):
            return
        before = estimate_tokens(message)
        message["content"] = _compact_tool_content(content, keep)
        after = estimate_tokens(message)
        if after >= before:       # already at the floor; rewriting would only churn
            message["content"] = content
            return
        total -= before - after
        touched.add(index)

    # Escalating tiers. Each stops as soon as the request fits, so a run only
    # loses as much detail as it actually has to.
    tiers: list[tuple[range, int]] = [
        # Truncate every result but the newest.
        (range(len(fitted) - 1), ELIDED_RESULT_CHARS),
        # Stubs are not free: enough of them will exceed the budget on their
        # own, which is what happens on a long run. Strip to citations only.
        (range(len(fitted) - 1), MINIMAL_RESULT_CHARS),
        # Last resort -- a single large search can blow the budget by itself.
        (range(len(fitted) - 1, -1, -1), MINIMAL_RESULT_CHARS),
    ]
    for indices, keep in tiers:
        if total <= budget:
            break
        for index in indices:
            if total <= budget:
                break
            compact(index, fitted[index], keep)

    elided = len(touched)
    if total > budget:
        log.warning(
            "context still %d tokens after eliding %d tool results (budget %d)",
            total, elided, budget,
        )
    return fitted, total, elided


@dataclass
class ToolInvocation:
    step: int
    name: str
    arguments: dict[str, Any]
    result: Any
    is_error: bool
    latency_ms: float


@dataclass
class Step:
    index: int
    kind: str                      # "tool_calls" | "final" | "error" | "truncated"
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    fell_back: bool = False
    thought: str = ""
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    # Wall-clock latency includes time spent waiting out provider rate limits,
    # which on a free tier can exceed the model's own work. Kept separate so a
    # latency figure can be read as a property of the system rather than of the
    # quota it happened to be running under.
    service_ms: float = 0.0
    throttle_ms: float = 0.0
    attempts: int = 1
    # Estimated prompt size and how many earlier tool results had to be
    # compacted to reach it. Surfaced so a run that fitted comfortably is
    # distinguishable from one that only fitted after losing detail.
    context_tokens: int = 0
    elided_results: int = 0


@dataclass
class Trace:
    """Everything that happened during one run. Rendered by the UI, scored by eval."""

    question: str
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    total_ms: float = 0.0
    # `total_ms` minus every provider rate-limit wait: the latency this system
    # would exhibit on an unthrottled quota. Reported alongside, never instead.
    total_service_ms: float = 0.0
    throttle_ms: float = 0.0
    provider: str = ""
    model: str = ""
    fell_back: bool = False
    grounded: bool | None = None
    truncated: bool = False
    error: str = ""
    # Total tool results compacted across the run, and the largest prompt sent.
    elided_results: int = 0
    peak_context_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_citations(result: Any, into: list[str]) -> None:
    """Collect every citation a tool returned, preserving first-seen order.

    Gathered from the tool results rather than parsed out of the model's prose,
    so the evaluation can check whether the answer cited something the system
    actually retrieved -- not merely something that looks like a citation.
    """
    if isinstance(result, dict):
        value = result.get("citation")
        if isinstance(value, str) and value not in into:
            into.append(value)
        for value in result.get("citations", []) or []:
            if isinstance(value, str) and value not in into:
                into.append(value)
        for value in result.values():
            _extract_citations(value, into)
    elif isinstance(result, list):
        for item in result:
            _extract_citations(item, into)


async def run_agent(
    question: str,
    client: MCPToolClient,
    history: list[dict[str, Any]] | None = None,
    max_steps: int | None = None,
) -> Trace:
    """Run one turn of the agent and return its full trace."""
    started = time.perf_counter()
    max_steps = max_steps or settings.agent_max_steps
    trace = Trace(question=question)

    tools = client.openai_tools()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(settings.as_of_date, len(tools))}
    ]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    citations: list[str] = []

    for index in range(1, max_steps + 1):
        # Bound the request before it is sent. Doing this per step rather than
        # once up front is the point: the transcript only exceeds the budget
        # after several tool results have accumulated.
        messages, context_tokens, elided = fit_context(
            messages, tools, settings.context_token_budget
        )
        trace.elided_results += elided
        trace.peak_context_tokens = max(trace.peak_context_tokens, context_tokens)

        try:
            response = await asyncio.to_thread(llm.chat, messages, tools)
        except llm.NoProviderConfigured as exc:
            trace.error = str(exc)
            trace.answer = (
                "No language model is configured. Set GROQ_API_KEY or GEMINI_API_KEY "
                "and restart the service."
            )
            trace.steps.append(Step(index=index, kind="error", thought=str(exc)))
            break
        except Exception as exc:
            log.exception("LLM call failed at step %d", index)
            trace.error = f"{type(exc).__name__}: {exc}"
            trace.answer = (
                "The language model is unavailable right now, so I could not complete "
                "this request. Everything else in the system is still running; please retry."
            )
            trace.steps.append(Step(index=index, kind="error", thought=trace.error))
            break

        trace.provider, trace.model = response.provider, response.model
        trace.fell_back = trace.fell_back or response.fell_back

        if not response.tool_calls:
            trace.steps.append(Step(
                index=index, kind="final", provider=response.provider, model=response.model,
                latency_ms=round(response.latency_ms, 1), fell_back=response.fell_back,
                service_ms=round(response.service_ms, 1),
                throttle_ms=round(response.throttle_ms, 1),
                attempts=response.attempts,
                context_tokens=context_tokens,
                elided_results=elided,
            ))
            trace.answer = (response.text or "").strip()
            break

        step = Step(
            index=index, kind="tool_calls", provider=response.provider, model=response.model,
            latency_ms=round(response.latency_ms, 1), fell_back=response.fell_back,
            thought=(response.text or "").strip(),
            service_ms=round(response.service_ms, 1),
            throttle_ms=round(response.throttle_ms, 1),
            attempts=response.attempts,
            context_tokens=context_tokens,
            elided_results=elided,
        )

        # The assistant turn must be echoed back verbatim (with its tool_calls)
        # before the tool results, or the provider rejects the next request.
        messages.append({
            "role": "assistant",
            "content": response.text or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in response.tool_calls
            ],
        })

        async def execute(call) -> tuple[Any, ToolCallResult]:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                return call, ToolCallResult(
                    name=call.function.name,
                    arguments={},
                    content={
                        "error": f"could not parse tool arguments: {exc}",
                        "hint": "Emit a JSON object matching the tool's schema.",
                    },
                    is_error=True,
                )
            return call, await client.call(call.function.name, arguments)

        for call, result in await asyncio.gather(*(execute(c) for c in response.tool_calls)):
            _extract_citations(result.content, citations)
            if result.name not in trace.tools_used:
                trace.tools_used.append(result.name)
            if isinstance(result.content, dict) and "grounded" in result.content:
                grounded = bool(result.content["grounded"])
                trace.grounded = grounded if trace.grounded is None else (trace.grounded or grounded)

            step.tool_calls.append(ToolInvocation(
                step=index,
                name=result.name,
                arguments=result.arguments,
                result=result.content,
                is_error=result.is_error,
                latency_ms=round(result.latency_ms, 1),
            ))
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": result.name,
                "content": result.as_message_content(),
            })

        trace.steps.append(step)
    else:
        # Loop exhausted without a text answer.
        trace.truncated = True
        trace.steps.append(Step(index=max_steps, kind="truncated"))
        trace.answer = (
            f"I could not finish this within {max_steps} steps. Here is what I established "
            f"so far using {', '.join(trace.tools_used) or 'no tools'}. Please narrow the "
            f"question, or contact HR directly."
        )

    trace.citations = citations
    trace.total_ms = round((time.perf_counter() - started) * 1000.0, 1)
    trace.throttle_ms = round(sum(s.throttle_ms for s in trace.steps), 1)
    trace.total_service_ms = round(trace.total_ms - trace.throttle_ms, 1)
    return trace
