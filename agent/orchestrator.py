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


@dataclass
class Trace:
    """Everything that happened during one run. Rendered by the UI, scored by eval."""

    question: str
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    total_ms: float = 0.0
    provider: str = ""
    model: str = ""
    fell_back: bool = False
    grounded: bool | None = None
    truncated: bool = False
    error: str = ""

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
            ))
            trace.answer = (response.text or "").strip()
            break

        step = Step(
            index=index, kind="tool_calls", provider=response.provider, model=response.model,
            latency_ms=round(response.latency_ms, 1), fell_back=response.fell_back,
            thought=(response.text or "").strip(),
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
    return trace
