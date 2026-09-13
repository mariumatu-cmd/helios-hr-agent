"""Run one real multi-step agent turn locally and report context pressure.

Used to verify that the per-step context budget keeps a full agentic workflow
inside the provider's tokens-per-minute bucket.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent.mcp_client import MCPToolClient  # noqa: E402
from agent.orchestrator import run_agent  # noqa: E402

QUESTION = (
    "I'm EMP-1007. I want to work from Portugal for three weeks starting 2026-10-05 "
    "and take two days of PTO during it. Can I, and what do I need to do?"
)


async def main() -> int:
    question = " ".join(sys.argv[1:]) or QUESTION
    client = MCPToolClient()
    await client.connect()
    try:
        trace = await run_agent(question, client)
    finally:
        await client.aclose()

    print(f"steps            : {len(trace.steps)}")
    print(f"tools used       : {', '.join(trace.tools_used) or 'none'}")
    print(f"citations        : {len(trace.citations)}")
    print(f"provider/model   : {trace.provider}/{trace.model}  fell_back={trace.fell_back}")
    print(f"peak ctx tokens  : {trace.peak_context_tokens}")
    print(f"elided results   : {trace.elided_results}")
    print(f"total / service  : {trace.total_ms:.0f} ms / {trace.total_service_ms:.0f} ms")
    print(f"throttle         : {trace.throttle_ms:.0f} ms")
    print(f"error            : {trace.error or 'none'}")
    for step in trace.steps:
        print(
            f"  step {step.index} {step.kind:<10} ctx={step.context_tokens:<6} "
            f"elided={step.elided_results} attempts={step.attempts} "
            f"tools={[c.name for c in step.tool_calls]}"
        )
    print("\n--- answer ---")
    print(trace.answer)
    return 0 if trace.answer and not trace.error else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
