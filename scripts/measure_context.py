"""Measure the fixed per-step context overhead, and where it goes.

Every agent step resends the system prompt and the full MCP tool manifest, so
those two set the floor that `CONTEXT_TOKEN_BUDGET` has to clear. On a free
tier metered per minute, that floor also sets how many LLM calls per minute the
agent can make at all -- which is the difference between a multi-step task
finishing in seconds and stalling for minutes.

    python scripts/measure_context.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent.mcp_client import MCPToolClient  # noqa: E402
from agent.orchestrator import estimate_tokens  # noqa: E402
from agent.prompts import build_system_prompt  # noqa: E402
from config import settings  # noqa: E402

GROQ_TPM = 8000


async def main() -> None:
    client = MCPToolClient()
    await client.connect()
    try:
        tools = client.openai_tools()
    finally:
        await client.aclose()

    system = build_system_prompt(settings.as_of_date, len(tools))
    sys_tokens = estimate_tokens(system)
    manifest_tokens = estimate_tokens(tools)
    floor = sys_tokens + manifest_tokens
    reserved = floor + settings.context_token_budget * 0 + 1600

    print(f"tools discovered      : {len(tools)}")
    print(f"system prompt         : {len(system):6d} chars  ~{sys_tokens:5d} tokens")
    print(f"tool manifest         : {len(json.dumps(tools)):6d} chars  ~{manifest_tokens:5d} tokens")
    print(f"fixed floor per step  : ~{floor} tokens")
    print(f"context budget        : {settings.context_token_budget}")
    print(f"room for transcript   : ~{settings.context_token_budget - floor} tokens")
    print(f"\nper-request reservation (floor + max_tokens): ~{reserved}")
    print(f"calls per minute at {GROQ_TPM} TPM          : ~{GROQ_TPM / max(reserved, 1):.1f}")

    print("\nmanifest cost by tool:")
    for spec in sorted(tools, key=lambda s: -estimate_tokens(s)):
        print(f"  {estimate_tokens(spec):5d}  {spec['function']['name']}")


if __name__ == "__main__":
    asyncio.run(main())
