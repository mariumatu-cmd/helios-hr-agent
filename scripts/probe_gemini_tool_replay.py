"""Can Gemini continue a tool loop at all? (Answer, measured: no.)

Gemini's OpenAI-compatible endpoint rejects a replayed assistant tool-call with
`400 Function call is missing a thought_signature in functionCall parts`, and
that signature is never exposed by the compat layer, so there is nothing to send
back. That kills the cross-provider fallback for precisely the multi-step runs
it existed to rescue, and it cost two evaluation cases before it was understood.

This probe exists so the claim in `design-and-evaluation.md` §6 is reproducible
rather than asserted. It tries every documented way to turn thinking off and
reports which, if any, lets a second turn through.

Result on 2026-09-13 against gemini-3.5-flash: all four variants fail
identically, so the exclusion in `agent/llm.py` is a limitation of the provider
rather than a tuning choice.

    python scripts/probe_gemini_tool_replay.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from config import settings  # noqa: E402

KEY = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
MODEL = settings.gemini_model
BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy_documents",
            "description": "Search HR policy documents.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]

VARIANTS: list[tuple[str, dict]] = [
    ("baseline (no thinking config)", {}),
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    (
        "extra_body thinking_budget=0",
        {"extra_body": {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}},
    ),
    (
        "extra_body include_thoughts=false",
        {"extra_body": {"extra_body": {"google": {"thinking_config": {"include_thoughts": False}}}}},
    ),
]


def attempt(label: str, kwargs: dict) -> None:
    client = OpenAI(api_key=KEY, base_url=BASE, timeout=60.0, max_retries=0)
    messages: list[dict] = [
        {"role": "system", "content": "Use the tools. Cite what you find."},
        {"role": "user", "content": "What is the PTO notice period?"},
    ]
    try:
        first = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", **kwargs
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  {label}: turn 1 failed -- {type(exc).__name__}: {str(exc)[:160]}")
        return

    msg = first.choices[0].message
    calls = list(msg.tool_calls or [])
    if not calls:
        print(f"  {label}: turn 1 returned no tool call; cannot test replay")
        return

    # Replay exactly the way the orchestrator does.
    messages.append(
        {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    },
                }
                for c in calls
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": calls[0].id,
            "name": calls[0].function.name,
            "content": json.dumps(
                {"hits": [{"text": "Ten business days.", "citation": "POL-PTO-001 s3.1"}]}
            ),
        }
    )

    try:
        second = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", **kwargs
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  {label}: turn 2 FAILED -- {str(exc)[:200]}")
        return

    text = (second.choices[0].message.content or "").strip()
    print(f"  {label}: turn 2 OK -- {text[:110]!r}")


if __name__ == "__main__":
    if not KEY:
        raise SystemExit("GEMINI_API_KEY is not set")
    print(f"model: {MODEL}\n")
    for label, kwargs in VARIANTS:
        attempt(label, kwargs)
