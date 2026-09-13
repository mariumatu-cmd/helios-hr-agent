"""Excluding models that cannot continue a tool loop.

Gemini's OpenAI-compatible endpoint accepts a request that *offers* tools, but
rejects one whose transcript already contains an assistant tool-call:

    400 Function call is missing a thought_signature in functionCall parts

The signature is never exposed by the compat layer, so there is nothing to send
back. Probed against gemini-3.5-flash four ways -- thinking left on,
``reasoning_effort="none"``, ``thinking_budget=0`` and ``include_thoughts=false``
-- and all four fail identically. It is a hard limitation, not a tuning problem.

That matters more than a single failed call. Reached mid-run, the 400 aborts the
whole agent turn, so a merely rate-limited Groq chain produced a *failed task*
rather than a slow one. Two evaluation cases failed exactly that way.

These tests pin the exclusion, and its two boundaries: it applies only when a
replay is actually required, and it never empties the chain.
"""
from __future__ import annotations

import pytest

from agent import llm


@pytest.fixture(autouse=True)
def _clean_cooldowns():
    llm.reset_cooldowns()
    yield
    llm.reset_cooldowns()


def _groq(model: str = "groq-model") -> llm.ProviderConfig:
    return llm.ProviderConfig("groq", "https://example.invalid", "k", model)


def _gemini() -> llm.ProviderConfig:
    return llm.ProviderConfig(
        "gemini", "https://example.invalid", "k", "gemini-model", replays_tool_calls=False
    )


FIRST_TURN = [
    {"role": "system", "content": "you are a helpful assistant"},
    {"role": "user", "content": "how many PTO days do I have left?"},
]

CONTINUED_TOOL_LOOP = FIRST_TURN + [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "check_pto_balance", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '{"remaining_days": 11}'},
]


def test_a_first_turn_does_not_require_replay():
    """Offering tools is not the trigger; only continuing a loop is.

    Excluding Gemini from every tools-enabled call would throw away a fallback
    that works perfectly well for the opening turn.
    """
    assert llm._replays_tool_calls_required(FIRST_TURN) is False


def test_a_transcript_with_an_assistant_tool_call_requires_replay():
    assert llm._replays_tool_calls_required(CONTINUED_TOOL_LOOP) is True


def test_a_tool_result_alone_does_not_count():
    """The rejected part is the assistant's ``functionCall``, not the result."""
    assert (
        llm._replays_tool_calls_required(
            [{"role": "tool", "tool_call_id": "x", "content": "{}"}]
        )
        is False
    )


def test_an_assistant_message_without_tool_calls_does_not_count():
    assert (
        llm._replays_tool_calls_required(
            FIRST_TURN + [{"role": "assistant", "content": "you have 11 days"}]
        )
        is False
    )


def _order_for(chain, messages):
    """Mirror the filtering `chat()` applies, without making a network call."""
    order = llm._ordered_chain(chain)
    if llm._replays_tool_calls_required(messages):
        usable = [entry for entry in order if entry[1].replays_tool_calls]
        if usable:
            order = usable
    return order


def test_gemini_is_dropped_when_the_transcript_replays_a_tool_call():
    chain = [_groq(), _gemini()]
    order = _order_for(chain, CONTINUED_TOOL_LOOP)
    assert [cfg.name for _, cfg in order] == ["groq"]


def test_gemini_is_kept_for_an_opening_turn():
    chain = [_groq(), _gemini()]
    order = _order_for(chain, FIRST_TURN)
    assert [cfg.name for _, cfg in order] == ["groq", "gemini"]


def test_a_gemini_only_chain_is_never_emptied():
    """Attempting a call that may fail beats refusing to call anyone.

    The first turn of a Gemini-only deployment succeeds; only continuation
    fails, and it should fail with the provider's own error rather than with a
    'no providers configured' error that misdescribes the cause.
    """
    chain = [_gemini()]
    order = _order_for(chain, CONTINUED_TOOL_LOOP)
    assert [cfg.name for _, cfg in order] == ["gemini"]


def test_dropping_the_primary_keeps_fallback_indexes_honest():
    """`fell_back` must keep meaning 'not the configured primary'.

    Filtering the *ordered* list rather than the chain preserves each entry's
    index in the configured chain, so a run that skipped an unusable primary is
    still reported as a fallback rather than silently looking like a clean
    primary call.
    """
    chain = [_gemini(), _groq("a"), _groq("b")]
    order = _order_for(chain, CONTINUED_TOOL_LOOP)
    assert [index for index, _ in order] == [1, 2]


def test_cooldowns_and_replay_exclusion_compose():
    """A cooling model is still reordered; an unusable one is still removed."""
    chain = [_groq("a"), _groq("b"), _gemini()]
    llm._mark_cooling(chain[0])
    order = _order_for(chain, CONTINUED_TOOL_LOOP)
    assert [cfg.model for _, cfg in order] == ["b", "a"]
