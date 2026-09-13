"""Keeping a reasoning model's scratchpad out of the trace.

The project brief asks for a visible trace of *operational* steps -- selected
tools, arguments, outputs, sources -- and is explicit that hidden
chain-of-thought must not be exposed. The step trace is rendered in the UI, and
the assistant text that accompanies a tool call used to be recorded there.

That is safe for the primary model and unsafe for two of the fallbacks: the
rotation chain reaches ``qwen/qwen3.8-27b`` and ``qwen/qwen3.6-27b`` whenever a
rate limit is hit, and those are reasoning models that can wrap a scratchpad in
``<think>`` tags. So the failure mode is not "a model behaves oddly" but "a
rate limit -- an infrastructure event the user never sees -- silently leaks raw
private reasoning into the answer".

The trace itself is now built from the agent's decisions rather than the
model's prose (see ``test_agent.py``), which closes the same hole from the
other side. This closes it at the provider boundary, so the answer text and the
replayed transcript are covered too.

Groq can be asked to hide reasoning per request, but that parameter is
model-specific and ignored by models that do not support it, which is precisely
the wrong property for a chain that mixes model families. Stripping the text on
the way out covers every model in the chain and any future addition to it.

These tests pin the stripping, and the two things it must not do: mangle
ordinary prose, and discard a real answer.
"""
from __future__ import annotations

import pytest

from agent.llm import strip_reasoning


@pytest.mark.parametrize("empty", [None, "", "   ", "\n\n"])
def test_absent_text_becomes_empty_string(empty):
    """`content` is legitimately null on a pure tool-call turn."""
    assert strip_reasoning(empty) == ""


def test_scratchpad_is_removed_and_the_answer_survives():
    text = "<think>The user wants PTO. I should call check_pto_balance.</think>Maya has 12 days."
    assert strip_reasoning(text) == "Maya has 12 days."


@pytest.mark.parametrize("tag", ["think", "thinking", "reasoning"])
def test_every_known_tag_is_stripped(tag):
    assert strip_reasoning(f"<{tag}>private</{tag}>answer") == "answer"


def test_tag_matching_is_case_insensitive_and_tolerates_whitespace():
    assert strip_reasoning("< THINK >private</ think >answer") == "answer"


def test_multiple_blocks_are_all_removed():
    text = "<think>one</think>first. <think>two</think>second."
    assert strip_reasoning(text) == "first. second."


def test_unterminated_block_is_dropped_to_the_end():
    """A scratchpad cut off by the token cap has no answer after it to keep."""
    assert strip_reasoning("Here goes.<think>reasoning that never clos") == "Here goes."


def test_ordinary_answer_is_untouched():
    """The strip must not become a general-purpose angle-bracket filter."""
    text = "Full-time employees accrue 15 days (POL-PTO-001 §2.1). See <hr@helios.example>."
    assert strip_reasoning(text) == text


def test_policy_text_containing_the_word_thinking_is_not_treated_as_a_tag():
    text = "If you are thinking about a sabbatical, POL-LEAVE-002 §4 applies."
    assert strip_reasoning(text) == text


def test_a_reasoning_only_turn_yields_no_text_rather_than_leaking():
    """Nothing but a scratchpad must produce nothing, not the scratchpad."""
    assert strip_reasoning("<think>deciding which tool to call</think>") == ""
