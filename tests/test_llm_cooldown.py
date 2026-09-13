"""Model-rotation cooldowns.

Groq meters its free tier per model, so the fallback chain is a chain of
budgets. But a chain with no memory re-tries a saturated model at the head of
the chain on every call: measured across an evaluation suite, every single case
burned two failed requests and roughly 35 seconds before reaching a model that
could answer.

These tests pin the memory, and the limits on it -- a cooldown is an informed
guess about a *refill*, so it must never make the system refuse to call anyone.
"""
from __future__ import annotations

import time

import pytest

from agent import llm


@pytest.fixture(autouse=True)
def _clean_cooldowns():
    llm.reset_cooldowns()
    yield
    llm.reset_cooldowns()


def _cfg(model: str) -> llm.ProviderConfig:
    return llm.ProviderConfig("groq", "https://example.invalid", "k", model)


def test_an_untouched_chain_keeps_its_configured_order():
    chain = [_cfg("a"), _cfg("b"), _cfg("c")]
    assert llm._ordered_chain(chain) == [(0, chain[0]), (1, chain[1]), (2, chain[2])]


def test_a_rate_limited_model_moves_to_the_back():
    chain = [_cfg("a"), _cfg("b"), _cfg("c")]
    llm._mark_cooling(chain[0])

    order = llm._ordered_chain(chain)

    assert [cfg.model for _, cfg in order] == ["b", "c", "a"]


def test_original_indexes_survive_reordering():
    """`fell_back` must mean "not the configured primary", not "not tried first".

    If the index were recomputed after reordering, a run that fell back to the
    third model would report itself as primary and the trace would misstate
    which model produced the answer.
    """
    chain = [_cfg("a"), _cfg("b")]
    llm._mark_cooling(chain[0])

    order = llm._ordered_chain(chain)

    assert order[0][0] == 1          # 'b' is still the second entry
    assert order[1][0] == 0          # 'a' is still the primary


def test_every_model_cooling_still_yields_a_full_chain():
    """A cooldown is a guess about a refill; it must never mean "give up"."""
    chain = [_cfg("a"), _cfg("b")]
    llm._mark_cooling(chain[0])
    llm._mark_cooling(chain[1])

    order = llm._ordered_chain(chain)

    assert len(order) == len(chain)


def test_the_soonest_to_recover_is_tried_first_when_all_are_cooling():
    chain = [_cfg("a"), _cfg("b")]
    llm._mark_cooling(chain[0], seconds=90)
    llm._mark_cooling(chain[1], seconds=5)

    order = llm._ordered_chain(chain)

    assert [cfg.model for _, cfg in order] == ["b", "a"]


def test_a_cooldown_expires():
    chain = [_cfg("a"), _cfg("b")]
    llm._cooldown_until[llm._cooldown_key(chain[0])] = time.monotonic() - 0.01

    order = llm._ordered_chain(chain)

    assert [cfg.model for _, cfg in order] == ["a", "b"]


def test_cooldowns_are_bounded():
    """A hostile or broken Retry-After must not sideline a model indefinitely."""
    chain = [_cfg("a")]
    llm._mark_cooling(chain[0], seconds=10_000)

    remaining = llm._cooldown_until[llm._cooldown_key(chain[0])] - time.monotonic()

    assert remaining <= llm.MAX_COOLDOWN_SECONDS + 1
