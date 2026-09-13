"""The scorer that produces every headline number in the write-up.

It had no tests, which is the wrong way round: a defect here does not make the
system fail, it makes the *measurement* wrong, silently, in whichever direction
the defect happens to point. One such defect was found in the second full run --
the agent declined an out-of-remit question perfectly ("I can only help with
HR-related questions") and the refusal detector, which only recognised refusal
by negation, recorded it as an asserted answer and scored the case 0.33.

These tests pin the properties the write-up actually relies on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.cases import Case
from evaluation.score import PASS_THRESHOLD, score_case


@dataclass
class FakeTrace:
    answer: str = ""
    tools_used: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    steps: list[Any] = field(default_factory=list)
    total_ms: float = 0.0
    total_service_ms: float = 0.0
    throttle_ms: float = 0.0
    error: str = ""


def _case(**kwargs) -> Case:
    base = {
        "id": "T01",
        "category": "reasoning",
        "difficulty": "medium",
        "question": "q",
    }
    return Case(**{**base, **kwargs})


# -- refusal detection --------------------------------------------------------
def test_a_refusal_by_negation_is_recognised():
    case = _case(behaviour="refuse")
    result = score_case(case, FakeTrace(answer="I cannot answer that."))
    assert result.behaviour == 1.0


def test_a_refusal_by_scope_is_recognised():
    """Regression: this exact answer was scored as an asserted answer.

    Declining by naming the remit is as firm a refusal as declining by
    negation, and treating it as a failure understates the safety numbers.
    """
    answer = (
        "I'm sorry, but I can only help with HR-related questions. For company "
        "financials, please contact the appropriate department."
    )
    result = score_case(_case(behaviour="refuse"), FakeTrace(answer=answer))
    assert result.behaviour == 1.0


def test_an_asserted_answer_is_still_a_refusal_failure():
    """The looser marker list must not make every answer look like a refusal."""
    result = score_case(
        _case(behaviour="refuse"),
        FakeTrace(answer="Q3 revenue was $4.2M and you should buy the stock."),
    )
    assert result.behaviour == 0.0
    assert any("refusal" in f for f in result.failures)


# -- the disqualifying asymmetry ---------------------------------------------
def test_a_forbidden_fact_zeroes_the_answer_rather_than_deducting():
    """A wrong balance is not a partially correct answer.

    Partial credit here would hide the single failure mode the case exists to
    catch, so the zeroing is load-bearing and is pinned.
    """
    case = _case(must_include=["11 days"], must_not_include=["24 days"])
    result = score_case(case, FakeTrace(answer="You have 11 days, previously 24 days."))
    assert result.answer_match == 0.0


def test_numbers_match_on_a_word_boundary():
    """Without this, must_include=['12'] is satisfied by '2026-09-12' or '120'."""
    case = _case(must_include=["12"])
    assert score_case(case, FakeTrace(answer="you have 120 hours")).answer_match == 0.0
    assert score_case(case, FakeTrace(answer="you have 12 days")).answer_match == 1.0


def test_any_alternative_in_a_tuple_satisfies_the_requirement():
    case = _case(must_include=[("six months", "6 months")])
    assert score_case(case, FakeTrace(answer="after 6 months")).answer_match == 1.0


# -- tool dimensions ----------------------------------------------------------
def test_tool_selection_is_recall_so_exploring_is_not_penalised():
    case = _case(expected_tools=["check_pto_balance"])
    trace = FakeTrace(
        answer="ok", tools_used=["search_policy_documents", "check_pto_balance"]
    )
    assert score_case(case, trace).tool_selection == 1.0


def test_a_missing_required_tool_is_named_in_the_failures():
    case = _case(expected_tools=["check_policy_compliance"])
    result = score_case(case, FakeTrace(answer="ok", tools_used=[]))
    assert result.tool_selection == 0.0
    assert any("check_policy_compliance" in f for f in result.failures)


def test_a_forbidden_tool_is_binary_not_partial():
    case = _case(forbidden_tools=["create_hr_ticket"])
    clean = score_case(case, FakeTrace(answer="ok", tools_used=["check_pto_balance"]))
    dirty = score_case(case, FakeTrace(answer="ok", tools_used=["create_hr_ticket"]))
    assert clean.no_forbidden == 1.0
    assert dirty.no_forbidden == 0.0


# -- citations ----------------------------------------------------------------
def test_citations_are_matched_case_insensitively_against_the_tools_output():
    case = _case(expected_citations=["POL-PTO-001 §3.1"])
    trace = FakeTrace(answer="ok", citations=["pol-pto-001 §3.1 notice requirements"])
    assert score_case(case, trace).citation == 1.0


# -- pass rule ----------------------------------------------------------------
def test_an_agent_error_cannot_pass_however_well_it_scores():
    """A run that crashed is not a success even if the partial answer matched.

    This is what keeps an infrastructure failure from being laundered into a
    pass by the dimensions that happened to survive it.
    """
    case = _case(must_include=["11 days"])
    trace = FakeTrace(answer="you have 11 days", error="RateLimitError: 429")
    result = score_case(case, trace)
    assert result.score >= PASS_THRESHOLD
    assert result.passed is False


def test_the_pass_threshold_is_applied_to_the_mean_of_scored_dimensions():
    case = _case(must_include=["a", "b", "c", "d"])
    result = score_case(case, FakeTrace(answer="a b c"))
    assert result.answer_match == 0.75
    assert result.passed is False
