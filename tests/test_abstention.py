"""Tests for the two-store abstention gate.

The property under test is the one the system's safety story rests on: a
question about something the system holds no record of must be marked
ungrounded, and a question it *can* answer must not be. Cosine similarity alone
fails this -- these tests pin the behaviour so a future tuning change cannot
silently reintroduce the failure.
"""
from __future__ import annotations

import pytest

from rag import retrieve
from rag.vocabulary import proper_nouns, unknown_terms

# Topics that exist in neither the policy corpus nor the mock HR data.
OUT_OF_CORPUS = [
    "What does the Helios pet insurance policy cover?",
    "How do I enrol in the Helios company car scheme?",
    "What is the tuition reimbursement cap for an MBA?",
    "What was Helios's Q3 revenue?",
    "Which Helios office has a rooftop swimming pool?",
    "What is the sabbatical policy after ten years of service?",
    "How many shares are in the employee stock purchase plan?",
]

IN_CORPUS = [
    "How much PTO does a full-time employee with three years of service accrue?",
    "How many company holidays does Helios observe?",
    "What is the maximum number of consecutive days I may work from another country?",
    "If my laptop is stolen while travelling, who do I have to notify and how quickly?",
    "Maya Rodriguez wants to work from Portugal for 42 days starting 5 October 2026.",
    "Tomas Silva asks to work remotely from Brazil for a month. What has to happen first?",
    "Marcus Doyle wants to switch to fully remote. Is that allowed right now?",
    "What is the per-night hotel cap for domestic travel, and what receipts do I need?",
]


@pytest.mark.parametrize("question", OUT_OF_CORPUS)
def test_out_of_corpus_questions_are_not_grounded(question):
    result = retrieve.search(question)
    assert not result.grounded, (
        f"{question!r} was accepted as grounded "
        f"(similarity {result.best_similarity:.3f}); the agent would answer from "
        f"topically-adjacent text it must not use"
    )
    assert result.reason, "an ungrounded result must explain itself"


@pytest.mark.parametrize("question", IN_CORPUS)
def test_in_corpus_questions_are_grounded(question):
    result = retrieve.search(question)
    assert result.grounded, (
        f"{question!r} was refused (similarity {result.best_similarity:.3f}, "
        f"unknown terms {result.unknown_terms}); this is a false refusal"
    )
    assert result.unknown_terms == []


def test_similarity_alone_would_not_separate_these_sets():
    """Documents *why* the lexical gate exists, and fails if that stops being true.

    If a future embedding model did separate the two sets on similarity alone,
    this test fails -- correctly -- as a prompt to revisit the extra machinery
    rather than carry it forever out of habit.
    """
    negative_best = max(retrieve.search(q).best_similarity for q in OUT_OF_CORPUS)
    positive_worst = min(retrieve.search(q).best_similarity for q in IN_CORPUS)
    assert negative_best > positive_worst, (
        "cosine similarity now separates in-corpus from out-of-corpus questions; "
        "the lexical gate may no longer be necessary"
    )


@pytest.mark.parametrize(
    "gibberish",
    ["asdkjhasd kjahsdkjh asdkjh", "qwertyuiop zxcvbnm", "?????", "the the the the the"],
)
def test_nonsense_input_is_refused_by_the_similarity_floor(gibberish):
    """The narrow job the similarity floor is still calibrated for.

    Some of these pass the lexical gate -- "?????" and "the the the the the"
    contain no unknown content terms because they contain no content terms at
    all -- so the floor is what stops them.
    """
    assert not retrieve.search(gibberish).grounded


def test_unknown_terms_names_the_missing_topic():
    assert "pet" in unknown_terms("What does the Helios pet insurance policy cover?")
    assert "sabbatical" in unknown_terms("Is there a sabbatical after ten years?")


def test_known_employees_are_not_flagged_as_unknown():
    """Employee names are absent from every policy document but present in the
    HR data, so they must not trigger a refusal."""
    assert unknown_terms("What is Jonas Weber's PTO balance?") == []
    assert unknown_terms("Which department is Marcus Doyle in?") == []


def test_proper_nouns_are_exempt_but_do_not_mask_a_missing_topic():
    """A capitalised name is exempt; a lowercase topic noun beside it is not."""
    assert "brazil" not in unknown_terms("Can I work from Brazil for a month?")
    assert "tuition" in unknown_terms("What is the tuition cap for an MBA?")


def test_sentence_initial_words_are_not_treated_as_names():
    names = proper_nouns("Sabbatical leave is what I want. Helios must have one.")
    assert "sabbatical" not in names
    assert "helios" not in names


def test_plural_and_hyphenated_forms_match_their_root():
    assert unknown_terms("What are the blackouts in the PTO policy?") == []
    assert unknown_terms("What is the per-night hotel cap?") == []


def test_search_still_returns_passages_when_it_refuses():
    """Refusal must be informative, not blank.

    The caller gets the verdict *and* the closest passages, so the agent can say
    "we have no pet insurance policy; here is what the benefits policy does
    cover" rather than stonewalling.
    """
    result = retrieve.search("What does the Helios pet insurance policy cover?")
    assert not result.grounded
    assert result.hits, "an ungrounded search should still surface its best attempts"
