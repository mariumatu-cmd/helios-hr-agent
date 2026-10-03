import pytest

from config import settings
from mcp_server import data
from rag.ingest.chunk import chunk_document
from rag.ingest.parse import load_corpus


def test_non_birthing_parent_receives_eight_weeks():
    result = data.check_parental_leave("David Okafor", "non_birthing", "2027-03-01")
    assert result["paid_weeks"] == 8
    assert result["eligible"]
    assert not result["pto_deducted"]
    assert result["pto_accrues_during_paid_leave"]


def test_parent_role_must_be_explicit():
    with pytest.raises(ValueError, match="parent_role"):
        data.check_parental_leave("David Okafor", "unknown", "2027-03-01")


def test_parental_leave_checks_tenure_and_part_time():
    assert not data.check_parental_leave("Jonas Weber", "non_birthing", "2026-10-01")["eligible"]
    assert data.check_parental_leave("Sofia Marino", "non_birthing", "2027-03-01")["paid_weeks"] == 4.8


def test_pto_uses_business_days_and_documents_alternatives():
    result = data.check_pto_request("Maya Rodriguez", "2026-10-09", 3)
    assert result["request"]["end_date"] == "2026-10-13"
    assert "3 business days notice" in result["alternatives"][0]["detail"]


@pytest.mark.parametrize("days", [float("nan"), float("inf"), -1, 261])
def test_invalid_pto_duration_is_rejected(days):
    with pytest.raises(ValueError):
        data.check_pto_request("Maya Rodriguez", "2026-10-09", days)


def test_notice_section_preserves_its_own_citation():
    doc = next(d for d in load_corpus(settings.corpus_dir) if d.doc_id == "POL-LEAVE-002")
    sections = {c["section_number"]: c for c in chunk_document(doc)}
    assert "2.4" in sections
    assert "§2.4 Notice" in sections["2.4"]["citation"]
    assert "30 days" in sections["2.4"]["text"]


def test_corpus_and_query_use_symmetric_word_roots():
    from rag.vocabulary import unknown_terms

    assert "harassing" not in unknown_terms("My manager is harassing me.")


def test_demo_instructions_are_not_unknown_policy_topics():
    from agent.demo_tasks import DEMO_WORKFLOWS
    from rag.vocabulary import unknown_terms

    for task in DEMO_WORKFLOWS:
        assert not unknown_terms(task["question"])


def test_punctuation_is_rejected_without_embedding(monkeypatch):
    from rag import retrieve

    def forbidden():
        raise AssertionError("No embedding/index call is needed for punctuation")

    monkeypatch.setattr(retrieve._Index, "get", forbidden)
    assert not retrieve.search("?????").grounded


@pytest.mark.parametrize("text", [
    "You are entitled to 16 weeks.",
    "As a birthing parent, you receive eight weeks.",
    "You are entitled to sixteen weeks.",
])
def test_incorrect_parental_claims_are_rejected_in_answers_and_drafts(text):
    from agent.orchestrator import ToolInvocation, _parental_claim_errors

    decision = ToolInvocation(1, "check_policy_compliance", {},
                              {"paid_weeks": 8, "parent_role": "non_birthing"}, False, 0)
    assert _parental_claim_errors(text, [decision], require_weeks=True)
    assert _parental_claim_errors(text, [decision], require_weeks=False)


def test_live_review_notice_error_is_rejected():
    from agent.orchestrator import ToolInvocation, _notice_claim_errors

    result = data.check_pto_request("Jonas Weber", "2026-09-21", 3)
    decision = ToolInvocation(1, "check_policy_compliance", {}, result, False, 0)
    wrong = "Reduce the length to 2 consecutive days, which only requires 5 business-day notice."
    assert _notice_claim_errors(wrong, [decision])
    assert not _notice_claim_errors(wrong.replace("5 business", "3 business"), [decision])
