"""The deployed verifier must stop spending quota at the first sign of trouble.

No network: the HTTP opener is replaced, so these tests make no model calls.
"""
from __future__ import annotations

import io
import json
import sys
from urllib.error import HTTPError

import pytest

from scripts import verify_deployed_tasks as verify

PASSING_L01 = {
    "answer": "Jonas Weber has 13 hours of PTO available.",
    "steps": [{"tool_calls": [{"name": "check_pto_balance", "is_error": False}]}],
    "tools_used": ["check_pto_balance"],
    "total_ms": 4200.0, "throttle_ms": 0.0, "api_calls": 2,
    "prompt_tokens": 900, "completion_tokens": 100, "model": "openai/gpt-oss-120b",
}


def quota_error():
    message = "QuotaExceeded: Every language model is rate-limited right now."
    return {"answer": message, "error": message, "api_calls": 1}


def hourly_limit():
    return HTTPError(
        "http://test/chat", 429, "Too Many Requests", {},
        io.BytesIO(b'{"detail": "The hourly limit of 12 chat requests has been reached."}'),
    )


class FakeOpener:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    @property
    def chats(self):
        return len(self.messages)

    def open(self, request, timeout):
        if request.full_url.endswith("/health"):
            return io.BytesIO(b'{"build_sha": "abc123"}')
        self.messages.append(json.loads(request.data)["message"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())


def run(monkeypatch, tmp_path, replies, *args):
    opener = FakeOpener(replies)
    monkeypatch.setattr(verify, "build_opener", lambda *handlers: opener)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    (tmp_path / "evidence").mkdir()
    monkeypatch.setattr(sys, "argv", ["verify", "--allow-live", "--base-url", "http://test", *args])
    code = verify.main()
    saved = next((tmp_path / "evidence").glob("deployed-*.json"))
    return code, opener, json.loads(saved.read_text(encoding="utf-8"))


def test_scores_a_case_and_records_its_cost(monkeypatch, tmp_path):
    code, opener, saved = run(monkeypatch, tmp_path, [PASSING_L01], "--case", "L01")

    assert code == 0 and opener.chats == 1
    assert saved["build_sha"] == "abc123"
    assert saved["summary"]["passed"] == 1 and saved["summary"]["stopped_early"] is None
    assert saved["tasks"][0]["metrics"] == {
        "agent_seconds": 4.2, "throttle_seconds": 0.0, "api_calls": 2,
        "tokens": 1000, "model": "openai/gpt-oss-120b", "fell_back": False,
    }


def test_demo_tasks_run_first_and_a_wrong_answer_does_not_stop_the_run(monkeypatch, tmp_path):
    wrong = {"answer": "Yes, approved.", "api_calls": 2}
    code, opener, saved = run(
        monkeypatch, tmp_path, [wrong, PASSING_L01], "--case", "L01", "--task", "pto"
    )

    assert opener.messages[0].startswith("Jonas Weber wants three business days")
    assert opener.chats == 2 and code == 1
    assert [t["passed"] for t in saved["tasks"]] == [False, True]
    assert saved["summary"]["stopped_early"] is None


@pytest.mark.parametrize("first_reply", [quota_error, hourly_limit])
def test_stops_at_the_first_quota_or_http_error(monkeypatch, tmp_path, first_reply):
    code, opener, saved = run(
        monkeypatch, tmp_path, [first_reply(), PASSING_L01], "--case", "L01", "--case", "R03"
    )

    assert code == 1
    assert opener.chats == 1, "the second case must not be sent"
    assert len(saved["tasks"]) == 1 and not saved["tasks"][0]["passed"]
    assert saved["summary"]["stopped_early"]
