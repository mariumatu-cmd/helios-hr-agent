"""HTTP API surface.

Uses the real app with its real lifespan, so the MCP session is genuinely opened
during these tests.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Helios HR Assistant" in response.text
    assert "Execution trace" in response.text


def test_health_reports_every_dependency(client):
    """Health must be diagnostic, not a bare 200: a degraded dependency should
    be visible without reading logs."""
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert body["mcp"]["connected"] is True
    assert body["mcp"]["tool_count"] == 12
    assert body["rag_index"]["ok"] is True
    assert body["rag_index"]["chunks"] > 0
    assert "providers_configured" in body["llm"]
    assert body["as_of_date"] == "2026-09-12"


def test_health_status_code_reflects_readiness(client):
    response = client.get("/health")
    body = response.json()
    assert response.status_code == (200 if body["status"] == "ok" else 503)


def test_health_reports_rather_than_crashes_when_startup_did_not_complete():
    """`/health` must diagnose a failed startup, not 500 on it.

    A readiness probe exists to say *which* dependency is down. If the lifespan
    never ran -- or died before attaching the MCP client to app state -- then
    reading `app.state.mcp` raises AttributeError and the endpoint returns 500,
    which is exactly the situation in which its answer matters most and tells
    you nothing.

    `app` is a module-level singleton shared with the `client` fixture, whose
    lifespan populates the same state, so the pre-startup condition is staged
    explicitly and restored rather than relying on test ordering.
    """
    saved = dict(app.state._state)
    app.state._state.pop("mcp", None)
    app.state._state.pop("mcp_error", None)
    try:
        response = TestClient(app).get("/health")
    finally:
        app.state._state.clear()
        app.state._state.update(saved)

    body = response.json()
    assert response.status_code == 503
    assert body["status"] == "degraded"
    assert body["mcp"]["connected"] is False
    assert body["mcp"]["tool_count"] == 0
    assert "startup" in (body["mcp"]["error"] or "")


def test_healthz_is_liveness_only(client):
    """`/healthz` must stay 200 even while `/health` reports degraded.

    The platform health check points at `/healthz`. If it ever started failing
    for a *readiness* reason -- a missing API key, say -- Render would kill and
    redeploy a perfectly functional process in a loop.
    """
    liveness = client.get("/healthz")
    assert liveness.status_code == 200
    assert liveness.json() == {"status": "alive"}

    readiness = client.get("/health")
    if readiness.json()["status"] != "ok":
        assert readiness.status_code == 503
        assert client.get("/healthz").status_code == 200


def test_assert_health_accepts_a_degraded_but_serviceable_payload(client):
    """The CI gate must pass on a keyless instance and fail on a broken one."""
    from scripts.assert_health import assess

    payload = client.get("/health").json()
    assert assess(payload, require_llm=False) == []

    # Built explicitly rather than read from the live payload: whether a key
    # happens to sit in the developer's .env must not decide whether this
    # assertion tests anything.
    keyless = {**payload, "llm": {**payload.get("llm", {}), "providers_configured": []}}
    assert assess(keyless, require_llm=False) == []
    assert assess(keyless, require_llm=True) != []

    keyed = {**payload, "llm": {**payload.get("llm", {}), "providers_configured": ["groq"]}}
    assert assess(keyed, require_llm=True) == []

    broken = {**payload, "mcp": {**payload["mcp"], "connected": False, "error": "boom"}}
    assert assess(broken, require_llm=False) != []

    short = {**payload, "mcp": {**payload["mcp"], "tool_count": 11}}
    assert assess(short, require_llm=False) != []


def test_tools_endpoint_exposes_the_discovered_catalogue(client):
    body = client.get("/tools").json()
    assert body["count"] == 12
    assert body["transport"] in ("stdio", "streamable-http")
    names = {t["name"] for t in body["tools"]}
    assert "search_policy_documents" in names
    assert all(t["description"] for t in body["tools"])


def test_tool_catalogue_marks_write_tools(client):
    tools = {t["name"]: t for t in client.get("/tools").json()["tools"]}
    assert tools["create_hr_ticket"]["read_only"] is False
    assert tools["search_policy_documents"]["read_only"] is True


def test_documents_endpoint_lists_the_corpus(client):
    body = client.get("/documents").json()
    # Range rather than a literal: the brief asks for 5-20 documents, and that
    # is the property worth asserting. Exact coverage is checked in
    # tests/test_retrieval.py against the corpus directory itself.
    assert 5 <= len(body["documents"]) <= 20
    assert body["index"]["chunks"] > 0


def test_demo_tasks_are_available(client):
    tasks = client.get("/demo-tasks").json()["tasks"]
    assert len(tasks) >= 3
    assert all(t["question"] and t["label"] for t in tasks)


def test_chat_rejects_an_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_an_oversized_message(client):
    assert client.post("/chat", json={"message": "x" * 5000}).status_code == 422


def test_chat_without_a_provider_returns_a_trace_not_a_crash(client, monkeypatch):
    """With no API key the run cannot succeed, but the service must still answer
    with a structured, honest trace."""
    from agent import llm

    monkeypatch.setattr(llm, "provider_chain", lambda: [])
    body = client.post("/chat", json={"message": "How much PTO do I have?"}).json()
    assert "answer" in body and "steps" in body
    assert body["error"]
