"""FastAPI application: chat UI, JSON API, and health.

One process serves the web app and owns a single long-lived MCP session, opened
on startup and closed on shutdown. Opening a session per request would spawn a
server process per request under stdio, which is both slow and a good way to
exhaust a 512 MB free-tier instance.

The MCP connection is treated as *optional at boot*: if it fails, the app still
starts and reports the failure on `/health` rather than crash-looping. A service
that will not start is much harder to diagnose than one that starts and tells
you what is broken.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import pathlib
import secrets
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field
from starlette.requests import Request

from agent import llm
from agent.demo_tasks import DEMO_WORKFLOWS
from agent.mcp_client import MCPToolClient, MCPUnavailable
from agent.orchestrator import run_agent
from config import apply_seeds, settings

log = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level.upper())

HERE = pathlib.Path(__file__).resolve().parent
MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("table")

DEMO_TASKS = DEMO_WORKFLOWS + [
    {
        "id": "benefits",
        "label": "Part-time benefits eligibility",
        "question": "Is Sofia Marino covered by short-term disability, and why?",
    },
    {
        "id": "out-of-scope",
        "label": "Out-of-scope question (refusal)",
        "question": "What was our Q3 revenue and which stock should I buy?",
    },
    {
        # The only seeded task that asks the agent to *act* rather than answer.
        # It is deliberately a two-turn task: the first click returns a preview
        # and a request for confirmation, and nothing is written until the user
        # replies. Surfaced as a button because a confirmation gate that is
        # never exercised in the demo is indistinguishable from one that does
        # not work.
        "id": "ticket",
        "label": "File an HR ticket (asks before acting)",
        "question": "Open an HR ticket for Jonas Weber about his PTO shortfall.",
    },
]


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=8)
    fresh: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    apply_seeds()
    app.state.mcp = MCPToolClient()
    app.state.mcp_error = ""
    app.state.pending = {}
    app.state.cache = {}
    app.state.chat_requests = []
    app.state.busy = False
    try:
        await app.state.mcp.connect()
    except MCPUnavailable as exc:
        app.state.mcp_error = str(exc)
        log.error("MCP unavailable at startup: %s", exc)

    if settings.warm_embedder:
        # Deliberately *not* warmed here. This process never embeds a query:
        # retrieval runs inside the MCP server subprocess, which warms its own
        # embedder at startup. Loading the weights here too would put a second
        # ~100 MB copy in a 512 MB instance and buy nothing.
        log.info("embedder warmup delegated to the MCP server process")

    yield
    await app.state.mcp.aclose()


app = FastAPI(
    title="Helios HR Assistant",
    description=(
        "Agentic RAG over a synthetic HR policy corpus, with tools exposed over the "
        "Model Context Protocol."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "demo_tasks": DEMO_TASKS,
            "as_of_date": settings.as_of_date,
            "tool_count": len(request.app.state.mcp.tools),
            "access_code_required": bool(settings.demo_access_code),
        },
    )


@app.get("/healthz")
async def healthz():
    """Liveness only: is the process up and serving?

    Separate from `/health` on purpose. `/health` is a *readiness* check and
    returns 503 when any dependency is missing, which is the right answer for a
    human or a CI gate. Pointing a platform health check at it would be wrong:
    a missing LLM key would make Render tear down and redeploy a process that is
    running perfectly well, forever. Liveness and readiness are different
    questions and get different endpoints.
    """
    return {"status": "alive"}


@app.get("/health")
async def health(request: Request):
    """Liveness plus a full readiness breakdown of every dependency.

    Never raises. A readiness probe that 500s tells you nothing about *which*
    dependency is down, which is the only reason to call it. If startup failed
    before the MCP client was attached to app state, that is itself the finding
    and is reported as a degraded MCP section rather than an exception.
    """
    client: MCPToolClient | None = getattr(request.app.state, "mcp", None)
    mcp_error = getattr(request.app.state, "mcp_error", None) or (
        None if client is not None else "MCP client not initialised (startup did not complete)"
    )
    mcp_ok = client is not None and client.session is not None and bool(client.tools)

    index_ok, index_detail = True, {}
    try:
        from rag import retrieve

        index_detail = {
            k: retrieve.index_info()[k] for k in ("model", "chunks", "documents", "built_at")
        }
    except Exception as exc:
        index_ok = False
        index_detail = {"error": str(exc)}

    providers = llm.available_providers()
    payload = {
        "status": "ok" if (mcp_ok and index_ok and providers and settings.llm_enabled) else "degraded",
        "as_of_date": settings.as_of_date,
        "build_sha": settings.build_sha,
        "quota_limits": {
            "per_turn": settings.llm_max_calls_per_turn,
            "per_hour": settings.llm_max_calls_per_hour,
            "per_day": settings.llm_max_calls_per_day,
        },
        "mcp": {
            "connected": mcp_ok,
            "transport": (client.transport if client else None) or settings.mcp_transport,
            "server": f"{client.server_name} v{client.server_version}" if mcp_ok else None,
            "tool_count": len(client.tools) if client else 0,
            "error": mcp_error,
        },
        "rag_index": {"ok": index_ok, **index_detail},
        "llm": {
            "providers_configured": providers,
            "primary": settings.llm_provider,
            "enabled": settings.llm_enabled,
            "ok": bool(providers) and settings.llm_enabled,
        },
    }
    return JSONResponse(payload, status_code=200 if payload["status"] == "ok" else 503)


@app.get("/tools")
async def tools(request: Request):
    """The live MCP tool catalogue, exactly as discovered from the server."""
    client: MCPToolClient = request.app.state.mcp
    if client.session is None:
        raise HTTPException(503, detail=request.app.state.mcp_error or "MCP not connected")
    return {
        "server": client.server_name,
        "version": client.server_version,
        "transport": client.transport,
        "count": len(client.tools),
        "tools": client.catalogue(),
    }


@app.get("/documents")
async def documents():
    """Catalogue of the indexed policy corpus."""
    from rag import retrieve

    return {"documents": retrieve.list_documents(), "index": retrieve.index_info()}


@app.get("/demo-tasks")
async def demo_tasks():
    return {"tasks": DEMO_TASKS}


@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    """Run one agent turn. Returns the answer and the full execution trace."""
    _authorize(request)
    client: MCPToolClient = request.app.state.mcp
    if client.session is None:
        raise HTTPException(
            503,
            detail=(
                request.app.state.mcp_error
                or "The HR tool server is not connected; see /health."
            ),
        )

    session = request.cookies.get("helios_session") or secrets.token_urlsafe(32)
    state = request.app.state
    now = time.monotonic()
    _expire(state.pending, now)
    _expire(state.cache, now)
    key = hashlib.sha256(
        json.dumps([session, body.message, [m.model_dump() for m in body.history]]).encode()
    ).hexdigest()
    if not body.fresh and key in state.cache:
        payload = {**state.cache[key]["payload"], "cached": True, "api_calls": 0}
        return _session_response(payload, session, request)
    if state.busy:
        raise HTTPException(429, "Another chat is running. Wait to preserve the shared API quota.")
    state.chat_requests = [t for t in state.chat_requests if t > now - 3600]
    if len(state.chat_requests) >= settings.chat_max_requests_per_hour:
        raise HTTPException(429, "Hourly chat limit reached. Wait before retrying.")
    state.chat_requests.append(now)
    state.busy = True
    try:
        trace = await run_agent(
            body.message, client, history=[m.model_dump() for m in body.history]
        )
    finally:
        state.busy = False
    if not trace.error and not trace.truncated:
        for step in trace.steps:
            for call in step.tool_calls:
                result = call.result
                if (
                    call.name in ("create_hr_ticket", "draft_hr_email")
                    and not call.is_error
                    and isinstance(result, dict)
                    and result.get("requires_confirmation")
                ):
                    action_id = secrets.token_urlsafe(24)
                    if len(state.pending) >= 100:
                        raise HTTPException(429, "Too many pending previews; let them expire.")
                    state.pending[action_id] = {
                        "session": session, "name": call.name, "arguments": call.arguments,
                        "expires": time.monotonic() + 600,
                    }
                    trace.pending_actions.append({
                        "id": action_id, "tool": call.name, "preview": result["preview"],
                    })
    payload = {**trace.to_dict(), "cached": False, "answer_html": MARKDOWN.render(trace.answer)}
    if (
        not trace.error and not trace.truncated and not trace.pending_actions
        and not set(trace.tools_used) & {"create_hr_ticket", "draft_hr_email", "list_hr_tickets"}
    ):
        if len(state.cache) >= 100:
            del state.cache[next(iter(state.cache))]
        state.cache[key] = {"expires": time.monotonic() + 600, "payload": payload}
    return _session_response(payload, session, request)


def _authorize(request: Request) -> None:
    if settings.demo_access_code and not hmac.compare_digest(
        request.headers.get("X-Demo-Code", ""), settings.demo_access_code
    ):
        raise HTTPException(403, "Enter the demo access code to use the shared model quota.")


def _expire(store: dict, now: float) -> None:
    for key in list(store):
        if store[key]["expires"] <= now:
            del store[key]


def _session_response(payload: dict, session: str, request: Request) -> JSONResponse:
    response = JSONResponse(payload)
    response.set_cookie(
        "helios_session", session, httponly=True, samesite="strict",
        secure=request.url.scheme == "https", max_age=3600,
    )
    return response


@app.post("/actions/{action_id}/confirm")
async def confirm_action(action_id: str, request: Request):
    """A distinct user action authorizes the stored payload, without an LLM call."""
    _authorize(request)
    state = request.app.state
    _expire(state.pending, time.monotonic())
    pending = state.pending.get(action_id)
    if pending is None or pending["session"] != request.cookies.get("helios_session"):
        raise HTTPException(404, "Preview expired, already used, or belongs to another session.")
    del state.pending[action_id]
    result = await state.mcp.confirm(pending["name"], pending["arguments"])
    if result.is_error:
        raise HTTPException(502, detail=result.content)
    # Invalidate cached reads after a mutation.
    state.cache.clear()
    return {
        "status": "completed", "mock": True, "tool": pending["name"],
        "arguments": {k: v for k, v in pending["arguments"].items() if k != "approval_token"},
        "result": result.content, "api_calls": 0,
    }
