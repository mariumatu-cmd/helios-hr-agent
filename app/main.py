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

import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from agent import llm
from agent.mcp_client import MCPToolClient, MCPUnavailable
from agent.orchestrator import run_agent
from config import apply_seeds, settings

log = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level.upper())

HERE = pathlib.Path(__file__).resolve().parent

DEMO_TASKS = [
    {
        "id": "international",
        "label": "International remote work request",
        "question": (
            "Maya Rodriguez wants to work from Portugal for six weeks, "
            "5 October to 15 November. Can she? If not, what are her options?"
        ),
    },
    {
        "id": "pto",
        "label": "PTO request with a short balance",
        "question": (
            "Jonas Weber wants to take three days off starting 21 September. "
            "Is that approvable, and what does he need to do?"
        ),
    },
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
]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict[str, Any]] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    apply_seeds()
    app.state.mcp = MCPToolClient()
    app.state.mcp_error = ""
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
        "status": "ok" if (mcp_ok and index_ok and providers) else "degraded",
        "as_of_date": settings.as_of_date,
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
            "ok": bool(providers),
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
    client: MCPToolClient = request.app.state.mcp
    if client.session is None:
        raise HTTPException(
            503,
            detail=(
                request.app.state.mcp_error
                or "The HR tool server is not connected; see /health."
            ),
        )

    trace = await run_agent(body.message, client, history=body.history)
    return trace.to_dict()
