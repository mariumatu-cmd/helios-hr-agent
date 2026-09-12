# Deployed Application

## URLs

| | |
|---|---|
| **Application** | `https://<service>.onrender.com` — *to be filled in after the Render deploy* |
| **Liveness** | `https://<service>.onrender.com/healthz` |
| **Readiness** | `https://<service>.onrender.com/health` |
| **Tool manifest** | `https://<service>.onrender.com/tools` |
| **Corpus manifest** | `https://<service>.onrender.com/documents` |

> **Status.** The deployment artifacts are complete and verified — `render.yaml`,
> `Dockerfile`, and a CI job that builds the image, runs it, and asserts health
> before any deploy is triggered. Creating the Render service requires signing in
> to a Render account, which is a manual step. The URLs above are filled in at
> that point, and this file is the only place that needs to change.

## Platform and shape

| | |
|---|---|
| Host | Render — Free instance type |
| Runtime | Docker (`Dockerfile` at the repository root) |
| Region | Oregon (`render.yaml`) |
| Services | **One.** Web app, agent orchestrator, MCP client, MCP server subprocess, RAG index, and mock data all run in a single container. |
| Health check path | `/healthz` |
| Workers | 1 uvicorn worker |

Only the LLM provider is external. Everything else — retrieval, tools, data —
is inside the container.

### Why one service

The free tier provides a single 512 MB instance. Deploying the MCP server
separately would add a second cold start to the critical path of the first
request and a second 512 MB allocation for a process that is idle most of the
time. The client supports an HTTP transport (`MCP_TRANSPORT=http`,
`MCP_SERVER_URL`), so the split is a configuration change if resources allow;
the deployed configuration uses a stdio subprocess.

### Why one worker

Each uvicorn worker owns its own MCP server subprocess and its own copy of the
embedding model. Measured steady-state footprint is 343 MB, so a second worker
would not fit in 512 MB. Concurrency within the single worker is handled by
`asyncio` — tool calls within a turn run in parallel.

## Environment variables

Set in the Render dashboard, never committed.

| Variable | Required | Value |
|---|---|---|
| `GROQ_API_KEY` | one of the two | Groq free-tier key. Primary provider. |
| `GEMINI_API_KEY` | one of the two | Google AI Studio free-tier key. Automatic fallback. |
| `LLM_PROVIDER` | no | `groq` (default) |
| `WARM_EMBEDDER` | set to `true` | Loads the embedder at startup rather than on the first user request. |
| `PORT` | no | Supplied by Render. |
| `LOG_LEVEL` | no | `INFO` |

With both keys set, the agent falls back transparently when the primary errors
or rate-limits, and the fallback is visible in the response trace. With neither,
the service still starts, `/healthz` returns 200, `/health` reports `degraded`,
and every non-LLM endpoint keeps working.

## Cold starts

**Free-tier instances spin down after roughly 15 minutes of inactivity.** The
next request wakes the instance, and the wake is visible to the user.

| Path | Expected | What is happening |
|---|---|---|
| Cold request (after spin-down) | **~50 s** | Render starts the container, the app boots, the MCP subprocess spawns, the index loads, and the embedder is warmed before the first query is served. |
| Warm request | **2–6 s** | Dominated entirely by LLM round trips — typically 2–4 model calls per agentic task. |
| `GET /healthz` when warm | < 50 ms | No work beyond a process-alive check. |

Two design choices exist specifically to bound the cold path:

- **The embedding weights are baked into the Docker image.** Startup performs no
  network I/O and cannot fail on a model download. Without this, a cold start
  would also include a ~133 MB fetch.
- **`WARM_EMBEDDER=true` loads the model during startup**, not on the first
  query. The ~10 s / ~164 MB load happens while Render is still bringing the
  service up, rather than landing on a live user.

**For the demo and for graders:** open `/healthz` once and wait for it to return
before using the chat UI. That absorbs the cold start on a URL where the wait is
obviously a platform wake-up rather than a slow agent. The demo video notes this
explicitly.

This is free-tier behaviour, not an application defect. A paid instance type with
no spin-down removes it entirely; no code change is involved.

## Health endpoints

Two endpoints that answer different questions, deliberately not merged:

**`GET /healthz` — liveness.** Returns `200` with `{"status": "alive"}` whenever
the process is running. Nothing else is checked. This is what Render polls.

**`GET /health` — readiness.** Checks that the RAG index is loaded, the MCP
session is live with its tools discovered, and a model provider is configured.
Returns `503` when any of those is missing.

```json
{
  "status": "ok",
  "as_of_date": "2026-09-12",
  "mcp": {
    "connected": true,
    "transport": "stdio",
    "server": "helios-hr v1.0.0",
    "tool_count": 12,
    "error": null
  },
  "rag_index": {
    "ok": true,
    "model": "BAAI/bge-small-en-v1.5",
    "chunks": 133,
    "documents": 12,
    "built_at": "2026-09-12T11:35:09Z"
  },
  "llm": {
    "providers_configured": ["groq"],
    "primary": "groq",
    "ok": true
  }
}
```

`/health` never raises. A readiness probe that returns 500 tells you nothing
about which dependency failed, which is the only reason to call it — so an
incomplete startup is reported as a degraded section rather than an exception.

Pointing the platform health check at `/health` would be a trap: a missing or
expired API key makes readiness fail, Render would read the 503 as a failed
deploy, and it would restart a perfectly healthy process indefinitely. Liveness
tells the platform whether to restart; readiness tells a human whether the
system can do its job.

## Verifying the deployment

```bash
BASE=https://<service>.onrender.com

curl -s $BASE/healthz                       # 200, immediately once awake
curl -s $BASE/health   | python -m json.tool # full readiness detail
curl -s $BASE/tools    | python -m json.tool # 12 MCP tools, discovered live
curl -s $BASE/documents| python -m json.tool # 12 documents, 133 chunks

curl -s -X POST $BASE/chat \
  -H 'content-type: application/json' \
  -d '{"message":"How much notice do I need for three days of PTO?","history":[]}' \
  | python -m json.tool
```

The `/chat` response carries the complete trace — every step, every tool name and
argument, every result, and the citations harvested from the tool output.

`/tools` is worth checking specifically: it renders the manifest the agent
discovered over MCP at runtime, not a hardcoded list, which is the simplest
demonstration that the protocol boundary is real.

## Deploying it yourself

1. Fork or push this repository to GitHub.
2. In Render: **New → Web Service**, connect the repository. `render.yaml` is
   detected and supplies runtime, region, plan, and health check path.
3. Add `GROQ_API_KEY` (and/or `GEMINI_API_KEY`) and `WARM_EMBEDDER=true` as
   environment variables.
4. Deploy. The first build takes roughly 5–8 minutes; most of it is installing
   dependencies and baking in the embedding weights.
5. Confirm `/health` returns `200` with `"status": "ok"`.

To enable automatic redeploys from CI, set the repository variable `APP_URL` to
the deployed URL and the repository secret `RENDER_DEPLOY_HOOK` to the Render
deploy hook. The `deploy` job in `.github/workflows/ci.yml` then runs on `main`
after both the test and container jobs pass, and verifies `/health` afterwards.
Without those two values the job is skipped, so the pipeline is green either way.
