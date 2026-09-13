# Deployed Application

## URLs

| | |
|---|---|
| **Application** | <https://helios-hr-assistant-wz3c.onrender.com> |
| **Liveness** | <https://helios-hr-assistant-wz3c.onrender.com/healthz> |
| **Readiness** | <https://helios-hr-assistant-wz3c.onrender.com/health> |
| **Tool manifest** | <https://helios-hr-assistant-wz3c.onrender.com/tools> |
| **Corpus manifest** | <https://helios-hr-assistant-wz3c.onrender.com/documents> |

Render service `srv-daj9a0mk1f9s73coho1g`, Free instance type, Oregon.

> **Cold start.** The instance sleeps after ~15 minutes idle. Open `/healthz`
> once and let it return before using the chat UI — see
> [Cold starts](#cold-starts) for the measured numbers.

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

Set in the Render dashboard, never committed. `.env.example` documents the same
set for local use.

| Variable | Required | Value on the deployed service | Why |
|---|---|---|---|
| `GROQ_API_KEY` | one of the two | Groq free-tier key | Primary provider. |
| `GEMINI_API_KEY` | one of the two | Google AI Studio free-tier key | Cross-provider fallback. |
| `LLM_PROVIDER` | no | `groq` | Which provider leads the chain. |
| `GROQ_FALLBACK_MODELS` | no | `openai/gpt-oss-20b,qwen/qwen3.8-27b,qwen/qwen3.6-27b` | Groq meters tokens **per model**, so each extra model is a fresh per-minute budget. This is what keeps a multi-step run alive on the free tier. |
| `LLM_MAX_TOKENS` | no | `1200` | Groq reserves this against the same per-minute bucket whether the completion uses it or not, so it is a prompt-budget decision, not just an output cap. |
| `CONTEXT_TOKEN_BUDGET` | no | `6200` | Ceiling on estimated prompt tokens per agent step. With `LLM_MAX_TOKENS` it must clear Groq's 8,000 TPM limit: 6,200 + 1,200 = 7,400. |
| `WARM_EMBEDDER` | yes | `true` | Loads the embedder during startup rather than on the first user request. |
| `FASTEMBED_CACHE_PATH` | yes | `/app/.fastembed_cache` | Points at the weights baked into the image, in a writable location. |
| `MCP_TRANSPORT` | no | `stdio` | The MCP server runs as a subprocess of the web app. |
| `PYTHONUNBUFFERED` | no | `1` | Makes logs appear in Render's stream immediately. |
| `LOG_LEVEL` | no | `INFO` | |
| `PORT` | no | supplied by Render | |

With both API keys set, the agent walks a chain of four Groq models and then
Gemini, and whichever model answered is reported in the response trace. With
neither, the service still starts, `/healthz` returns 200, `/health` reports
`degraded`, and every non-LLM endpoint keeps working.

### Why the free tier needs three of these

Groq's free tier meters **8,000 tokens per minute**, and that bucket covers the
prompt *and* the completion together. Every agent step resends the full tool
manifest plus every prior tool result, so an unbounded multi-step run grows
past the limit and stalls — and a request larger than the bucket can never
succeed, because retrying only waits for a refill that is already big enough
and switching model relocates the identical failure.

Measured fixed cost per step (`python scripts/measure_context.py`):

| Component | Tokens |
|---|---|
| System prompt | ~595 |
| 12-tool MCP manifest (resent every step) | ~2,709 |
| **Fixed floor** | **~3,304** |

`CONTEXT_TOKEN_BUDGET` bounds the prompt, `LLM_MAX_TOKENS` bounds what is
reserved for the answer, and `GROQ_FALLBACK_MODELS` multiplies the available
budget across models. Together they are the difference between a multi-step
task completing and timing out. See `design-and-evaluation.md` for the full
analysis.

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
  service up, rather than landing on a live user. The warmup runs in the **MCP
  server subprocess**, which is the only process that embeds queries; the web
  app never does, so loading the weights there as well would put a second
  ~164 MB copy into a 512 MB instance and exhaust it.

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
    "chunks": 184,
    "documents": 16,
    "built_at": "2026-09-13T11:36:23Z"
  },
  "llm": {
    "providers_configured": ["groq", "gemini"],
    "primary": "groq",
    "ok": true
  }
}
```

*Captured from the live service.*

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
BASE=https://helios-hr-assistant-wz3c.onrender.com

curl -s $BASE/healthz                        # 200, immediately once awake
curl -s $BASE/health    | python -m json.tool # full readiness detail
curl -s $BASE/tools     | python -m json.tool # 12 MCP tools, discovered live
curl -s $BASE/documents | python -m json.tool # 16 documents, 184 chunks

curl -s -X POST $BASE/chat \
  -H 'content-type: application/json' \
  -d '{"message":"How much notice do I need for three days of PTO?","history":[]}' \
  | python -m json.tool
```

To reproduce the two graded end-to-end tasks against the live service:

```bash
python scripts/verify_deployed_tasks.py            # both tasks
python scripts/verify_deployed_tasks.py parental   # just one
```

It asserts each run is multi-step, uses more than one tool, returns citations,
and did **not** exhaust its step budget, then writes the full traces to
`evidence/deployed-tasks.json`.

The `/chat` response carries the complete trace — every step, every tool name and
argument, every result, and the citations harvested from the tool output.

`/tools` is worth checking specifically: it renders the manifest the agent
discovered over MCP at runtime, not a hardcoded list, which is the simplest
demonstration that the protocol boundary is real.

## Deploying it yourself

1. Fork or push this repository to GitHub.
2. In Render: **New → Web Service**, connect the repository. `render.yaml` is
   detected and supplies runtime, region, plan, and health check path.
3. Add `GROQ_API_KEY` (and/or `GEMINI_API_KEY`) plus `WARM_EMBEDDER=true` and
   `FASTEMBED_CACHE_PATH=/app/.fastembed_cache` as environment variables.
4. Deploy. The first build takes roughly 5–8 minutes; most of it is installing
   dependencies and baking in the embedding weights.
5. Confirm `/health` returns `200` with `"status": "ok"`.

### Continuous deployment

The `deploy` job in `.github/workflows/ci.yml` runs on `main` after both the
test and container jobs pass. It needs:

| Kind | Name | Value |
|---|---|---|
| Secret | `RENDER_API_KEY` | Render account API key |
| Variable | `RENDER_SERVICE_ID` | `srv-daj9a0mk1f9s73coho1g` |
| Variable | `APP_URL` | the deployed base URL |

The job calls Render's REST API rather than a deploy hook. That is deliberate:
Render only fires its own auto-deploy webhook when its GitHub App is installed
on the repository, and a plain deploy hook returns nothing identifying, so the
pipeline would have to *assume* the next healthy response belonged to its own
deploy. The API returns a deploy id, so CI polls **that** deploy to `live` and
only then asserts `/health` on the running service with `--require-llm`.

If the credentials are absent — on a fork, for example — the job emits a
GitHub warning annotation and succeeds rather than failing someone else's
build. The annotation matters: an earlier version logged a plain line, and a
run where the secrets had not yet been created reported a green "Deploy to
Render" that had silently deployed nothing.
