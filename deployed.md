# Deployment and demonstration

| Endpoint | URL |
|---|---|
| Application | https://helios-hr-assistant-wz3c.onrender.com |
| Readiness | https://helios-hr-assistant-wz3c.onrender.com/health |
| Liveness | https://helios-hr-assistant-wz3c.onrender.com/healthz |
| Discovered tools | https://helios-hr-assistant-wz3c.onrender.com/tools |
| Corpus | https://helios-hr-assistant-wz3c.onrender.com/documents |

`/health.build_sha` reports the deployed commit, which CI checks against the
commit it tested.

## Runtime

Render Free, Docker, Oregon; one service and one uvicorn worker. The web app
owns a long-lived stdio MCP subprocess. Local embeddings, index and synthetic
records stay inside the container; the LLM is external.

The Dockerfile bakes the embedding weights into `/opt/fastembed_cache`, owned by
the non-root runtime user. Both Docker and `render.yaml` use that path.
`WARM_EMBEDDER=true` warms the MCP process, not another copy in the web process.

No email is sent by this app, and created tickets and drafts disappear on
process restart.

## Configuration

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Required for the multi-step agent |
| `GROQ_MODEL`, `GROQ_FALLBACK_MODELS` | Tool-capable model chain; see `.env.example` |
| `GEMINI_API_KEY` | Optional non-tool provider; not a reliable tool-loop fallback |
| `MCP_TRANSPORT=stdio` | Single-container MCP |
| `WARM_EMBEDDER=true` | Warm local model at startup |
| `FASTEMBED_CACHE_PATH=/opt/fastembed_cache` | Baked model cache |
| `CONTEXT_TOKEN_BUDGET=6200` | Estimated prompt ceiling |
| `LLM_MAX_TOKENS=1200` | Completion ceiling |
| `LLM_ENABLED` | Set false to stop all real model calls |
| `LLM_MAX_CALLS_PER_TURN=8` | Actual HTTP attempt cap per turn |
| `LLM_MAX_CALLS_PER_HOUR=60` | Process-local hourly cap |
| `LLM_MAX_CALLS_PER_DAY=100` | Process-local daily cap |
| `CHAT_MAX_REQUESTS_PER_HOUR=12` | Shared endpoint limit |
| `DEMO_ACCESS_CODE` | Optional quota-protection code; share privately |
| `RENDER_GIT_COMMIT` | Render-provided revision, exposed as `build_sha` |

Never put secrets in repository files or model prompts. Stdio automatically
generates a private approval secret for its child process. Separate HTTP
services require the same `MCP_APPROVAL_SECRET` on both sides.

## CI-gated deployment

GitHub Actions needs secret `RENDER_API_KEY`, repository variable
`RENDER_SERVICE_ID`, and repository variable `APP_URL`.

The `quality` and `container` jobs must succeed before `deploy`. The Render
request specifies the tested `GITHUB_SHA`; CI polls that deploy ID, then checks
readiness and exact revision. Missing deployment configuration fails explicitly.

`render.yaml` disables Render's independent auto-deployment, so only revisions
that pass CI are deployed.

## Running the live demonstration

1. Open `/healthz` and wait for the platform to wake, then check `/health` and
   its SHA. Neither endpoint calls an LLM.
2. Enter the demo code if one is configured. Select **Force live call**, then use
   the two numbered canonical demo buttons.
3. The trace shows the tool names, exact arguments and results, source snippets
   and the answer. For PTO, the preview's confirmation button creates the mock
   ticket and reports its ID with `api_calls: 0`.
4. After a 429, wait for the provider window to recover before retrying.
   Cached results are labelled and are never presented as live runs.

Local counters cannot see account-wide usage or consumption by another
application; the provider's dashboard shows remaining quota.

The UI disables concurrent submissions. Identical read-only requests can use a
labelled ten-minute session cache; previews, writes and mutable ticket reads
do not. A new conversation clears conversational context, not provider quota.
`LLM_ENABLED=false` is useful for checking the UI and no-model error path, but
does not simulate a successful live demo.

The opt-in verifier makes real LLM calls and consumes quota:

```powershell
python scripts/verify_deployed_tasks.py --allow-live --task international
python scripts/verify_deployed_tasks.py --allow-live --task pto --confirm-mock-actions
```

Set `DEMO_ACCESS_CODE` in the shell if the deployed service requires it.
The verifier preserves cookies for the separate confirmation request and saves
full responses under a new timestamped evidence filename.

## Cold starts and expected waits

Free instances may sleep after roughly 15 minutes idle. The historical cold-start
measurement was **52.5 seconds**; allow a minute or more rather than assuming
this remains an exact bound.

Warm chat is **not a 2-6 second operation**. Historical deployed tasks took
about 189-207 seconds; the historical evaluation p50/p95 were 65.4/248.2 seconds.
These measurements predate the current code, whose live latency has not been
measured. Health response time is a different metric and cannot stand in for
agent latency.
