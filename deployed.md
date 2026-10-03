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
| `LLM_MAX_CALLS_PER_TURN=8` | Per-turn cap on billable requests; refused requests are refunded |
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
2. If a demo access code is configured, enter it in the **Demo access code** field.
   Tick **Run live (skip cache)**, then use the two buttons under
   **Agentic demo tasks**.
3. The **Execution trace** panel opens with the final answer basis: outcome,
   selected tools, cited and retrieved policy sources, evidence check, escalation
   decision and model usage. Each step below it lists the tool calls with their
   arguments, an output summary and the full JSON. For PTO, review the ticket
   preview card and select **Create mock ticket**; the reply gives the ticket ID
   and reports zero model calls for the confirmation.
4. A rate limit is normally absorbed: the call moves to another model, or waits
   up to a minute. If an answer still reports one, wait before retrying.
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
full responses, with per-run timing, model calls and tokens, under a new
timestamped evidence filename. `--task` and `--case` are repeatable; the run
stops at the first provider or HTTP error instead of spending more quota.

## Cold starts and expected waits

Free instances may sleep after roughly 15 minutes idle. The historical cold-start
measurement was **52.5 seconds**; allow a minute or more rather than assuming
this remains an exact bound.

Warm chat is **not a 2-6 second operation**. On 3 October 2026, eight
back-to-back requests to build `291b50a` took a median of 15.1 seconds and at
most 46.9 seconds, the PTO demonstration, with 95.5 seconds in total spent
waiting on provider rate limits. September's deployed tasks took about 189-207
seconds. Health response time is a different metric and cannot stand in for
agent latency.
