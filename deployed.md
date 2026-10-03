# Deployment and demonstration

| Endpoint | URL |
|---|---|
| Application | https://helios-hr-assistant-wz3c.onrender.com |
| Readiness | https://helios-hr-assistant-wz3c.onrender.com/health |
| Liveness | https://helios-hr-assistant-wz3c.onrender.com/healthz |
| Discovered tools | https://helios-hr-assistant-wz3c.onrender.com/tools |
| Corpus | https://helios-hr-assistant-wz3c.onrender.com/documents |

The fixes in this branch are **not deployed merely because these URLs respond**.
After merging, compare `/health.build_sha` to the commit GitHub Actions tested.

## Runtime

Render Free, Docker, Oregon; one service and one uvicorn worker. The web app
owns a long-lived stdio MCP subprocess. Local embeddings, index and synthetic
records stay inside the container; the LLM is external.

The Dockerfile bakes the embedding weights into `/opt/fastembed_cache`, owned by
the non-root runtime user. Both Docker and `render.yaml` use that path.
`WARM_EMBEDDER=true` warms the MCP process, not another copy in the web process.

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
| `DEMO_ACCESS_CODE` | Optional quota-protection code; share privately with grader |
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

`render.yaml` disables independent auto-deployment. **For an existing service,
confirm Auto-Deploy is also disabled in the Render dashboard**, otherwise its
previous setting can still bypass GitHub's test gate. Do not manually deploy
untested revisions for the recording.

## Recording without exhausting quota

1. Complete offline checks before the recording. Do not run the 30-case live suite.
2. Check the provider's own dashboard for remaining quota. Local counters cannot
   see account-wide usage or consumption by another application.
3. Open `/healthz` and wait for the platform to wake. Check `/health` and its SHA.
   Neither endpoint calls an LLM.
4. Enter the optional demo code. Select **Force live call** for the recorded run,
   then use the two numbered canonical demo buttons once each.
5. Show the tool names, exact arguments/results, source snippets and answer.
   For PTO, inspect the preview and click its confirmation button. Show the
   created mock ticket ID and `api_calls: 0` for confirmation.
6. Do not rerun immediately after a 429. Allow the provider window to recover.
   Keep any recorded run's errors visible; do not describe cached evidence as live.

The UI disables concurrent submissions. Identical read-only requests can use a
labelled ten-minute session cache; previews, writes and mutable ticket reads
do not. A new conversation clears conversational context, not provider quota.
`LLM_ENABLED=false` is useful for checking the UI and no-model error path, but
does not simulate a successful live demo.

Example opt-in verifier, only when spare quota is available:

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
These are old-code measurements, and the revised code has not yet been measured
live to avoid consuming the recording quota. Health response time is a different
metric and cannot stand in for agent latency.

Budget for those waits in the required 7-10-minute recording. Explain platform
and quota effects accurately rather than silently replaying cached results.

## Submission checks

The recording link and grader access must be supplied/confirmed by the owner.
The prior read-invitation statement could not be verified with the review token.
Keep all identity verification out of the public repository. No email is sent
by this app, and all created tickets/drafts disappear on process restart.
