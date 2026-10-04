# Helios HR Assistant

An agentic HR assistant for the fictional **Helios Dynamics**: policy retrieval,
synthetic employee records, deterministic compliance checks, and reversible mock
actions, all accessed through a real Model Context Protocol (MCP) server.

**Application:** https://helios-hr-assistant-wz3c.onrender.com

**Readiness:** https://helios-hr-assistant-wz3c.onrender.com/health

**Design and evaluation:** [design-and-evaluation.md](design-and-evaluation.md)

**Deployment:** [deployed.md](deployed.md)

**AI tooling disclosure:** [ai-tooling.md](ai-tooling.md)

`/health.build_sha` reports the commit currently deployed.

The free instance sleeps when idle, so the first visit can take about a minute
while it wakes. See [Cold starts and expected waits](#cold-starts-and-expected-waits).

## At a glance

- 16 policy documents in Markdown, HTML, PDF and TXT, indexed into 221 chunks.
- 12 MCP tools covering policy search, employee records, compliance rules and mock actions.
- 30 evaluation cases and two canonical demo workflows.

Live evaluation scores recorded in September are **historical**: the older citation
metric measured retrieved-label presence rather than factual support. See the
design report for the current scoring method and results.

## Setup

Use **Python 3.12+**. From the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

On macOS/Linux, activate with `source .venv/bin/activate` and copy the template
using `cp .env.example .env`.

Set `GROQ_API_KEY` in `.env` or the host environment. A Groq tool-capable model
is required for multi-step workflows. Gemini's configured compatibility endpoint
is only available for non-tool calls; it is not a replacement for Groq here.
Never commit API keys, demo codes, or approval secrets.

Direct dependencies are pinned in `pyproject.toml`; transitive dependencies are
not locked.

## Run

```powershell
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000. The app starts and discovers the MCP subprocess.
All dates are evaluated against the clearly labelled **2026-09-12 synthetic
snapshot**, not today's date. This prevents the example answers drifting.

| Endpoint | Purpose |
|---|---|
| `GET /` | Chat with demo task buttons, cited source snippets, mock-action previews and an execution trace (answer basis, tool calls, arguments, outputs) |
| `GET /healthz` | Process liveness; no LLM call |
| `GET /health` | Configured dependencies, tool count, index, revision and local quota limits; no LLM call |
| `GET /tools` | Discovered MCP catalogue |
| `GET /documents` | Indexed corpus |
| `GET /demo-tasks` | Reproducible UI tasks |
| `POST /chat` | Message, optional user/assistant history, and optional `fresh`; answer, cited/retrieved labels, snippets, trace, usage and previews |
| `POST /actions/{id}/confirm` | Execute the exact stored mock preview, once, in its originating browser session; **zero LLM calls** |

`/health` checks configuration, not whether a provider key is valid or has quota.
Only a live chat can establish that. The UI shows cached results explicitly;
their trace/timings belong to the original run.

## API quota protection

- Unit/integration tests use scripted models and prohibit real chat-completion calls.
  Indexing and retrieval evaluation use the local embedding model, not an LLM API.
- Live evaluation and deployed-task scripts require explicit `--allow-live`.
- The app admits one chat at a time and rejects duplicate/concurrent submissions.
  The default hourly chat allowance is 12 requests.
- Provider requests are capped at **8 per turn, 60 per hour and 100 per day** by
  default. The deployed service allows 16 per turn and 16 agent steps, because
  demo runs on a fallback model can need more steps.
  A request the provider refuses before doing any work (rate limit, oversized
  request, retired model) costs no tokens and is refunded; anything it may have
  processed counts.
- A rate-limited model is passed over for the next model in the chain, which has
  its own per-minute budget. A call waits only when every model is rate-limited,
  for the soonest to recover (honouring `Retry-After`, at most 60 s per call).
  Longer limits are reported, not waited out. Retired models are skipped.
- Read-only identical requests within the same session/context may reuse a
  labelled **10-minute cache**. Mutating workflows and ticket listings are not cached.
  **Run live (skip cache)** bypasses the cache and uses quota.
- Set `DEMO_ACCESS_CODE` to protect a public deployment from casual quota consumption.
  Share it privately; users enter it in the **Demo access code** field, not the chat.
- Set `LLM_ENABLED=false` to disable all real model calls.
  Chat explicitly reports that calls are disabled; it does not fake an answer.

These are process-local application limits, **not provider quota readings**.
They reset on restart and cannot protect against other applications using the
same API key. Provider daily limits, and per-minute limits that outlast the wait,
can still interrupt a task.

## The two demo workflows

The canonical prompts live in `agent/demo_tasks.py` and are reused by the UI
and deployed verifier:

1. **International remote work:** Maya's 42-day Portugal request; read her
   rolling usage, check compliance, read POL-INTL-001 §2 The 30-Day Rule and
   POL-REMOTE-001 §6 Temporary Work Outside the Home Country, and explain,
   citing both, that 12 + 42 exceeds 30 by 24 days.
2. **PTO and ticket preview:** Jonas's three-day request; check 13 available hours
   against 24 requested and 6 days' notice against 10, read POL-PTO-001 §3.1
   Notice Requirements and §3.3 Insufficient Balance, show a mock ticket preview,
   and explain the manager/skip-level exception routes.
   Select **Create mock ticket** on the preview to create the in-memory ticket and show its ID.

Every policy claim, including alternatives, needs evidence. A preview is not a
completed write, and a tool call alone is not a correct answer.

## Offline checks

```powershell
pytest -q
ruff check .
python scripts/validate_mock_data.py
python scripts/check_rules.py
python scripts/healthcheck.py
python -m rag.ingest.build_index --check
python -m evaluation.run_retrieval_eval
```

The committed index avoids embedding downloads at application startup. When
ingestion code or policies change, rebuild locally:

```powershell
python -m rag.ingest.build_index
```

The embedding weights must already be cached for fully offline execution.

## Live evaluation (opt-in)

These commands make real LLM calls and consume quota:

```powershell
python -m evaluation.run_eval --allow-live --case C01
python -m evaluation.run_eval --allow-live --resume
python scripts/verify_deployed_tasks.py --allow-live --task pto --confirm-mock-actions
```

The last command performs a real agent turn and then explicitly authorizes the
mock ticket. It saves the complete response and confirmation result in `evidence/`
with each run's wall time, rate-limit waits, model calls and tokens. `--task` and
`--case` can be repeated, and the run stops at the first provider or HTTP error
rather than spending quota on requests that would fail the same way.
Current evaluation records contain scorer version, configuration/code fingerprint
and full traces. A checkpoint from another revision is rejected.

## Architecture

```text
Browser -> FastAPI -> agent orchestrator -> LLM provider
                         |
                      MCP client
                         | stdio JSON-RPC
                      MCP server
                      /        \
             local RAG index   synthetic HR records + rule engine
```

The folder is called `mcp_server/`, not `mcp/`, to avoid shadowing the SDK package.
All employee data is synthetic; tickets and email drafts disappear on restart.

## Deployment

The live service runs on Render Free as one Docker service with one worker. The
web app, agent, local vector index, synthetic data and stdio MCP server share the
container; only the LLM is external. GitHub Actions checks the code, the real MCP
transport, app startup and the container, then deploys **that exact commit**.
`render.yaml` disables Render's own auto-deploy, so only revisions that pass CI are deployed.

To deploy your own copy:

1. Fork the repository and create a Render **Blueprint** from `render.yaml`.
   Render builds the `Dockerfile`, which bakes the embedding model into the image,
   and the app listens on Render's `$PORT`.
2. In the Render dashboard, set `GROQ_API_KEY`. `GEMINI_API_KEY` and
   `DEMO_ACCESS_CODE` are optional; `render.yaml` supplies the other settings.
3. In GitHub, add the secret `RENDER_API_KEY` and the repository variables
   `RENDER_SERVICE_ID` (the `srv-…` ID) and `APP_URL` (the service URL).
4. Push to `main`. CI runs lint, tests, the MCP smoke test and the container check,
   triggers the Render deploy for that commit, and waits until `/health` reports it.

To run the production image locally with the settings from `.env`:

```powershell
docker build -t helios-hr .
docker run --rm -p 8000:8000 --env-file .env helios-hr
```

[deployed.md](deployed.md) lists every environment variable and the demo steps.

## Cold starts and expected waits

Render's free plan puts the service to sleep after about 15 minutes without
traffic, and the next request has to wake it. The measured cold start was
**52.5 seconds**, so allow about a minute. Before a demo, open `/healthz` and wait
for it to respond; neither health endpoint calls an LLM.

A multi-step agent task is slower than a single answer even when the service is
warm, because each model step waits for the provider's per-minute token budget.
On 3 October 2026, eight back-to-back requests to the deployed service took a
median of 15 seconds and at most 47 seconds. Sent about a minute apart, the two
demo tasks on build `c88530a` took 7 and 5 seconds;
[design-and-evaluation.md](design-and-evaluation.md#deployed-measurement-3-october-2026)
has the details.
