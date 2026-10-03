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
| `GET /` | Chat, canonical demo buttons, source snippets and operational trace |
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
- Actual provider HTTP attempts, including failures and fallbacks, are capped at
  **8 per turn, 60 per hour and 100 per day**. No automatic same-model 429 retry.
- Read-only identical requests within the same session/context may reuse a
  labelled **10-minute cache**. Mutating workflows and ticket listings are not cached.
  **Force live call** bypasses the cache and uses quota.
- Set `DEMO_ACCESS_CODE` to protect a public deployment from casual quota consumption.
  Share it privately and enter it in the UI, not the chat.
- Set `LLM_ENABLED=false` to disable all real model calls.
  Chat explicitly reports that calls are disabled; it does not fake an answer.

These are process-local application limits, **not provider quota readings**.
They reset on restart and cannot protect against other applications using the
same API key. Token-per-minute and provider daily limits can still interrupt a task.

## The two demo workflows

The canonical prompts live in `agent/demo_tasks.py` and are reused by the UI
and deployed verifier:

1. **International remote work:** Maya's 42-day Portugal request; retrieve
   international and remote-work policies, inspect her profile and rolling usage,
   check compliance, and explain that 12 + 42 exceeds 30 by 24 days.
2. **PTO and ticket preview:** Jonas's three-day request; retrieve PTO policy,
   check 13 available hours against 24 requested and 6 days' notice against 10,
   explain manager/skip-level exception routes, and show a mock ticket preview.
   Click its confirmation button to create the in-memory ticket and show its ID.

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
mock ticket. It saves the complete response and confirmation result in `evidence/`.
Current evaluation records contain scorer version, configuration/code fingerprint
and full traces. A checkpoint from another revision is rejected.

## Architecture and deployment

```text
Browser -> FastAPI -> agent orchestrator -> LLM provider
                         |
                      MCP client
                         | stdio JSON-RPC
                      MCP server
                      /        \
             local RAG index   synthetic HR records + rule engine
```

Render runs one Docker service and one worker. GitHub Actions checks the code,
real MCP transport, app startup and container, then deploys **that exact commit**.
`render.yaml` disables Render's own auto-deploy, so only revisions that pass CI are deployed.

The folder is called `mcp_server/`, not `mcp/`, to avoid shadowing the SDK package.
All employee data is synthetic; tickets and email drafts disappear on restart.
