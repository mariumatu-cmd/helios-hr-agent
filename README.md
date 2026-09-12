# Helios HR Assistant

An agentic AI system that answers HR policy and operations questions for a
fictional company, **Helios Dynamics**. It combines retrieval over a policy
corpus with live tool calls against mock HR records, exposed through a **Model
Context Protocol (MCP)** server, and it cites every claim it makes.

The system is built to give a defensible answer to questions that a naive RAG
chatbot gets wrong, because answering them correctly requires *both* the written
policy *and* a specific employee's data:

> *"Maya Rodriguez wants to work from Portugal for six weeks, 5 October to 15
> November. Can she?"*

The policy allows 30 days abroad per rolling 12 months. Maya has 12 days already
used inside the window — not the 24 a careless reading of her travel history
suggests, because her Spain trip closed before the window opened. The agent must
retrieve the rule, look up the history, apply the rolling-window arithmetic, and
cite the section. The answer is **no, by 24 days** — and the useful part is the
alternatives it offers, which are only reachable by combining retrieval with
structured data.

**Deployed application:** see [`deployed.md`](deployed.md)
**Design rationale and evaluation results:** see [`design-and-evaluation.md`](design-and-evaluation.md)
**How AI coding tools were used:** see [`ai-tooling.md`](ai-tooling.md)

---

## What it does

| Capability | Where |
|---|---|
| Hybrid RAG (dense + BM25 + reciprocal rank fusion) over 12 policy documents in 4 file formats | `rag/` |
| 12 MCP tools over policy search, employee records, PTO, benefits, travel history, and ticketing | `mcp_server/` |
| Hand-written agent loop with full step-by-step tracing and provider fallback | `agent/` |
| FastAPI chat app that renders every tool call, argument, and result | `app/` |
| Two-store abstention gate that refuses questions the system genuinely cannot answer | `rag/vocabulary.py` |
| 28-case evaluation suite with a deterministic scorer, plus a 6-way retrieval ablation | `evaluation/` |

---

## Architecture at a glance

```
                    ┌──────────────────────────────────────────┐
  browser  ───────► │  FastAPI web app        (app/main.py)    │
                    │  chat UI + /chat + /health + /healthz     │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
                    │  Agent orchestrator (agent/orchestrator) │
                    │  plan → call tools → observe → answer    │
                    └──────┬──────────────────────┬────────────┘
                           │                      │
                  ┌────────▼────────┐   ┌─────────▼──────────────┐
                  │ LLM provider     │   │ MCP client            │
                  │ Groq → Gemini    │   │ (agent/mcp_client.py) │
                  │ automatic fallback│  └─────────┬─────────────┘
                  └──────────────────┘             │ stdio (JSON-RPC)
                                                   │
                                    ┌──────────────▼─────────────┐
                                    │ MCP server (mcp_server/)   │
                                    │ 12 tools, annotated        │
                                    └───┬────────────────┬───────┘
                                        │                │
                              ┌─────────▼──────┐  ┌──────▼────────────┐
                              │ RAG index      │  │ Mock HR data      │
                              │ numpy + BM25   │  │ 6 JSON datasets   │
                              │ 133 chunks     │  │ + rule engine     │
                              └────────────────┘  └───────────────────┘
```

Everything runs inside **one process tree**, so it fits a single free-tier web
service. The MCP server is a genuine child process speaking JSON-RPC over stdio —
not an in-process function call dressed up as a protocol.

Full detail, including transport rationale and tool schemas, is in
[`design-and-evaluation.md`](design-and-evaluation.md).

---

## Setup

Requires **Python 3.11+** (developed on 3.12).

```bash
git clone <this repo>
cd helios-hr-assistant

python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### Configure a model provider

Copy the template and add at least one key. Both providers have free tiers.

```bash
cp .env.example .env
```

| Variable | Where to get it | Notes |
|---|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys | Primary. Fast, free tier. |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | Automatic fallback. |

Set either one, or both. With both configured the agent falls back
transparently when the primary provider errors or rate-limits, and the fallback
is recorded in the trace so you can see it happened. With neither, the app still
starts and every non-LLM endpoint works — `/health` reports `degraded` and the
chat endpoint says plainly that no model is configured.

### Build the retrieval index

The index is committed, so this is only needed if you change the corpus:

```bash
python -m rag.ingest.build_index
```

It parses all 12 documents (Markdown, HTML, plain text, and PDF), chunks them on
heading boundaries, embeds them with `BAAI/bge-small-en-v1.5` running locally via
ONNX, and writes `rag/index/`. Takes about 30 seconds. Deterministic — the same
corpus always produces the same 133 chunks, verified by a fingerprint in
`rag/index/index_info.json`.

---

## Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000. The app starts its own MCP server subprocess, so
there is nothing else to launch.

Four demo questions are seeded in the UI, including one the system is expected
to **refuse**. Every response shows the full trace: which tools ran, with which
arguments, what came back, and which policy sections were cited.

### Useful endpoints

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Liveness. Always 200 while the process is up. Used by the host. |
| `GET /health` | Readiness. 503 when the index or a model provider is missing. |
| `GET /tools` | The live MCP tool manifest, as discovered from the server. |
| `GET /documents` | The indexed corpus. |
| `POST /chat` | `{"message": "...", "history": []}` → answer plus full trace. |

### Running the MCP server on its own

The server is a standard MCP server and works with any MCP client, including
Claude Desktop and MCP Inspector:

```bash
python mcp_server/hr_server.py          # stdio
python scripts/mcp_smoketest.py         # connect, list tools, call a few
```

To run it as a separate HTTP service instead of a subprocess, set
`MCP_TRANSPORT=http` and `MCP_SERVER_URL`. The client supports both; stdio is the
default because it keeps the free-tier deployment to a single service.

---

## Tests and checks

```bash
pytest -q                                # 143 tests, ~50 s
ruff check .                             # lint
python scripts/validate_mock_data.py     # referential integrity of the mock data
python scripts/check_rules.py            # hand-verified rule-engine edge cases
python scripts/healthcheck.py            # end-to-end smoke test, no API key needed
```

The test suite covers ingestion determinism, retrieval and abstention, all 12
MCP tools over a real client session, the rule engine's date arithmetic, the
agent loop against a scripted model, and the HTTP layer.

---

## Evaluation

```bash
# No API key required -- retrieval quality and a 6-way ablation
python -m evaluation.run_retrieval_eval

# Requires an API key -- full agent evaluation over 28 cases
python -m evaluation.run_eval
python -m evaluation.run_eval --category refusal     # one slice
python -m evaluation.run_eval --min-pass-rate 0.85   # gate for CI
```

Results, methodology, and an honest reading of what the numbers do and do not
show are in [`design-and-evaluation.md`](design-and-evaluation.md). Raw output
lands in `evaluation/results/`.

---

## Deployment

The repository deploys as a **single Docker service on Render's free tier**
(`render.yaml` + `Dockerfile`). The image bakes in the embedding model weights so
that startup does no network I/O.

```bash
docker build -t helios-hr .
docker run -p 8000:8000 -e GROQ_API_KEY=... helios-hr
```

Measured memory is **343 MB against the 512 MB free-tier cap**
(`evaluation/results/memory_footprint.txt`).

Free-tier instances spin down after ~15 minutes idle; the first request
afterwards takes roughly 50 seconds. See [`deployed.md`](deployed.md) for the
live URL and the cold-start details.

---

## Repository layout

```
agent/           orchestrator, MCP client, LLM providers with fallback, prompts
app/             FastAPI application, chat UI, static assets
corpus/          12 HR policy documents in md / html / txt / pdf
mcp_server/      MCP server, 12 tool definitions, and the HR rule engine
mock_data/       6 JSON datasets: employees, PTO, benefits, travel, offices, tickets
rag/             chunking, indexing, hybrid retrieval, abstention vocabulary
evaluation/      26 scored cases, deterministic scorer, harnesses, results
scripts/         calibration, diagnostics, validation, smoke tests
tests/           143 tests
```

## A note on the `mcp_server/` directory name

It is not called `mcp/`. The official MCP Python SDK installs a top-level package
with that exact name, and a local `mcp/` directory shadows it on `sys.path`, so
`from mcp.server import ...` would import this project instead of the SDK. The
assignment suggests `mcp/ or equivalent`; this is the equivalent.
