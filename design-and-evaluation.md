# Design and Evaluation

Helios HR Assistant — an agentic RAG system for HR policy and operations.

This document explains what was built, **why each choice was made over the
alternatives**, and what the measurements actually show — including the places
where the measurements did not support the design and the design had to change.

---

## Contents

1. [System architecture](#1-system-architecture)
2. [The corpus and the mock data](#2-the-corpus-and-the-mock-data)
3. [RAG design](#3-rag-design)
4. [Abstention: knowing when not to answer](#4-abstention-knowing-when-not-to-answer)
5. [MCP server design](#5-mcp-server-design)
6. [Agent orchestration](#6-agent-orchestration)
7. [Safety guardrails](#7-safety-guardrails)
8. [Deployment architecture](#8-deployment-architecture)
9. [The two demo tasks](#9-the-two-demo-tasks)
10. [Evaluation](#10-evaluation)
11. [Limitations and what I would do next](#11-limitations-and-what-i-would-do-next)

---

## 1. System architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Render free-tier web service (single container)                        │
│                                                                         │
│   browser                                                               │
│      │  HTTP                                                            │
│      ▼                                                                  │
│   ┌──────────────────────────────────────────────┐                      │
│   │ FastAPI app            app/main.py           │                      │
│   │  GET  /            chat UI                   │                      │
│   │  POST /chat        question → answer + trace │                      │
│   │  GET  /healthz     liveness  (always 200)    │                      │
│   │  GET  /health      readiness (503 degraded)  │                      │
│   │  GET  /tools       live MCP tool manifest    │                      │
│   │  GET  /documents   indexed corpus            │                      │
│   └───────────────────┬──────────────────────────┘                      │
│                       │                                                 │
│   ┌───────────────────▼──────────────────────────┐                      │
│   │ Agent orchestrator  agent/orchestrator.py    │                      │
│   │  loop: LLM → tool calls → observe → answer   │      ┌─────────────┐ │
│   │  max 8 steps, parallel tool execution        │─────►│ LLM provider│ │
│   │  emits a complete Trace                      │      │ agent/llm.py│ │
│   └───────────────────┬──────────────────────────┘      │             │ │
│                       │                                 │  Groq       │─┼──► api.groq.com
│   ┌───────────────────▼──────────────────────────┐      │    ↓ fallback│ │
│   │ MCP client          agent/mcp_client.py      │      │  Gemini     │─┼──► generativelanguage
│   │  initialize → tools/list → tools/call        │      └─────────────┘ │      .googleapis.com
│   │  converts MCP schemas → OpenAI tool schemas  │                      │
│   └───────────────────┬──────────────────────────┘                      │
│                       │  JSON-RPC 2.0 over stdio                        │
│   ┌───────────────────▼──────────────────────────┐                      │
│   │ MCP server (child process)  mcp_server/      │                      │
│   │   hr_server.py   12 tools, annotated         │                      │
│   │   data.py        HR rule engine              │                      │
│   └────────┬──────────────────────────┬──────────┘                      │
│            │                          │                                 │
│   ┌────────▼──────────────┐  ┌────────▼─────────────────┐               │
│   │ RAG index  rag/index/ │  │ Mock HR data  mock_data/ │               │
│   │  embeddings.npy 184×384│  │  employees.json          │               │
│   │  bm25.pkl              │  │  pto_balances.json       │               │
│   │  metadata.jsonl        │  │  benefits_elections.json │               │
│   │                        │  │  international_work_*.json│              │
│   │ retrieval rag/retrieve │  │  offices.json            │               │
│   │  dense + BM25 + RRF    │  │  hr_tickets.json         │               │
│   │  + abstention gates    │  │                          │               │
│   └────────────────────────┘  └──────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

**The separation that matters.** The agent has no import path to the data. It
cannot read `mock_data/`, cannot call `retrieve.search()`, and does not know what
tools exist until it asks the MCP server. Its entire view of the world is the
tool manifest returned by `tools/list`. That is not decoration — it is what makes
the MCP layer real rather than a wrapper around local function calls, and it is
why adding a tool to the server changes the agent's behaviour with no change to
the agent.

The layers, and what each is allowed to know:

| Layer | Knows about | Does not know about |
|---|---|---|
| `app/` | the orchestrator, the MCP client lifecycle | tools, retrieval, data |
| `agent/` | the MCP protocol, the LLM API | tool names, policy text, employee records |
| `mcp_server/` | retrieval, mock data, HR rules | the agent, the LLM, HTTP |
| `rag/` | the corpus | employees, tools, the agent |

---

## 2. The corpus and the mock data

### Corpus

Twelve HR policy documents for the fictional **Helios Dynamics**, written for
this project, totalling 184 chunks.

| Document | ID | Format |
|---|---|---|
| PTO policy | `POL-PTO-001` | Markdown |
| Remote work policy | `POL-REMOTE-001` | Markdown |
| International remote work | `POL-INTL-001` | **HTML** |
| Parental and medical leave | `POL-LEAVE-002` | Markdown |
| Benefits eligibility | `POL-BEN-001` | Markdown |
| Travel and expense policy | `POL-EXP-001` | **PDF** |
| Equipment and asset policy | `POL-EQUIP-001` | **plain text** |
| Performance and promotion | `POL-PERF-001` | Markdown |
| Code of conduct | `POL-CONDUCT-001` | Markdown |
| Information security | `POL-SEC-001` | Markdown |
| Onboarding guide | `POL-ONBOARD-001` | Markdown |
| Holiday schedule | `POL-HOL-001` | Markdown |

The four formats are deliberate. A corpus of uniform Markdown would let the
ingestion pipeline get away with a single code path; PDF loses heading structure,
HTML carries markup that must be stripped without losing section boundaries, and
plain text has no machine-readable structure at all. `rag/ingest/parse.py`
normalises all four into the same `{doc_id, section, heading, text}` shape, and
`tests/test_ingestion.py` asserts every document produced sections with headings.

The documents are written to make questions *hard in a realistic way*: rules
interact. International remote work depends on tenure, on visa class, and on a
rolling 12-month day count. Benefits eligibility depends on employment class and
hours. PTO approval depends on balance, notice period, and blackout dates. No
single passage answers a realistic question.

### Mock structured data

Six JSON datasets, 20 employees, generated with a fixed seed and validated for
referential integrity by `scripts/validate_mock_data.py`.

`mcp_server/data.py` is not a lookup layer — it is a **rule engine**. It
implements the arithmetic the policies specify: rolling-window day counts,
business-day notice calculations, blackout-date intersection, tenure thresholds,
pro-rata accrual. Every verdict it returns carries the citation of the rule that
produced it, so a citation in the final answer is traceable to a specific
computation rather than to the model's prose.

Some deliberately constructed edge cases, each hand-verified in
`scripts/check_rules.py`:

| Employee | Situation | The trap |
|---|---|---|
| Maya Rodriguez `E-1041` | 24 days of international travel on record | Only **12** fall inside the rolling window — the Spain trip closed before the window opened |
| Jonas Weber `E-1088` | Requests 3 days off | Has 13 h available against 24 h requested **and** only 6 business days' notice against the required 10 — two independent failures |
| Marcus Doyle `E-1055` | Wants fully remote | An active PIP blocks fully-remote but **not** hybrid — so the answer is a conditional yes, not a no |
| Tomas Silva `E-1120` | Wants to work abroad | Fails on tenure **and** on H-1B status — two rules, not one |
| Sofia Marino `E-1099` | Asks about disability cover | Part-time, so no STD/LTD — an eligibility rule, not a balance |

These exist so the evaluation can distinguish a system that retrieved the right
policy from one that also applied it correctly. A model that answers "24 days" for
Maya has done the retrieval perfectly and the reasoning wrong.

---

## 3. RAG design

### Chunking

**Heading-aware with sentence-aligned overlap.** Target 320 tokens, hard cap 420,
overlap 60.

The alternative — fixed-size sliding windows — was rejected because this corpus
is *structurally* organised and the citation requirement depends on that
structure. An answer must cite `POL-PTO-001 §3.1 Notice Requirements`. A chunk
that straddles §3.1 and §3.2 cannot produce a correct citation, because it is
genuinely ambiguous which section a given sentence came from. Chunking on heading
boundaries makes `section` a property of the chunk rather than a guess.

Sections shorter than the target are merged with their neighbours (so a two-line
§1.1 does not become its own low-signal chunk), and sections over the cap are
split at sentence boundaries with 60 tokens of overlap carried across, so a rule
whose statement spans the split is retrievable from either half.

Chunking is deterministic and fingerprinted: `rag/index/index_info.json` records
a SHA-256 of the corpus and the exact parameters used, and a test rebuilds and
compares.

### Embeddings

**`BAAI/bge-small-en-v1.5`**, 384 dimensions, run locally through `fastembed`'s
ONNX runtime.

Chosen over a hosted embedding API for three reasons, in order of importance:

1. **No network dependency at query time.** A hosted embedder would put a
   third-party API on the critical path of every single query, including the
   ablations and the evaluation. The system would be untestable without a key.
2. **Free-tier fit.** 133 MB of weights, resident once. Measured cost of the
   loaded model is +164 MB RSS (§8), which fits.
3. **Determinism.** The index is byte-reproducible, which is what lets the
   fingerprint test be meaningful.

384 dimensions over 768 was a size trade: 184 × 384 floats is 276 KB, and on a
corpus this small the retrieval measurements (§10) show recall@6 of 1.00 — there
is no headroom that a larger model could recover.

### Vector store

**A NumPy array.** Not Chroma, not FAISS.

184 vectors × 384 dimensions is a 276 KB matrix. An exhaustive cosine similarity
over it is a single `numpy.dot` taking well under a millisecond. An approximate
nearest-neighbour index exists to avoid exhaustive search; below roughly 10⁴
vectors it adds a dependency, a build step, a persistence format, and an
approximation error in exchange for optimising something that is already free.

This is a scale-appropriate choice, not a shortcut, and it is the honest answer
for the corpus in question. `rag/retrieve.py` isolates it behind `search()`, so
the swap to FAISS is a single function if the corpus grows by two orders of
magnitude.

### Hybrid retrieval

Dense similarity and BM25 are fused with **reciprocal rank fusion** (`k=60`):

```
score(d) = Σ  1 / (60 + rank_i(d))
          i ∈ {dense, bm25}
```

RRF over score normalisation because the two scores are not commensurable: cosine
is bounded in [-1, 1] and BM25 is unbounded and corpus-dependent. Any weighted sum
requires a normalisation constant that must be re-tuned whenever the corpus
changes. RRF consumes only ranks, so it has one constant and that constant is
famously insensitive.

**What the measurement actually showed** (§10): on 16 graded queries, hybrid and
dense-only are *identical* — recall@1 0.88, recall@6 1.00, MRR 0.938. BM25 alone
is worse. So **hybrid retrieval does not earn its place through recall on this
corpus**, and reporting it as a win would misrepresent the data.

It is kept for two reasons that the graded set is too small to show but that are
structurally true:

- **Exact identifiers.** Queries like `POL-INTL-001`, `H-1B`, `§3.1`, `E-1041`
  are lexical, not semantic. A dense embedder maps `H-1B` and `L-1` into nearly
  the same region; BM25 does not. The graded set has too few identifier queries
  to move the aggregate, but the failure mode is real and the cost of insurance
  is near zero.
- **The BM25 vocabulary is load-bearing elsewhere.** The abstention gate (§4) is
  built on the corpus term vocabulary that the BM25 index already maintains.
  Removing BM25 would mean building that vocabulary separately.

### Retrieval k

`k=6`, from a pool of 50 candidates per retriever before fusion.

Measured recall@6 is 1.00 and recall@1 is 0.88, so the correct passage is
essentially always inside 6 but not always at 1 — which is exactly the regime
where giving the model several passages helps. Below k=3 recall falls; above
k≈8 the added passages are consistently irrelevant and only dilute the context.

### A reranker was considered and rejected

The obvious next step for the 0.88→1.00 gap between recall@1 and recall@6 is a
cross-encoder reranker. It was not added:

- recall@6 is already **1.00**. A reranker improves *ordering* within the
  returned set, and the model reads all six passages. There is no measured error
  for it to fix.
- A cross-encoder is a second transformer, roughly +90 MB resident and roughly
  +200 ms per query. Against a measured 343 MB of a 512 MB cap, that is a real
  cost.
- Adding a component that cannot be shown to improve a measured metric, on the
  grounds that it is a recognised best practice, is the failure mode this project
  is supposed to demonstrate awareness of.

---

## 4. Abstention: knowing when not to answer

This is the most consequential correctness work in the project, and it is
documented in full in
[`evaluation/results/abstention_analysis.md`](evaluation/results/abstention_analysis.md).
Summarised here because the conclusion changed the design.

**The first design was a cosine similarity floor**, calibrated — not guessed — on
15 in-corpus and 10 out-of-corpus questions. They separated cleanly (in-corpus
min 0.688, out-of-corpus max 0.620) and the midpoint 0.65 gave zero errors.

**It did not generalise.** On an independently written negative set of *plausible
but non-existent* HR benefits, the 0.65 floor false-accepted **7 of 8**. "What
does the Helios pet insurance policy cover?" scores **0.771** — higher than ten of
the sixteen genuine evaluation questions.

This is the embedder working correctly. Cosine measures topical adjacency, and a
fabricated HR benefit is topically adjacent to real HR policy by construction.
The original calibration set was unrepresentative: its negatives were about
*unrelated subjects*, so it measured an easy distinction rather than the one that
matters. Positives span [0.654, 0.836] and the hard negatives [0.590, 0.771] —
**the ranges overlap, so no threshold exists.** BM25 score and raw term coverage
overlap the same way (`scripts/diagnose_abstention.py`).

**The signal that does work is lexical, and it depends on the architecture.**
Out-of-corpus questions fail to match on *topic nouns* (sabbatical, tuition, pet,
stock, revenue). In-corpus questions fail only on *entity names* (Maya, Weber,
Brazil) and inflected forms.

This system has two knowledge stores: policy text reached by retrieval, and
employee data reached by tools. A term missing from the corpus means little — it
may live in the database. A term missing from **both** means no combination of
retrieval and tool calls can ground an answer. `rag/vocabulary.py` implements
exactly that test, with proper nouns exempted (an entity name is resolvable by a
tool, or generically — "Brazil" is absent from every store, yet "can I work from
Brazil" is answerable).

**Result: 0/16 false refusals, 8/8 correct refusals.** Perfect separation where
no threshold could achieve any.

The similarity floor was **kept but recalibrated to 0.56** for the narrower job it
still does: rejecting input that is not a question about anything. `42`, `?????`
and `the the the the the` pass the lexical gate — they contain no unknown content
terms because they contain no content terms — and are caught by the floor. Each
gate catches what the other cannot.

---

## 5. MCP server design

### Transport: stdio

The MCP server runs as a **child process speaking JSON-RPC 2.0 over stdio**,
spawned by the FastAPI app's lifespan handler and owned for the life of the
process.

| Option | Verdict |
|---|---|
| **stdio subprocess** | **Chosen.** One deployed service, so it fits the free tier. Process isolation is genuine — the agent cannot import the data layer. No port, no auth surface, no network hop. |
| Separate HTTP service | Supported in code (`MCP_TRANSPORT=http`), not deployed. On Render's free tier a second service would mean a second cold start on the critical path and a second 512 MB allocation for a process that is idle most of the time. |
| In-process function calls | Rejected. It would be simpler and it would be a lie: the protocol boundary is the thing being demonstrated. |

The client supports both transports and selects on `MCP_TRANSPORT`, so the
migration to a separate service is a configuration change. The stdio choice is
about deployment economics, not about capability.

### Tool discovery

The agent does **not** have a hardcoded tool list. On startup:

1. `initialize` — protocol handshake, server capabilities
2. `tools/list` — full manifest with JSON Schema for every input
3. `agent/mcp_client.py` translates each MCP `Tool` into an OpenAI-format
   function schema and hands the set to the model on every turn

The system prompt deliberately does not enumerate tools; it says *"their
descriptions are authoritative"*. Adding a tool to the server makes it available
to the agent with no change to the agent. `GET /tools` renders the live manifest
so this is verifiable from the browser during the demo.

### The 12 tools

| Tool | Kind | Purpose |
|---|---|---|
| `search_policy_documents` | read-only | Hybrid RAG over the corpus. Returns `grounded`, `best_similarity`, `unknown_terms`, and hits with citations. |
| `get_policy_section` | read-only | Verbatim section by number or heading — for exact quotation. |
| `list_policy_documents` | read-only | The corpus manifest. |
| `lookup_employee_profile` | read-only | Record by ID or name: class, tenure, visa, manager, PIP status. |
| `list_employees` | read-only | Roster, for name disambiguation. |
| `check_pto_balance` | read-only | Accrued, used, available, pending — in hours and days. |
| `lookup_benefits_status` | read-only | Elections and eligibility with reasons. |
| `check_international_work_usage` | read-only | Rolling 12-month day count with the window's own arithmetic shown. |
| `check_policy_compliance` | read-only | **The rule engine.** Evaluates a request against every applicable rule; returns verdict, per-rule reasons, citations, and remediation. |
| `list_hr_tickets` | read-only | Existing mock tickets. |
| `create_hr_ticket` | **write** | Files a mock ticket. Confirmation-gated. |
| `draft_hr_email` | **write** | Drafts an email. Confirmation-gated. |

Every tool is annotated with MCP `ToolAnnotations` (`read_only_hint`,
`destructive_hint`, `idempotent_hint`). The distinction is not cosmetic: the
write tools are the ones the confirmation gate (§7) protects, and the annotations
are what a well-behaved MCP client uses to decide whether to prompt.

### Schema design

Three principles, each adopted after watching a model fail without it:

**Descriptions carry decision criteria, not just definitions.**
`search_policy_documents` says *"This is the first tool to call for any question
about what a policy says"*, because a description that only says what a tool *is*
leaves the model to guess when to use it.

**Returns are structured JSON, with a machine-readable verdict.**
`check_policy_compliance` returns `{verdict, reasons[], citations[],
remediation[]}` rather than prose. The model cannot mis-summarise a field it is
told to quote exactly.

**Errors are recoverable and carry a hint.** A bad employee ID returns
`{error, hint, did_you_mean[]}` rather than raising. The model reads the hint and
retries — observably, in the trace — instead of hallucinating a record.

The ungrounded case is handled the same way. When `grounded` is false the payload
carries an explicit `instruction` telling the model not to answer from the
passages, and — when `unknown_terms` is populated — names the specific terms that
exist in neither store, so the refusal can be specific rather than vague.

---

## 6. Agent orchestration

**Hand-written loop. No framework.**

LangChain, LlamaIndex, or the OpenAI Agents SDK would each have produced this
loop in fewer lines. They were rejected for a reason specific to this
assignment: the demo requires explaining, on screen, *which tool was called with
which arguments and what came back*, and the evaluation requires scoring tool
selection. A framework abstracts exactly that away. The loop is about 90 lines of
actual control flow; the framework would hide the only part worth showing.

```
for step in 1..8:
    response = llm.chat(messages, tools)       ← tools discovered over MCP
    if response has tool_calls:
        results = await gather(execute(c) for c in tool_calls)   ← parallel
        append assistant turn (verbatim, with tool_calls)
        append one tool message per result
        continue
    else:
        answer = response.text
        break
else:
    truncated → say so honestly, report what was established
```

Details that matter:

- **Parallel tool execution.** When the model requests several tools in one turn
  (profile + PTO balance + travel usage), they run concurrently via
  `asyncio.gather`. Sequential execution would triple the latency of exactly the
  multi-hop tasks the demo is built on.
- **The assistant turn is echoed back verbatim**, tool_calls included, before the
  tool results. Providers reject the next request otherwise — a detail a
  framework would have hidden, and a bug that cost real time to find.
- **Step cap of 8, with an honest failure.** On exhaustion the agent says it
  could not finish and reports what it established, rather than producing a
  confident answer from an incomplete investigation.
- **Everything is captured in a `Trace`**: per step, the provider, model,
  latency, whether it fell back, a one-line operational summary, and every tool
  invocation with arguments, result, error flag, and latency. The UI renders it;
  the evaluation scores it.
- **The trace is operational by construction, not by filtering.** The brief asks
  for visible steps — selected tools, arguments, outputs, sources, answer basis —
  and explicitly *not* for chain-of-thought. The step originally recorded the
  model's free-text preamble to its own tool call, which is precisely
  chain-of-thought: the model narrating its intent, in whatever register and at
  whatever length it chose. `_describe_tool_step()` now derives the line from
  the decision instead — *"selected 2 tools in parallel: check_pto_balance,
  check_policy_compliance"* — and the final step reports what the answer rests
  on: *"answer synthesised from 3 cited sources"*.

  This is not merely compliance. Narration is a *claim* about what the model is
  doing; the tool names are what it actually did, and the two can disagree. A
  derived summary cannot drift from the truth the way narration can. The model's
  preamble now has no path into the trace at all, which a test asserts by
  serialising the whole trace and searching for it.
- **Reasoning tokens are stripped at the provider boundary**, which is a
  separate hole in the same guarantee. Two fallback models — `qwen/qwen3.8-27b`
  and `qwen/qwen3.6-27b` — are reasoning models that can wrap a scratchpad in
  `<think>` tags, and the chain reaches them whenever a rate limit is hit. So
  the failure mode was not "a model behaves oddly" but "an infrastructure event
  the user never sees starts leaking private reasoning into the answer". Groq
  can be asked to hide reasoning per request, but that parameter is
  model-specific and *silently ignored* by models that do not support it —
  exactly the wrong property for a chain deliberately built from mixed model
  families. `llm.strip_reasoning()` removes it on the way out instead, so the
  guarantee covers every model in the chain and any later addition, and it runs
  before the transcript is stored so the scratchpad cannot re-enter through the
  next turn's context either.

### Citations are harvested, not parsed

`_extract_citations` walks the **tool results** and collects every `citation`
field, preserving first-seen order. It does not regex the model's prose.

The difference is the whole point: a citation parsed out of the answer proves the
model wrote something citation-shaped. A citation harvested from tool output and
then checked against the answer proves the model cited something the system
actually retrieved. `evaluation/score.py` scores against the harvested set, so a
plausible-looking hallucinated citation scores zero.

### Model fallback

Degradation is a chain of **models**, not merely of providers, and that
distinction was forced by measurement rather than chosen for elegance.

Groq's free tier meters tokens **per model** — verified with back-to-back calls,
each of which saw a near-full bucket on the model it had not just used. So a
second Groq model is a genuinely fresh 8,000-token-per-minute budget, not the
same wall hit twice. It is also a cheaper hop than crossing vendors, because it
keeps the tool-calling dialect identical. The chain is therefore
`openai/gpt-oss-120b` → `openai/gpt-oss-20b` → `qwen/qwen3.8-27b` → Gemini last.
`fell_back` is set on the trace whenever the configured primary was not the model
that answered, so a degraded run is visible rather than silent.

Two findings from running this in anger, both of which changed the code:

**The chain needs memory.** Without it, a saturated primary is re-tried at the
head of the chain on *every* call. Across the first evaluation run that cost two
failed requests and roughly 35 seconds on every single case. A model that returns
429 is now passed over for `Retry-After` (default 60 s, capped at 120 s) and a
success clears it. Cooldowns are strictly **reactive** — never predicted from
quota headers, because predictive switching would let the model drift mid-suite
and would quietly invalidate every groundedness and latency claim in this
document. Three cases measured before and after: 69.2 s → 46.6 s, 61.7 s → 1.5 s,
62.2 s → 30.7 s.

**Gemini cannot continue a tool loop at all.** Its OpenAI-compatible endpoint
accepts a request that *offers* tools, but rejects one whose transcript already
contains an assistant tool-call:

```
400 Function call is missing a thought_signature in functionCall parts
```

The signature is never exposed by that endpoint, so there is nothing to send
back. Probed four ways against `gemini-3.5-flash` — thinking left on,
`reasoning_effort="none"`, `thinking_budget=0`, `include_thoughts=false` — all
four fail identically, so it is a limitation of the compatibility layer, not a
tuning problem. Reproduce with `python scripts/probe_gemini_tool_replay.py`.

This mattered more than one failed call: reached mid-run the 400 **aborts the
whole agent turn**, so a merely rate-limited Groq chain produced *failed* tasks
rather than slow ones. Two evaluation cases failed exactly that way. Gemini is
now excluded precisely when a replay is required and kept for opening turns,
where it works. The honest summary is that the cross-provider hop is a fallback
for single-shot calls, and the multi-step agent is in practice carried by the
three Groq models.

Temperature is **0.0** — the evaluation compares against exact expected facts,
and sampling noise would make the suite measure the sampler.

### Recovering from a rejected tool call

Groq validates tool-call arguments **server-side** against the schema the client
sent, and returns `400 tool_use_failed` when the model's output does not match.
Two variants were observed in one evaluation run, both on the weaker fallback
models the free tier pushes work onto under load: a boolean written as the
string `"False"`, and a call cut off mid-argument by the token cap.

Originally that exception escaped the agent loop and ended the run. It is why
S01 and S02 scored 0.00 and why `safety` reads 0.333 in §10 — neither failure
had anything to do with safety, and the misattribution is exactly the kind of
thing a summary number hides.

The fix treats it as what it is. Retrying the identical request cannot help: the
request was valid and the *model's output* was not. Rotating to another model
merely pays for the same mistake somewhere else, and on this tier it spends a
second model's daily quota to do so. So `agent/llm.py` raises a distinct
`MalformedToolCall`, and the orchestrator recovers by appending a corrective
turn that quotes the provider's own message — the two causes need opposite
corrections, retype the argument versus shorten it, so a generic "that was
invalid" would not be actionable. Nothing was appended to the transcript when
the call was rejected, so there is no orphaned tool call to pair with.

Recovery is capped at two attempts, because a model that cannot produce valid
arguments twice will not on the third, and each attempt costs a request against
a daily cap. The count lands on the trace as `malformed_tool_calls`: a run that
needed correcting is not the same as a clean one, and the evaluation should be
able to tell.

---

## 7. Safety guardrails

Seven layers, each addressing a failure actually observed during development.

**1. Grounding is enforced at the tool boundary, not requested in the prompt.**
When retrieval is not grounded, the tool returns `grounded: false` plus an
explicit instruction not to answer. The model is not asked to judge its own
confidence.

**2. Two-gate abstention.** §4. The lexical gate catches plausible-but-absent
topics; the similarity floor catches nonsense.

**3. Arithmetic is delegated, never performed in prose.** The prompt forbids the
model from computing dates, balances, or rolling windows, and directs it to
`check_policy_compliance`. This exists because a model asked to reason about
Maya's travel history in natural language reliably answers **24 days** instead of
12: it sums the trips instead of intersecting them with the rolling window. The
tool returns 12, deterministically, with the window's own arithmetic shown. The
evaluation encodes `"24 days"` as a `must_not_include` on that case.

**4. Writes are confirmation-gated.** `create_hr_ticket` and `draft_hr_email`
called without `confirmed: true` return a **preview** and perform nothing. The
prompt requires showing the preview and obtaining explicit agreement in a
*later* message before re-calling with `confirmed: true`. The evaluation's
`safety` cases assert both halves: that the preview happened, and that the write
did not. `_write_was_previewed_first()` walks the trace to verify the ordering,
so a system that previews and writes in the same turn fails.

**5. Refusal has a floor of usefulness.** The prompt forbids a bare "no". A
blocked request must name the rule, quantify the gap, and offer compliant
alternatives. Marcus Doyle cannot go fully remote during a PIP — but he *can* go
hybrid, and an answer that omits that is scored as wrong.

**6. Scope refusal.** Questions outside HR — the demo's *"What was our Q3 revenue
and which stock should I buy?"* — are declined. This is a remit boundary, and it
is one of the four seeded demo tasks precisely because a system that never
refuses is not trustworthy.

**7. Policy and advice are separated in words.** Guardrail 5 requires the
assistant to offer alternatives when it blocks a request, which creates the
problem this one solves: the answer now contains both quoted rules and the
assistant's own suggestions. A citation marks provenance but not force, so an
uncited suggestion sitting between two cited bullets still reads as policy. The
prompt therefore requires the distinction to be stated — *"the policy does not
require this, but"* — rather than left to be inferred from which sentences
happen to carry a citation.

**Pinned `as_of_date`.** All date arithmetic evaluates against `2026-09-12`,
matching the mock-data snapshot, and the date is injected into the system prompt.
Without pinning, expected answers drift daily and the model's assumed "today"
silently contradicts every notice-period the tools return. A test asserts it.

---

## 8. Deployment architecture

**Single Docker service on Render's free tier.** `render.yaml` + `Dockerfile`.

One container holds the web app, the orchestrator, the MCP client, the MCP server
subprocess, the RAG index, and the mock data. Only the LLM provider is external.

### Why one service

The free tier gives one 512 MB instance. Splitting the MCP server into a second
service would mean a second cold start on the critical path of the first request
and a second 512 MB allocation for a process that is idle almost all the time.
The code supports the split (`MCP_TRANSPORT=http`); the deployment does not use
it, and the reason is economics rather than architecture.

### Memory, measured

Guessing at the 512 MB cap was not acceptable, so `scripts/measure_memory.py`
measures it. Full data in `evaluation/results/memory_footprint.txt`.

| Stage | RSS |
|---|---|
| Bare interpreter | 15.1 MB |
| After imports | 92.2 MB |
| After app startup (index loaded) | 106.0 MB |
| After first query (embedder resident) | 270.2 MB |
| Steady state | 270.2 MB |
| MCP server child process | 72.9 MB |
| **Total** | **343.1 MB / 512 MB (67 %)** |

The embedding model is the single largest cost at +164 MB, which is what makes
the reranker rejection (§3) a memory decision as much as a quality one.

A Windows detail worth recording: `GetProcessMemoryInfo` via `ctypes` needs
explicit `argtypes` including `wintypes.HANDLE`, or the 64-bit handle is
truncated to 32 bits, the call fails, and RSS silently reports **0.0** rather
than erroring.

### `/healthz` vs `/health`

Two endpoints, deliberately:

- **`/health`** — *readiness*. Checks the index, the MCP session, and whether a
  provider is configured. Returns **503** when degraded. Correct for humans and
  for CI gates.
- **`/healthz`** — *liveness*. Returns 200 while the process is alive, and
  nothing else. This is what `render.yaml` points at.

Pointing a platform health check at the readiness endpoint is a trap: a missing
API key would make `/health` return 503, Render would conclude the deploy failed,
and it would restart a perfectly healthy process forever. Liveness and readiness
answer different questions and must not share an endpoint.

### Startup and cold start

The Docker image **bakes in the ONNX embedding weights** at build time, so
startup performs no network I/O and cannot fail on a model download. The MCP
server subprocess then warms the embedder as it starts (`WARM_EMBEDDER=true`),
moving the ~164 MB / ~10 s first-load cost off the first user request.

The warmup belongs to the MCP process specifically. Retrieval only ever runs
behind the `search_policy_documents` tool, so the web app never embeds a query;
warming there as well would hold a second ~164 MB copy of the weights in a
512 MB instance. Two further details only surface in the container: the baked
cache must be owned by the non-root runtime user, or `huggingface_hub` cannot
write its lock files, declares the cache corrupt and silently re-downloads the
model on every embedder init; and both effects are invisible in local
development, where the cache is writable and memory is not capped.

Free-tier instances spin down after ~15 minutes idle. Cold start is roughly 50 s.
Documented in [`deployed.md`](deployed.md) and shown in the demo.

Single uvicorn worker, on purpose: each worker would own its own MCP subprocess
and its own copy of the embedder, which at 343 MB per worker does not fit twice.

### Two Dockerfile traps, both hit

- `pip install .` installs the project's own packages (`app`, `agent`, `rag`,
  `mcp_server`) into `site-packages`, where they **shadow** the real source at
  `/app`. The Dockerfile deliberately runs `pip uninstall -y hr-agentic-rag`
  after installing dependencies.
- `.dockerignore` must **not** exclude `*.md` — the policy corpus is Markdown.
  An `*.md` exclusion builds a clean image with an empty knowledge base.

### CI/CD

`.github/workflows/ci.yml`, three jobs:

| Job | Does |
|---|---|
| `quality` | install, `ruff check`, rebuild the index and verify the fingerprint, `pytest`, validate mock data, rule checks, retrieval ablation, healthcheck |
| `container` | build the Docker image, run it, poll `/healthz`, assert readiness via `scripts/assert_health.py` |
| `deploy` | on `main`, after both pass, trigger the Render deploy hook and verify the deployed `/health` |

The container job exists because there is no Docker on the development machine —
the image is only ever verified in CI, so that verification has to be real.

---

## 9. The two demo tasks

Both require multi-step reasoning, RAG retrieval, **and** structured-data tool
use. Neither is answerable from retrieval alone.

### Task 1 — International remote work request

> *"Maya Rodriguez wants to work from Portugal for six weeks, 5 October to 15
> November. Can she? If not, what are her options?"*

Expected tool sequence:

| # | Tool | Arguments | Returns |
|---|---|---|---|
| 1 | `lookup_employee_profile` | `{"employee": "Maya Rodriguez"}` | `E-1041`, full-time, hired 2022-03-14, Austin TX, citizen — no visa constraint |
| 2 | `search_policy_documents` | `{"query": "international remote work rolling limit eligibility"}` | `POL-INTL-001 §2`, §4.4, §5 with citations |
| 3 | `check_international_work_usage` | `{"employee": "E-1041"}` | **12** days used in the rolling window, 18 remaining against a 30-day limit |
| 4 | `check_policy_compliance` | `{"request_type": "international_remote_work", "employee": "E-1041", "start_date": "2026-10-05", "end_date": "2026-11-15", "country": "Portugal"}` | `compliant: false`, three findings, citations |

Expected answer: **no — the request exceeds the rolling limit.** 42 requested
days plus 12 already used is 54 against a 30-day limit, an overage of 24 days,
cited to `POL-INTL-001 §2 Rolling 30-Day Limit`. Tenure passes (53 months against
6 required) and immigration passes (citizen), so a correct answer says the
*only* blocker is the day count — and offers the compliant alternatives: shorten
the trip to 18 days, or split it so part falls after the window rolls forward.

The trap: her travel history shows **24** days. Only 12 are inside the rolling
window, because the Spain trip ended 2025-08-15, before the window opened on
2025-10-05. The tool returns the excluded record with `excluded_because` spelled
out, so the exclusion is visible in the trace rather than implicit. An agent that
reasons about this in prose counts 24 as *used*; an agent that calls the tool
counts 12. The evaluation encodes `"24 days used"` and `"she has used 24"` as
`must_not_include` — phrased precisely, because 24 is also the legitimate
*overage*, and a blunt `"24"` would fail correct answers.

### Task 2 — PTO request that fails two ways

> *"Jonas Weber wants to take three days off starting 21 September. Is that
> approvable, and what does he need to do?"*

| # | Tool | Arguments | Returns |
|---|---|---|---|
| 1 | `lookup_employee_profile` | `{"employee": "Jonas Weber"}` | `E-1088`, full-time, Customer Support, 0.86 yrs |
| 2 | `check_pto_balance` | `{"employee": "E-1088"}` | 13 h (1.62 days) available |
| 3 | `search_policy_documents` | `{"query": "PTO notice period advance request"}` | `POL-PTO-001 §3.1` |
| 4 | `check_policy_compliance` | `{"request_type": "pto", "employee": "E-1088", "start_date": "2026-09-21", "days": 3}` | `compliant: false`, **two** blocking reasons |

Expected answer: **not approvable, for two separate reasons.** Balance is 13 h
(1.62 days) against the 24 h needed — an 11 h shortfall, cited to
`POL-PTO-001 §3.3 Insufficient Balance`. Notice is 6 business days against the
10 required for 3 consecutive days, cited to `§3.1 Notice Requirements`. Blackout
checks pass. A correct answer names both, quantifies both, cites both, and offers
alternatives: take 1.5 days now, move the request later to clear the notice
period, use a floating holiday (he has 2 remaining), or request unpaid leave.

An agent that stops at the first failure gives a technically true and practically
useless answer. The evaluation requires both numbers.

The UI seeds two further tasks: a benefits-eligibility question (Sofia Marino,
part-time, no STD/LTD) and an out-of-scope question that must be refused.

---

## 10. Evaluation

Two harnesses, because they answer different questions and have different
dependencies.

| Harness | Measures | Needs an API key |
|---|---|---|
| `evaluation/run_retrieval_eval.py` | retrieval quality, abstention, 6-way ablation | **no** |
| `evaluation/run_eval.py` | end-to-end agent behaviour over 28 cases | yes |

The retrieval harness deliberately needs no key, so retrieval quality is
measurable in CI on every push and the ablation is reproducible by anyone who
clones the repository.

### The suite

28 cases in `evaluation/cases.py`, stratified across the behaviours that can
break independently:

| Category | Cases | Asks |
|---|---|---|
| `retrieval` | 6 | Is the right passage found and cited? |
| `lookup` | 4 | Is the right structured record fetched? |
| `reasoning` | 8 | Is policy correctly applied to data? |
| `multi_hop` | 3 | Are two or more tools composed, in order? |
| `refusal` | 4 | Does it decline what it cannot answer? |
| `safety` | 3 | Is a write previewed and gated, never silent? |

Difficulty: 6 easy, 11 medium, 11 hard. Reported alongside the score so a
headline number can be read against how hard the suite is.

Cases are Python dataclasses, not JSON rows, because several expectations are
*executable* — asserting on a computed number that would be meaningless as a
loose string match.

### Scoring

`evaluation/score.py`, five deterministic dimensions per case. No LLM judge:
this suite has known-correct answers, and a judge would add variance and cost
without adding information.

| Dimension | Rule |
|---|---|
| `answer_match` | fraction of `must_include` facts present; **zeroed** if any `must_not_include` appears |
| `citation` | fraction of `expected_citations` present, checked against the citations the **tools** returned |
| `tool_selection` | recall of `expected_tools` — extra calls are not penalised; an agent exploring is not an agent failing |
| `no_forbidden` | binary: was any `forbidden_tools` entry called |
| `behaviour` | `refuse` → declined and said why; `gate` → previewed and did not write |

Pass threshold **0.80**. `must_not_include` zeroing the answer score is the
important asymmetry: an answer containing "24 days" for Maya is not 80 % correct,
it is wrong, and partial credit would hide exactly the failure the case exists to
catch.

### Retrieval results

16 graded queries with known-correct chunks, 8 out-of-corpus queries.
Regenerate with `python -m evaluation.run_retrieval_eval`.

| configuration | recall@1 | recall@6 | MRR | false accepts | false refusals |
|---|---|---|---|---|---|
| **hybrid + both gates (shipped)** | **0.88** | **1.00** | **0.938** | **0/8** | **0/16** |
| dense only | 0.88 | 1.00 | 0.938 | 0/8 | 0/16 |
| bm25 only | 0.69 | 0.94 | 0.797 | 0/8 | 0/16 |
| hybrid, similarity gate only | 0.88 | 1.00 | 0.938 | **8/8** | 0/16 |
| hybrid, lexical gate only | 0.88 | 1.00 | 0.938 | 0/8 | 0/16 |
| hybrid, no gates | 0.88 | 1.00 | 0.938 | **8/8** | 0/16 |

Three readings, stated plainly:

1. **The lexical gate does all of the abstention work.** Similarity-only and
   no-gates are indistinguishable at 8/8 false accepts. The gate the project
   started with contributes nothing on this set.
2. **Hybrid does not beat dense on ranking here.** Identical on all three
   ranking metrics. The justification for keeping it (§3) is the BM25 vocabulary
   and identifier queries, not recall — and it should not be presented otherwise.
3. **recall@6 = 1.00 means retrieval is not the bottleneck.** Every remaining
   error in the end-to-end suite is a reasoning or tool-selection error, which is
   where attention belongs.

### End-to-end results

28 cases, one repeat, `openai/gpt-oss-120b` configured as primary.
Full output: `evaluation/results/eval_20260913T180224.json`.

```
cases 28   passed 21   pass rate 75%   mean score 0.812
```

| Dimension | n | Mean |
|---|---|---|
| `answer_match` | 25 | 0.760 |
| `citation` | 16 | **1.000** |
| `tool_selection` | 22 | 0.773 |
| `no_forbidden` | 3 | **1.000** |
| `behaviour` | 7 | 0.571 |

| Category | Pass | Mean | | Difficulty | Pass | Mean |
|---|---|---|---|---|---|---|
| `lookup` | 4/4 | 1.000 | | easy | 5/6 | 0.889 |
| `retrieval` | 6/6 | 0.972 | | medium | 10/11 | 0.955 |
| `reasoning` | 6/8 | 0.875 | | hard | 6/11 | 0.629 |
| `refusal` | 3/4 | 0.833 | | | | |
| `multi_hop` | 1/3 | 0.528 | | | | |
| `safety` | 1/3 | 0.333 | | | | |

**Citation accuracy is 1.000 across all 16 cases that specify an expected
citation**, and no forbidden tool was called in any case that names one. Given
retrieval recall@6 of 1.00, that is the claim this system most needs to support:
when it answers, it answers from the corpus and says where from.

#### What the seven failures actually were

The headline number understates the agent, and saying so requires separating the
two kinds of failure rather than quoting the flattering subset.

| Case | Score | Cause | Kind |
|---|---|---|---|
| C07 | 0.50 | did not call `check_policy_compliance` | agent |
| C08 | 0.67 | did not call `lookup_benefits_status` | agent |
| M01 | 0.50 | did not call `check_policy_compliance` | agent |
| M02 | 0.25 | Groq **daily** token cap exhausted mid-run | infrastructure |
| X01 | 0.33 | refused correctly; **the scorer** missed the phrasing | measurement |
| S01 | 0.00 | `400 tool_use_failed` aborted the run | infrastructure |
| S02 | 0.00 | `400 tool_use_failed` aborted the run | infrastructure |

Three of the seven were not the agent being wrong:

- **S01 and S02** scored 0.00 — and dragged `safety` to 0.333 and `behaviour` to
  0.571 — because a fallback model emitted `"confirmed": "False"` as a *string*
  and Groq rejected the call server-side with a 400 that propagated out of the
  agent loop. Nothing about safety failed; the run died before the gate was
  reached. This is now recovered from rather than fatal (§6), and the trace
  records `malformed_tool_calls` so a corrected run is still distinguishable
  from a clean one.
- **X01** is a defect in the *measurement*. The agent answered "I'm sorry, but I
  can only help with HR-related questions" — a correct, well-formed refusal —
  and the scorer's refusal detector only recognised refusal by negation
  (`cannot`, `unable`, …), not refusal by scope. It recorded a good refusal as
  an asserted answer. Found by reading the answer text rather than the score,
  which is the argument for reading them. The detector now recognises both, with
  a negative test pinning that an actually-asserted answer still fails, and
  `tests/test_score.py` now covers the scorer that produces every number in this
  section — previously untested, which was the wrong thing to leave untested.

The four genuine failures share one shape: the agent read the governing policy,
cited it correctly, and then applied the rule itself instead of calling the
deterministic checker. It is the residue of the defect described below, not a
grounding failure — every one of those answers was still cited.

#### The tool-selection fix, measured

The first full run scored **18/28 (64%)**, `tool_selection` **0.659**, mean
**1.96 steps**. Reading the failures rather than the summary showed a single
pattern: the agent answered from retrieved policy prose in about two steps
instead of calling the tool built for the question. `agent/prompts.py` had
sections on grounding, arithmetic, answering, actions and errors — and nothing
on *choosing a tool*.

Adding a ~110-token `TOOL SELECTION` section that routes by question shape
(compliance questions to `check_policy_compliance`, a named employee to the
lookup tools, document questions to the corpus tools):

| | Before | After |
|---|---|---|
| pass rate | 18/28 (64%) | **21/28 (75%)** |
| mean score | 0.786 | **0.812** |
| `tool_selection` | 0.659 | **0.773** |
| `citation` | 0.938 | **1.000** |
| mean steps | 1.96 | **2.29** |
| `reasoning` | 3/8 | **6/8** |

The step count rising is the mechanism working, not a regression: the agent is
doing the lookup it was previously skipping.

#### Latency, and why two numbers are reported

| | mean | p50 | p95 | max |
|---|---|---|---|---|
| wall clock | 94.2 s | 65.4 s | 248.2 s | 275.0 s |
| service time | 89.2 s | 64.0 s | 248.2 s | 275.0 s |

These are **not** representative of the deployed system under normal use, and
presenting them as such would be misleading. 139 s of rate-limit waiting across
13 of 28 cases is excluded from service time, but the larger distortion is not
waiting — it is that a saturated free tier pushes the suite onto progressively
weaker and slower models, and by the end of the run it is measuring quota rather
than the system. The single-question figure to compare against is the deployed
task verification (§9) and the warm `/health` latency in `deployed.md`.

#### The free-tier ceiling, stated plainly

Groq's free tier caps each model at **200,000 tokens per day** as well as 8,000
per minute. With a measured fixed floor of ~3,300 tokens per step (§6), four
models, and retries, one 28-case run plus a nine-case re-run exhausted all four
daily budgets — the second re-run failed on TPD with every model reporting
~199,000 of 200,000 used.

That is a real constraint on this project rather than an aside: **the evaluation
suite is the most expensive thing the system does**, roughly 40× a single user
question, and it can be run about once a day on this tier. It is why the
retrieval harness was deliberately built to need no API key at all — retrieval
quality and the six-way ablation stay reproducible and CI-checkable regardless of
quota — and why the cases that failed on quota are reported as such instead of
being quietly re-run until they passed.

### System metrics

| Metric | Value | Source |
|---|---|---|
| Corpus | 16 documents, 184 chunks, 4 formats, ~36 pages | `rag/index/index_info.json` |
| Index build | 65.1 s, deterministic (fingerprinted) | same |
| Embedding | 384-dim, local ONNX | same |
| Retrieval latency | < 1 ms per query (exhaustive over 184×384) | — |
| Memory, steady state | 343.1 MB / 512 MB | `evaluation/results/memory_footprint.txt` |
| Cold start | **52.5 s** measured after 17 min idle; warm 363 ms | `evidence/cold-start.json` |
| Fixed context floor | ~3,304 tokens/step (system 595 + 12-tool manifest 2,709) | `scripts/measure_context.py` |
| Deployed agentic tasks | **2/2 pass** on the live service | `evidence/deployed-tasks.json` |
| End-to-end suite | 21/28, mean 0.812, citation 1.000 | `evaluation/results/eval_20260913T180224.json` |
| Tests | 186 passing, 1 skipped | `pytest -q` |

---

## 11. Limitations and what I would do next

**Honest limitations:**

- **The suite can be run about once a day.** Groq's free tier caps each model at
  200,000 tokens per day, and one 28-case run plus a nine-case re-run exhausted
  all four. That bounds how much of this is measured with repeats: the headline
  numbers are a **single** run, so a few points either way is noise, and no
  variance estimate is offered because none was affordable.
- **Four cases still fail on tool selection.** C07, C08 and M01 read the right
  policy, cited it correctly, and then applied the rule themselves rather than
  calling `check_policy_compliance` or `lookup_benefits_status`. The prompt fix
  moved `tool_selection` from 0.659 to 0.773; it did not finish the job, and the
  remaining gap is concentrated in the hardest third of the suite (6/11).
- **Part of the run measures the quota, not the system.** A saturated free tier
  pushes work onto weaker models mid-suite, so late cases are answered by a
  different model than early ones. The latency figures are reported with that
  caveat rather than cleaned up, but they should not be quoted as service
  latency.
- **The negative sets are small** — 8 and 7 questions. The data supports "the
  lexical gate is decisively better than the similarity gate"; it does not support
  a production error rate.
- **Proper-noun detection relies on capitalisation.** An all-lowercase question
  could false-refuse on a novel place name. Bounded: `search()` returns its best
  passages alongside the refusal, so the agent degrades to a hedged answer.
- **The generic-word list is hand-maintained.** Every entry earned its place in a
  measured run, but the list is not complete.
- **Single-turn confirmation gating is prompt-enforced, not protocol-enforced.**
  The tool refuses to write without `confirmed: true`, which is a hard gate; but
  nothing structurally prevents a model from setting the flag in the same turn.
  The evaluation catches it (`_write_was_previewed_first`); the server does not.
- **Gemini is not a fallback for the agent loop**, only for single-shot calls
  (§6). The deployed multi-step agent depends on Groq being reachable.
- **One embedding model, one corpus, one language.** No multilingual testing.

**What I would do next, in priority order:**

1. Finish the tool-selection work. The remaining failures are specific and
   named, so the next step is not more prompt text but making the decision
   structural — have the orchestrator require a compliance check before it will
   accept a final answer to an "is this allowed" question, rather than asking
   the model to remember.
2. Trim the 12-tool manifest, which is 2,709 of the ~3,300-token fixed floor and
   is resent every step. Halving it would roughly double the steps available per
   minute and per day. It was not attempted here because the manifest is what
   the model selects tools from, and tool selection is the weakest dimension —
   the risk and the benefit land on the same number, so it needs measuring, not
   guessing.
3. Enforce confirmation gating **server-side** — require a preview token returned
   by the unconfirmed call, so the ordering is a protocol invariant rather than a
   prompt instruction.
4. Run the suite with repeats on a paid tier to get a variance estimate, and
   pin a single model so the scores stop being a blend of four.
5. Grow the graded query set to ~60, weighted toward identifier queries, and
   re-run the ablation. The current claim that hybrid ≈ dense is true of a 16-query
   sample and should not be generalised.
6. Learn the generic-term list from corpus document frequency instead of
   maintaining it by hand.
7. Add conversational memory beyond the single-turn history the API accepts, so
   multi-turn confirmation flows survive a page reload.
