# Design and evaluation

## Evidence status

The current code has not been evaluated end to end against a live LLM. Live
measurements in this report come from the September revision and are labelled
historical; they are not measurements of the current code.

`evidence/deployed-tasks.json` is a superseded September capture. Its verifier
checked tool and step counts, not answers, and so labelled David's
non-birthing-parent scenario successful while answering 16 weeks instead of the
policy's 8. It is retained as a documented failure.

## Architecture

```text
Browser: chat, source snippets, operational trace, explicit confirmation button
  |
FastAPI: session cookie, bounded input, concurrency/rate guards, read-only cache
  |                                  |
Agent orchestrator                   confirmation endpoint (no model call)
  | LLM provider                     | signed exact-payload capability
MCP client --------------------------+
  | JSON-RPC over stdio
MCP server: discovery + twelve tools
  |                           |
RAG: local ONNX, NumPy, BM25   synthetic JSON records + deterministic rules
```

The handwritten agent loop makes model/tool decisions, errors, context pressure,
and provider usage inspectable. It uses `tools/list` and `tools/call`; it does not
replace MCP calls with direct data-function invocation.

The web confirmation endpoint shares the MCP client, but cannot authorize an
arbitrary model-generated payload: it retrieves an exact previously displayed
preview from server-side session state. No approval secret enters the model.

One process tree keeps the deployment suitable for a small free-tier instance.
Stdio avoids a second service and cold start. Streamable HTTP is also supported;
it needs an explicitly configured shared approval secret for mock writes.

## Corpus, metadata and ingestion

The corpus has 16 synthetic policies spanning PTO, remote work, international work,
parental/medical leave, benefits, expenses, equipment, performance, conduct,
security, onboarding, holidays, payroll, grievance, learning and safety.
Markdown, HTML, plain text and PDF normalize into canonical structured documents.
The `_source_` Markdown used to generate the PDF is excluded from ingestion.

The index has **221 chunks**. Small original sections remain independently
addressable: merging parental-leave section 2.4 into 2.3 previously broke exact
lookup and citations. Large sections use sentence-aligned windows with overlap,
target 320 estimated tokens, cap 420, overlap 60.

Each record retains document ID/title, version, effective date, source format/file,
section number/title, text, ordinal and chunk ID. Index metadata fingerprints both
the corpus and ingestion implementation. CI refuses a stale index.

The embedded representation includes a heading breadcrumb. `BAAI/bge-small-en-v1.5`
runs locally through fastembed/ONNX with 384-dimensional normalized vectors.
An exhaustive NumPy index is appropriate for this size: introducing an approximate
vector service would add operational complexity without a measured need.

## Retrieval and answer evidence

Dense ranking and BM25 are combined by reciprocal rank fusion (constant 60).
Top-k and optional document filtering are supported; exact section lookup can
follow a search. The normal MCP search default is five passages, clamped to 1-10.
The retrieval ablation uses six.

A similarity floor and vocabulary-based abstention detect unsupported queries.
They are heuristics, not proof of claim-level groundedness. The vocabulary gate
can reject valid paraphrases, and a relevant passage can still be misinterpreted.

The agent receives evidence as MCP tool results. Final output distinguishes:

| Field | Meaning |
|---|---|
| `retrieved_citations` | Labels found in tool results |
| `citations` | Retrieved labels actually cited inline in the accepted answer |
| `sources` | Retrieved passage snippets and metadata |
| `grounded` | Retrieval-gate status, **not** a semantic certification of the answer |

Citation identity normalizes Unicode hyphens and matches document plus section.
An answer with invented citations, no inline citations despite policy evidence,
or a successful compliance decision without retrieved passages receives one
bounded correction opportunity. If still unsupported, it fails explicitly.
An ungrounded search leads to a refusal rather than a speculative policy answer.
This validates provenance, not entailment of every sentence.

## Deterministic policy decisions

The rules layer computes PTO balance/notice/blackouts, international rolling
usage, remote-arrangement conditions and parental-leave eligibility.

Parenting role is explicit: `birthing` or `non_birthing`, never inferred from
name or gender. "My partner is due" is non-birthing. The parental checker requires
a start date, evaluates 12-month tenure and employment type, prorates part-time
entitlement, and returns paid weeks plus PTO interaction with citations.
The agent validates numeric week claims against this result before accepting
the final answer. David's 2027-03-01 scenario returns **8 weeks**.

PTO duration is counted across weekdays rather than adding calendar days.
Non-finite or excessive durations and weekend starts are rejected. The checker
returns the actual notice bands and exception routes; recommendations must
cite evidence and must be rechecked before being described as compliant.
Short notice requires documented manager approval and justification, rather
than being described as an unconditional denial. Unpaid leave needs skip-level approval.

These rules remain a deliberately limited model, not a production HRIS:
holidays, team coverage and approvals need careful interpretation and expansion
before using the system for real employment decisions.

## MCP tools and schemas

Full argument schemas are discovered from the running server. The LLM receives
those schemas plus the first descriptive paragraph of each tool, reducing repeated
prompt overhead without changing required arguments.

| Tool | Key arguments | Purpose |
|---|---|---|
| `search_policy_documents` | query, k, optional doc_id | Retrieve passages |
| `get_policy_section` | doc_id, section | Exact original section |
| `list_policy_documents` | none | Document versions/catalogue |
| `lookup_employee_profile` | employee | Resolve identity and profile |
| `list_employees` | optional department | Filter synthetic roster |
| `check_pto_balance` | employee | Available balance, not merely total |
| `lookup_benefits_status` | employee | Elections and eligibility |
| `check_international_work_usage` | employee, optional as_of | Rolling usage and exclusions |
| `check_policy_compliance` | request_type, employee, type-specific fields | Deterministic decision |
| `create_hr_ticket` | employee, category, subject, body, priority | Preview; signed approval to create |
| `draft_hr_email` | employee, subject, body | Preview; signed approval to store draft |
| `list_hr_tickets` | optional employee | Seeded and process-local tickets |

`check_policy_compliance` supports `pto`, `international_remote_work`,
`remote_arrangement`, and `parental_leave`; the latter requires `parent_role`
and `start_date`. Expected argument/lookup failures return structured errors.

## Actions and safety

All writes are mock, in-memory and reversible by process restart. A tool call
with `confirmed=true` alone is rejected.

The first call returns a preview. FastAPI stores the exact arguments, session
owner and ten-minute expiry and returns an opaque action ID. A separate click
posts to `/actions/{id}/confirm`. The endpoint atomically consumes that preview,
signs the exact action and arguments, and invokes the MCP tool. The MCP server
checks the signature, expiry and single-use nonce before mutating state.
The signature is not advertised in the model's tool schema.

The confirmation performs **zero LLM calls** and returns the mock ticket/draft ID.
Cross-session, expired, altered, unsigned and replayed approvals fail explicitly.
Text saying "yes" does not bypass the button.

Operational traces expose selected tools, arguments, results and outcome, not
hidden chain-of-thought. Provider reasoning blocks are stripped. User-supplied
history accepts only bounded user/assistant messages, not system/tool roles.
This is a synthetic demonstrator, not an authenticated multi-user HR product.

## Quota and latency engineering

`LLM_MAX_CALLS_PER_TURN=8`, hourly 60, daily 100 count actual HTTP attempts,
including failed calls and fallbacks. `CHAT_MAX_REQUESTS_PER_HOUR=12` and
single-chat admission prevent overlapping runs on the shared key. SDK automatic
retries are disabled, and rate-limited models are not immediately retried.
All-compatible-models-cooling produces a clear quota response.

The maximum orchestration length remains 12 steps; the API-attempt budget may
stop it earlier. Context has a 6,200 estimated-token ceiling, with a 1,200-token
completion cap. Oversized contexts are refused before transmission. Compliance
results are not discarded during context compaction.

Read-only results can be cached for ten minutes per session and exact
question/history. The response and UI label cached traces; `api_calls` is zero
on cache hits, while old trace timings are retained and labelled. Writes, previews
and mutable ticket listings are never cached; confirming a write clears the cache.
The user can request `fresh=true` to bypass the cache.

Counters are **process-local**, reset on restart, and do not measure account-wide
remaining tokens. Provider quotas and other users of the key still matter.
An optional `DEMO_ACCESS_CODE` protects casual public access. Do not publish it.

## Canonical demonstrations

`agent/demo_tasks.py` is the single source of the UI/verifier prompts.

**International workflow:** look up Maya, retrieve international and remote-work
policies, read rolling usage, check 2026-10-05 through 2026-11-15 in Portugal.
Expected: 12 prior days plus 42 requested exceeds the 30-day limit by 24.
Explain remaining allowance and conditional alternatives without inventing rules.

**PTO workflow:** retrieve PTO policy, look up Jonas/balance, check three business
days beginning 2026-09-21, prepare a mock ticket preview. Expected: 13 available
hours versus 24 requested; 6 days' notice versus 10 required. Explain exception
routes accurately. The user confirms the exact preview in a separate click,
which returns the ticket ID with zero additional LLM calls.

Both must show successful MCP retrieval and structured-data calls, supporting
inline citations, final answers and operational traces. The verifier rejects
missing evidence and incorrect gold facts, rather than accepting counts alone.

## Evaluation methodology and results

There are **30 cases** across retrieval, lookup, reasoning, multi-hop, refusal,
ambiguity and action safety, with executable gold facts in `evaluation/cases.py`.
The suite includes a parental-leave regression and a two-document security/
equipment question that requires both document citations.

Scorer version **2** reports:

| Metric | Definition / limitation |
|---|---|
| Answer match | Required gold facts present, forbidden facts absent; not full semantic equivalence |
| Citation coverage | Expected documents/sections actually cited inline and present in tool evidence |
| Citation precision | Fraction of inline section identities present in retrieved evidence; provenance, not entailment |
| Tool selection | Required tools with successful recorded invocations, not a claimed tools-used list |
| Workflow completion | All required checks pass; errors, truncation and any critical failure disqualify |
| Clarification/escalation | Rubric phrases, question form and routing facts; still heuristic |
| Action safety | Successful preview and no confirmed write during the model turn |
| Latency | Wall-clock p50/p95 plus explicit retry-wait accounting |

**Groundedness still needs claim-level review.** A citation from the right policy
can support the wrong conclusion. Read the saved answer beside the actual source;
do not rename provenance precision as semantic groundedness. The historical
parental answer is an explicit failed example.

Each live case uses a fresh stdio MCP server, so writes cannot leak between cases.
Results include full traces, scorer version and code/configuration/index fingerprint.
Checkpoint resume rejects different revisions. Quota interruptions are not scored
as successful completions or silently merged with a new experiment.

### Historical measurements (superseded scorer and index)

The September 28-case run recorded 21 passes; later individual cases were combined
into 23/30. These are **legacy scorer outputs**, not validated current completion
rates. The legacy "citation 1.000" meant a returned label mentioned the expected
document, even if the answer did not cite it.

That run recorded wall-clock p50 **65.4 seconds**, p95 **248.2 seconds**.
Saved deployed workflows took **206.8** and **189.4 seconds**, and a later PTO
request took about **68 seconds**. Warm chat is not a 2-6 second operation, and
health-endpoint latency must not be substituted for chat latency.
`evidence/cold-start.json` records a historical 52.5-second platform cold start;
it is not a measurement of the current code.

### Current offline evidence

`evaluation/results/retrieval_eval.json` and `.md` are generated by the local,
six-configuration retrieval ablation. They measure document-level hit rate/MRR
and abstention, **not** answer quality. Consult their timestamps/index context
rather than mixing them with old full-agent results.

With the 221-chunk index, the offline sample contains 17 graded queries
and eight out-of-corpus queries:

| Retrieval | Document hit@1 | Document hit@6 | MRR | False accepts | False refusals |
|---|---|---|---|---|---|
| Hybrid + both gates | 0.88 | 1.00 | 0.941 | 0/8 | 0/17 |
| Dense + both gates | 1.00 | 1.00 | 1.000 | 0/8 | 0/17 |
| BM25 + both gates | 0.59 | 0.94 | 0.728 | 0/8 | 0/17 |
| Hybrid without lexical gate | 0.88 | 1.00 | 0.941 | 8/8 | 0/17 |

Dense wins top-one ranking on this small sample; hybrid has no demonstrated
ranking advantage here. The retained hybrid configuration should not be
described as outperforming dense. These hit rates count any expected document,
not proof that all evidence for a multi-document answer was retrieved.

The automated suite covers the real MCP transport and app lifecycle, policy rules,
preserved section addresses, scoring false positives, session-bound single-use
confirmation, cache behavior and quota limits. Chat-completion calls are blocked
inside tests; a green test run consumes no LLM quota.

Live answer quality, workflow completion and warm-chat latency of the current
code have not been measured; the live figures above are historical.

## Deployment and reproducibility

Docker bakes the local embedding weights and committed index into one service.
Single-worker operation avoids duplicating model memory. The 343 MB
footprint is a historical measurement, not one of the current image.

CI runs lint, index freshness, data/rule checks, automated tests, MCP smoke,
application startup and a container build/start check. Only successful prerequisite
jobs may trigger Render. The deploy request names the tested commit; the readiness
verification asserts the live `build_sha` equals it. `render.yaml` disables
Render's independent auto-deploy.

Direct dependencies are pinned to tested versions; transitive dependencies are
not locked.

## Remaining limitations

This system is not authorized to make real employment decisions. It has synthetic
data, a pinned snapshot date, limited calendar/rule coverage and no production
employee authentication. Citation validation and gold-fact scoring cannot prove
every natural-language statement. Vocabulary abstention needs broader paraphrase
testing. Gemini cannot carry the configured tool loop. Free-tier availability
and provider token limits are external dependencies.
