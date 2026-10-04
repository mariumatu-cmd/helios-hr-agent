# Design and evaluation

## Evidence status

`evidence/deployed-20261003T152745.json` measures the deployed service (build
`291b50a`, 3 October 2026): both demonstrations and six evaluation cases, sent
back to back on the free tier. Scorer 2 passed 4 of 8;
[Deployed measurement](#deployed-measurement-3-october-2026) explains each
failure and the change it led to. Older live figures come from the September
revision and are labelled historical. `evidence/deployed-20261003T163450.json`
is a later failed check of the international task, described in the same
section. After the resulting fixes, both demonstrations passed on build
`c88530a` (`evidence/deployed-20261003T172000.json` and
`evidence/deployed-20261003T172114.json`).

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
bounded correction opportunity, which asks for the complete answer again rather
than a note about the fix. If still unsupported, it fails explicitly. When the
user explicitly asked for a ticket or email preview and none was produced, the
agent asks once for it; if that follow-up fails validation or runs out of
budget, the answer that had already passed stands.
An ungrounded turn ends in a refusal rather than a speculative policy answer:
either no search found a grounded passage, or the question uses a term that
neither the corpus nor the HR data contains. A narrower follow-up search that
finds nothing quotable does not undo an earlier grounded result. Section
headings count as corpus words, because chunk text does not repeat them.
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
and `start_date`. Expected argument/lookup failures return structured errors
that list each request type with its required arguments. A remote-work request
that carries a country and both dates is evaluated as `international_remote_work`.

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

`LLM_MAX_CALLS_PER_TURN=8` (16 on the deployed service), hourly 60, daily 100 count every provider request
that may have been processed, including server errors and fallbacks. A request
refused before any work (a 429, a 413 oversized request or a retired model)
costs no tokens and is refunded, so routine rate-limit churn cannot end a turn
that still has budget. `CHAT_MAX_REQUESTS_PER_HOUR=12` and single-chat admission
prevent overlapping runs on the shared key.

SDK automatic retries are disabled; `agent/llm.py` decides every retry. A
rate-limited model is passed over for the next model, which has its own
per-minute budget, and is not led with again until its `Retry-After` has passed.
A call waits only when every compatible model is rate-limited, for the soonest
to recover, at most 60 seconds per call. Longer limits produce a clear quota
response instead of a held request. Retired models (`model_decommissioned`,
`model_not_found`) are skipped until restart. `tests/test_llm_resilience.py`
replays the longest demo task against a simulated free tier (8,000 TPM per
model, plus one retired model still configured) at four response speeds. Every
simulated run completes within 7 of the 8 per-turn calls. The simulation
scripts correct tool arguments, so it does not cover a fallback model retrying
a wrong one, which is what ended the follow-up check in the
[deployed measurement](#deployed-measurement-3-october-2026).

The default maximum orchestration length is 12 steps (16 on the deployed
service); the API-attempt budget may stop it earlier. Context has a 6,200 estimated-token ceiling, with a 1,200-token
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

`agent/demo_tasks.py` is the single source of the UI/verifier prompts. Each
names its checks in order: the free tier's per-model token limit usually moves a
multi-step task onto a smaller fallback model, which skipped the compliance
check or the ticket preview when left to choose (see the
[deployed measurement](#deployed-measurement-3-october-2026)). They also name
each policy section by number and heading. A paraphrased search can fall just
below the similarity floor ("outside home country" scores 0.554 against 0.56),
while the heading itself scores 0.754, and the number can be read directly with
`get_policy_section`.

**International workflow:** read Maya's rolling usage, check 2026-10-05 through
2026-11-15 in Portugal, and read POL-INTL-001 §2 The 30-Day Rule and
POL-REMOTE-001 §6 Temporary Work Outside the Home Country.
Expected: 12 prior days plus 42 requested exceeds the 30-day limit by 24.
Explain remaining allowance and conditional alternatives without inventing
rules, citing both policies.

**PTO workflow:** check Jonas's balance and three business days beginning
2026-09-21, read POL-PTO-001 §3.1 Notice Requirements and §3.3 Insufficient
Balance, then
prepare a mock ticket preview. Expected: 13 available hours versus 24 requested;
6 days' notice versus 10 required. Explain exception routes accurately. The user
confirms the exact preview in a separate click, which returns the ticket ID with
zero additional LLM calls.

Both must show successful MCP retrieval and structured-data calls, supporting
inline citations, final answers and operational traces. The verifier scores
them as cases C01 and C02 and also requires citations to both policies for the
international task and a ticket preview for PTO. It rejects missing evidence
and incorrect gold facts, rather than accepting counts alone.

## Evaluation methodology and results

There are **30 cases** across retrieval, lookup, reasoning, multi-hop, refusal,
ambiguity and action safety, with executable gold facts in `evaluation/cases.py`.
The suite includes a parental-leave regression and a two-document security/
equipment question that requires both document citations.

Scorer version **3** reports the metrics below. Version 3 accepts a value
written with a zero fraction ("13.0 hours" for 13) and the orchestrator's own
decline wording as a refusal; both were false negatives in the deployed
measurement.

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
request took about **68 seconds**, far from a 2-6 second reply, and
health-endpoint latency must not be substituted for chat latency.
`evidence/cold-start.json` records a historical 52.5-second platform cold start;
it is not a measurement of the current code.

### Deployed measurement (3 October 2026)

`scripts/verify_deployed_tasks.py` sent eight requests back to back to the
deployed build `291b50a`: both demonstrations and cases R03, L01, C03, X03, A01
and S01. All eight shared one free-tier per-minute token budget.

| Measure | Result |
|---|---|
| Wall-clock time per request | median 15.1 s, maximum 46.9 s (PTO demonstration) |
| Waiting on provider rate limits | 95.5 s across the eight requests |
| Model calls and tokens | 28 calls, 86,820 tokens |
| Passed at capture, scorer 2 | 4/8 (R03, C03, A01, S01) |
| Same responses re-scored, scorer 3 | 6/8 (adds L01 and X03) |

Each failure was traced to a cause:

- **L01 and X03** were scorer false negatives: "13.0 hours" did not match the
  expected 13, and the orchestrator's own decline did not count as a refusal.
- **International** answered correctly but skipped the compliance check and
  cited only POL-INTL-001, although a POL-REMOTE-001 section had been retrieved.
- **PTO** ran on the fallback models. The first answer claimed 20 days' notice
  instead of 10. The evidence check caught it, but the model replied to the
  correction with a note about notice alone, which replaced the answer and lost
  the balance finding. It never drafted the ticket preview.

Changes made in response: scorer 3; a correction turn that asks for the complete
answer; a one-time follow-up for a requested preview that cannot cost a verified
answer; and demonstration prompts that name their checks. The demonstration
traces in that evidence file predate these changes.

A follow-up check of the international task on build `d0e8733`
(`evidence/deployed-20261003T163450.json`) failed before answering. The
fallback model looked up the profile unasked, and its document-scoped search
for "outside home country" scored 0.554 against the 0.56 floor. It then sent
the compliance check as `remote_work` with a country and dates, which the
server rejected as an arrangement change. Two retries with invented arguments
used the remaining calls, and the ninth call reached the per-turn budget.
Changes made in response: such a request is evaluated as international remote
work; errors list each request type with its arguments; a narrow search miss no
longer overrides an earlier grounded result; section headings count as corpus
words; and the prompts name each section by number and heading.

Build `c88530a` then passed both demonstrations at the first attempt, one
request each, on the fallback model `qwen/qwen3.8-27b`
(`evidence/deployed-20261003T172000.json`, `evidence/deployed-20261003T172114.json`).
International took 7.4 seconds and PTO 5.4 seconds, six model calls each, with
no rate-limit waits. The international compliance check succeeded on its first
call, and confirming the PTO preview created the mock ticket in a separate
request. Two passing runs show that the workflows can complete, not a success rate.

During the video recording on 4 October (same code, build `79c1444`), both
tasks passed in several takes, but three runs stopped early, each with an
explicit message rather than a guessed answer:

- A PTO run reached the per-turn call cap of 8. The cap was raised to 12, then
  16, as a Render environment change with no code change.
- A PTO run on a fallback model called extra tools (`lookup_employee_profile`,
  `list_policy_documents`, `search_policy_documents`) and reached the 12-step
  limit. The step limit was raised to 16 the same way.
- One later run exceeded the context budget and was stopped before an
  oversized request was sent.

A take was also refused by the local hourly limit of 60 calls, as designed.
The longer runs came from the fallback models: lower stop rates need better
behaviour from those models, not higher limits.

### Current offline evidence

`evaluation/results/retrieval_eval.json` and `.md` are generated by the local,
six-configuration retrieval ablation. They measure document-level hit rate/MRR
and abstention, **not** answer quality. Consult their timestamps/index context
rather than mixing them with old full-agent results.

With the 221-chunk index, the offline sample contains 17 graded queries
and eight out-of-corpus queries:

| Retrieval | Document hit@1 | Document hit@6 | All expected docs in top 6 | MRR | False accepts | False refusals |
|---|---|---|---|---|---|---|
| Hybrid + both gates | 0.88 | 1.00 | 17/17 | 0.941 | 0/8 | 0/17 |
| Dense + both gates | 1.00 | 1.00 | 16/17 | 1.000 | 0/8 | 0/17 |
| BM25 + both gates | 0.59 | 0.94 | 16/17 | 0.728 | 0/8 | 0/17 |
| Hybrid without lexical gate | 0.88 | 1.00 | 17/17 | 0.941 | 8/8 | 0/17 |

Dense wins top-one ranking on this small sample; hybrid has no demonstrated
ranking advantage here. The retained hybrid configuration should not be
described as outperforming dense. Hit rates count any expected document, so the
"all expected docs" column, computed from `per_query` in the same JSON, checks
multi-document coverage. Hybrid is kept because it is the only retriever that
returned every expected document for all 17 queries. Dense missed POL-PERF-001
on the two-document case C04, and BM25 missed POL-INTL-001 on C05.

The automated suite covers the real MCP transport and app lifecycle, policy rules,
preserved section addresses, scoring false positives, session-bound single-use
confirmation, cache behavior and quota limits. Chat-completion calls are blocked
inside tests; a green test run consumes no LLM quota.

The 30-case suite has not been run live against the current code. The deployed
measurement covers eight requests and is not a completion-rate estimate.

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
