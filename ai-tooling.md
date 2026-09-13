# AI Tooling

Required by the assignment: a description of which AI code tools were used, how,
and what worked well and what did not.

## What was used

| Tool | Role |
|---|---|
| **GitHub Copilot CLI** (agentic mode, Claude models) | The primary tool. Used for essentially the whole build: scaffolding, the corpus, the MCP server, the agent loop, the tests, the diagnostics, and this documentation. |
| GitHub Copilot in-editor completion | Minor. Line-level completion while reading and adjusting generated code. |

No other code-generation tool was used. The LLM providers the *application*
calls at runtime (Groq, Gemini) are part of the system, not part of the
toolchain, and are described in [`design-and-evaluation.md`](design-and-evaluation.md).

## How it was used

The work was driven as a sequence of scoped phases rather than one large
generation. Phase 0 was a written plan with a rubric checklist; each subsequent
phase produced a runnable artifact that was executed and checked before moving
on — corpus, then ingestion, then retrieval, then the MCP server, then the agent,
then the web app, then tests, CI, deployment artifacts, evaluation, and docs.

Two working rules shaped the result more than any prompt did:

**Nothing was accepted without running it.** Every phase ended with a command
that either passed or failed: `pytest`, `scripts/validate_mock_data.py`,
`scripts/check_rules.py`, `scripts/healthcheck.py`, `ruff check`. Code that
looked correct and was never executed was treated as not written.

**Design claims had to be measured, not asserted.** Where the assistant proposed
a parameter — a similarity threshold, a chunk size, a value of *k* — the response
was to ask for a script that measures it. That produced
`scripts/calibrate_threshold.py`, `scripts/diagnose_abstention.py`,
`scripts/recalibrate_threshold.py`, `scripts/measure_memory.py`, and
`evaluation/run_retrieval_eval.py`. Several of those measurements contradicted
the original design, which is the point of having them.

## What worked well

**Volume with structure.** Twelve policy documents across four file formats,
twenty internally consistent employee records, six cross-referenced JSON
datasets, and 186 tests are more artifact than the timebox allowed by hand. The
corpus in particular benefited: the documents needed *interlocking rules* —
international work depending on tenure and visa class and a rolling day count —
and generating them together kept them consistent in a way that writing them
one at a time would not have.

**Rapid, disposable diagnostics.** The most valuable use was not writing
application code but writing throwaway measurement scripts. When the abstention
threshold was suspect, `scripts/diagnose_abstention.py` — which scores three
signals across two populations and prints the unmatched terms per query — took
minutes to produce. That script is what revealed that dense score, BM25 score,
and term coverage all overlap while the *identity* of unmatched terms separates
cleanly. That insight became `rag/vocabulary.py`, the project's strongest
technical result. Writing that diagnostic by hand would plausibly not have
happened, and the flawed threshold would have shipped.

**Obscure-detail recall.** Two examples that cost real time and would have cost
much more alone: the MCP Python SDK 2.1.1 returns snake_case attributes
(`server_info`, `input_schema`, `is_error`) while accepting camelCase aliases in
constructors; and Windows `GetProcessMemoryInfo` via `ctypes` silently reports
0.0 RSS unless `argtypes` is declared with `wintypes.HANDLE`, because the 64-bit
handle is otherwise truncated.

**Explaining its own failures.** When the rewritten `tests/test_mcp_server.py`
raised `RuntimeError: Attempted to exit cancel scope in a different task`, the
diagnosis — module-scoped async-generator fixtures yielding an MCP
`ClientSession` cross task boundaries under pytest-asyncio + anyio — was correct
and led straight to the fix (an `@asynccontextmanager` used inside each test
body).

## What did not work well

**Confident wrong architecture, early.** The first MCP server was written
against `FastMCP` from an older SDK. The installed version is 2.1.1, where the
class is `MCPServer` in `mcp.server.mcpserver`. The generated code was internally
coherent and did not exist. The fix was to stop trusting recalled API surface and
introspect the installed package. That pattern — plausible code against an API
that changed — was the single most common failure.

**Shadowed imports it could not see.** The MCP server package was initially
proposed as `mcp/`, which shadows the installed SDK on `sys.path`. The same class
of error appeared again in the Dockerfile, where `pip install .` puts the
project's own package names into `site-packages` ahead of `/app`, and again in
`.dockerignore`, where an `*.md` exclusion would have silently produced a clean
image with an empty policy corpus. None of these fail loudly. All three were
found by running things, not by reading them.

**Premature convergence on best practice.** Left alone, the assistant reliably
suggested adding components with good reputations — a cross-encoder reranker, a
vector database, an LLM-as-judge evaluator — without first checking whether any
measurement justified them. recall@6 is 1.00, so a reranker has no measured error
to fix; 184 vectors is a 276 KB NumPy array, so Chroma adds a dependency to
optimise something already free; the evaluation has known-correct answers, so a
judge adds variance and cost without information. Each rejection is documented in
`design-and-evaluation.md`. Each required deliberately asking "what would this
improve, and how would we see it?" rather than accepting the suggestion.

**Calibration that looked rigorous and was not.** The similarity threshold was
not guessed — it was calibrated on 15 positive and 10 negative questions that
separated cleanly. It still failed, because the *negatives were generated by the
same process that had the same blind spot*: they were questions about unrelated
subjects rather than about plausible benefits that do not exist. The measurement
was real; the sample was not representative. This is the sharpest lesson from the
project. An AI-generated evaluation set inherits the generator's assumptions, and
it will validate a design against the cases the generator found salient. The
failure only surfaced because a *second*, independently written negative set went
into the evaluation suite for a different purpose. The full account is in
[`evaluation/results/abstention_analysis.md`](evaluation/results/abstention_analysis.md).

**Stale self-reference.** Generated docstrings said "twenty-six cases" after the
suite had grown to 28. Small, but a reminder that generated prose about generated
code drifts, and that counts in documentation should be checked against the code
rather than trusted.

## Division of responsibility

The AI tooling produced most of the source text. The direction, the acceptance
criteria, and every decision that survived a measurement were mine:

- choosing that the system needs an abstention mechanism at all
- refusing to accept the threshold until an independent negative set tested it
- deciding the lexical gate should key on **two** stores rather than one, which
  follows from the architecture rather than from the data
- rejecting the reranker, the vector database, and the judge on measured grounds
- separating `/healthz` from `/health`, because a readiness check wired to a
  platform health probe restarts healthy processes forever
- pinning `as_of_date` so date-sensitive expectations do not drift

Correctness, security, and academic integrity of the submitted work are my
responsibility, and every claim in `design-and-evaluation.md` is reproducible
from a script in this repository.
