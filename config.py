"""Central configuration. Every tunable is an environment variable so that the
deployed service can be reconfigured without a code change (project req. §1, §7).

Secrets are read from the environment only and are never committed; `.env` is
git-ignored and `.env.example` documents the contract.
"""
from __future__ import annotations

import os
import pathlib
import random

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = pathlib.Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # -- LLM ------------------------------------------------------------------
    llm_provider: str = "groq"
    groq_api_key: str = ""
    # Groq retired the Llama 3.3 endpoints; `openai/gpt-oss-120b` is the
    # strongest tool-calling model on the current free-tier catalogue and was
    # verified to emit well-formed tool calls against this project's schemas.
    # `qwen/qwen3.8-27b` also works and is the obvious fallback within Groq.
    groq_model: str = "openai/gpt-oss-120b"
    # Groq meters tokens-per-minute *per model*, so each entry here is a
    # genuinely separate budget rather than the same wall hit twice. That
    # matters more than it sounds: at ~6,000 tokens per request against an
    # 8,000 TPM bucket, a single model allows roughly one agent step per
    # minute, which is not enough to finish a multi-step task. Rotating across
    # several models multiplies the available budget. Comma-separated, tried in
    # order after `groq_model`; every entry must support tool calling
    # (verified against the live catalogue -- Groq's `compound` models reject a
    # caller-supplied tool manifest with HTTP 400 and cannot drive this agent).
    groq_fallback_models: str = "openai/gpt-oss-20b,qwen/qwen3.8-27b,qwen/qwen3.6-27b"
    gemini_api_key: str = ""
    # Verified against the live catalogue on 2026-09-13. The 2.x Flash models
    # are now 404 "no longer available to new users", so a stale default here
    # would turn the fallback provider into a second point of failure rather
    # than a mitigation for the first.
    gemini_model: str = "gemini-3.5-flash"

    # -- Agent ----------------------------------------------------------------
    # A realistic HR request spans several sub-questions -- entitlement, the
    # employee's own balance or usage history, a policy condition the rule
    # engine does not encode, and then an action such as filing a ticket. Each
    # is a separate tool call plus a final synthesis turn, so a genuinely
    # multi-part task needs headroom. Measured: a four-part international
    # remote-work request exhausted an 8-step budget and returned an apology
    # instead of filing the ticket; the same request completes well inside 12.
    # The ceiling still exists to stop a confused run looping indefinitely.
    agent_max_steps: int = 12
    agent_temperature: float = 0.0

    # Cap on completion length. This is not only an output-shaping knob: Groq
    # reserves `max_tokens` against the per-minute budget whether or not the
    # model uses them, so every 1,000 here is 1,000 fewer available for the
    # prompt. 1,200 tokens is roughly 900 words -- ample for a cited HR answer
    # -- and buys back a meaningful slice of an 8,000 TPM bucket.
    llm_max_tokens: int = 1200

    # Groq's free tier meters tokens-per-minute, and that bucket covers the
    # prompt *and* the completion. A single request larger than the bucket can
    # therefore never succeed: retrying it waits for a refill that is already
    # big enough, and switching model just relocates the same failure. Because
    # every agent step resends the whole tool manifest plus every prior tool
    # result, an unbounded four-step run reliably grows past the limit and
    # stalls. The orchestrator compacts the oldest tool results to keep each
    # request under this budget.
    #
    # The value is derived, not guessed. Measured by `scripts/measure_context.py`:
    #
    #     system prompt        ~  595 tokens
    #     12-tool MCP manifest ~2,709 tokens   (resent every single step)
    #     fixed floor          ~3,304 tokens
    #
    # Budget + `llm_max_tokens` must clear the 8,000 TPM ceiling with room for
    # estimation error: 6,200 + 1,200 = 7,400. That leaves ~2,900 tokens of
    # transcript per step, which fits one full retrieval plus several structured
    # tool results before compaction has to discard anything.
    context_token_budget: int = 6200

    # -- Retrieval ------------------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    retrieval_k: int = 6
    retrieval_pool: int = 50
    retrieval_mode: str = "hybrid"

    # -- MCP ------------------------------------------------------------------
    mcp_transport: str = "stdio"
    mcp_server_url: str = "http://127.0.0.1:8765/mcp"

    # -- App ------------------------------------------------------------------
    port: int = 8000
    log_level: str = "INFO"
    seed: int = 42

    # Load the embedding model during startup instead of on the first query.
    # Off by default so tests and CLI scripts stay fast; on in the deployed
    # service, where the ~164 MB / ~10 s first-query cost would otherwise land
    # on a live user. Measured container total with the model resident is
    # 343 MB against Render's 512 MB cap -- see evaluation/results/
    # memory_footprint.txt -- so paying it eagerly is affordable.
    warm_embedder: bool = False

    # The mock datasets are a snapshot taken on this date. Evaluating date-
    # sensitive rules (PTO notice periods, the 12-month rolling international
    # window) against the real `date.today()` would make the expected answers
    # drift from one run to the next, so "today" is pinned and configurable.
    # Every tool that reasons about dates reads it from here.
    as_of_date: str = "2026-09-12"

    # -- Paths (not env-driven) ----------------------------------------------
    root: pathlib.Path = Field(default=ROOT, exclude=True)

    @property
    def corpus_dir(self) -> pathlib.Path:
        return self.root / "corpus"

    @property
    def mock_data_dir(self) -> pathlib.Path:
        return self.root / "mock_data"

    @property
    def index_dir(self) -> pathlib.Path:
        return self.root / "rag" / "index"

    @property
    def mcp_server_script(self) -> pathlib.Path:
        return self.root / "mcp_server" / "hr_server.py"


settings = Settings()


def apply_seeds(seed: int | None = None) -> int:
    """Fix seeds so chunking, evaluation sampling and any stochastic step are
    reproducible (project req. §1)."""
    s = settings.seed if seed is None else seed
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    try:
        import numpy as np

        np.random.seed(s)
    except ImportError:
        pass
    return s
