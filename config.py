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
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # -- Agent ----------------------------------------------------------------
    agent_max_steps: int = 8
    agent_temperature: float = 0.0

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
