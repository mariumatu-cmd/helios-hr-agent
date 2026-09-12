"""Shared fixtures."""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server import data  # noqa: E402


@pytest.fixture(autouse=True)
def clean_writes():
    """Mock writes live in memory; clear them so cases cannot leak into each other."""
    data.reset_writes()
    yield
    data.reset_writes()


@pytest.fixture(scope="session")
def corpus():
    from config import settings
    from rag.ingest.parse import load_corpus

    return load_corpus(settings.corpus_dir)
