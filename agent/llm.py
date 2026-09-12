"""LLM provider abstraction.

Both Groq and Google Gemini expose OpenAI-compatible chat-completions endpoints,
so a single client shape covers them; only `base_url`, key and model differ.

The primary provider is configured by `LLM_PROVIDER`; the other is used as an
automatic fallback on rate-limit / server errors. That fallback is not just
convenience -- it is the documented graceful-degradation path required by
project req. §4 (handle failures gracefully) and is surfaced in the agent trace.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from openai import APIStatusError, OpenAI

from config import settings

log = logging.getLogger(__name__)

_ENDPOINTS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


class NoProviderConfigured(RuntimeError):
    """Raised when no provider has an API key set."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str


def _provider_config(name: str) -> ProviderConfig | None:
    if name == "groq" and settings.groq_api_key:
        return ProviderConfig("groq", _ENDPOINTS["groq"], settings.groq_api_key, settings.groq_model)
    if name == "gemini" and settings.gemini_api_key:
        return ProviderConfig(
            "gemini", _ENDPOINTS["gemini"], settings.gemini_api_key, settings.gemini_model
        )
    return None


def provider_chain() -> list[ProviderConfig]:
    """Configured providers, primary first. Empty if nothing is configured."""
    primary = settings.llm_provider.lower().strip()
    order = [primary] + [n for n in _ENDPOINTS if n != primary]
    return [c for c in (_provider_config(n) for n in order) if c is not None]


def available_providers() -> list[str]:
    return [c.name for c in provider_chain()]


@dataclass
class LLMResponse:
    """Normalised result of one chat-completion call."""

    provider: str
    model: str
    message: Any                      # the raw assistant message (may carry tool_calls)
    tool_calls: list[Any]
    text: str | None
    latency_ms: float
    fell_back: bool


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int = 1600,
) -> LLMResponse:
    """Call the primary provider, falling back to the next on transient failure."""
    import time

    chain = provider_chain()
    if not chain:
        raise NoProviderConfigured(
            "Set GROQ_API_KEY or GEMINI_API_KEY (see .env.example)."
        )

    temp = settings.agent_temperature if temperature is None else temperature
    last_error: Exception | None = None

    for attempt, cfg in enumerate(chain):
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60.0)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code in (408, 429, 500, 502, 503, 504) and attempt + 1 < len(chain):
                log.warning("provider %s failed (%s); falling back", cfg.name, exc.status_code)
                continue
            raise
        except Exception as exc:  # network / timeout
            last_error = exc
            if attempt + 1 < len(chain):
                log.warning("provider %s errored (%s); falling back", cfg.name, exc)
                continue
            raise

        latency_ms = (time.perf_counter() - started) * 1000.0
        msg = completion.choices[0].message
        return LLMResponse(
            provider=cfg.name,
            model=cfg.model,
            message=msg,
            tool_calls=list(msg.tool_calls or []),
            text=msg.content,
            latency_ms=latency_ms,
            fell_back=attempt > 0,
        )

    raise RuntimeError(f"all LLM providers failed: {last_error}")
