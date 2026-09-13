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
import time
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


def active_model() -> str:
    """The model that a call would actually use right now, provider-qualified.

    Reported in evaluation output: a score is only interpretable next to the
    model that produced it.
    """
    chain = provider_chain()
    return f"{chain[0].name}/{chain[0].model}" if chain else "none"


@dataclass
class LLMResponse:
    """Normalised result of one chat-completion call.

    Three timings, not one, because on a free tier they measure different
    things and conflating them makes the evaluation report the provider's
    quota rather than the system:

      service_ms   the successful HTTP call alone -- the model's actual work
      throttle_ms  time spent waiting out 429s before that call succeeded
      latency_ms   wall clock, service + throttle: what a user would feel
    """

    provider: str
    model: str
    message: Any                      # the raw assistant message (may carry tool_calls)
    tool_calls: list[Any]
    text: str | None
    latency_ms: float
    fell_back: bool
    service_ms: float = 0.0
    throttle_ms: float = 0.0
    attempts: int = 1


# Groq's free tier is capped on tokens-per-minute, and every agent step resends
# the full tool manifest plus the transcript so far, so 429s are routine rather
# than exceptional. They are retried here instead of by the SDK so that the wait
# can be measured and reported separately -- and so the server's own Retry-After
# is honoured rather than guessed at with blind exponential backoff.
MAX_RATE_LIMIT_RETRIES = 5
MAX_BACKOFF_SECONDS = 30.0


def _retry_after_seconds(exc: APIStatusError, attempt: int) -> float:
    """How long to wait before retrying, preferring the server's own answer."""
    header = ""
    try:
        header = exc.response.headers.get("retry-after", "") or ""
    except Exception:
        header = ""
    try:
        if header:
            return min(float(header), MAX_BACKOFF_SECONDS)
    except ValueError:
        pass
    return min(2.0 ** attempt, MAX_BACKOFF_SECONDS)


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int = 1600,
) -> LLMResponse:
    """Call the primary provider, falling back to the next on transient failure."""
    chain = provider_chain()
    if not chain:
        raise NoProviderConfigured(
            "Set GROQ_API_KEY or GEMINI_API_KEY (see .env.example)."
        )

    temp = settings.agent_temperature if temperature is None else temperature
    last_error: Exception | None = None

    for provider_index, cfg in enumerate(chain):
        # max_retries=0: retries are handled below so their cost is observable.
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60.0, max_retries=0)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        throttle_s = 0.0
        attempts = 0
        service_ms = 0.0
        completion = None
        started = time.perf_counter()

        for retry in range(MAX_RATE_LIMIT_RETRIES + 1):
            attempts += 1
            call_started = time.perf_counter()
            try:
                completion = client.chat.completions.create(**kwargs)
                service_ms = (time.perf_counter() - call_started) * 1000.0
                break
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code == 429 and retry < MAX_RATE_LIMIT_RETRIES:
                    wait = _retry_after_seconds(exc, retry)
                    log.info(
                        "provider %s rate-limited; waiting %.1fs (attempt %d)",
                        cfg.name, wait, attempts,
                    )
                    time.sleep(wait)
                    throttle_s += wait
                    continue
                if (
                    exc.status_code in (408, 429, 500, 502, 503, 504)
                    and provider_index + 1 < len(chain)
                ):
                    log.warning("provider %s failed (%s); falling back", cfg.name, exc.status_code)
                    break
                raise
            except Exception as exc:  # network / timeout
                last_error = exc
                if provider_index + 1 < len(chain):
                    log.warning("provider %s errored (%s); falling back", cfg.name, exc)
                    break
                raise

        if completion is None:
            continue  # exhausted this provider; try the next one

        latency_ms = (time.perf_counter() - started) * 1000.0
        msg = completion.choices[0].message
        return LLMResponse(
            provider=cfg.name,
            model=cfg.model,
            message=msg,
            tool_calls=list(msg.tool_calls or []),
            text=msg.content,
            latency_ms=latency_ms,
            fell_back=provider_index > 0,
            service_ms=service_ms,
            throttle_ms=throttle_s * 1000.0,
            attempts=attempts,
        )

    raise RuntimeError(f"all LLM providers failed: {last_error}")
