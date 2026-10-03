"""LLM provider abstraction.

Both Groq and Google Gemini expose OpenAI-compatible chat-completions endpoints,
so a single client shape covers them; only `base_url`, key and model differ.

Degradation is a chain of *models*, not merely of providers. Groq meters its
free tier per model -- measured directly: two back-to-back calls to different
Groq models each reported `x-ratelimit-remaining-tokens` against a full 8,000
bucket rather than a shared, cumulatively drained one. A second Groq model is
therefore a genuinely fresh budget, and rotating to it costs one request where
waiting out a 429 costs tens of seconds. The cross-provider hop to Gemini sits
behind it as the last resort.

Rotation is strictly reactive: it happens because a call failed, never because
a quota header looked close. Predictive switching would make the agent
non-deterministic, and an evaluation whose model silently drifts mid-suite
cannot support a groundedness or latency claim. The chain does, however,
*remember* a failure it already observed -- a model that returns 429 is passed
over for a cooldown rather than re-tried at the head of the chain on the very
next call. Without that memory an evaluation suite paid two failed requests and
roughly 35 seconds of dead time on every single case.

This is the documented graceful-degradation path required by project req. §4
(handle failures gracefully) and is surfaced in the agent trace.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from openai import APIStatusError, OpenAI

from config import settings

log = logging.getLogger(__name__)

_requests: deque[float] = deque()
_quota_lock = threading.Lock()
_turn_calls: ContextVar[list[int] | None] = ContextVar("turn_calls", default=None)


class QuotaExceeded(RuntimeError):
    """A provider limit or a local spending guard stopped the request."""

    status_code = 429


@contextmanager
def call_budget():
    count = [0]
    token = _turn_calls.set(count)
    try:
        yield count
    finally:
        _turn_calls.reset(token)


def reserve_call() -> float:
    """Reserve one provider request against the local guards before it is sent.

    Returns the reservation, so `release_call` can refund it if the provider
    refuses the request without doing any work.
    """
    if not settings.llm_enabled:
        raise QuotaExceeded("LLM calls are disabled (LLM_ENABLED=false); no quota was used.")
    now = time.monotonic()
    with _quota_lock:
        while _requests and _requests[0] <= now - 86400:
            _requests.popleft()
        turn = _turn_calls.get()
        if turn is not None and turn[0] >= settings.llm_max_calls_per_turn:
            raise QuotaExceeded("Per-turn API-call budget reached. Narrow the request.")
        if len(_requests) >= settings.llm_max_calls_per_day:
            raise QuotaExceeded("Local daily API-call budget reached; wait before demonstrating.")
        if sum(t > now - 3600 for t in _requests) >= settings.llm_max_calls_per_hour:
            raise QuotaExceeded("Local hourly API-call budget reached; wait before retrying.")
        _requests.append(now)
        if turn is not None:
            turn[0] += 1
    return now


def release_call(stamp: float) -> None:
    """Refund a reservation the provider refused before spending any tokens.

    A 429, an oversized-request 413 or a retired model costs no tokens.
    Counting them would let rate-limit churn, which is routine on a free tier,
    end a turn that still had budget. Anything the provider may have processed
    stays counted.
    """
    with _quota_lock:
        try:
            _requests.remove(stamp)
        except ValueError:
            return
        turn = _turn_calls.get()
        if turn is not None and turn[0] > 0:
            turn[0] -= 1


_ENDPOINTS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


class NoProviderConfigured(RuntimeError):
    """Raised when no provider has an API key set."""


class MalformedToolCall(RuntimeError):
    """The model emitted a tool call the provider rejected as invalid.

    Groq validates tool-call arguments server-side against the schema we sent
    and returns `400 tool_use_failed` when they do not match. Two variants were
    observed, both from the weaker fallback models the free tier pushes work
    onto under load: a boolean written as the string `"False"`, and a call cut
    off mid-argument by the token cap.

    It is worth a distinct exception because the usual responses are both
    wrong. Retrying the identical request cannot help -- the request was fine,
    the model's output was not -- and rotating to another model merely pays for
    the same mistake somewhere else while burning a second model's daily quota.
    The orchestrator recovers instead by telling the model what it got wrong,
    which is cheap, keeps the run on the model that has context, and is the
    difference between a failed task and a corrected one.
    """

    def __init__(self, message: str, failed_generation: str = "") -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


def _error_detail(exc: APIStatusError) -> dict[str, Any]:
    """The provider's error object, whether or not it is still enveloped.

    The SDK unwraps an OpenAI-style `{"error": {...}}` body before storing it
    on the exception, so production sees the inner object. Accepting both
    shapes keeps the checks below independent of that SDK detail.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return {}
    inner = body.get("error", body)
    return inner if isinstance(inner, dict) else {}


def _error_code(exc: APIStatusError) -> str:
    return str(getattr(exc, "code", None) or _error_detail(exc).get("code") or "")


def _malformed_tool_call(exc: APIStatusError) -> MalformedToolCall | None:
    """Recognise a server-side tool-argument rejection, or return None."""
    if _error_code(exc) != "tool_use_failed":
        return None
    error = _error_detail(exc)
    return MalformedToolCall(
        str(error.get("message") or exc), str(error.get("failed_generation") or "")
    )


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    # Whether this model can be handed a transcript that already contains an
    # assistant tool-call to replay.
    #
    # Gemini cannot, through its OpenAI-compatible endpoint. Replaying one
    # returns 400 "Function call is missing a thought_signature in functionCall
    # parts", and the signature is not exposed by the compat layer, so there is
    # nothing to send back. Verified against gemini-3.5-flash with thinking
    # left on, `reasoning_effort="none"`, `thinking_budget=0` and
    # `include_thoughts=false` -- all four fail identically. The first turn of a
    # conversation is fine, because there is nothing to replay yet; only
    # continuation fails.
    replays_tool_calls: bool = True


def _provider_config(name: str) -> ProviderConfig | None:
    if name == "groq" and settings.groq_api_key:
        return ProviderConfig("groq", _ENDPOINTS["groq"], settings.groq_api_key, settings.groq_model)
    if name == "gemini" and settings.gemini_api_key:
        return ProviderConfig(
            "gemini",
            _ENDPOINTS["gemini"],
            settings.gemini_api_key,
            settings.gemini_model,
            replays_tool_calls=False,
        )
    return None


def provider_chain() -> list[ProviderConfig]:
    """Configured fallback targets, primary first. Empty if nothing is configured.

    This is a chain of *models*, not just of providers. Groq meters its free
    tier per model, so a second Groq model is a separate token budget rather
    than the same wall hit twice -- which makes it a real mitigation and a
    cheaper one than crossing to another vendor. The cross-provider hop is kept
    behind it as the last resort.
    """
    primary = settings.llm_provider.lower().strip()
    order = [primary] + [n for n in _ENDPOINTS if n != primary]
    chain = [c for c in (_provider_config(n) for n in order) if c is not None]

    alternates = [
        m.strip() for m in settings.groq_fallback_models.split(",") if m.strip()
    ]
    if settings.groq_api_key and alternates:
        # Directly after the primary when Groq leads, otherwise appended: the
        # point is a fresh budget, not a particular vendor ordering.
        at = 1 if chain and chain[0].name == "groq" else len(chain)
        for offset, model in enumerate(m for m in alternates if m != settings.groq_model):
            chain.insert(
                at + offset,
                ProviderConfig("groq", _ENDPOINTS["groq"], settings.groq_api_key, model),
            )
    return chain


def available_providers() -> list[str]:
    seen: list[str] = []
    for cfg in provider_chain():
        if cfg.name not in seen:
            seen.append(cfg.name)
    return seen


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
    prompt_tokens: int = 0
    completion_tokens: int = 0


# Groq's free tier is capped on tokens-per-minute, and every agent step resends
# the full tool manifest plus the transcript so far, so 429s are routine rather
# than exceptional. They are handled here instead of by the SDK so that the
# wait can be measured and reported separately -- and so the server's own
# Retry-After is honoured rather than guessed at with blind exponential backoff.
#
# When another model is still available, waiting is the worse option: the next
# entry in the chain has its own token budget, so rotating to it costs one
# request instead of tens of seconds. A call waits only when every compatible
# model is rate-limited, and then for whichever recovers first. An empty bucket
# refills within a minute, so the wait is capped at one; anything longer is a
# daily limit or an outage, and is reported instead of holding the request open.
MAX_WAIT_PER_CALL_SECONDS = 60.0

# Backstop against a provider that keeps refusing with tiny Retry-After values.
# A healthy call needs at most two attempts per model.
MAX_ATTEMPTS_PER_CALL = 12

# How long a model is passed over after it rate-limits. Groq's bucket is
# per-minute, so a minute is the natural refill horizon; the server's own
# Retry-After wins when it sends one.
DEFAULT_COOLDOWN_SECONDS = 60.0
MAX_COOLDOWN_SECONDS = 120.0

# Models observed to be rate-limited, and when they are worth trying again.
# Keyed "provider/model".
#
# Without this, a saturated primary is re-tried at the head of the chain on
# *every* call: measured over an evaluation suite, each case burned two failed
# requests and ~35s before reaching a model that could answer. The chain still
# rotates only because a call actually failed -- nothing here predicts a limit
# from a quota header, which would make model choice drift mid-suite and
# invalidate any latency or groundedness claim. This only stops the system
# forgetting a failure it already observed.
_cooldown_until: dict[str, float] = {}

# Models the provider has retired or does not serve. Unlike a rate limit this
# does not refill, so the model is skipped until the process restarts. Groq
# retires free-tier models on a published schedule; a shutdown date passing
# must not turn a configured fallback into a failed task.
_unavailable: set[str] = set()
_RETIRED_CODES = frozenset({"model_decommissioned", "model_not_found"})

# Failures specific to one model or one moment, so another model may succeed.
# 413 is Groq's "request too large for this model's per-minute limit": refused
# before any work, like a 429, but the same model will refuse it again.
_TRANSIENT_STATUSES = frozenset({408, 413, 500, 502, 503, 504})


def reset_cooldowns() -> None:
    """Forget every recorded rate limit and retired model. Used by tests."""
    _cooldown_until.clear()
    _unavailable.clear()


def _cooldown_key(cfg: ProviderConfig) -> str:
    return f"{cfg.name}/{cfg.model}"


def _mark_cooling(cfg: ProviderConfig, seconds: float | None = None) -> None:
    wait = DEFAULT_COOLDOWN_SECONDS if seconds is None else seconds
    wait = max(1.0, min(wait, MAX_COOLDOWN_SECONDS))
    _cooldown_until[_cooldown_key(cfg)] = time.monotonic() + wait


def _replays_tool_calls_required(messages: list[dict[str, Any]]) -> bool:
    """True when the transcript already contains an assistant tool-call.

    That is the precise condition under which Gemini's OpenAI-compatible
    endpoint rejects the request, so it is the precise condition for excluding
    it -- not merely "tools were offered". A first turn that offers tools is
    fine on any provider; only continuing a tool loop is not.
    """
    return any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in messages
    )


def _ordered_chain(chain: list[ProviderConfig]) -> list[tuple[int, ProviderConfig]]:
    """Chain order, with recently rate-limited models moved to the back.

    Returns (original_index, cfg) so `fell_back` keeps meaning "not the
    configured primary" rather than "not first after reordering".

    Models still cooling are not dropped -- a cooldown is an informed guess,
    and if every model is cooling the call must still be attempted. They are
    just tried last, soonest-to-recover first.
    """
    now = time.monotonic()
    ready: list[tuple[int, ProviderConfig]] = []
    cooling: list[tuple[float, int, ProviderConfig]] = []
    for index, cfg in enumerate(chain):
        until = _cooldown_until.get(_cooldown_key(cfg), 0.0)
        if until > now:
            cooling.append((until, index, cfg))
        else:
            ready.append((index, cfg))
    cooling.sort(key=lambda item: item[0])
    return ready + [(index, cfg) for _, index, cfg in cooling]


def _retry_after_seconds(exc: APIStatusError) -> float | None:
    """The server's own Retry-After in seconds, or None if it sent none usable."""
    try:
        return float(exc.response.headers.get("retry-after", ""))
    except (AttributeError, TypeError, ValueError):
        return None


# Reasoning models wrap their scratchpad in these. Groq can be asked to hide it
# per-request, but the parameter is model-specific and silently ignored by the
# models that do not support it, so it cannot be relied on across a rotation
# chain that mixes reasoning and non-reasoning models.
_REASONING_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning)\s*>.*?<\s*/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_UNCLOSED_REASONING = re.compile(r"<\s*(think|thinking|reasoning)\s*>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str | None) -> str:
    """Remove a reasoning model's scratchpad from assistant text.

    The trace is shown to the user, and the project brief is explicit that it
    must carry concise operational detail rather than chain-of-thought. The
    Groq fallback chain includes reasoning models, so without this a rotation
    triggered by a rate limit -- an infrastructure event the user never sees --
    would start rendering raw private reasoning in the UI.

    Stripping the text here rather than in the view layer means the transcript
    replayed to the provider on the next turn is also clean, so the scratchpad
    cannot re-enter through the context either. An unterminated block is
    dropped to the end: a scratchpad truncated by the token cap has no answer
    after it to preserve.
    """
    if not text:
        return ""
    cleaned = _REASONING_BLOCK.sub("", text)
    cleaned = _UNCLOSED_REASONING.sub("", cleaned)
    return cleaned.strip()


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Call the first model in the chain that can serve the request.

    A model that rate-limits is passed over for the next, which has its own
    budget; a retired one is skipped for good; one that fails transiently is
    skipped for this call. The call waits only when every compatible model is
    rate-limited, and then for the soonest to recover, within
    MAX_WAIT_PER_CALL_SECONDS. Requests refused before any work was done -- a
    429, a 413 or a retired model -- are refunded to the local guards.
    """
    chain = provider_chain()
    if not chain:
        raise NoProviderConfigured(
            "Set GROQ_API_KEY or GEMINI_API_KEY (see .env.example)."
        )

    temp = settings.agent_temperature if temperature is None else temperature
    cap = settings.llm_max_tokens if max_tokens is None else max_tokens
    request: dict[str, Any] = {"messages": messages, "temperature": temp, "max_tokens": cap}
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"

    # Exclude models that cannot continue this particular transcript. Doing it
    # up front matters: reached mid-run, Gemini raises a 400 that aborts the
    # whole agent turn, so a rate-limited Groq chain turned a recoverable
    # slowdown into a failed task. Two evaluation cases failed exactly that way.
    # Entries keep their index in the configured chain, so `fell_back` still
    # means "not the configured primary" even when the primary is excluded.
    needs_replay = bool(tools) or _replays_tool_calls_required(messages)
    if needs_replay and not any(cfg.replays_tool_calls for cfg in chain):
        raise NoProviderConfigured("Configure a Groq key for multi-step tool workflows.")

    run_started = time.perf_counter()
    attempts = 0
    waited_s = 0.0
    tried: set[str] = set()
    failed: set[str] = set()
    last_error: Exception | None = None

    while True:
        order = [
            (index, cfg) for index, cfg in _ordered_chain(chain)
            if (cfg.replays_tool_calls or not needs_replay)
            and _cooldown_key(cfg) not in _unavailable
            and _cooldown_key(cfg) not in failed
        ]
        if not order:
            break
        if attempts >= MAX_ATTEMPTS_PER_CALL:
            raise QuotaExceeded(
                f"The model provider refused this request {attempts} times in a row. "
                "Wait a minute, then try again."
            ) from last_error

        # Ready models come first, then the soonest to recover.
        provider_index, cfg = order[0]
        delay = _cooldown_until.get(_cooldown_key(cfg), 0.0) - time.monotonic()
        if delay > 0 and waited_s + delay <= MAX_WAIT_PER_CALL_SECONDS:
            log.info("every compatible model is rate-limited; waiting %.1fs for %s/%s",
                     delay, cfg.name, cfg.model)
            time.sleep(delay)
            waited_s += delay
        elif delay > 0:
            # Nothing recovers within the wait budget. A cooldown is only an
            # estimate, so each model not yet asked during this call is asked
            # once: only the provider's own refusal may end the call.
            untried = [entry for entry in order if _cooldown_key(entry[1]) not in tried]
            if not untried:
                raise QuotaExceeded(
                    "Every language model is rate-limited right now, so I stopped rather "
                    "than keep you waiting. Try again in a minute; if this keeps happening, "
                    "the provider's daily allowance is used up."
                ) from last_error
            provider_index, cfg = untried[0]

        key = _cooldown_key(cfg)
        stamp = reserve_call()
        attempts += 1
        tried.add(key)
        # max_retries=0: every retry is decided here, so its cost is observable.
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60.0, max_retries=0)
        try:
            completion = client.chat.completions.create(model=cfg.model, **request)
        except APIStatusError as exc:
            last_error = exc
            malformed = _malformed_tool_call(exc)
            if malformed is not None:
                # Not a transport failure: the request was valid and the
                # model's own output was not. Surface it so the caller can
                # correct the model rather than rotating.
                raise malformed from exc
            if exc.status_code == 404 or _error_code(exc) in _RETIRED_CODES:
                release_call(stamp)
                _unavailable.add(key)
                log.warning("%s is no longer served (%s); skipping it", key, exc.status_code)
                continue
            if exc.status_code == 429:
                # Observed, not predicted: this model has just said its bucket
                # is empty, so it is not led with again until it has refilled.
                release_call(stamp)
                _mark_cooling(cfg, _retry_after_seconds(exc))
                log.info("%s rate-limited; moving on", key)
                continue
            if exc.status_code in _TRANSIENT_STATUSES:
                if exc.status_code == 413:
                    release_call(stamp)
                failed.add(key)
                log.warning("%s failed (%s); moving on", key, exc.status_code)
                continue
            raise
        except Exception as exc:  # network or timeout: the request may have been served
            last_error = exc
            failed.add(key)
            log.warning("%s errored (%s); moving on", key, exc)
            continue

        latency_ms = (time.perf_counter() - run_started) * 1000.0
        throttle_ms = waited_s * 1000.0
        # A success proves the bucket has room again, whatever was assumed.
        _cooldown_until.pop(key, None)
        msg = completion.choices[0].message
        usage = getattr(completion, "usage", None)
        return LLMResponse(
            provider=cfg.name,
            model=cfg.model,
            message=msg,
            tool_calls=list(msg.tool_calls or []),
            text=strip_reasoning(msg.content),
            latency_ms=latency_ms,
            fell_back=provider_index > 0,
            service_ms=latency_ms - throttle_ms,
            throttle_ms=throttle_ms,
            attempts=attempts,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    if last_error is None:
        raise RuntimeError(
            "no configured model is served any more; update GROQ_MODEL or GROQ_FALLBACK_MODELS"
        )
    raise RuntimeError(f"every model in the fallback chain failed: {last_error}") from last_error
