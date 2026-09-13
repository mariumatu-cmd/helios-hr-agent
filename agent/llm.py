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


def _malformed_tool_call(exc: APIStatusError) -> MalformedToolCall | None:
    """Recognise a server-side tool-argument rejection, or return None."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != "tool_use_failed":
        return None
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


# Groq's free tier is capped on tokens-per-minute, and every agent step resends
# the full tool manifest plus the transcript so far, so 429s are routine rather
# than exceptional. They are retried here instead of by the SDK so that the wait
# can be measured and reported separately -- and so the server's own Retry-After
# is honoured rather than guessed at with blind exponential backoff.
#
# When another model is still available, waiting is the worse option: the next
# entry in the chain has its own token budget, so rotating to it costs one
# request instead of tens of seconds. The long retry budget is therefore spent
# only on the last entry, where there is nowhere else to go.
MAX_RATE_LIMIT_RETRIES = 5
RETRIES_BEFORE_ROTATING = 1
MAX_BACKOFF_SECONDS = 30.0

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


def reset_cooldowns() -> None:
    """Forget every recorded rate-limit. Used by tests to isolate cases."""
    _cooldown_until.clear()


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
    max_tokens: int | None = None,
) -> LLMResponse:
    """Call the primary model, rotating through the chain on transient failure."""
    chain = provider_chain()
    if not chain:
        raise NoProviderConfigured(
            "Set GROQ_API_KEY or GEMINI_API_KEY (see .env.example)."
        )

    temp = settings.agent_temperature if temperature is None else temperature
    cap = settings.llm_max_tokens if max_tokens is None else max_tokens
    last_error: Exception | None = None

    # Drop models that cannot continue this particular transcript. Doing it up
    # front matters: reached mid-run, Gemini raises a 400 that aborts the whole
    # agent turn, so a rate-limited Groq chain turned a recoverable slowdown
    # into a failed task. Two evaluation cases failed exactly that way. If the
    # filter would leave nothing, keep the chain as it is -- an attempt that
    # might fail still beats refusing to call anyone.
    #
    # Filtering the ordered list rather than the chain keeps `provider_index`
    # measured against the configured chain, so `fell_back` still means "not
    # the configured primary" even when the primary is the one removed.
    order = _ordered_chain(chain)
    if _replays_tool_calls_required(messages):
        usable = [entry for entry in order if entry[1].replays_tool_calls]
        if usable:
            order = usable

    for position, (provider_index, cfg) in enumerate(order):
        # max_retries=0: retries are handled below so their cost is observable.
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60.0, max_retries=0)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": cap,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        throttle_s = 0.0
        attempts = 0
        service_ms = 0.0
        completion = None
        started = time.perf_counter()
        has_next = position + 1 < len(order)
        max_retries = RETRIES_BEFORE_ROTATING if has_next else MAX_RATE_LIMIT_RETRIES

        for retry in range(max_retries + 1):
            attempts += 1
            call_started = time.perf_counter()
            try:
                completion = client.chat.completions.create(**kwargs)
                service_ms = (time.perf_counter() - call_started) * 1000.0
                break
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code == 400:
                    malformed = _malformed_tool_call(exc)
                    if malformed is not None:
                        # Not a transport failure: the request was valid and
                        # the model's own output was not. Surface it so the
                        # caller can correct the model rather than rotating.
                        raise malformed from exc
                if exc.status_code == 429:
                    # Observed, not predicted: this model has just told us its
                    # bucket is empty, so stop leading with it until it refills.
                    _mark_cooling(cfg, _retry_after_seconds(exc, retry))
                if exc.status_code == 429 and retry < max_retries:
                    wait = _retry_after_seconds(exc, retry)
                    log.info(
                        "%s/%s rate-limited; waiting %.1fs (attempt %d)",
                        cfg.name, cfg.model, wait, attempts,
                    )
                    time.sleep(wait)
                    throttle_s += wait
                    continue
                if exc.status_code in (408, 429, 500, 502, 503, 504) and has_next:
                    log.warning(
                        "%s/%s failed (%s); rotating to the next model",
                        cfg.name, cfg.model, exc.status_code,
                    )
                    break
                raise
            except Exception as exc:  # network / timeout
                last_error = exc
                if has_next:
                    log.warning(
                        "%s/%s errored (%s); rotating to the next model",
                        cfg.name, cfg.model, exc,
                    )
                    break
                raise

        if completion is None:
            continue  # exhausted this provider; try the next one

        latency_ms = (time.perf_counter() - started) * 1000.0
        # A success proves the bucket has room again, whatever we assumed.
        _cooldown_until.pop(_cooldown_key(cfg), None)
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

    raise RuntimeError(f"every model in the fallback chain failed: {last_error}")
