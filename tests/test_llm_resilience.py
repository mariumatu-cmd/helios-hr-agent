"""Keeping a multi-step task alive on a free tier.

The longest demo task makes seven model calls against Groq's free tier, where
each model has its own 8,000 tokens-per-minute bucket and one step costs most of
it. Rate limits during such a task are routine, and the provider also retires
models on its own schedule. Neither may end a task while some model can still
serve it, and neither may spend the local call budget, because a refused request
costs no tokens.

The provider is faked, but every error is built by the real SDK from a real HTTP
response, so the code under test sees exactly what production sees -- including
the SDK unwrapping the ``{"error": {...}}`` envelope. Time is virtual. Nothing
here makes a network call or spends quota.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import httpx2
import openai
import pytest

from agent import llm
from config import settings

GROQ = "https://api.groq.com/openai/v1"
_SDK = openai.OpenAI(api_key="test", base_url=GROQ, max_retries=0)

MESSAGES = [{"role": "user", "content": "How many PTO days do I have?"}]
TOOLS = [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}]


def sdk_error(status: int, code: str, message: str = "", headers=None, **extra):
    body = {"error": {"message": message or code, "type": "invalid_request_error",
                      "code": code, **extra}}
    response = httpx2.Response(
        status,
        headers=headers or {},
        content=json.dumps(body).encode(),
        request=httpx2.Request("POST", f"{GROQ}/chat/completions"),
    )
    return _SDK._make_status_error_from_response(response)


def rate_limited(retry_after: int | None = None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return sdk_error(429, "rate_limit_exceeded", "Rate limit reached", headers=headers)


def completion(prompt_tokens: int = 100):
    message = SimpleNamespace(content="done", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=10),
    )


class Clock:
    """Virtual time, so a wait is asserted rather than slept."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    perf_counter = monotonic

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class ScriptedProvider:
    """Replaces the SDK client; each model answers from its own script.

    The last outcome in a script repeats, so ``[error, completion()]`` means
    "refuse once, then serve".
    """

    def __init__(self, script: dict[str, list]) -> None:
        self.script = script
        self.calls: list[str] = []

    def __call__(self, **_client_options):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def create(self, *, model, **_request):
        self.calls.append(model)
        outcomes = self.script[model]
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def clock(monkeypatch):
    llm.reset_cooldowns()
    fake = Clock()
    monkeypatch.setattr(llm, "time", fake)
    yield fake
    llm.reset_cooldowns()


def use_chain(monkeypatch, *models: str) -> list[llm.ProviderConfig]:
    chain = [llm.ProviderConfig("groq", GROQ, "k", model) for model in models]
    monkeypatch.setattr(llm, "provider_chain", lambda: chain)
    return chain


def use_provider(monkeypatch, script: dict[str, list]) -> ScriptedProvider:
    provider = ScriptedProvider(script)
    monkeypatch.setattr(llm, "OpenAI", provider)
    return provider


# -- Malformed tool calls ------------------------------------------------------


def test_a_malformed_tool_call_is_recognised_in_the_shape_the_sdk_delivers():
    """The SDK strips the ``error`` envelope, so the check must not expect it."""
    exc = sdk_error(400, "tool_use_failed", "Failed to call a function",
                    failed_generation='{"confirmed": "False"}')

    malformed = llm._malformed_tool_call(exc)

    assert malformed is not None
    assert malformed.failed_generation == '{"confirmed": "False"}'


def test_chat_hands_a_malformed_tool_call_back_instead_of_rotating(monkeypatch, clock):
    use_chain(monkeypatch, "a", "b")
    provider = use_provider(monkeypatch, {
        "a": [sdk_error(400, "tool_use_failed", failed_generation="{}")],
        "b": [completion()],
    })

    with pytest.raises(llm.MalformedToolCall):
        llm.chat(MESSAGES, TOOLS)

    assert provider.calls == ["a"]


# -- Retired models ------------------------------------------------------------


@pytest.mark.parametrize("status, code", [(400, "model_decommissioned"), (404, "model_not_found")])
def test_a_retired_model_is_skipped_for_good_and_costs_no_budget(monkeypatch, clock, status, code):
    use_chain(monkeypatch, "retired", "live")
    provider = use_provider(monkeypatch, {
        "retired": [sdk_error(status, code)],
        "live": [completion()],
    })

    with llm.call_budget() as count:
        first = llm.chat(MESSAGES, TOOLS)
        second = llm.chat(MESSAGES, TOOLS)

    assert first.model == second.model == "live"
    assert provider.calls == ["retired", "live", "live"]
    assert count[0] == 2


def test_an_unrelated_bad_request_is_not_mistaken_for_a_retired_model(monkeypatch, clock):
    use_chain(monkeypatch, "a", "b")
    provider = use_provider(monkeypatch, {
        "a": [sdk_error(400, "invalid_request_error")],
        "b": [completion()],
    })

    with pytest.raises(openai.BadRequestError):
        llm.chat(MESSAGES, TOOLS)

    assert provider.calls == ["a"]


# -- Rate limits ---------------------------------------------------------------


def test_a_rate_limit_rotates_honours_retry_after_and_is_refunded(monkeypatch, clock):
    use_chain(monkeypatch, "a", "b")
    use_provider(monkeypatch, {"a": [rate_limited(7)], "b": [completion()]})

    with llm.call_budget() as count:
        response = llm.chat(MESSAGES, TOOLS)

    assert response.model == "b"
    assert response.fell_back
    assert response.attempts == 2
    assert count[0] == 1
    assert len(llm._requests) == 1
    assert llm._cooldown_until["groq/a"] == pytest.approx(clock.now + 7)
    assert clock.slept == []


def test_a_rate_limit_on_the_last_model_is_waited_out(monkeypatch, clock):
    use_chain(monkeypatch, "only")
    use_provider(monkeypatch, {"only": [rate_limited(12), completion()]})

    with llm.call_budget() as count:
        response = llm.chat(MESSAGES, TOOLS)

    assert response.model == "only"
    assert clock.slept == [pytest.approx(12)]
    assert response.throttle_ms == pytest.approx(12_000)
    assert count[0] == 1


def test_when_every_model_is_cooling_the_soonest_is_awaited(monkeypatch, clock):
    chain = use_chain(monkeypatch, "a", "b")
    llm._mark_cooling(chain[0], seconds=40)
    llm._mark_cooling(chain[1], seconds=9)
    use_provider(monkeypatch, {"a": [completion()], "b": [completion()]})

    response = llm.chat(MESSAGES, TOOLS)

    assert response.model == "b"
    assert clock.slept == [pytest.approx(9)]


def test_a_cooldown_alone_never_refuses_a_call(monkeypatch, clock):
    """A cooldown is an estimate; only the provider's own 429 may refuse."""
    chain = use_chain(monkeypatch, "only")
    llm._mark_cooling(chain[0], seconds=llm.MAX_COOLDOWN_SECONDS)
    use_provider(monkeypatch, {"only": [completion()]})

    response = llm.chat(MESSAGES, TOOLS)

    assert response.model == "only"
    assert clock.slept == []


def test_a_long_outage_is_reported_instead_of_holding_the_request(monkeypatch, clock):
    use_chain(monkeypatch, "only")
    provider = use_provider(monkeypatch, {"only": [rate_limited(1800)]})

    with llm.call_budget() as count, pytest.raises(llm.QuotaExceeded, match="rate-limited"):
        llm.chat(MESSAGES, TOOLS)

    assert provider.calls == ["only"]
    assert clock.slept == []
    assert count[0] == 0


def test_a_request_too_large_for_one_model_moves_on_without_charge(monkeypatch, clock):
    use_chain(monkeypatch, "a", "b")
    provider = use_provider(monkeypatch, {
        "a": [sdk_error(413, "rate_limit_exceeded", "Request too large")],
        "b": [completion()],
    })

    with llm.call_budget() as count:
        response = llm.chat(MESSAGES, TOOLS)

    assert response.model == "b"
    assert provider.calls == ["a", "b"]
    assert count[0] == 1


# -- Other failures ------------------------------------------------------------


def test_a_server_error_rotates_and_still_counts(monkeypatch, clock):
    """The provider may have done the work, so the attempt stays charged."""
    use_chain(monkeypatch, "a", "b")
    use_provider(monkeypatch, {"a": [sdk_error(503, "service_unavailable")], "b": [completion()]})

    with llm.call_budget() as count:
        response = llm.chat(MESSAGES, TOOLS)

    assert response.model == "b"
    assert count[0] == 2


def test_a_chain_that_fails_everywhere_reports_the_cause(monkeypatch, clock):
    use_chain(monkeypatch, "a", "b")
    provider = use_provider(monkeypatch, {
        "a": [sdk_error(503, "service_unavailable")],
        "b": [sdk_error(503, "service_unavailable")],
    })

    with pytest.raises(RuntimeError, match="503"):
        llm.chat(MESSAGES, TOOLS)

    assert provider.calls == ["a", "b"]


def test_a_refused_reservation_is_refunded(monkeypatch):
    monkeypatch.setattr(settings, "llm_max_calls_per_turn", 1)
    with llm.call_budget() as count:
        llm.release_call(llm.reserve_call())
        llm.reserve_call()
        with pytest.raises(llm.QuotaExceeded, match="Per-turn"):
            llm.reserve_call()
    assert count[0] == 1
    assert len(llm._requests) == 1


# -- End to end ----------------------------------------------------------------


class FreeTierGroq:
    """Groq's free tier, modelled per model.

    Each model has an 8,000 tokens-per-minute bucket that refills continuously
    and is charged the prompt plus ``max_tokens`` up front. A request the bucket
    cannot cover is refused with 429 and the Retry-After its deficit implies. A
    retired model answers 400 ``model_decommissioned``, as Groq does after a
    shutdown date.
    """

    TPM = 8_000

    def __init__(self, clock: Clock, retired: set[str], service_seconds: float) -> None:
        self.clock = clock
        self.retired = retired
        self.service_seconds = service_seconds
        self.level: dict[str, float] = {}
        self.updated: dict[str, float] = {}
        self.served: list[str] = []

    def __call__(self, **_client_options):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def create(self, *, model, messages, max_tokens, **_request):
        if model in self.retired:
            raise sdk_error(400, "model_decommissioned", f"The model `{model}` has been decommissioned")
        now = self.clock.now
        refill = (now - self.updated.get(model, now)) * self.TPM / 60
        level = min(self.TPM, self.level.get(model, self.TPM) + refill)
        self.level[model], self.updated[model] = level, now
        need = messages[-1]["prompt_tokens"] + max_tokens
        if level < need:
            raise rate_limited(math.ceil((need - level) * 60 / self.TPM))
        self.level[model] = level - need
        self.clock.now += self.service_seconds
        self.served.append(model)
        return completion(messages[-1]["prompt_tokens"])


# Context size of each model call in the international remote-work demo task,
# as recorded by the deployed service in evidence/deployed-tasks.json.
LONGEST_DEMO_TASK = [3419, 3841, 4265, 6098, 5107, 6174, 5710]


@pytest.mark.parametrize("service_seconds", [0.5, 2.0, 10.0, 30.0])
def test_the_longest_demo_task_finishes_on_the_free_tier(monkeypatch, clock, service_seconds):
    """Seven calls, a retired model still configured, and the deployed caps."""
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(
        settings, "groq_fallback_models", "openai/gpt-oss-20b,qwen/qwen3.8-27b,qwen/qwen3.6-27b"
    )
    monkeypatch.setattr(settings, "llm_max_tokens", 1200)
    monkeypatch.setattr(settings, "llm_max_calls_per_turn", 8)
    monkeypatch.setattr(settings, "llm_max_calls_per_hour", 60)
    monkeypatch.setattr(settings, "llm_max_calls_per_day", 100)
    groq = FreeTierGroq(clock, retired={"qwen/qwen3.6-27b"}, service_seconds=service_seconds)
    monkeypatch.setattr(llm, "OpenAI", groq)

    with llm.call_budget() as count:
        for tokens in LONGEST_DEMO_TASK:
            llm.chat([{"role": "user", "content": "step", "prompt_tokens": tokens}], TOOLS)

    assert len(groq.served) == len(LONGEST_DEMO_TASK)
    assert count[0] == len(LONGEST_DEMO_TASK)
    assert "qwen/qwen3.6-27b" not in groq.served
    assert sum(clock.slept) <= llm.MAX_WAIT_PER_CALL_SECONDS * len(LONGEST_DEMO_TASK)
