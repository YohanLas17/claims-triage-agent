"""Swappable LLM client interface used by the agent loop.

The agent (``agent.py``) never talks to a specific vendor SDK directly.
It depends only on the ``LLMClient`` protocol defined here, so the whole
tool-calling loop can be exercised in tests with ``FakeLLMClient`` and no
network access or API key, and swapped for a real provider in production
by passing a different implementation into ``ClaimsTriageAgent``.

The wire format follows the now-common OpenAI-style function-calling
shape (a list of ``tool_calls`` with ``name`` + JSON ``arguments`` on the
assistant message), since that shape is what most LLM providers'
tool-use APIs converge on today.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, cast


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model.

    ``provider_extra`` carries opaque, vendor-specific fields the agent
    loop must round-trip back to the provider unmodified but has no
    reason to understand -- e.g. Gemini's OpenAI-compatible endpoint
    attaches an ``extra_content.google.thought_signature`` to every
    function-call part and rejects the next turn if it isn't echoed
    back verbatim. Empty for providers (and FakeLLMClient) that don't
    need this.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    provider_extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """The model's response to one turn of the conversation.

    Exactly one of ``tool_calls`` or ``content`` is expected to be
    meaningful for a given turn, mirroring how tool-calling APIs behave:
    either the model wants to call tools, or it produced a final text
    answer.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str = ""


class LLMClient(Protocol):
    """Protocol every LLM backend (real or fake) must satisfy."""

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        """Given the running conversation and the available tool schemas,
        return the model's next turn."""
        ...


class FakeLLMClient:
    """A scripted ``LLMClient`` used in tests and eval.

    ``script`` is a list of ``LLMResponse`` objects returned in order, one
    per call to ``complete``. This lets tests exercise the agent loop
    deterministically -- including edge cases like "the model keeps
    calling tools forever" or "the model never cites a passage" -- without
    any network access or real model non-determinism.
    """

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self._calls = 0

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        if self._calls >= len(self._script):
            raise AssertionError(
                f"FakeLLMClient script exhausted after {self._calls} calls; "
                "the agent asked for another turn but the test did not "
                "script one. This usually means the agent looped more "
                "than the test expected."
            )
        response = self._script[self._calls]
        self._calls += 1
        return response

    @property
    def call_count(self) -> int:
        return self._calls


class ReferenceScriptLLMClient:
    """A stateless ``LLMClient`` that replays known-correct scripted
    trajectories, keyed by claim id, for the no-API-key demo app
    (``claims_triage_agent.demo``).

    Unlike ``FakeLLMClient`` (which tracks a call count on the instance and
    is built fresh per test), this client is shared across concurrent HTTP
    requests by ``api.create_app``. It has no mutable state: on every call
    it re-derives which claim and which turn it is by reading the
    conversation history handed to it -- the claim id from the first
    ``user`` message (``agent._build_claim_message`` always JSON-embeds the
    claim there) and the turn index from how many ``assistant`` messages
    already appear in ``messages``. That makes it safe to use for multiple
    concurrent ``/adjudicate`` requests without any locking.
    """

    def __init__(self, scripts: dict[str, list[LLMResponse]]) -> None:
        self._scripts = scripts

    def _claim_id_from_messages(self, messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            brace_index = content.find("{")
            if brace_index == -1:
                continue
            claim = json.loads(content[brace_index:])
            return claim["claim_id"]
        raise ValueError(
            "ReferenceScriptLLMClient could not find a claim in the "
            "conversation's user message."
        )

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        claim_id = self._claim_id_from_messages(messages)
        script = self._scripts.get(claim_id)
        if script is None:
            supported = ", ".join(sorted(self._scripts))
            raise ValueError(
                f"Demo mode does not have a scripted trajectory for claim "
                f"{claim_id!r}. It only supports these {len(self._scripts)} "
                f"synthetic demo claims: {supported}. Use `--llm openai` "
                f"in eval/run_eval.py (or wire up a real LLMClient) to "
                f"adjudicate arbitrary claims."
            )
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        if turn >= len(script):
            raise AssertionError(
                f"ReferenceScriptLLMClient script for {claim_id!r} exhausted "
                f"after {turn} turns; the agent asked for another turn than "
                f"the script provides."
            )
        return script[turn]


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the agent's vendor-neutral transcript into OpenAI wire format.

    ``agent.py`` (and ``agent_langgraph.py`` via the same helper) keeps
    the running conversation in a deliberately simple internal shape:
    assistant tool calls as ``{"id", "name", "arguments": dict}`` and tool
    results with a ``"name"`` key for readability. Neither is valid on
    the wire: the Chat Completions API (and OpenAI-compatible endpoints
    such as Gemini's) requires assistant tool calls as
    ``{"id", "type": "function", "function": {"name", "arguments": <JSON
    string>}}`` and rejects an unexpected ``"name"`` key on ``tool``
    messages. This function does that rewrite, and only that rewrite --
    it never mutates its input, so ``agent.py`` stays vendor-neutral and
    callers can keep using the internal shape for the audit trail.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant" and "tool_calls" in message:
            converted.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                            **tc.get("provider_extra", {}),
                        }
                        for tc in message["tool_calls"]
                    ],
                }
            )
        elif message.get("role") == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                }
            )
        else:
            converted.append(dict(message))
    return converted


def _parse_tool_call_arguments(raw: str) -> dict[str, Any]:
    """Parse one tool call's JSON arguments defensively.

    Real models occasionally emit malformed JSON, or valid JSON that
    isn't an object (e.g. a bare string or list). Either case is handed
    back to the agent as a sentinel dict rather than raising, so a bad
    tool call becomes a tool error the model can recover from (see
    ``agent.py``'s argument validation) instead of crashing the run.
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__invalid_json__": raw}
    if not isinstance(parsed, dict):
        return {"__invalid_json__": raw}
    return parsed


_MAX_RETRY_ATTEMPTS = 6
_RETRY_BASE_DELAY_SECONDS = 1.0
_RETRY_MAX_DELAY_SECONDS = 30.0


def _is_retryable_status_error(exc: BaseException) -> bool:
    """True for rate-limit (429) and server (5xx) errors.

    Duck-typed on a ``status_code`` attribute rather than importing and
    matching ``openai.APIStatusError`` subclasses directly, which keeps
    this function trivially unit-testable with a plain fake exception
    and works unchanged against any OpenAI-compatible provider whose SDK
    raises its own status-carrying exception type.
    """
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        return False
    return status_code == 429 or 500 <= status_code < 600


def _backoff_delay_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter, capped at 30s."""
    base = min(_RETRY_BASE_DELAY_SECONDS * (2**attempt), _RETRY_MAX_DELAY_SECONDS)
    return base * (0.5 + random.random())


class QuotaExceededError(Exception):
    """Raised instead of retrying when a 429 is a hard, day-scale quota.

    Discovered running eval against Gemini's free tier: its
    ``generate_content_free_tier_requests`` quota caps some models at as
    few as 20 requests *per day*, and a 429 for that violation is
    wire-identical to an ordinary per-minute rate limit except for which
    ``quotaId`` it names. Retrying a per-day cap with the backoff below
    (max ~30s a step) can never succeed and would just burn the
    remaining attempts for nothing, so ``complete()`` special-cases it
    and raises immediately instead. Callers that run many cases in a
    loop (``eval/run_eval.py``) can catch this specifically to stop
    cleanly and save partial progress, rather than crashing mid-run.
    """


def _is_daily_quota_exceeded(exc: BaseException) -> bool:
    """True for a 429 whose Google quota violation is a per-day cap.

    Duck-typed on ``status_code`` and ``body`` like
    ``_is_retryable_status_error`` above -- costs nothing for providers
    that set neither. Google's Gemini API reports which quota was hit in
    ``body["details"]``, a list of typed objects; a ``QuotaFailure``
    entry's ``violations[].quotaId`` names the specific quota (e.g.
    ``GenerateRequestsPerDayPerProjectPerModel-FreeTier`` vs. the
    per-minute equivalent) -- only the former can't be outlasted by
    in-process retrying.
    """
    if getattr(exc, "status_code", None) != 429:
        return False
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    for detail in body.get("details") or []:
        if not isinstance(detail, dict):
            continue
        for violation in detail.get("violations") or []:
            if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


class OpenAIChatCompletionsClient:
    """A thin ``LLMClient`` adapter over the OpenAI Chat Completions API.

    This is the "real" production implementation. It is intentionally
    isolated behind the same ``LLMClient`` protocol as ``FakeLLMClient``
    so nothing else in the codebase needs to know it exists; none of the
    test suite imports this class, so the ``openai`` package is only
    required if you actually construct one of these.

    Because it only talks to ``openai.OpenAI(base_url=...)``, this same
    class works against any OpenAI-compatible Chat Completions endpoint,
    not just OpenAI itself -- Gemini's ``v1beta/openai/`` endpoint, a
    local Ollama server, or anything else that speaks the same wire
    format.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "OpenAIChatCompletionsClient requires the 'openai' package. "
                "Install it with `pip install openai`, or use FakeLLMClient "
                "for tests."
            ) from exc
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        openai_messages = to_openai_messages(messages)
        openai_tools = [{"type": "function", "function": schema} for schema in tools]

        response = None
        for attempt in range(_MAX_RETRY_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=cast(Any, openai_messages),
                    tools=cast(Any, openai_tools),
                    temperature=self._temperature,
                )
                break
            except Exception as exc:
                if _is_daily_quota_exceeded(exc):
                    raise QuotaExceededError(str(exc)) from exc
                if not _is_retryable_status_error(exc) or attempt == _MAX_RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(_backoff_delay_seconds(attempt))
        assert response is not None  # the loop above always returns or raises

        message = response.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            function = getattr(tc, "function", None)
            if function is None:  # a non-function ("custom") tool call; not supported here
                continue
            provider_extra = dict(getattr(tc, "model_extra", None) or {})
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=function.name,
                    arguments=_parse_tool_call_arguments(function.arguments),
                    provider_extra=provider_extra,
                )
            )
        return LLMResponse(tool_calls=tool_calls, content=message.content or "")
