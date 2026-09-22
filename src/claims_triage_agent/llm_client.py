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
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


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


class OpenAIChatCompletionsClient:
    """A thin ``LLMClient`` adapter over the OpenAI Chat Completions API.

    This is the "real" production implementation. It is intentionally
    isolated behind the same ``LLMClient`` protocol as ``FakeLLMClient``
    so nothing else in the codebase needs to know it exists; none of the
    test suite imports this class, so the ``openai`` package is only
    required if you actually construct one of these.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "OpenAIChatCompletionsClient requires the 'openai' package. "
                "Install it with `pip install openai`, or use FakeLLMClient "
                "for tests."
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        openai_tools = [
            {"type": "function", "function": schema} for schema in tools
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=openai_tools,
        )
        message = response.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (message.tool_calls or [])
        ]
        return LLMResponse(tool_calls=tool_calls, content=message.content or "")
