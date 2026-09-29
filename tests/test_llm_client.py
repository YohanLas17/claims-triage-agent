"""Tests for the vendor-neutral <-> OpenAI wire-format conversion and the
retry/backoff and malformed-JSON handling in ``OpenAIChatCompletionsClient``.

None of these tests need network access or an API key: ``openai.OpenAI`` is
monkeypatched with a small fake so the retry loop can be exercised
deterministically, and ``time.sleep`` is monkeypatched to a no-op so the
tests run instantly regardless of the real backoff delay.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from claims_triage_agent.llm_client import (
    OpenAIChatCompletionsClient,
    _backoff_delay_seconds,
    _is_retryable_status_error,
    _parse_tool_call_arguments,
    to_openai_messages,
)

# ---------------------------------------------------------------------------
# to_openai_messages
# ---------------------------------------------------------------------------


def test_to_openai_messages_converts_assistant_tool_calls():
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Adjudicate this claim."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "name": "lookup_policy", "arguments": {"policy_id": "POL-1001"}}
            ],
        },
    ]
    converted = to_openai_messages(messages)

    assert converted[0] == {"role": "system", "content": "You are an assistant."}
    assert converted[1] == {"role": "user", "content": "Adjudicate this claim."}
    assert converted[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {
                    "name": "lookup_policy",
                    "arguments": '{"policy_id": "POL-1001"}',
                },
            }
        ],
    }


def test_to_openai_messages_converts_tool_result_and_drops_name():
    messages = [
        {
            "role": "tool",
            "tool_call_id": "1",
            "name": "lookup_policy",
            "content": '{"plan_name": "Meridian"}',
        }
    ]
    converted = to_openai_messages(messages)

    assert converted == [
        {"role": "tool", "tool_call_id": "1", "content": '{"plan_name": "Meridian"}'}
    ]
    assert "name" not in converted[0]


def test_to_openai_messages_does_not_mutate_input():
    original: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "1", "name": "lookup_policy", "arguments": {"policy_id": "POL-1001"}}
            ],
        },
        {"role": "tool", "tool_call_id": "1", "name": "lookup_policy", "content": "{}"},
    ]
    snapshot = copy.deepcopy(original)

    to_openai_messages(original)

    assert original == snapshot


def test_to_openai_messages_passes_through_plain_messages_unchanged():
    messages = [{"role": "user", "content": "hello"}]
    converted = to_openai_messages(messages)
    assert converted == messages
    assert converted[0] is not messages[0]  # a copy, not the same object


# ---------------------------------------------------------------------------
# malformed tool-call argument parsing
# ---------------------------------------------------------------------------


def test_parse_tool_call_arguments_valid_json_object():
    assert _parse_tool_call_arguments('{"policy_id": "POL-1001"}') == {
        "policy_id": "POL-1001"
    }


def test_parse_tool_call_arguments_malformed_json_returns_sentinel():
    raw = '{"policy_id": "POL-1001"'  # missing closing brace
    result = _parse_tool_call_arguments(raw)
    assert result == {"__invalid_json__": raw}


def test_parse_tool_call_arguments_non_dict_json_returns_sentinel():
    raw = '["not", "a", "dict"]'
    result = _parse_tool_call_arguments(raw)
    assert result == {"__invalid_json__": raw}


def test_parse_tool_call_arguments_empty_string_is_empty_dict():
    assert _parse_tool_call_arguments("") == {}


# ---------------------------------------------------------------------------
# retry classification and backoff
# ---------------------------------------------------------------------------


class _FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_is_retryable_status_error_for_429_and_5xx():
    assert _is_retryable_status_error(_FakeStatusError(429)) is True
    assert _is_retryable_status_error(_FakeStatusError(500)) is True
    assert _is_retryable_status_error(_FakeStatusError(503)) is True


def test_is_retryable_status_error_false_for_other_4xx():
    assert _is_retryable_status_error(_FakeStatusError(400)) is False
    assert _is_retryable_status_error(_FakeStatusError(404)) is False


def test_is_retryable_status_error_false_without_status_code():
    assert _is_retryable_status_error(ValueError("boom")) is False


def test_backoff_delay_grows_and_is_capped():
    # Base delay doubles per attempt but is capped at 30s before jitter is
    # applied; jitter multiplies by [0.5, 1.5), so the capped delay's
    # ceiling is 30 * 1.5.
    assert _backoff_delay_seconds(0) <= 1.5
    assert _backoff_delay_seconds(10) <= 30.0 * 1.5
    assert _backoff_delay_seconds(0) >= 0.0


# ---------------------------------------------------------------------------
# OpenAIChatCompletionsClient.complete retry behavior, via a fake openai.OpenAI
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str = "") -> None:
        self.content = content
        self.tool_calls: list[Any] = []


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeResponse:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [_FakeChoice(message)]


class _FailThenSucceedCompletions:
    """Fails with the given status codes in order, then succeeds."""

    def __init__(self, failures: list[int]) -> None:
        self._failures = list(failures)
        self.call_count = 0

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.call_count += 1
        if self._failures:
            status = self._failures.pop(0)
            raise _FakeStatusError(status)
        return _FakeResponse(_FakeMessage(content="ok"))


class _AlwaysFailCompletions:
    def __init__(self, status_code: int) -> None:
        self._status_code = status_code
        self.call_count = 0

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.call_count += 1
        raise _FakeStatusError(self._status_code)


class _FakeChat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, completions: Any) -> None:
        self.chat = _FakeChat(completions)


def _make_client(monkeypatch: pytest.MonkeyPatch, completions: Any) -> OpenAIChatCompletionsClient:
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: _FakeOpenAIClient(completions))
    monkeypatch.setattr("claims_triage_agent.llm_client.time.sleep", lambda seconds: None)
    return OpenAIChatCompletionsClient(model="fake-model", api_key="fake-key")


def test_complete_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch):
    completions = _FailThenSucceedCompletions(failures=[429, 429])
    client = _make_client(monkeypatch, completions)

    response = client.complete([{"role": "user", "content": "hi"}], tools=[])

    assert response.content == "ok"
    assert completions.call_count == 3


def test_complete_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch):
    completions = _AlwaysFailCompletions(status_code=503)
    client = _make_client(monkeypatch, completions)

    with pytest.raises(_FakeStatusError):
        client.complete([{"role": "user", "content": "hi"}], tools=[])

    assert completions.call_count == 6  # _MAX_RETRY_ATTEMPTS


def test_complete_does_not_retry_on_400(monkeypatch: pytest.MonkeyPatch):
    completions = _AlwaysFailCompletions(status_code=400)
    client = _make_client(monkeypatch, completions)

    with pytest.raises(_FakeStatusError):
        client.complete([{"role": "user", "content": "hi"}], tools=[])

    assert completions.call_count == 1


def test_complete_exposes_model_name(monkeypatch: pytest.MonkeyPatch):
    completions = _FailThenSucceedCompletions(failures=[])
    client = _make_client(monkeypatch, completions)
    assert client.model == "fake-model"
