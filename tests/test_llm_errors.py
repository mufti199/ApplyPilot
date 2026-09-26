"""Tests for LLM HTTP error handling: non-retryable quota errors, retries, messages."""

import httpx
import pytest

from applypilot import llm
from applypilot.llm import LLMClient, _error_detail

OK_BODY = {"choices": [{"message": {"content": "hello"}}]}


def _error_body(code, message):
    return {"error": {"code": code, "message": message, "type": "invalid_request_error"}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)


def _client(responses):
    """LLMClient whose HTTP transport returns the given responses in order."""
    calls = []

    def handler(request):
        calls.append(request)
        return responses[len(calls) - 1]

    client = LLMClient("https://api.openai.com/v1", "m", "k")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, calls


def test_insufficient_quota_fails_fast_without_retry():
    client, calls = _client([
        httpx.Response(429, json=_error_body("insufficient_quota", "You exceeded your current quota")),
    ])
    with pytest.raises(RuntimeError) as exc_info:
        client.ask("hi")
    assert len(calls) == 1
    assert "OpenAI" in str(exc_info.value)
    assert "insufficient_quota" in str(exc_info.value)


def test_plain_rate_limit_is_retried_then_succeeds():
    client, calls = _client([
        httpx.Response(429, json=_error_body("rate_limit_exceeded", "Slow down")),
        httpx.Response(200, json=OK_BODY),
    ])
    assert client.ask("hi") == "hello"
    assert len(calls) == 2


def test_client_error_raises_with_api_message():
    client, calls = _client([
        httpx.Response(400, json=_error_body(None, "Unsupported parameter: max_tokens")),
    ])
    with pytest.raises(RuntimeError, match="Unsupported parameter") as exc_info:
        client.ask("hi")
    assert "HTTP 400" in str(exc_info.value)
    assert len(calls) == 1


def test_rate_limit_exhausting_retries_raises_runtime_error():
    client, calls = _client([httpx.Response(429, json=_error_body(None, "Slow down"))] * llm._MAX_RETRIES)
    with pytest.raises(RuntimeError, match="HTTP 429: Slow down"):
        client.ask("hi")
    assert len(calls) == llm._MAX_RETRIES


def test_error_detail_parses_openai_shape():
    resp = httpx.Response(429, json=_error_body("insufficient_quota", "No quota"))
    assert _error_detail(resp) == ("insufficient_quota", "No quota")


def test_error_detail_falls_back_to_body_text():
    resp = httpx.Response(502, text="<html>Bad Gateway</html>")
    assert _error_detail(resp) == (None, "<html>Bad Gateway</html>")


def test_error_detail_empty_body_uses_reason_phrase():
    resp = httpx.Response(502)
    assert _error_detail(resp) == (None, "Bad Gateway")
