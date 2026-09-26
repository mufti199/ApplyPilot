"""Tests for truncated-reply detection and the reasoning_effort setting."""

import json

import httpx
import pytest

from applypilot import llm
from applypilot.llm import LLMClient, TruncatedResponse, _read_reasoning_effort
from applypilot.scoring import scorer

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


def _reply(content, finish_reason="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def _client(responses, **kwargs):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return responses[len(requests) - 1]

    client = LLMClient(GEMINI_BASE, "gemini-3.8-flash", "k", **kwargs)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, requests


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)


# -- truncation --------------------------------------------------------------

def test_length_finish_reason_raises_truncated():
    client, requests = _client([httpx.Response(200, json=_reply("SCORE: 5\nKEYWORDS: a,", "length"))])

    with pytest.raises(TruncatedResponse, match="512-token output limit"):
        client.chat([{"role": "user", "content": "hi"}], max_tokens=512)

    assert len(requests) == 1  # not retried: the same request would be cut off again


def test_complete_reply_is_returned():
    client, _ = _client([httpx.Response(200, json=_reply("OK"))])
    assert client.chat([{"role": "user", "content": "hi"}]) == "OK"


def test_native_gemini_max_tokens_raises_truncated():
    client, _ = _client([])
    native = {"candidates": [{"content": {"parts": [{"text": "SCORE: 5"}]}, "finishReason": "MAX_TOKENS"}]}
    client._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=native)))

    with pytest.raises(TruncatedResponse):
        client._chat_native_gemini([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=100)


def test_truncated_score_leaves_job_unscored(monkeypatch):
    class Cut:
        def chat(self, *a, **k):
            raise TruncatedResponse("cut off")

    monkeypatch.setattr(scorer, "get_client", Cut)
    result = scorer.score_job({"software": "r"}, {"title": "t", "site": "s", "full_description": "d"})
    assert result["score"] == 0  # 0 = failure; run_scoring leaves it unscored for retry


# -- reasoning_effort --------------------------------------------------------

def test_reasoning_effort_sent_when_set():
    client, requests = _client([httpx.Response(200, json=_reply("OK"))], reasoning_effort="low")
    client.chat([{"role": "user", "content": "hi"}])
    assert requests[0]["reasoning_effort"] == "low"


def test_reasoning_effort_omitted_by_default():
    client, requests = _client([httpx.Response(200, json=_reply("OK"))])
    client.chat([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in requests[0]


@pytest.mark.parametrize(("raw", "expected"), [("", None), ("low", "low"), (" HIGH ", "high")])
def test_read_reasoning_effort(monkeypatch, raw, expected):
    monkeypatch.setenv("LLM_REASONING_EFFORT", raw)
    assert _read_reasoning_effort() == expected


def test_read_reasoning_effort_rejects_unknown(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "extreme")
    with pytest.raises(ValueError, match="LLM_REASONING_EFFORT"):
        _read_reasoning_effort()
