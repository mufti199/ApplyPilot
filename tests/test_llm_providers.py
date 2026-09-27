"""Tests for provider selection (incl. Z.ai), GLM thinking switch, doctor, and reply parsing."""

import json

import httpx
import pytest
from typer.testing import CliRunner

from applypilot import cli, config, llm
from applypilot.llm import LLMClient, _detect_provider, _read_thinking_off, provider_label
from applypilot.scoring import scorer

ZAI = "https://api.z.ai/api/paas/v4"
_VARS = ("LLM_PROVIDER", "LLM_MODEL", "LLM_URL", "LLM_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
         "ZAI_API_KEY", "LLM_THINKING")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)


# -- provider selection --------------------------------------------------------

def test_explicit_zai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "zai")
    monkeypatch.setenv("ZAI_API_KEY", "z")
    monkeypatch.setenv("GEMINI_API_KEY", "g")  # present but not chosen
    assert _detect_provider() == (ZAI, "glm-4.7-flash", "z")


def test_model_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "zai")
    monkeypatch.setenv("ZAI_API_KEY", "z")
    monkeypatch.setenv("LLM_MODEL", "glm-5.3-flash")
    assert _detect_provider()[1] == "glm-5.3-flash"


def test_auto_detect_keeps_old_order(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ZAI_API_KEY", "z")
    assert _detect_provider()[2] == "g"


def test_auto_detect_zai_alone(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "z")
    assert _detect_provider()[0] == ZAI


def test_local_url_wins_in_auto_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("LLM_URL", "http://localhost:11434/v1/")
    assert _detect_provider()[0] == "http://localhost:11434/v1"


@pytest.mark.parametrize(("env", "error"), [
    ({"LLM_PROVIDER": "claude"}, ValueError),
    ({"LLM_PROVIDER": "zai"}, RuntimeError),       # key missing
    ({"LLM_PROVIDER": "local"}, RuntimeError),     # URL missing
    ({}, RuntimeError),                            # nothing configured
])
def test_provider_errors(monkeypatch, env, error):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(error):
        _detect_provider()


@pytest.mark.parametrize(("url", "label"), [
    (ZAI, "Z.ai"), ("https://api.openai.com/v1", "OpenAI"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "Gemini"),
    ("http://localhost:11434/v1", "Local (http://localhost:11434/v1)"),
])
def test_provider_label(url, label):
    assert provider_label(url) == label


# -- GLM thinking switch -------------------------------------------------------

@pytest.mark.parametrize(("raw", "off"), [("", False), ("on", False), ("off", True), ("OFF", True)])
def test_read_thinking(monkeypatch, raw, off):
    monkeypatch.setenv("LLM_THINKING", raw)
    assert _read_thinking_off() is off


def test_read_thinking_rejects_bad_value(monkeypatch):
    monkeypatch.setenv("LLM_THINKING", "maybe")
    with pytest.raises(ValueError, match="LLM_THINKING"):
        _read_thinking_off()


def _payload(base_url, thinking_off):
    sent = []
    client = LLMClient(base_url, "m", "k", thinking_off=thinking_off)
    client._client = httpx.Client(transport=httpx.MockTransport(
        lambda r: sent.append(json.loads(r.content)) or httpx.Response(
            200, json={"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]})))
    client.chat([{"role": "user", "content": "hi"}])
    return sent[0]


def test_thinking_disabled_sent_to_zai():
    assert _payload(ZAI, thinking_off=True)["thinking"] == {"type": "disabled"}


def test_thinking_never_sent_to_other_providers():
    assert "thinking" not in _payload("https://api.openai.com/v1", thinking_off=True)
    assert "thinking" not in _payload(ZAI, thinking_off=False)


# -- doctor --------------------------------------------------------------------

def test_doctor_reports_zai(monkeypatch):
    monkeypatch.setattr(cli, "_bootstrap", lambda: None)
    monkeypatch.setattr(config, "load_env", lambda: None)  # never read the real ~/.applypilot/.env
    monkeypatch.setenv("LLM_PROVIDER", "zai")
    monkeypatch.setenv("ZAI_API_KEY", "z")
    result = runner_invoke(["doctor"])
    assert result.exit_code == 0, result.output
    assert "Z.ai (glm-4.7-flash)" in result.output


def test_doctor_reports_missing_provider(monkeypatch):
    monkeypatch.setattr(cli, "_bootstrap", lambda: None)
    monkeypatch.setattr(config, "load_env", lambda: None)  # never read the real ~/.applypilot/.env
    result = runner_invoke(["doctor"])
    assert result.exit_code == 0 and "No LLM provider configured" in result.output


def runner_invoke(args):
    return CliRunner().invoke(cli.app, args)


# -- parsing tolerant of markdown / shared-line mistakes -----------------------

def test_parse_tolerates_markdown():
    parsed = scorer._parse_score_response(
        "**TRACK:** devops\n- **SCORE:** 8\n**KEYWORDS:** a, b\nREASONING : fine\nCOMPANY: Acme")
    assert (parsed["score"], parsed["track"], parsed["keywords"], parsed["reasoning"]) == (8, "devops", "a, b", "fine")


def test_grouped_prompt_asks_for_track_in_every_block():
    prompt = scorer._build_score_prompt(["software", "devops"]) + scorer.BATCH_SUFFIX.format(count=5)
    assert "FIRST in your response" not in prompt
    assert "Every block must" in prompt and "directly before the SCORE line" in prompt


def test_setup_keeps_llm_module_singleton_clean():
    llm._instance = None  # other tests must not inherit a client built from this file's env
