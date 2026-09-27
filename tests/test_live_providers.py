"""Live checks against the real providers (skipped unless run with `pytest --live`).

Each provider gets one tiny chat call and one real scoring call with the same
settings the app uses. Keys come from ~/.applypilot/.env. Gemini's free tier
allows ~20 requests/day, so this file uses 2 of them.
"""

import os

import pytest

from applypilot import config, llm
from applypilot.llm import LLMClient
from applypilot.scoring import scorer

pytestmark = pytest.mark.live

JOB = {
    "url": "https://example.test/dba", "title": "Senior Database Administrator (PostgreSQL)", "site": "indeed",
    "company": "Example Bank", "location": "Melbourne VIC",
    "full_description": (
        "Permanent full-time role. We run PostgreSQL on Azure across multiple regions. You will own major "
        "version upgrades, replication, backups, performance tuning and on-call support, automate with "
        "Terraform and Python, and work with our SRE team on incident response."
    ),
}
RESUMES = {
    "software": "Software engineer: .NET, Angular, React, APIs, AI features on GCP Vertex AI.",
    "devops": "DevOps engineer: Azure, Terraform, AKS, PostgreSQL major upgrades across 8 regions, "
              "PgBouncer cutovers, on-call incident response, Python automation.",
}
GUIDES = {"software": "application, backend and AI development",
          "devops": "infrastructure, SRE, platform and database administration"}


@pytest.fixture(scope="module", autouse=True)
def real_env():
    config.load_env()


def _client(provider: str) -> LLMClient:
    model_var = "LLM_MODEL" if os.environ.get("LLM_PROVIDER") == provider else "LLM_FALLBACK_MODEL"
    try:
        base_url, model, key = llm._resolve_provider(provider, os.environ.get(model_var, ""), setting="test")
    except RuntimeError as e:
        pytest.skip(str(e))
    thinking_off = provider == "zai" and llm._read_thinking_off()
    return LLMClient(base_url, model, key, reasoning_effort=llm._read_reasoning_effort(), thinking_off=thinking_off)


@pytest.mark.parametrize("provider", ["zai", "gemini"])
def test_provider_answers(provider):
    reply = _client(provider).chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=2048)
    assert "OK" in reply.upper()


@pytest.mark.parametrize("provider", ["zai", "gemini"])
def test_provider_scores_a_job(provider, monkeypatch):
    client = _client(provider)
    monkeypatch.setattr(scorer, "get_client", lambda: client)

    result = scorer.score_job(RESUMES, JOB, GUIDES)

    assert 1 <= result["score"] <= 10, result["reasoning"]
    assert result["track"] == "devops", f"{provider} picked {result['track']}: {result['reasoning']}"
    assert result["employment"] in ("permanent", None)
    assert result["model"] == client.model
