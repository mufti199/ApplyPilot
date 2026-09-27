"""Tests for the main + fallback provider setup (Z.ai with Gemini as fallback)."""

import json

import httpx
import pytest

from applypilot import database, llm
from applypilot.database import init_db
from applypilot.llm import DailyQuotaExceeded, FallbackClient, LLMClient, TruncatedResponse
from applypilot.scoring import scorer

ZAI = "https://api.z.ai/api/paas/v4"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
RESUMES = {"software": "SOFTWARE RESUME", "devops": "DEVOPS RESUME"}
JOB = {"url": "https://x/1", "title": "Database Administrator", "site": "indeed", "location": "Melbourne",
       "full_description": "PostgreSQL on Azure"}
GOOD = "TRACK: devops\nSCORE: 6\nKEYWORDS: PostgreSQL\nREASONING: Solid DBA overlap.\nCOMPANY: unknown\nEMPLOYMENT: unknown"


def ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


def overloaded():
    return httpx.Response(429, json={"error": {"code": "1305", "message": "The service may be temporarily overloaded"}})


def gemini_daily_quota():
    return httpx.Response(429, json=[{"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
                                                "details": [{"violations": [{"quotaId": "GenerateRequestsPerDay"}]}]}}])


def fake(base_url, model, responses):
    """A real LLMClient whose HTTP calls return the given responses in order."""
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        item = responses[min(len(calls), len(responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    client = LLMClient(base_url, model, "k")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, calls


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)


# -- FallbackClient behaviour ----------------------------------------------------

def test_main_model_used_when_it_works():
    zai, _ = fake(ZAI, "glm-4.7-flash", [ok("hi")])
    gem, gem_calls = fake(GEMINI, "gemini-3.8-flash", [ok("unused")])
    client = FallbackClient(zai, gem)

    assert client.chat([{"role": "user", "content": "x"}]) == "hi"
    assert client.last_model == "glm-4.7-flash" and len(gem_calls) == 0


def test_persistent_overload_falls_back_to_gemini():
    zai, zai_calls = fake(ZAI, "glm-4.7-flash", [overloaded()] * 5)
    gem, gem_calls = fake(GEMINI, "gemini-3.8-flash", [ok("from gemini")])
    client = FallbackClient(zai, gem)

    assert client.chat([{"role": "user", "content": "x"}]) == "from gemini"
    assert len(zai_calls) == llm._MAX_RETRIES and len(gem_calls) == 1
    assert client.last_model == "gemini-3.8-flash"


def test_truncated_reply_falls_back():
    cut = httpx.Response(200, json={"choices": [{"message": {"content": "SCORE: 5"}, "finish_reason": "length"}]})
    zai, _ = fake(ZAI, "glm-4.7-flash", [cut])
    gem, _ = fake(GEMINI, "gemini-3.8-flash", [ok("complete")])
    assert FallbackClient(zai, gem).chat([{"role": "user", "content": "x"}], max_tokens=10) == "complete"


def test_network_error_falls_back():
    zai, _ = fake(ZAI, "glm-4.7-flash", [httpx.ConnectError("no route")])
    gem, _ = fake(GEMINI, "gemini-3.8-flash", [ok("ok")])
    assert FallbackClient(zai, gem).chat([{"role": "user", "content": "x"}]) == "ok"


def test_main_quota_sticks_to_fallback():
    zai_quota = httpx.Response(429, text='{"error": {"message": "limit PerDay reached"}}')
    zai, zai_calls = fake(ZAI, "glm-4.7-flash", [zai_quota])
    gem, gem_calls = fake(GEMINI, "gemini-3.8-flash", [ok("a"), ok("b")])
    client = FallbackClient(zai, gem)

    assert client.chat([{"role": "user", "content": "1"}]) == "a"
    assert client.chat([{"role": "user", "content": "2"}]) == "b"
    assert len(zai_calls) == 1 and len(gem_calls) == 2  # main not retried after its quota ran out


def test_fallback_quota_disables_fallback_and_main_error_surfaces():
    zai, _ = fake(ZAI, "glm-4.7-flash", [overloaded()] * 5 + [ok("main again")])
    gem, gem_calls = fake(GEMINI, "gemini-3.8-flash", [gemini_daily_quota()])
    client = FallbackClient(zai, gem)

    with pytest.raises(RuntimeError) as exc:
        client.chat([{"role": "user", "content": "1"}])
    assert not isinstance(exc.value, DailyQuotaExceeded)  # a normal failure: the run carries on

    assert client.chat([{"role": "user", "content": "2"}]) == "main again"
    assert len(gem_calls) == 1  # fallback not tried again


def test_both_out_of_quota_stops_the_run():
    zai_quota = httpx.Response(429, text='{"error": {"message": "GenerateRequestsPerDay"}}')
    zai, _ = fake(ZAI, "glm-4.7-flash", [zai_quota])
    gem, _ = fake(GEMINI, "gemini-3.8-flash", [gemini_daily_quota()])
    with pytest.raises(DailyQuotaExceeded, match="fallback also out of quota"):
        FallbackClient(zai, gem).chat([{"role": "user", "content": "x"}])


def test_ask_and_close():
    zai, _ = fake(ZAI, "glm-4.7-flash", [ok("asked")])
    gem, _ = fake(GEMINI, "gemini-3.8-flash", [ok("x")])
    client = FallbackClient(zai, gem)
    assert client.ask("hello") == "asked"
    client.close()


# -- get_client wiring -------------------------------------------------------------

_ENV = ("LLM_PROVIDER", "LLM_MODEL", "LLM_FALLBACK_PROVIDER", "LLM_FALLBACK_MODEL", "LLM_FALLBACK_RPM",
        "GEMINI_API_KEY", "ZAI_API_KEY", "OPENAI_API_KEY", "LLM_URL", "LLM_RPM", "LLM_THINKING",
        "LLM_REASONING_EFFORT")


@pytest.fixture
def env(monkeypatch):
    for var in _ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_instance", None)
    yield monkeypatch
    llm._instance = None


def test_get_client_builds_zai_with_gemini_fallback(env):
    env.setenv("LLM_PROVIDER", "zai")
    env.setenv("ZAI_API_KEY", "z")
    env.setenv("GEMINI_API_KEY", "g")
    env.setenv("LLM_FALLBACK_PROVIDER", "gemini")
    env.setenv("LLM_FALLBACK_MODEL", "gemini-3.8-flash")
    env.setenv("LLM_FALLBACK_RPM", "5")

    client = llm.get_client()

    assert isinstance(client, FallbackClient)
    assert client.primary.base_url == ZAI and client.primary.model == "glm-4.7-flash"
    assert client.fallback.base_url == GEMINI and client.fallback.model == "gemini-3.8-flash"
    assert client.fallback._min_interval == 12.0


def test_get_client_without_fallback_is_plain(env):
    env.setenv("LLM_PROVIDER", "zai")
    env.setenv("ZAI_API_KEY", "z")
    assert isinstance(llm.get_client(), LLMClient)


@pytest.mark.parametrize(("extra", "error"), [
    ({"LLM_FALLBACK_PROVIDER": "claude"}, ValueError),
    ({"LLM_FALLBACK_PROVIDER": "gemini"}, RuntimeError),  # GEMINI_API_KEY missing
    ({"LLM_FALLBACK_PROVIDER": "gemini", "GEMINI_API_KEY": "g", "LLM_FALLBACK_RPM": "0"}, ValueError),
])
def test_bad_fallback_settings(env, extra, error):
    env.setenv("LLM_PROVIDER", "zai")
    env.setenv("ZAI_API_KEY", "z")
    for k, v in extra.items():
        env.setenv(k, v)
    with pytest.raises(error):
        llm.get_client()


def test_zai_thinking_setting_not_sent_to_gemini_fallback(env):
    env.setenv("LLM_PROVIDER", "zai")
    env.setenv("ZAI_API_KEY", "z")
    env.setenv("GEMINI_API_KEY", "g")
    env.setenv("LLM_FALLBACK_PROVIDER", "gemini")
    env.setenv("LLM_THINKING", "off")
    client = llm.get_client()
    assert client.primary.thinking_off is True and client.fallback.thinking_off is False


# -- scoring through both models ---------------------------------------------------

def _use(monkeypatch, client):
    monkeypatch.setattr(scorer, "get_client", lambda: client)


def test_score_from_main_model_records_it(monkeypatch):
    zai, _ = fake(ZAI, "glm-4.7-flash", [ok(GOOD)])
    gem, gem_calls = fake(GEMINI, "gemini-3.8-flash", [ok("unused")])
    _use(monkeypatch, FallbackClient(zai, gem))

    result = scorer.score_job(RESUMES, JOB)

    assert (result["score"], result["track"], result["model"]) == (6, "devops", "glm-4.7-flash")
    assert gem_calls == []


def test_unreadable_main_reply_is_rescored_by_fallback(monkeypatch):
    zai, _ = fake(ZAI, "glm-4.7-flash", [ok("I think this is a decent match overall.")])
    gem, gem_calls = fake(GEMINI, "gemini-3.8-flash", [ok(GOOD)])
    _use(monkeypatch, FallbackClient(zai, gem))

    result = scorer.score_job(RESUMES, JOB)

    assert (result["score"], result["model"]) == (6, "gemini-3.8-flash")
    assert len(gem_calls) == 1


def test_wrong_track_from_main_is_rescored_by_fallback(monkeypatch):
    zai, _ = fake(ZAI, "glm-4.7-flash", [ok(GOOD.replace("devops", "data"))])
    gem, _ = fake(GEMINI, "gemini-3.8-flash", [ok(GOOD)])
    _use(monkeypatch, FallbackClient(zai, gem))
    assert scorer.score_job(RESUMES, JOB)["track"] == "devops"


def test_both_unreadable_leaves_job_unscored(monkeypatch):
    zai, _ = fake(ZAI, "glm-4.7-flash", [ok("no idea")])
    gem, _ = fake(GEMINI, "gemini-3.8-flash", [ok("also no idea")])
    _use(monkeypatch, FallbackClient(zai, gem))
    result = scorer.score_job(RESUMES, JOB)
    assert result["score"] == 0 and "model" not in result


def test_plain_client_unreadable_reply_is_not_retried(monkeypatch):
    zai, zai_calls = fake(ZAI, "glm-4.7-flash", [ok("no idea")])
    _use(monkeypatch, zai)
    assert scorer.score_job(RESUMES, JOB)["score"] == 0 and len(zai_calls) == 1


def test_scored_by_is_saved(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    conn = init_db(path)
    conn.execute("INSERT INTO jobs (url, title, site, full_description) VALUES ('https://x/1', 'DBA', 'indeed', 'd')")
    conn.commit()
    result = {"score": 6, "keywords": "", "reasoning": "", "track": "devops", "company": None,
              "employment": None, "model": "gemini-3.8-flash"}

    scorer._save_score(conn, "https://x/1", result)

    assert conn.execute("SELECT scored_by FROM jobs").fetchone()[0] == "gemini-3.8-flash"
    database.close_connection(path)


def test_truncation_class_is_a_runtime_error():
    assert issubclass(TruncatedResponse, RuntimeError) and issubclass(DailyQuotaExceeded, RuntimeError)
