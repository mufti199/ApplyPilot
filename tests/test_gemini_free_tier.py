"""Tests for free-tier handling: request pacing, daily quota stop, retry delays, scoring order."""

import httpx
import pytest

from applypilot import database, llm
from applypilot.database import init_db
from applypilot.llm import DailyQuotaExceeded, LLMClient, _error_detail, _read_rpm, _retry_wait
from applypilot.scoring import scorer

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
OK_BODY = {"choices": [{"message": {"content": "hello"}}]}


def _gemini_429(quota_id, retry_delay="30s"):
    """A Gemini-compat 429 body, wrapped in a list as the compat layer does."""
    return [{"error": {
        "code": 429,
        "message": "You exceeded your current quota.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": quota_id}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ],
    }}]


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(llm.time, "sleep", recorded.append)
    return recorded


def _gemini_client(responses, rpm=None):
    calls = []

    def handler(request):
        calls.append(request)
        return responses[len(calls) - 1]

    client = LLMClient(GEMINI_BASE, "gemini-3.8-flash", "k", rpm=rpm)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client, calls


# -- LLM_RPM parsing ---------------------------------------------------------

@pytest.mark.parametrize(("raw", "expected"), [("", None), ("  ", None), ("5", 5.0), ("2.5", 2.5)])
def test_read_rpm(monkeypatch, raw, expected):
    monkeypatch.setenv("LLM_RPM", raw)
    assert _read_rpm() == expected


@pytest.mark.parametrize("raw", ["fast", "0", "-3"])
def test_read_rpm_rejects_bad_values(monkeypatch, raw):
    monkeypatch.setenv("LLM_RPM", raw)
    with pytest.raises(ValueError, match="LLM_RPM"):
        _read_rpm()


# -- pacing ------------------------------------------------------------------

def test_pacing_spaces_requests(monkeypatch, sleeps):
    clock = [100.0]
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])
    client, _ = _gemini_client([httpx.Response(200, json=OK_BODY)] * 3, rpm=6)  # 10s apart

    for _ in range(3):
        client.chat([{"role": "user", "content": "hi"}])

    assert sleeps == [10.0, 20.0]  # 2nd waits 10s, 3rd 20s (clock frozen)


def test_no_pacing_when_rpm_unset(sleeps):
    client, _ = _gemini_client([httpx.Response(200, json=OK_BODY)] * 2)
    client.chat([{"role": "user", "content": "hi"}])
    client.chat([{"role": "user", "content": "hi"}])
    assert sleeps == []


# -- daily quota and retry delay ---------------------------------------------

def test_daily_quota_raises_without_retry(sleeps):
    client, calls = _gemini_client([
        httpx.Response(429, json=_gemini_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier")),
    ])

    with pytest.raises(DailyQuotaExceeded, match="midnight Pacific"):
        client.chat([{"role": "user", "content": "hi"}])

    assert len(calls) == 1
    assert sleeps == []


def test_per_minute_quota_waits_retry_delay_then_succeeds(sleeps):
    client, calls = _gemini_client([
        httpx.Response(429, json=_gemini_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "17s")),
        httpx.Response(200, json=OK_BODY),
    ])

    assert client.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert len(calls) == 2
    assert sleeps == [18.0]  # retryDelay + 1s margin


def test_retry_wait_prefers_header():
    resp = httpx.Response(429, headers={"Retry-After": "7"}, json=_gemini_429("PerMinute", "30s"))
    assert _retry_wait(resp, attempt=0) == 7.0


def test_retry_wait_falls_back_to_backoff():
    resp = httpx.Response(429, text="slow down")
    assert _retry_wait(resp, attempt=1) == llm._RATE_LIMIT_BASE_WAIT * 2


def test_error_detail_handles_gemini_list_shape():
    resp = httpx.Response(429, json=_gemini_429("PerMinute"))
    assert _error_detail(resp) == ("RESOURCE_EXHAUSTED", "You exceeded your current quota.")


# -- scoring -----------------------------------------------------------------

@pytest.fixture
def jobs_db(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    conn = init_db(path)
    rows = [
        ("https://a/sydney-new", "Sydney NSW", "2026-01-03"),
        ("https://b/melbourne-old", "Melbourne VIC", "2026-01-01"),
        ("https://c/perth", "Perth WA", "2026-01-02"),
    ]
    for url, location, discovered in rows:
        conn.execute(
            "INSERT INTO jobs (url, title, site, location, full_description, discovered_at) "
            "VALUES (?, 'Engineer', 'indeed', ?, 'desc', ?)",
            (url, location, discovered),
        )
    conn.commit()

    resume = tmp_path / "r.txt"
    resume.write_text("resume", encoding="utf-8")
    monkeypatch.setattr(scorer, "load_search_config", lambda: {"location_preferred": ["VIC", "Melbourne"]})
    monkeypatch.setattr(scorer, "get_resume_tracks", lambda _cfg: {"software": {"text": resume, "pdf": None}})
    yield conn
    database.close_connection(path)


def test_scoring_starts_with_preferred_locations(jobs_db, monkeypatch):
    seen = []

    def fake(resumes, job):
        seen.append(job["url"])
        return {"score": 7, "keywords": "", "reasoning": "", "track": "software"}

    monkeypatch.setattr(scorer, "score_job", fake)
    scorer.run_scoring()

    assert seen[0] == "https://b/melbourne-old"          # preferred, despite being oldest
    assert seen[1:] == ["https://a/sydney-new", "https://c/perth"]  # then newest first


def test_scoring_stops_on_daily_quota_and_keeps_earlier_scores(jobs_db, monkeypatch):
    calls = []

    def fake(resumes, job):
        calls.append(job["url"])
        if len(calls) == 2:
            raise DailyQuotaExceeded("daily quota reached")
        return {"score": 8, "keywords": "", "reasoning": "", "track": "software"}

    monkeypatch.setattr(scorer, "score_job", fake)
    stats = scorer.run_scoring()

    assert len(calls) == 2                      # third job never attempted
    assert stats["scored"] == 1 and stats["aborted"] is True
    scores = dict(jobs_db.execute("SELECT url, fit_score FROM jobs").fetchall())
    assert scores["https://b/melbourne-old"] == 8
    assert scores["https://a/sydney-new"] is None


def test_score_job_propagates_daily_quota(monkeypatch):
    class Quota:
        def chat(self, *a, **k):
            raise DailyQuotaExceeded("daily")

    monkeypatch.setattr(scorer, "get_client", Quota)
    with pytest.raises(DailyQuotaExceeded):
        scorer.score_job({"software": "r"}, {"title": "t", "site": "s", "full_description": "d"})
