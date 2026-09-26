"""Tests for scoring several jobs per LLM request."""

import pytest

from applypilot import database
from applypilot.config import get_score_group_size
from applypilot.database import init_db
from applypilot.llm import DailyQuotaExceeded
from applypilot.scoring import scorer

RESUMES = {"software": "SOFTWARE RESUME", "devops": "DEVOPS RESUME"}


def _job(n):
    return {"url": f"https://x/{n}", "title": f"Engineer {n}", "site": "indeed",
            "location": "Melbourne VIC", "full_description": f"desc {n}"}


def _block(n, score, track="devops"):
    return f"JOB: {n}\nTRACK: {track}\nSCORE: {score}\nKEYWORDS: k{n}\nREASONING: r{n}"


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _use(monkeypatch, response):
    client = FakeClient(response)
    monkeypatch.setattr(scorer, "get_client", lambda: client)
    return client


# -- config ------------------------------------------------------------------

@pytest.mark.parametrize(("cfg", "expected"), [({}, 1), ({"score_jobs_per_request": 5}, 5)])
def test_group_size(cfg, expected):
    assert get_score_group_size(cfg) == expected


@pytest.mark.parametrize(("value", "error"), [(0, ValueError), (11, ValueError), ("5", TypeError), (True, TypeError)])
def test_group_size_rejects_bad_values(value, error):
    with pytest.raises(error):
        get_score_group_size({"score_jobs_per_request": value})


# -- score_jobs --------------------------------------------------------------

def test_one_request_scores_every_job(monkeypatch):
    client = _use(monkeypatch, "\n\n".join(_block(n, 5 + n) for n in (1, 2, 3)))

    results = scorer.score_jobs(RESUMES, [_job(1), _job(2), _job(3)])

    assert [r["score"] for r in results] == [6, 7, 8]
    assert [r["keywords"] for r in results] == ["k1", "k2", "k3"]
    assert len(client.calls) == 1
    messages, kwargs = client.calls[0]
    assert messages[1]["content"].count("RESUME (TRACK:") == 2       # resumes sent once
    assert "JOB 3:" in messages[1]["content"]
    assert kwargs["max_tokens"] == 3 * scorer._TOKENS_PER_JOB


def test_out_of_order_and_decorated_headers(monkeypatch):
    reply = "**JOB 2**\nTRACK: software\nSCORE: 4\nKEYWORDS: b\nREASONING: two\n\nJob: 1\n" \
            "TRACK: devops\nSCORE: 9\nKEYWORDS: a\nREASONING: one"
    _use(monkeypatch, reply)

    results = scorer.score_jobs(RESUMES, [_job(1), _job(2)])

    assert [(r["score"], r["track"]) for r in results] == [(9, "devops"), (4, "software")]


def test_missing_job_fails_only_that_job(monkeypatch):
    _use(monkeypatch, _block(1, 7) + "\n\n" + _block(3, 8))

    results = scorer.score_jobs(RESUMES, [_job(1), _job(2), _job(3)])

    assert [r["score"] for r in results] == [7, 0, 8]
    assert "missing" in results[1]["reasoning"]


def test_bad_track_fails_only_that_job(monkeypatch):
    _use(monkeypatch, _block(1, 7) + "\n\n" + _block(2, 8, track="data"))

    results = scorer.score_jobs(RESUMES, [_job(1), _job(2)])

    assert [r["score"] for r in results] == [7, 0]


def test_llm_error_fails_whole_group(monkeypatch):
    _use(monkeypatch, RuntimeError("HTTP 500"))
    results = scorer.score_jobs(RESUMES, [_job(1), _job(2)])
    assert [r["score"] for r in results] == [0, 0]


def test_daily_quota_propagates(monkeypatch):
    _use(monkeypatch, DailyQuotaExceeded("daily"))
    with pytest.raises(DailyQuotaExceeded):
        scorer.score_jobs(RESUMES, [_job(1), _job(2)])


def test_group_of_one_uses_single_job_prompt(monkeypatch):
    client = _use(monkeypatch, "TRACK: software\nSCORE: 6\nKEYWORDS: x\nREASONING: y")
    results = scorer.score_jobs(RESUMES, [_job(1)])
    assert results[0]["score"] == 6
    assert "MULTIPLE JOBS" not in client.calls[0][0][0]["content"]


# -- run_scoring with groups ---------------------------------------------------

@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    conn = init_db(path)
    for n in range(1, 8):
        conn.execute(
            "INSERT INTO jobs (url, title, site, location, full_description, discovered_at) "
            "VALUES (?, ?, 'indeed', 'Melbourne', 'desc', ?)",
            (f"https://x/{n}", f"Engineer {n}", f"2026-01-{30 - n:02d}"),
        )
    conn.commit()
    monkeypatch.setattr(scorer, "load_search_config", lambda: {"score_jobs_per_request": 3})
    resume = tmp_path / "r.txt"
    resume.write_text("r", encoding="utf-8")
    monkeypatch.setattr(scorer, "get_resume_tracks", lambda _c: {"software": {"text": resume, "pdf": None}})
    yield conn
    database.close_connection(path)


def test_run_scoring_groups_jobs(db, monkeypatch):
    sizes = []

    def fake(resumes, jobs):
        sizes.append(len(jobs))
        return [{"score": 7, "keywords": "", "reasoning": "", "track": "software"} for _ in jobs]

    monkeypatch.setattr(scorer, "score_jobs", fake)
    stats = scorer.run_scoring()

    assert sizes == [3, 3, 1]
    assert stats["scored"] == 7


def test_partial_group_saves_successes_and_resets_failure_count(db, monkeypatch):
    def fake(resumes, jobs):
        return [{"score": 0 if n == 0 else 6, "keywords": "", "reasoning": "x", "track": "software"}
                for n, _ in enumerate(jobs)]

    monkeypatch.setattr(scorer, "score_jobs", fake)
    stats = scorer.run_scoring()

    assert stats["scored"] == 4 and stats["errors"] == 3 and stats["aborted"] is False


def test_quota_mid_run_keeps_earlier_groups(db, monkeypatch):
    calls = []

    def fake(resumes, jobs):
        calls.append(jobs)
        if len(calls) == 2:
            raise DailyQuotaExceeded("daily")
        return [{"score": 8, "keywords": "", "reasoning": "", "track": "software"} for _ in jobs]

    monkeypatch.setattr(scorer, "score_jobs", fake)
    stats = scorer.run_scoring()

    assert stats["scored"] == 3 and stats["aborted"] is True
    assert db.execute("SELECT COUNT(*) FROM jobs WHERE fit_score IS NOT NULL").fetchone()[0] == 3
