"""Tests for run_scoring: per-job persistence, failed jobs left unscored, early abort."""

import sqlite3

import pytest

from applypilot import database
from applypilot.database import init_db
from applypilot.scoring import scorer

URLS = [f"https://e/job-{i}" for i in range(8)]
FAIL = {"score": 0, "keywords": "", "reasoning": "LLM error: boom", "track": None}


def _ok(score, track="software"):
    return {"score": score, "keywords": "python", "reasoning": "good fit", "track": track}


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    connection = init_db(path)
    for i, url in enumerate(URLS):
        # Descending discovered_at so jobs are scored in URLS order.
        connection.execute(
            "INSERT INTO jobs (url, title, site, location, full_description, discovered_at) "
            "VALUES (?, 'Engineer', 'indeed', 'Melbourne', 'desc', ?)",
            (url, f"2026-01-{20 - i:02d}"),
        )
    connection.commit()

    resume = tmp_path / "software.txt"
    resume.write_text("software resume", encoding="utf-8")
    monkeypatch.setattr(scorer, "load_search_config", dict)
    monkeypatch.setattr(scorer, "get_resume_tracks", lambda _cfg: {"software": {"text": resume, "pdf": None}})
    yield path
    database.close_connection(path)


def _fake_scores(monkeypatch, results):
    """Make score_job return the given results in order; returns the URLs it was called with."""
    seen = []

    def fake(resumes, job):
        seen.append(job["url"])
        return dict(results[len(seen) - 1])

    monkeypatch.setattr(scorer, "score_job", fake)
    return seen


def _committed_scores(db_path):
    """Read scores through a separate connection, so only committed rows are visible."""
    other = sqlite3.connect(db_path)
    try:
        rows = other.execute("SELECT url, fit_score, resume_track FROM jobs").fetchall()
    finally:
        other.close()
    return {url: (score, track) for url, score, track in rows}


def test_successful_scores_persisted_and_failures_left_null(db_path, monkeypatch):
    _fake_scores(monkeypatch, [_ok(8), FAIL, _ok(5), FAIL, _ok(3), _ok(9), _ok(7), _ok(6)])

    result = scorer.run_scoring()

    assert set(result) == {"scored", "errors", "elapsed", "distribution", "aborted"}
    assert result["scored"] == 6
    assert result["errors"] == 2
    assert result["aborted"] is False
    scores = _committed_scores(db_path)
    assert scores[URLS[0]] == (8, "software")
    assert scores[URLS[1]] == (None, None)
    assert scores[URLS[3]] == (None, None)
    row = database.get_connection().execute(
        "SELECT score_reasoning, scored_at FROM jobs WHERE url = ?", (URLS[0],)
    ).fetchone()
    assert row[0] == "python\ngood fit"
    assert row[1] is not None


def test_failed_jobs_are_retried_on_next_run(db_path, monkeypatch):
    _fake_scores(monkeypatch, [_ok(8), FAIL] + [_ok(5)] * 6)
    scorer.run_scoring()

    seen = _fake_scores(monkeypatch, [_ok(4)])
    result = scorer.run_scoring()

    assert seen == [URLS[1]]
    assert result["scored"] == 1
    assert _committed_scores(db_path)[URLS[1]] == (4, "software")


def test_all_failures_abort_after_limit_and_write_nothing(db_path, monkeypatch):
    seen = _fake_scores(monkeypatch, [FAIL] * len(URLS))

    result = scorer.run_scoring()

    assert len(seen) == scorer.MAX_CONSECUTIVE_FAILURES
    assert result["aborted"] is True
    assert result["scored"] == 0
    assert result["errors"] == scorer.MAX_CONSECUTIVE_FAILURES
    assert all(score is None for score, _ in _committed_scores(db_path).values())


def test_success_resets_consecutive_failure_count(db_path, monkeypatch):
    seen = _fake_scores(monkeypatch, [FAIL] * 4 + [_ok(7)] + [FAIL] * 3)

    result = scorer.run_scoring()

    assert len(seen) == len(URLS)
    assert result["aborted"] is False
    assert result["scored"] == 1
    assert result["errors"] == 7


def test_score_committed_before_later_abort(db_path, monkeypatch):
    _fake_scores(monkeypatch, [_ok(9)] + [FAIL] * 5)

    result = scorer.run_scoring()

    assert result["aborted"] is True
    assert result["scored"] == 1
    assert _committed_scores(db_path)[URLS[0]] == (9, "software")


def test_score_committed_before_crash(db_path, monkeypatch):
    calls = []

    def fake(resumes, job):
        calls.append(job["url"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return _ok(6)

    monkeypatch.setattr(scorer, "score_job", fake)
    with pytest.raises(KeyboardInterrupt):
        scorer.run_scoring()

    assert _committed_scores(db_path)[URLS[0]] == (6, "software")


def test_rescore_overwrites_existing_scores(db_path, monkeypatch):
    conn = database.get_connection()
    conn.execute("UPDATE jobs SET fit_score = 2, resume_track = 'old'")
    conn.commit()
    _fake_scores(monkeypatch, [_ok(8)] * len(URLS))

    result = scorer.run_scoring(rescore=True, limit=3)

    assert result["scored"] == 3
    assert sorted(score for score, _ in _committed_scores(db_path).values()) == [2] * 5 + [8] * 3
