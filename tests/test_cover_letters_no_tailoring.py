"""Tests for cover letters without tailoring: track resumes, stage wiring, per-job saves."""

import sqlite3

import pytest

from applypilot import database, pipeline
from applypilot.config import get_tailor_resumes
from applypilot.database import cover_letter_pending_where, init_db
from applypilot.llm import DailyQuotaExceeded
from applypilot.scoring import cover_letter

TRACKS = ["software", "devops"]


# -- config ------------------------------------------------------------------

@pytest.mark.parametrize(("cfg", "expected"), [({}, True), ({"tailor_resumes": False}, False)])
def test_get_tailor_resumes(cfg, expected):
    assert get_tailor_resumes(cfg) is expected


def test_get_tailor_resumes_rejects_non_bool():
    with pytest.raises(TypeError, match="tailor_resumes"):
        get_tailor_resumes({"tailor_resumes": "no"})


# -- stage wiring ------------------------------------------------------------

def test_all_skips_tailor_when_off():
    assert pipeline._resolve_stages(["all"], tailoring=False) == ["discover", "enrich", "score", "cover", "pdf"]


def test_all_keeps_tailor_by_default():
    assert "tailor" in pipeline._resolve_stages(["all"])


def test_explicit_tailor_still_runs_when_off():
    assert pipeline._resolve_stages(["tailor", "cover"], tailoring=False) == ["tailor", "cover"]


def test_cover_follows_score_when_tailoring_off():
    assert pipeline._upstream("cover", tailoring=False) == "score"
    assert pipeline._upstream("cover", tailoring=True) == "tailor"


# -- pending rule ------------------------------------------------------------

def test_pending_where_requires_tailored_resume_only_when_on():
    on, _ = cover_letter_pending_where(7, 5, True, TRACKS)
    off, params = cover_letter_pending_where(7, 5, False, TRACKS)
    assert "tailored_resume_path IS NOT NULL" in on
    assert "tailored_resume_path" not in off
    assert params == [7, 5, "software", "devops"]


def test_pending_where_single_track_does_not_filter_tracks():
    where, params = cover_letter_pending_where(7, 5, False, ["default"])
    assert "resume_track" not in where
    assert params == [7, 5]


# -- run_cover_letters against a temp DB -------------------------------------

JOBS = [
    # url, score, track, location
    ("https://a/devops-vic", 8, "devops", "Melbourne VIC"),
    ("https://b/software-nsw", 8, "software", "Sydney NSW"),
    ("https://c/no-track", 9, None, "Melbourne VIC"),
    ("https://d/low-score", 5, "software", "Melbourne VIC"),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    conn = init_db(db_path)
    for url, score, track, location in JOBS:
        conn.execute(
            "INSERT INTO jobs (url, title, site, location, fit_score, resume_track, full_description) "
            "VALUES (?, 'Software Engineer', 'indeed', ?, ?, ?, 'desc')",
            (url, location, score, track),
        )
    conn.commit()

    resumes = {}
    for name in TRACKS:
        path = tmp_path / f"{name}.txt"
        path.write_text(f"{name.upper()} RESUME", encoding="utf-8")
        resumes[name] = {"text": path, "pdf": None}

    out_dir = tmp_path / "letters"
    cfg = {"tailor_resumes": False, "location_preferred": ["VIC"]}
    monkeypatch.setattr(cover_letter, "COVER_LETTER_DIR", out_dir)
    monkeypatch.setattr(cover_letter, "load_profile", lambda: {"personal": {}})
    monkeypatch.setattr(cover_letter, "load_search_config", lambda: cfg)
    monkeypatch.setattr(cover_letter, "get_resume_tracks", lambda _cfg: resumes)
    monkeypatch.setattr(cover_letter, "_save_pdf_best_effort", lambda _p: None)
    yield {"db": db_path, "out": out_dir}
    database.close_connection(db_path)


def _letters(db_path):
    other = sqlite3.connect(db_path)
    try:
        rows = other.execute("SELECT url, cover_letter_path, cover_attempts FROM jobs").fetchall()
    finally:
        other.close()
    return {url: (path, attempts) for url, path, attempts in rows}


def test_letters_use_track_resume_without_tailoring(env, monkeypatch):
    used = {}

    def fake(resume_text, job, profile, validation_mode):
        used[job["url"]] = resume_text
        return f"Letter for {job['url']}"

    monkeypatch.setattr(cover_letter, "generate_cover_letter", fake)
    stats = cover_letter.run_cover_letters(min_score=7)

    assert used == {"https://a/devops-vic": "DEVOPS RESUME", "https://b/software-nsw": "SOFTWARE RESUME"}
    assert list(used) == ["https://a/devops-vic", "https://b/software-nsw"]  # VIC first on tie
    assert stats["generated"] == 2 and stats["skipped_no_track"] == 1

    letters = _letters(env["db"])
    assert letters["https://c/no-track"] == (None, 0)  # left for after re-scoring
    assert letters["https://d/low-score"] == (None, 0)


def test_same_title_jobs_get_distinct_files(env, monkeypatch):
    monkeypatch.setattr(cover_letter, "generate_cover_letter", lambda r, j, p, validation_mode: "x")
    cover_letter.run_cover_letters(min_score=7)

    paths = {path for path, _ in _letters(env["db"]).values() if path}
    assert len(paths) == 2  # both are "indeed_Software_Engineer", but files differ


def test_each_letter_saved_before_a_later_failure(env, monkeypatch):
    calls = []

    def fake(resume_text, job, profile, validation_mode):
        calls.append(job["url"])
        if len(calls) == 2:
            raise DailyQuotaExceeded("daily quota")
        return "ok"

    monkeypatch.setattr(cover_letter, "generate_cover_letter", fake)
    cover_letter.run_cover_letters(min_score=7)

    letters = _letters(env["db"])
    assert letters["https://a/devops-vic"][0] is not None
    assert letters["https://b/software-nsw"] == (None, 0)  # quota stop is not counted as an attempt


def test_generation_error_counts_attempt(env, monkeypatch):
    def boom(resume_text, job, profile, validation_mode):
        raise RuntimeError("validation failed")

    monkeypatch.setattr(cover_letter, "generate_cover_letter", boom)
    stats = cover_letter.run_cover_letters(min_score=7)

    assert stats["errors"] == 2
    assert _letters(env["db"])["https://a/devops-vic"] == (None, 1)


def test_pipeline_pending_count_matches_stage(env, monkeypatch):
    monkeypatch.setattr(pipeline, "load_search_config", lambda: {"tailor_resumes": False})
    monkeypatch.setattr(pipeline, "get_resume_tracks", lambda _cfg: dict.fromkeys(TRACKS))
    assert pipeline._count_pending("cover", min_score=7) == 2  # no-track and low-score excluded
