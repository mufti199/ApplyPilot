"""Tests for storing and using the real employer name instead of the job board."""

import pandas as pd
import pytest

from applypilot import database
from applypilot.database import init_db
from applypilot.discovery.jobspy import store_jobspy_results
from applypilot.scoring import scorer


@pytest.fixture
def conn(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    connection = init_db(path)
    yield connection
    database.close_connection(path)


def test_jobspy_company_is_stored(conn):
    df = pd.DataFrame([{
        "job_url": "https://www.linkedin.com/jobs/view/1", "title": "Cloud Engineer", "company": "Acme Bank",
        "location": "Melbourne, VIC", "site": "linkedin", "description": "x" * 300,
    }])
    store_jobspy_results(conn, df, "linkedin")
    row = conn.execute("SELECT company, site FROM jobs").fetchone()
    assert (row["company"], row["site"]) == ("Acme Bank", "linkedin")


def test_missing_company_stored_as_null(conn):
    df = pd.DataFrame([{"job_url": "https://x/2", "title": "Dev", "company": float("nan"),
                        "location": "Sydney", "site": "indeed", "description": "y" * 300}])
    store_jobspy_results(conn, df, "indeed")
    assert conn.execute("SELECT company FROM jobs").fetchone()[0] is None


# -- prompts -----------------------------------------------------------------

def test_scoring_prompt_uses_company_not_board():
    text = scorer._format_job({"title": "Dev", "site": "linkedin", "company": "Acme", "full_description": "d"})
    assert "COMPANY: Acme" in text and "linkedin" not in text


def test_scoring_prompt_without_company_says_not_stated():
    text = scorer._format_job({"title": "Dev", "site": "indeed", "company": None, "full_description": "d"})
    assert "COMPANY: not stated" in text


@pytest.mark.parametrize(("line", "expected"), [
    ("COMPANY: Acme Bank", "Acme Bank"),
    ("COMPANY: [Acme]", "Acme"),
    ("COMPANY: unknown", None),
    ("COMPANY: Unknown", None),
])
def test_parse_company(line, expected):
    parsed = scorer._parse_score_response(f"SCORE: 7\nKEYWORDS: k\nREASONING: r\n{line}")
    assert parsed["company"] == expected


def test_parse_without_company_line():
    assert scorer._parse_score_response("SCORE: 7\nKEYWORDS: k\nREASONING: r")["company"] is None


# -- backfill during scoring -------------------------------------------------

def _insert(conn, url, company):
    conn.execute("INSERT INTO jobs (url, title, site, company, full_description) VALUES (?, 'Dev', 'indeed', ?, 'd')",
                 (url, company))
    conn.commit()


def _result(company):
    return {"score": 7, "keywords": "", "reasoning": "", "track": "software", "company": company}


def test_scoring_fills_missing_company(conn):
    _insert(conn, "https://x/1", None)
    scorer._save_score(conn, "https://x/1", _result("Acme"))
    assert conn.execute("SELECT company FROM jobs").fetchone()[0] == "Acme"


def test_scoring_never_overwrites_stored_company(conn):
    _insert(conn, "https://x/1", "Real Employer")
    scorer._save_score(conn, "https://x/1", _result("Something Else"))
    assert conn.execute("SELECT company FROM jobs").fetchone()[0] == "Real Employer"


# -- cover letter ------------------------------------------------------------

def test_cover_letter_prompt_handles_unnamed_company():
    from applypilot.scoring.cover_letter import _build_cover_letter_prompt
    prompt = _build_cover_letter_prompt({"personal": {"full_name": "Sam"}})
    assert 'If COMPANY is "not stated"' in prompt
