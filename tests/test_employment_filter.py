"""Tests for the permanent full-time filter: rules, scorer answer, storage, and effects."""

import pandas as pd
import pytest

from applypilot import database
from applypilot.config import get_permanent_full_time_only
from applypilot.database import get_jobs_by_stage, init_db
from applypilot.discovery.jobspy import store_jobspy_results
from applypilot.scoring import scorer
from applypilot.scoring.employment import apply_rules, normalise_llm_employment, rule_exclusion_reason

LONG = "Build and run Azure platforms with Terraform for a growing product team. " * 5


# -- config ------------------------------------------------------------------

@pytest.mark.parametrize(("cfg", "expected"), [
    ({}, False), ({"employment": {}}, False), ({"employment": {"permanent_full_time_only": True}}, True),
])
def test_setting(cfg, expected):
    assert get_permanent_full_time_only(cfg) is expected


@pytest.mark.parametrize("cfg", [{"employment": True}, {"employment": {"permanent_full_time_only": "yes"}}])
def test_setting_rejects_bad_shape(cfg):
    with pytest.raises(TypeError):
        get_permanent_full_time_only(cfg)


# -- rules -------------------------------------------------------------------

@pytest.mark.parametrize(("title", "desc", "salary", "job_type"), [
    ("Cloud Engineer", LONG, None, "contract"),
    ("Cloud Engineer", LONG, None, "fulltime, parttime"),
    ("DevSecOps SME - Contract", LONG, None, None),
    ("System Engineer - 6-month Fixed Term", LONG, None, None),
    ("Software Development Expert ($100/hr, remote)", LONG, None, None),
    ("Quality Auditor - 12 Month Parental Leave Cover", LONG, None, None),
    ("Data Engineer", LONG, "$800-$1,000/day", None),
    ("Full Stack Engineer", LONG + " This is an initial 12-month contract.", None, None),
    ("Senior Cloud Engineer", LONG + " This is a part-time role, 3 days a week.", None, None),
    ("Platform Engineer", LONG + " Paid at a competitive day rate.", None, None),
])
def test_rules_exclude_non_permanent(title, desc, salary, job_type):
    assert rule_exclusion_reason(title, desc, salary, job_type) is not None


@pytest.mark.parametrize(("title", "desc", "job_type"), [
    ("Cloud Engineer", LONG, "fulltime"),
    ("Internal Tools Engineer", LONG, None),            # 'internal' is not 'intern'
    ("Temple Systems Developer", LONG, None),           # 'temple' is not 'temp'
    ("Platform Engineer", LONG + " You will work alongside contractors and vendors.", None),
    ("DevOps Engineer", LONG + " Permanent full-time role with 6 months parental leave.", None),
])
def test_rules_keep_permanent(title, desc, job_type):
    assert rule_exclusion_reason(title, desc, None, job_type) is None


@pytest.mark.parametrize(("raw", "expected"), [
    ("permanent full-time", "permanent"), ("Full-time", "permanent"), ("Contract", "contract"),
    ("part time", "part-time"), ("[casual]", "casual"), ("unknown", None), ("", None), (None, None),
])
def test_normalise_llm_employment(raw, expected):
    assert normalise_llm_employment(raw) == expected


# -- database effects --------------------------------------------------------

@pytest.fixture
def conn(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    connection = init_db(path)
    yield connection
    database.close_connection(path)


def _add(conn, url, title, desc=LONG, job_type=None):
    conn.execute("INSERT INTO jobs (url, title, site, full_description, job_type) VALUES (?, ?, 'indeed', ?, ?)",
                 (url, title, desc, job_type))
    conn.commit()


def test_jobspy_job_type_is_stored(conn):
    df = pd.DataFrame([{"job_url": "https://x/1", "title": "Dev", "company": "Acme", "location": "Melbourne",
                        "site": "indeed", "description": LONG, "job_type": "contract"}])
    store_jobspy_results(conn, df, "indeed")
    assert conn.execute("SELECT job_type FROM jobs").fetchone()[0] == "contract"


def test_apply_rules_excludes_and_hides_from_scoring(conn):
    _add(conn, "https://x/perm", "Cloud Engineer")
    _add(conn, "https://x/contract", "Cloud Engineer - Contract")

    assert apply_rules(conn) == 1
    assert apply_rules(conn) == 0  # idempotent

    reason = conn.execute("SELECT excluded_reason FROM jobs WHERE url = 'https://x/contract'").fetchone()[0]
    assert "contract" in reason.lower()
    pending = get_jobs_by_stage(conn=conn, stage="pending_score", limit=0)
    assert [j["url"] for j in pending] == ["https://x/perm"]


def test_scorer_answer_excludes_when_enabled(conn):
    _add(conn, "https://x/1", "Cloud Engineer")
    result = {"score": 8, "keywords": "", "reasoning": "", "track": "devops", "company": None,
              "employment": "contract"}

    scorer._save_score(conn, "https://x/1", result, exclude_non_permanent=True)

    row = conn.execute("SELECT fit_score, excluded_reason FROM jobs").fetchone()
    assert row["fit_score"] == 8 and row["excluded_reason"] == "scorer says contract"


@pytest.mark.parametrize("employment", ["permanent", None])
def test_scorer_answer_keeps_permanent_and_unknown(conn, employment):
    _add(conn, "https://x/1", "Cloud Engineer")
    result = {"score": 8, "keywords": "", "reasoning": "", "track": "devops", "company": None,
              "employment": employment}
    scorer._save_score(conn, "https://x/1", result, exclude_non_permanent=True)
    assert conn.execute("SELECT excluded_reason FROM jobs").fetchone()[0] is None


def test_scorer_answer_ignored_when_filter_off(conn):
    _add(conn, "https://x/1", "Cloud Engineer")
    result = {"score": 8, "keywords": "", "reasoning": "", "track": "devops", "company": None,
              "employment": "contract"}
    scorer._save_score(conn, "https://x/1", result, exclude_non_permanent=False)
    assert conn.execute("SELECT excluded_reason FROM jobs").fetchone()[0] is None


def test_parse_employment_line():
    parsed = scorer._parse_score_response("SCORE: 7\nKEYWORDS: k\nREASONING: r\nEMPLOYMENT: Contract")
    assert parsed["employment"] == "contract"


def test_run_scoring_applies_rules_first_when_enabled(conn, monkeypatch, tmp_path):
    _add(conn, "https://x/perm", "Cloud Engineer")
    _add(conn, "https://x/contract", "Cloud Engineer (12 month contract)")
    resume = tmp_path / "r.txt"
    resume.write_text("r", encoding="utf-8")
    monkeypatch.setattr(scorer, "load_search_config", lambda: {"employment": {"permanent_full_time_only": True}})
    monkeypatch.setattr(scorer, "get_resume_tracks", lambda _c: {"software": {"text": resume, "pdf": None}})
    seen = []

    def fake(resumes, job):
        seen.append(job["url"])
        return {"score": 7, "keywords": "", "reasoning": "", "track": "software", "company": None, "employment": None}

    monkeypatch.setattr(scorer, "score_job", fake)
    scorer.run_scoring()

    assert seen == ["https://x/perm"]


def test_dashboard_lists_excluded_with_reason(conn):
    from applypilot import dashboard_server

    _add(conn, "https://x/contract", "Cloud Engineer - Contract")
    apply_rules(conn)

    data = dashboard_server.dashboard_data(conn)
    assert data["stats"]["excluded"] == 1
    assert data["jobs"][0]["excluded"].startswith("title says")
