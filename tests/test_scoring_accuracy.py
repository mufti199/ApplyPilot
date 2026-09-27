"""Tests for the accuracy settings: track guides, deal-breaker rule, single-job budget."""

import pytest

from applypilot.config import get_resume_tracks
from applypilot.scoring import scorer

RESUMES = {"software": "SOFTWARE RESUME", "devops": "DEVOPS RESUME"}
GUIDES = {"software": "application and AI roles", "devops": "infrastructure, SRE and DBA roles"}
JOB = {"title": "Database Administrator", "site": "indeed", "location": "Melbourne VIC", "full_description": "PostgreSQL"}


@pytest.fixture
def files(tmp_path):
    out = {}
    for name in RESUMES:
        (tmp_path / f"{name}.txt").write_text(name, encoding="utf-8")
        out[name] = {"text": str(tmp_path / f"{name}.txt")}
    return out


# -- focus in config -----------------------------------------------------------

def test_focus_is_read_and_trimmed(files):
    files["devops"]["focus"] = "  infrastructure roles  "
    tracks = get_resume_tracks({"resumes": files})
    assert tracks["devops"]["focus"] == "infrastructure roles"
    assert tracks["software"]["focus"] is None


@pytest.mark.parametrize("bad", ["", "   ", 5, ["a"]])
def test_focus_must_be_text(files, bad):
    files["software"]["focus"] = bad
    with pytest.raises(TypeError, match="focus"):
        get_resume_tracks({"resumes": files})


# -- prompt ------------------------------------------------------------------

def test_prompt_lists_track_guides_and_duties_rule():
    prompt = scorer._build_score_prompt(["software", "devops"], GUIDES)
    assert "- software: application and AI roles" in prompt
    assert "- devops: infrastructure, SRE and DBA roles" in prompt
    assert "main day-to-day duties, not by its title or the order" in prompt


def test_prompt_without_guides_still_valid():
    prompt = scorer._build_score_prompt(["software", "devops"])
    assert "Which TRACK fits" not in prompt and "TRACK: [exactly one of: software, devops]" in prompt


def test_single_track_prompt_unchanged():
    assert scorer._build_score_prompt(["default"], GUIDES) == scorer.SCORE_PROMPT


def test_deal_breakers_are_major_gaps():
    assert "security clearance" in scorer.SCORE_PROMPT and "major gaps" in scorer.SCORE_PROMPT


# -- request shape -------------------------------------------------------------

class Recorder:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "TRACK: devops\nSCORE: 6\nKEYWORDS: PostgreSQL\nREASONING: ok"


def test_single_job_gets_large_output_budget_and_guides(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(scorer, "get_client", lambda: rec)

    result = scorer.score_jobs(RESUMES, [JOB], GUIDES)

    messages, kwargs = rec.calls[0]
    assert kwargs["max_tokens"] == scorer._SINGLE_JOB_TOKENS == 4096
    assert "infrastructure, SRE and DBA roles" in messages[0]["content"]
    assert result[0]["track"] == "devops"


def test_run_scoring_passes_guides(tmp_path, monkeypatch):
    from applypilot import database
    from applypilot.database import init_db

    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    conn = init_db(db_path)
    conn.execute("INSERT INTO jobs (url, title, site, full_description) VALUES ('https://x/1', 'DBA', 'indeed', 'd')")
    conn.commit()
    resume = tmp_path / "r.txt"
    resume.write_text("r", encoding="utf-8")
    monkeypatch.setattr(scorer, "load_search_config", dict)
    monkeypatch.setattr(scorer, "get_resume_tracks", lambda _c: {
        "software": {"text": resume, "pdf": None, "focus": "apps"},
        "devops": {"text": resume, "pdf": None, "focus": None},
    })
    seen = []

    def fake(resumes, jobs, guides=None):
        seen.append(guides)
        return [{"score": 5, "keywords": "", "reasoning": "", "track": "software", "company": None, "employment": None}]

    monkeypatch.setattr(scorer, "score_jobs", fake)
    scorer.run_scoring()
    database.close_connection(db_path)

    assert seen == [{"software": "apps"}]  # tracks without focus are left out
