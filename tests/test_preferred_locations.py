"""Tests for the location_preferred tie-break on fit_score."""

import pytest

from applypilot import config, database
from applypilot.config import get_preferred_locations
from applypilot.database import get_jobs_by_stage, init_db, preferred_location_order

# -- get_preferred_locations -------------------------------------------------

@pytest.mark.parametrize("cfg", [None, {}, {"queries": []}])
def test_no_setting_means_no_preference(cfg):
    assert get_preferred_locations(cfg) == []


def test_patterns_are_trimmed():
    assert get_preferred_locations({"location_preferred": [" Victoria ", "VIC"]}) == ["Victoria", "VIC"]


@pytest.mark.parametrize("value", ["Victoria", ["Victoria", 3], {"a": "b"}])
def test_non_list_of_strings_rejected(value):
    with pytest.raises(TypeError, match="list of strings"):
        get_preferred_locations({"location_preferred": value})


def test_blank_pattern_rejected():
    with pytest.raises(ValueError, match="blank"):
        get_preferred_locations({"location_preferred": ["Victoria", "  "]})


# -- preferred_location_order ------------------------------------------------

def test_empty_patterns_leave_order_unchanged():
    assert preferred_location_order([]) == ("NULL", [])


def test_like_wildcards_are_escaped():
    _, params = preferred_location_order(["50%_off"])
    assert params == ["%50\\%\\_off%"]


# -- ordering against a real SQLite DB ---------------------------------------

JOBS = [
    # (url, fit_score, location)
    ("https://a/sydney", 8, "Sydney NSW"),
    ("https://b/melbourne", 8, "Melbourne VIC"),
    ("https://c/none", 8, None),
    ("https://d/perth-high", 9, "Perth WA"),
    ("https://e/geelong-low", 7, "Geelong, Victoria"),
]
PREFERRED = ["Victoria", "VIC", "Melbourne", "Geelong"]


@pytest.fixture
def conn(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    connection = init_db(db_path)
    for url, score, location in JOBS:
        connection.execute(
            "INSERT INTO jobs (url, title, site, location, fit_score, full_description, tailored_resume_path) "
            "VALUES (?, 'Engineer', 'indeed', ?, ?, 'desc', '/tmp/r.txt')",
            (url, location, score),
        )
    connection.commit()
    yield connection
    database.close_connection(db_path)


def _urls(rows):
    return [r["url"] for r in rows]


def test_preferred_location_wins_ties_but_not_higher_scores(conn):
    rows = get_jobs_by_stage(conn=conn, stage="tailored", min_score=1, preferred_locations=PREFERRED)
    urls = _urls(rows)

    assert urls[0] == "https://d/perth-high"          # higher score still first
    assert urls[1] == "https://b/melbourne"           # wins the score-8 tie
    assert urls[-1] == "https://e/geelong-low"        # lower score still last


def test_without_preference_all_jobs_returned_by_score(conn):
    rows = get_jobs_by_stage(conn=conn, stage="tailored", min_score=1)
    assert [r["fit_score"] for r in rows] == [9, 8, 8, 8, 7]


def test_acquire_job_prefers_victoria_on_tie(conn, monkeypatch):
    from applypilot.apply import launcher

    monkeypatch.setattr(config, "load_search_config", lambda: {"location_preferred": PREFERRED})
    conn.execute("UPDATE jobs SET fit_score = 8")  # make every job tie
    conn.commit()

    job = launcher.acquire_job(min_score=7)

    assert job["url"] in {"https://b/melbourne", "https://e/geelong-low"}
