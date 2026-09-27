"""Tests for grouping the same job posted on several boards (or reposted)."""

import pytest

from applypilot import database
from applypilot.database import cover_letter_pending_where, dedupe_key, get_jobs_by_stage, init_db, mark_duplicates

DESC = "We are hiring a Cloud Engineer to build Azure platforms with Terraform and Kubernetes. " * 5


@pytest.fixture
def conn(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    connection = init_db(path)
    yield connection
    database.close_connection(path)


def _add(conn, url, site, title="Cloud Engineer", desc=DESC, score=None):
    conn.execute(
        "INSERT INTO jobs (url, title, site, full_description, fit_score) VALUES (?, ?, ?, ?, ?)",
        (url, title, site, desc, score),
    )
    conn.commit()


def _dup_of(conn):
    return dict(conn.execute("SELECT url, duplicate_of FROM jobs").fetchall())


# -- key ---------------------------------------------------------------------

def test_key_ignores_case_and_punctuation():
    assert dedupe_key("Cloud Engineer!", DESC) == dedupe_key("cloud engineer", DESC.upper().replace(",", ""))


def test_key_differs_for_different_description():
    assert dedupe_key("Cloud Engineer", DESC) != dedupe_key("Cloud Engineer", "A different role entirely. " * 20)


@pytest.mark.parametrize(("title", "desc"), [("Cloud Engineer", "too short"), ("", DESC), (None, DESC)])
def test_no_key_for_short_description_or_missing_title(title, desc):
    assert dedupe_key(title, desc) is None


# -- grouping ----------------------------------------------------------------

def test_cross_board_copies_point_at_earliest(conn):
    _add(conn, "https://indeed/1", "indeed")
    _add(conn, "https://linkedin/1", "linkedin")
    _add(conn, "https://linkedin/2", "linkedin")  # repost

    assert mark_duplicates(conn) == 2
    assert _dup_of(conn) == {"https://indeed/1": None, "https://linkedin/1": 1, "https://linkedin/2": 1}


def test_scored_copy_becomes_main(conn):
    _add(conn, "https://indeed/1", "indeed")
    _add(conn, "https://linkedin/1", "linkedin", score=8)

    mark_duplicates(conn)

    assert _dup_of(conn) == {"https://indeed/1": 2, "https://linkedin/1": None}


def test_different_jobs_not_grouped(conn):
    _add(conn, "https://a", "indeed", title="Cloud Engineer")
    _add(conn, "https://b", "indeed", title="DevOps Engineer")
    _add(conn, "https://c", "linkedin", title="Cloud Engineer", desc="Different employer and duties. " * 20)

    assert mark_duplicates(conn) == 0


def test_idempotent(conn):
    _add(conn, "https://indeed/1", "indeed")
    _add(conn, "https://linkedin/1", "linkedin")
    assert mark_duplicates(conn) == 1
    assert mark_duplicates(conn) == 1
    assert _dup_of(conn)["https://linkedin/1"] == 1


# -- effect on the pipeline ----------------------------------------------------

def test_duplicates_not_offered_for_scoring(conn):
    _add(conn, "https://indeed/1", "indeed")
    _add(conn, "https://linkedin/1", "linkedin")
    mark_duplicates(conn)

    pending = get_jobs_by_stage(conn=conn, stage="pending_score", limit=0)
    assert [j["url"] for j in pending] == ["https://indeed/1"]


def test_duplicates_not_offered_for_cover_letters(conn):
    _add(conn, "https://indeed/1", "indeed", score=8)
    _add(conn, "https://linkedin/1", "linkedin", score=8)
    mark_duplicates(conn)

    where, params = cover_letter_pending_where(7, 5, False, ["software"])
    urls = [r[0] for r in conn.execute(f"SELECT url FROM jobs WHERE {where}", params)]
    assert urls == ["https://indeed/1"]


def test_dashboard_shows_copies_on_main_card(conn):
    from applypilot import dashboard_server

    _add(conn, "https://indeed/1", "indeed", score=8)
    _add(conn, "https://linkedin/1", "linkedin")
    mark_duplicates(conn)

    jobs = dashboard_server.dashboard_data(conn)["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["also_on"] == [{"site": "linkedin", "url": "https://linkedin/1"}]
