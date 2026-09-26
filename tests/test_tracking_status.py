"""Tests for manual tracking statuses: DB helpers, CLI commands, auto-apply exclusion."""

import pytest
from typer.testing import CliRunner

from applypilot import cli, config, database
from applypilot.database import TRACKING_STATUSES, find_job, init_db, set_tracking_status

runner = CliRunner()


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    monkeypatch.setattr(cli, "_bootstrap", lambda: None)
    conn = init_db(path)
    for n, score in ((1, 9), (2, 8), (3, 5)):
        conn.execute(
            "INSERT INTO jobs (url, application_url, title, site, location, fit_score, tailored_resume_path) "
            "VALUES (?, ?, ?, 'indeed', 'Melbourne VIC', ?, '/r.txt')",
            (f"https://x/{n}", f"https://apply/{n}", f"Engineer {n}", score),
        )
    conn.commit()
    yield conn
    database.close_connection(path)


def test_statuses_match_the_agreed_list():
    assert TRACKING_STATUSES == ("applied", "skipped", "interviewing", "rejected", "offer", "cold-call")


@pytest.mark.parametrize("ref", ["1", "https://x/1", "https://apply/1"])
def test_find_job_by_number_or_url(db, ref):
    assert find_job(ref, db)["url"] == "https://x/1"


def test_find_job_missing(db):
    assert find_job("999", db) is None


def test_set_status_and_clear(db):
    set_tracking_status("2", "interviewing", db)
    row = db.execute("SELECT tracking_status, tracking_updated_at, applied_at FROM jobs WHERE rowid = 2").fetchone()
    assert row["tracking_status"] == "interviewing" and row["tracking_updated_at"]
    assert row["applied_at"] is None  # only 'applied' sets applied_at

    set_tracking_status("2", None, db)
    row = db.execute("SELECT tracking_status, tracking_updated_at FROM jobs WHERE rowid = 2").fetchone()
    assert row["tracking_status"] is None and row["tracking_updated_at"] is None


def test_applied_sets_applied_at_once(db):
    set_tracking_status("1", "applied", db)
    first = db.execute("SELECT applied_at FROM jobs WHERE rowid = 1").fetchone()[0]
    set_tracking_status("1", "applied", db)
    assert db.execute("SELECT applied_at FROM jobs WHERE rowid = 1").fetchone()[0] == first


def test_set_status_rejects_unknown_status_and_job(db):
    with pytest.raises(ValueError, match="Unknown status"):
        set_tracking_status("1", "hired", db)
    with pytest.raises(LookupError):
        set_tracking_status("999", "applied", db)


# -- CLI ---------------------------------------------------------------------

def test_cli_mark(db):
    result = runner.invoke(cli.app, ["mark", "3", "cold-call"])
    assert result.exit_code == 0, result.output
    assert db.execute("SELECT tracking_status FROM jobs WHERE rowid = 3").fetchone()[0] == "cold-call"


def test_cli_mark_clear(db):
    set_tracking_status("3", "skipped", db)
    assert runner.invoke(cli.app, ["mark", "3", "clear"]).exit_code == 0
    assert db.execute("SELECT tracking_status FROM jobs WHERE rowid = 3").fetchone()[0] is None


@pytest.mark.parametrize("args", [["mark", "1", "hired"], ["mark", "999", "applied"]])
def test_cli_mark_errors_exit_nonzero(db, args):
    assert runner.invoke(cli.app, args).exit_code == 1


def test_cli_jobs_lists_and_filters(db):
    set_tracking_status("1", "applied", db)

    listed = runner.invoke(cli.app, ["jobs", "--min-score", "1"])
    assert listed.exit_code == 0 and "Engineer 3" in listed.output

    untracked = runner.invoke(cli.app, ["jobs", "--status", "none"])
    assert "Engineer 2" in untracked.output and "Engineer 1" not in untracked.output


# -- auto-apply exclusion ------------------------------------------------------

def test_auto_apply_skips_tracked_jobs(db, monkeypatch):
    from applypilot.apply import launcher

    monkeypatch.setattr(config, "load_search_config", dict)
    set_tracking_status("1", "skipped", db)

    job = launcher.acquire_job(min_score=7)

    assert job["url"] == "https://x/2"  # highest score (job 1) is tracked, so skipped
