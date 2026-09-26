"""Tests for the live dashboard server: data, security checks, status updates, files."""

import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from applypilot import dashboard_server as ds
from applypilot import database
from applypilot.database import init_db


@pytest.fixture
def env(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    conn = init_db(db_path)

    letters = tmp_path / "letters"
    letters.mkdir()
    (letters / "a_CL.txt").write_text("letter", encoding="utf-8")
    (letters / "a_CL.pdf").write_bytes(b"%PDF-cover")
    outside = tmp_path / "outside_CL.pdf"
    outside.write_bytes(b"%PDF-secret")
    monkeypatch.setattr(ds, "COVER_LETTER_DIR", letters)

    rows = [
        ("https://x/1", "DevOps Engineer", 8, "devops", "Plan\nGood fit", str(letters / "a_CL.txt"), "full desc 1"),
        ("https://x/2", "Software Engineer", 6, "software", "Py\nOk", str(tmp_path / "outside_CL.txt"), "full desc 2"),
        ("https://x/3", "Unscored Engineer", None, None, None, None, "full desc 3"),
    ]
    for url, title, score, track, reasoning, cl, desc in rows:
        conn.execute(
            "INSERT INTO jobs (url, title, site, location, fit_score, resume_track, score_reasoning, "
            "cover_letter_path, full_description) VALUES (?, ?, 'indeed', 'Melbourne VIC', ?, ?, ?, ?, ?)",
            (url, title, score, track, reasoning, cl, desc),
        )
    conn.commit()

    resume_pdf = tmp_path / "devops.pdf"
    resume_pdf.write_bytes(b"%PDF-resume")
    monkeypatch.setattr(ds, "load_search_config", dict)
    monkeypatch.setattr(ds, "get_resume_tracks", lambda _c: {
        "devops": {"text": tmp_path / "d.txt", "pdf": resume_pdf},
        "software": {"text": tmp_path / "s.txt", "pdf": None},
    })
    yield conn
    database.close_connection(db_path)


@pytest.fixture
def server(env):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ds.DashboardHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    with httpx.Client(base_url=base, timeout=5) as client:
        yield client
    srv.shutdown()
    srv.server_close()


# -- data --------------------------------------------------------------------

def test_dashboard_data_lists_scored_jobs_only(env):
    data = ds.dashboard_data(env)
    assert [j["title"] for j in data["jobs"]] == ["DevOps Engineer", "Software Engineer"]
    first = data["jobs"][0]
    assert first["keywords"] == "Plan" and first["reasoning"] == "Good fit" and first["track"] == "devops"
    assert first["has_cover_letter"] is True
    assert data["jobs"][1]["has_cover_letter"] is False  # PDF outside the letters folder
    assert data["stats"]["scored"] == 2 and data["stats"]["unscored"] == 1
    assert "full_description" not in first


# -- HTTP --------------------------------------------------------------------

def test_page_and_data_endpoints(server):
    page = server.get("/")
    assert page.status_code == 200 and "ApplyPilot Live" in page.text
    data = server.get("/api/data").json()
    assert len(data["jobs"]) == 2 and data["statuses"][0] == "applied"


def test_description_endpoint(server):
    assert server.get("/api/job/1/description").json() == {"description": "full desc 1"}
    assert server.get("/api/job/999/description").status_code == 404


def test_foreign_host_header_is_rejected(server):
    assert server.get("/api/data", headers={"Host": "evil.example"}).status_code == 403


def test_status_update_needs_custom_header(server, env):
    r = server.post("/api/job/1/status", json={"status": "applied"})
    assert r.status_code == 403
    assert env.execute("SELECT tracking_status FROM jobs WHERE rowid = 1").fetchone()[0] is None


def test_status_update_and_clear(server, env):
    headers = {"X-ApplyPilot": "1"}
    r = server.post("/api/job/1/status", json={"status": "interviewing"}, headers=headers)
    assert r.status_code == 200 and r.json() == {"id": 1, "status": "interviewing"}
    assert env.execute("SELECT tracking_status FROM jobs WHERE rowid = 1").fetchone()[0] == "interviewing"

    assert server.post("/api/job/1/status", json={"status": None}, headers=headers).status_code == 200
    assert env.execute("SELECT tracking_status FROM jobs WHERE rowid = 1").fetchone()[0] is None


@pytest.mark.parametrize(("job", "body", "code"), [
    (1, {"status": "hired"}, 400),
    (1, {"status": 5}, 400),
    (999, {"status": "applied"}, 404),
])
def test_status_update_errors(server, job, body, code):
    assert server.post(f"/api/job/{job}/status", json=body, headers={"X-ApplyPilot": "1"}).status_code == code


def test_resume_pdf_served_only_for_configured_tracks(server):
    ok = server.get("/files/resume/devops")
    assert ok.status_code == 200 and ok.content == b"%PDF-resume"
    assert server.get("/files/resume/software").status_code == 404  # no pdf configured
    assert server.get("/files/resume/..%2F..%2Fsecret").status_code == 404


def test_cover_pdf_only_from_letters_folder(server):
    ok = server.get("/files/cover/1")
    assert ok.status_code == 200 and ok.content == b"%PDF-cover"
    assert server.get("/files/cover/2").status_code == 404  # outside folder: never served
