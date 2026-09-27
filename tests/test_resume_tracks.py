"""Tests for multiple resume tracks: config parsing and track-aware scoring."""

import pytest

from applypilot import config
from applypilot.config import get_resume_tracks
from applypilot.scoring import scorer

JOB = {"title": "Cloud Engineer", "site": "indeed", "location": "Melbourne VIC", "full_description": "Terraform"}


@pytest.fixture
def resume_files(tmp_path):
    files = {}
    for name in ("software", "devops"):
        (tmp_path / f"{name}.txt").write_text(f"{name} resume", encoding="utf-8")
        (tmp_path / f"{name}.pdf").write_bytes(b"%PDF-1.4")
        files[name] = {"text": str(tmp_path / f"{name}.txt"), "pdf": str(tmp_path / f"{name}.pdf")}
    return files


# -- get_resume_tracks -------------------------------------------------------

def test_no_setting_falls_back_to_single_default_track():
    tracks = get_resume_tracks({})
    assert list(tracks) == ["default"]
    assert tracks["default"]["text"] == config.RESUME_PATH


def test_configured_tracks_are_loaded(resume_files):
    tracks = get_resume_tracks({"resumes": resume_files})
    assert list(tracks) == ["software", "devops"]
    assert tracks["devops"]["text"].name == "devops.txt"
    assert tracks["devops"]["pdf"].name == "devops.pdf"


def test_pdf_is_optional(resume_files):
    tracks = get_resume_tracks({"resumes": {"software": {"text": resume_files["software"]["text"]}}})
    assert tracks["software"]["pdf"] is None


def test_missing_file_rejected(resume_files, tmp_path):
    cfg = {"resumes": {"software": {"text": str(tmp_path / "nope.txt")}}}
    with pytest.raises(FileNotFoundError, match=r"resumes\.software\.text not found"):
        get_resume_tracks(cfg)


@pytest.mark.parametrize("name", ["Software", "dev ops", ""])
def test_invalid_track_name_rejected(resume_files, name):
    with pytest.raises(ValueError, match="Invalid resume track name"):
        get_resume_tracks({"resumes": {name: resume_files["software"]}})


@pytest.mark.parametrize("value", [[], {}, "resume.txt"])
def test_wrong_shape_rejected(value):
    with pytest.raises(TypeError):
        get_resume_tracks({"resumes": value})


def test_missing_text_path_rejected(resume_files):
    with pytest.raises(ValueError, match="text must be a file path"):
        get_resume_tracks({"resumes": {"software": {"pdf": resume_files["software"]["pdf"]}}})


# -- score_job ---------------------------------------------------------------

class FakeClient:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return self.response


def _use_client(monkeypatch, response):
    client = FakeClient(response)
    monkeypatch.setattr(scorer, "get_client", lambda: client)
    return client


RESUMES = {"software": "SOFTWARE RESUME", "devops": "DEVOPS RESUME"}


def test_multi_track_picks_track_from_response(monkeypatch):
    client = _use_client(monkeypatch, "TRACK: devops\nSCORE: 8\nKEYWORDS: Terraform\nREASONING: Good fit.")

    result = scorer.score_job(RESUMES, JOB)

    assert result == {"score": 8, "keywords": "Terraform", "reasoning": "Good fit.", "track": "devops",
                      "company": None, "employment": None}
    system, user = client.messages[0]["content"], client.messages[1]["content"]
    assert "software, devops" in system
    assert "RESUME (TRACK: software)" in user and "RESUME (TRACK: devops)" in user


def test_track_is_normalised(monkeypatch):
    _use_client(monkeypatch, "TRACK: [DevOps]\nSCORE: 7\nKEYWORDS: x\nREASONING: y")
    assert scorer.score_job(RESUMES, JOB)["track"] == "devops"


@pytest.mark.parametrize("response", [
    "SCORE: 8\nKEYWORDS: x\nREASONING: no track line",
    "TRACK: data\nSCORE: 8\nKEYWORDS: x\nREASONING: unknown track",
])
def test_missing_or_unknown_track_is_an_error(monkeypatch, response):
    _use_client(monkeypatch, response)

    result = scorer.score_job(RESUMES, JOB)

    assert result["score"] == 0
    assert result["track"] is None


def test_single_track_uses_original_prompt(monkeypatch):
    client = _use_client(monkeypatch, "SCORE: 6\nKEYWORDS: x\nREASONING: y")

    result = scorer.score_job({"default": "ONLY RESUME"}, JOB)

    assert result["track"] == "default"
    assert client.messages[0]["content"] == scorer.SCORE_PROMPT
    assert client.messages[1]["content"].startswith("RESUME:\nONLY RESUME")


def test_llm_error_returns_zero_score(monkeypatch):
    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("429")
    monkeypatch.setattr(scorer, "get_client", Boom)

    result = scorer.score_job(RESUMES, JOB)

    assert result["score"] == 0
    assert result["track"] is None
