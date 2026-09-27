"""Tests for the discovery_sources toggle in searches.yaml."""

import pytest

from applypilot import pipeline
from applypilot.config import get_discovery_sources

# -- get_discovery_sources ---------------------------------------------------

@pytest.mark.parametrize("cfg", [None, {}, {"queries": []}])
def test_all_sources_enabled_by_default(cfg):
    assert get_discovery_sources(cfg) == {"jobspy": True, "workday": True, "smartextract": True}


def test_listed_sources_override_defaults():
    cfg = {"discovery_sources": {"workday": False, "smartextract": False}}
    assert get_discovery_sources(cfg) == {"jobspy": True, "workday": False, "smartextract": False}


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="Unknown discovery source 'seek'"):
        get_discovery_sources({"discovery_sources": {"seek": True}})


@pytest.mark.parametrize("value", ["false", 0, None])
def test_non_boolean_value_rejected(value):
    with pytest.raises(TypeError, match="must be true or false"):
        get_discovery_sources({"discovery_sources": {"workday": value}})


def test_non_mapping_rejected():
    with pytest.raises(TypeError, match="must be a mapping"):
        get_discovery_sources({"discovery_sources": ["jobspy"]})


# -- _run_discover -----------------------------------------------------------

@pytest.fixture
def fake_runners(monkeypatch, tmp_path):
    """Replace the three discovery runners with call recorders, on a temp database."""
    from applypilot import database

    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db(db_path)

    from applypilot.discovery import jobspy, smartextract, workday

    calls = []
    monkeypatch.setattr(jobspy, "run_discovery", lambda: calls.append("jobspy"))
    monkeypatch.setattr(workday, "run_workday_discovery", lambda workers: calls.append("workday"))
    monkeypatch.setattr(smartextract, "run_smart_extract", lambda workers: calls.append("smartextract"))
    return calls


def test_run_discover_skips_disabled_sources(monkeypatch, fake_runners):
    cfg = {"discovery_sources": {"workday": False, "smartextract": False}}
    monkeypatch.setattr(pipeline, "load_search_config", lambda: cfg)

    stats = pipeline._run_discover()

    assert fake_runners == ["jobspy"]
    assert stats == {"jobspy": "ok", "workday": "skipped", "smartextract": "skipped"}


def test_run_discover_runs_everything_without_setting(monkeypatch, fake_runners):
    monkeypatch.setattr(pipeline, "load_search_config", dict)

    stats = pipeline._run_discover()

    assert fake_runners == ["jobspy", "workday", "smartextract"]
    assert set(stats.values()) == {"ok"}


def test_run_discover_invalid_config_runs_nothing(monkeypatch, fake_runners):
    cfg = {"discovery_sources": {"workday": "no"}}
    monkeypatch.setattr(pipeline, "load_search_config", lambda: cfg)

    stats = pipeline._run_discover()

    assert fake_runners == []
    assert all(v.startswith("error:") for v in stats.values())
