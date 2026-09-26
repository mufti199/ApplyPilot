"""Tests for the auto_apply switch that blocks `applypilot apply` from submitting."""

import pytest
from typer.testing import CliRunner

from applypilot import cli, config
from applypilot.config import get_auto_apply

runner = CliRunner()


@pytest.mark.parametrize(("cfg", "expected"), [({}, True), ({"auto_apply": False}, False)])
def test_get_auto_apply(cfg, expected):
    assert get_auto_apply(cfg) is expected


def test_get_auto_apply_rejects_non_bool():
    with pytest.raises(TypeError, match="auto_apply"):
        get_auto_apply({"auto_apply": "off"})


@pytest.fixture
def no_bootstrap(monkeypatch):
    monkeypatch.setattr(cli, "_bootstrap", lambda: None)


def _fail_if_called(*_a, **_k):
    raise AssertionError("auto-apply should not have started")


def test_apply_refuses_when_switched_off(no_bootstrap, monkeypatch):
    monkeypatch.setattr(config, "load_search_config", lambda: {"auto_apply": False})
    monkeypatch.setattr(config, "check_tier", _fail_if_called)

    result = runner.invoke(cli.app, ["apply"])

    assert result.exit_code == 1
    assert "Auto-apply is turned off" in result.output


def test_apply_reports_bad_setting(no_bootstrap, monkeypatch):
    monkeypatch.setattr(config, "load_search_config", lambda: {"auto_apply": "no"})
    result = runner.invoke(cli.app, ["apply"])
    assert result.exit_code == 1 and "Config error" in result.output


def test_mark_applied_still_works_when_off(no_bootstrap, monkeypatch):
    from applypilot.apply import launcher

    monkeypatch.setattr(config, "load_search_config", lambda: {"auto_apply": False})
    marked = []
    monkeypatch.setattr(launcher, "mark_job", lambda url, status, reason=None: marked.append((url, status)))

    result = runner.invoke(cli.app, ["apply", "--mark-applied", "https://x/1"])

    assert result.exit_code == 0
    assert marked == [("https://x/1", "applied")]


def test_apply_proceeds_when_on(no_bootstrap, monkeypatch):
    monkeypatch.setattr(config, "load_search_config", dict)
    reached = []

    def stop_at_tier_check(*_a, **_k):
        reached.append(True)
        raise SystemExit(3)

    monkeypatch.setattr(config, "check_tier", stop_at_tier_check)
    runner.invoke(cli.app, ["apply"])
    assert reached == [True]  # got past the switch to the normal checks
