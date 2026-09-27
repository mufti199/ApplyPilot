"""Shared pytest setup: the opt-in --live flag for tests that call real LLM APIs."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", default=False,
        help="also run tests that call the real LLM providers (uses API quota)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls a real LLM provider; run with --live")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live API test: run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
