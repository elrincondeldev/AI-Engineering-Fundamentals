"""Test configuration shared by every lab.

Network access is blocked by default so that no test can silently depend on
the internet, an API key or a running model server. Tests that talk to a real
model are marked ``ollama`` or ``openai`` and only run with RUN_LIVE_TESTS=1.
"""

from __future__ import annotations

import os
import socket

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "ollama: needs a running Ollama server (RUN_LIVE_TESTS=1)")
    config.addinivalue_line("markers", "openai: needs an OpenAI-compatible endpoint (RUN_LIVE_TESTS=1)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live model test: set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if "ollama" in item.keywords or "openai" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if "ollama" in request.keywords or "openai" in request.keywords:
        return

    def guard(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Network access is disabled in tests (mark the test 'ollama' or 'openai').")

    monkeypatch.setattr(socket.socket, "connect", guard)
