"""Smoke tests: every documented command runs and prints what the lessons show."""

from __future__ import annotations

import pytest

from m07_agents_tools.cli import main


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    assert main(list(argv)) == 0
    return capsys.readouterr().out


def test_ladder(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "ladder")
    assert "4 agent (can act, approved)   the model           3      2    3314  yes" in out


@pytest.mark.parametrize("pattern", ["chain", "route", "section", "vote", "reflect", "orchestrate"])
def test_workflow_patterns(capsys: pytest.CaptureFixture[str], pattern: str) -> None:
    out = run(capsys, "workflow", pattern)
    assert out.startswith(f"== {pattern} ==")


def test_tools_list(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "tools")
    assert "create_return       yes           order_id, sku, quantity, reason" in out
    assert "All four specs together: 872 tokens" in out


def test_tools_demos(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "tools", "--demo", "all")
    for name in ("compare", "fix-args", "return", "return-denied", "not-yours", "retries", "timeout"):
        assert f"== {name} ==" in out
    assert "[approval] a person rejected this call" in out
    assert "Returns in the store: 2" in out  # idempotency keys switched off


def test_mcp_demo(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "mcp-demo")
    assert "<- isError=true" in out and "JSON-RPC messages exchanged: 13" in out
    out = run(capsys, "mcp-demo", "--poisoned")
    assert "quarantined: ['check_stock']" in out


def test_live_flag_reports_unreachable_server(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from m07_agents_tools import providers

    def unreachable(url, *args, **kwargs):
        raise providers.ProviderError(f"Cannot reach {url}: connection refused")

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(providers, "_post_json", unreachable)
    assert main(["tools", "--demo", "compare", "--live"]) == 2
    assert "error: Cannot reach http://localhost:11434/api/chat" in capsys.readouterr().err
