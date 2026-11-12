"""The orchestrator end to end, the traces it writes and the CLI commands."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from m03_playground import cli
from m03_playground.errors import ErrorKind
from m03_playground.playground import DataBoundaryError, Playground, Settings
from m03_playground.providers import OpenAICompatibleProvider
from m03_playground.resilience import GatewayError
from m03_playground.tracing import TraceWriter, read_traces

PII = "My order #HW-640183 is late. Best regards, Laura Gómez, +34 612 345 678, laura@mail.example"


def _pg(tmp_path: Path, **kw: object) -> Playground:
    return Playground(Settings(provider="fake", **kw), tracer=TraceWriter(tmp_path / "t.jsonl"))


def test_lifecycle_redacts_before_the_model_and_restores_after(tmp_path: Path) -> None:
    pg = _pg(tmp_path)
    r = pg.ask(PII)
    sent = "\n".join(m.content for m in r.prompt)
    for secret in ("Laura Gómez", "+34 612 345 678", "laura@mail.example", "#HW-640183"):
        assert secret not in sent
    assert r.redactions == {"ORDER": 1, "NAME": 1, "PHONE": 1, "EMAIL": 1}
    assert all("Laura" not in m.content for m in pg.conversation.history)


def test_trace_has_numbers_but_no_raw_pii(tmp_path: Path) -> None:
    pg = _pg(tmp_path)
    pg.ask(PII)
    pg.ask("How much is Hearthwise Plus?")
    raw = (tmp_path / "t.jsonl").read_text(encoding="utf-8")
    assert "Laura" not in raw and "612 345" not in raw
    traces = read_traces(tmp_path / "t.jsonl")
    assert [t["turn"] for t in traces] == [1, 2]
    for key in (
        "ts",
        "trace_id",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "ttft_ms",
        "latency_ms",
        "cost_eur",
        "attempts",
        "redactions",
        "output_flags",
        "error",
    ):
        assert key in traces[-1]
    assert pg.totals.requests == 2
    assert pg.totals.cost_eur == pytest.approx(sum(t["cost_eur"] for t in traces))


def test_failed_requests_are_traced_too(tmp_path: Path) -> None:
    pg = _pg(tmp_path, flaky=[ErrorKind.BAD_REQUEST])
    with pytest.raises(GatewayError):
        pg.ask("hello")
    assert read_traces(tmp_path / "t.jsonl")[0]["error"] == "bad_request"


def test_external_provider_without_redaction_is_refused() -> None:
    with pytest.raises(DataBoundaryError):
        Playground(Settings(provider="fake", redact=False), providers=[OpenAICompatibleProvider(model="m")])
    Playground(Settings(provider="fake", redact=False))  # local: allowed


def test_history_survives_a_model_switch(tmp_path: Path) -> None:
    pg = _pg(tmp_path)
    pg.ask("How much is Hearthwise Plus?")
    pg.reconfigure(model="fake-large", temperature=0.7)
    r = pg.ask("And the warranty?")
    assert r.completion.model == "fake-large"
    assert len(r.prompt) == 4


@pytest.mark.parametrize(
    "argv",
    [
        ["ask", "How long is the warranty?", "--stream"],
        ["ask", "How long is the warranty?", "--flaky", "server,rate_limit", "--fallback", "fake-large"],
        ["compare", "--samples", "3", "--show"],
        ["retry", "--requests", "50"],
        ["budget", "--show-summary"],
        ["redact", "--ticket", "T00008"],
        ["guard"],
        ["cost", "--price-in", "1", "--price-out", "2"],
    ],
)
def test_cli_commands_run(argv: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    if argv[0] == "ask":
        argv = [*argv, "--trace-file", str(tmp_path / "t.jsonl")]
    assert cli.main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_cli_ask_reports_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "ask",
            "How long is the warranty?",
            "--max-tokens",
            "1000",
            "--trace-file",
            str(tmp_path / "t.jsonl"),
        ]
    )
    assert code == 1
    assert "context_length" in capsys.readouterr().out


def test_chat_repl_from_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = (
        "How much is Hearthwise Plus?\n/temp 0.7\n/system Answer in one sentence.\n/cost\n/reset\n/quit\n"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(script))
    assert cli.main(["chat", "--trace-file", str(tmp_path / "t.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "bot> Hearthwise Plus costs" in out
    assert "temperature = 0.7" in out and "requests 1" in out and "bye" in out
    assert json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])["session"] == "chat"
