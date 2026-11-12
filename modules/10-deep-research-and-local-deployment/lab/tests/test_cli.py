"""Every CLI command runs offline and prints something sensible."""

from __future__ import annotations

import pytest

from m10_deep_research.cli import main


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["search", "Corvid Thermo X standby power", "-k", "2"], "cred"),
        (["grade", "rq3"], "rq3"),
        (["memory", "--ram", "16"], "fits in 12 GB"),
        (["memory", "--sweep"], "70B"),
        (["quantize"], "int4 group-32"),
        (["serve-sim"], "continuous (batch 16)"),
        (["kv-cache"], "paged"),
        (["speculate"], "speed-up"),
        (["cost", "--peak-tokens-s", "400"], "capacity: 4 server(s)"),
    ],
)
def test_commands(argv: list[str], expected: str, capsys: pytest.CaptureFixture[str]) -> None:
    main(argv)
    assert expected in capsys.readouterr().out


def test_research_writes_a_report(capsys: pytest.CaptureFixture[str]) -> None:
    main(["research", "rq3"])
    out = capsys.readouterr().out
    assert "report written to outputs/rq3.md" in out
