"""The command line runs end to end (fake provider, no network)."""

from __future__ import annotations

import pytest

from m08_web_agent.cli import main


def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("search", "open", "ask", "task", "eval", "compare", "ablate", "judge", "replay"):
        assert cmd in out


def test_task_command_grades(capsys: pytest.CaptureFixture[str]) -> None:
    main(["task", "task04", "--naive", "--trace"])
    out = capsys.readouterr().out
    assert "Final Answer: 2 million" in out and "violations: ['2 million']" in out


def test_open_command_reports_sanitiser(capsys: pytest.CaptureFixture[str]) -> None:
    main(["open", "https://forum.homeautomators.example/t/ember-t3-wifi-drops"])
    assert "(sanitiser applied: 1 sentence(s) removed, 2 item(s) masked)" in capsys.readouterr().out
