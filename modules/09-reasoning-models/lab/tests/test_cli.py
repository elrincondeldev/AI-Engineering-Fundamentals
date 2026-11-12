"""The command line runs offline and lists its commands."""

from __future__ import annotations

import pytest

from m09_reasoning.cli import main


def test_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in ("solve", "scale", "revise", "search", "tot", "star", "rl"):
        assert command in out


def test_single_problem_views(capsys: pytest.CaptureFixture[str]) -> None:
    main(["solve", "--id", "r076", "--mode", "think-medium"])
    main(["search", "--id", "r076"])
    main(["tot", "--id", "r113"])
    out = capsys.readouterr().out
    assert "Re-reading the question" in out
    assert "depth 1:" in out
    assert "found:" in out


def test_unknown_problem_is_a_clear_error() -> None:
    with pytest.raises(SystemExit):
        main(["solve", "--id", "r999"])
