"""The command line runs and prints what the lessons show."""

from __future__ import annotations

import pytest

from m04_adaptation.cli import COMMANDS, build_parser, main


def test_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for name in COMMANDS:
        assert name in out


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["show-prompt"], "-> predicted: product_question · true: connectivity_issue"),
        (["context"], "examples kept 4"),
        (["lora-params"], "65,536"),
        (["cache"], "hit rate 0.998"),
    ],
)
def test_commands_print_expected_lines(argv: list[str], expected: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 0
    assert expected in capsys.readouterr().out
