"""Every number quoted in the module's Markdown (lessons, README, quiz, cheatsheet, lab README).

Table rows are checked verbatim against the CLI output, exactly as they are pasted in the docs.
Numbers derived in the prose (ratios, differences) are recomputed from the experiment results.
If you change the simulator or an algorithm, these tests tell you which documents to update.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from math import comb
from pathlib import Path

import pytest

from m09_reasoning import experiments as ex
from m09_reasoning.cli import main
from m09_reasoning.problems import check_answer, load_problems, numeric_problems, target_problems
from m09_reasoning.scaling import answer_key
from m09_reasoning.training import group_advantages, split_train_test

MODULE = Path(__file__).resolve().parents[2]


def run(capsys: pytest.CaptureFixture[str], *args: str) -> str:
    main(list(args))
    return capsys.readouterr().out


def assert_lines(out: str, expected: list[str]) -> None:
    lines = out.splitlines()
    for line in expected:
        assert line in lines, f"missing line:\n{line}"


def by_id(pid: str):
    return next(p for p in load_problems() if p.id == pid)


# --------------------------------------------------------------------------- #
# Dataset facts quoted in the docs                                            #
# --------------------------------------------------------------------------- #


def test_dataset_counts() -> None:
    problems = load_problems()
    assert len(problems) == 160
    assert len(numeric_problems()) == 110 and len(target_problems()) == 50
    hub = [p for p in problems if p.type == "hub_capacity"]
    assert len(hub) == 35 and {p.difficulty for p in hub} == {5}
    assert {p.difficulty for p in problems if p.type == "order_total" and "discount code" in p.question} == {
        4
    }
    assert min(p.difficulty for p in problems) == 2 and max(p.difficulty for p in problems) == 5
    train, test = split_train_test(problems)
    assert (len(train), len(test)) == (80, 80)
    assert 100 / len(target_problems()) == 2.0  # one puzzle moves the score by 2 points


def test_jordans_hub_question() -> None:
    p = by_id("r076")
    assert (
        "2 floors with 3 rooms each" in p.question
        and "1 motion sensor(s), 1 door sensor(s) and 5 bulbs" in p.question
    )
    assert "up to 64 HomeMesh devices" in p.question
    assert p.answer == "52"
    r113 = by_id("r113")
    assert r113.numbers == (2, 4, 11, 12) and r113.target == 12


# --------------------------------------------------------------------------- #
# Lesson 9.1 — lab solve                                                      #
# --------------------------------------------------------------------------- #

SOLVE = [
    "mode          accuracy  diff 2  diff 3  diff 4  diff 5  tokens/answer  of which thinking",
    "direct           33.1%   44.7%   28.1%   40.0%   22.9%            6.5                0.0",
    "cot              49.4%   57.9%   38.6%   63.3%   45.7%           72.7                0.0",
    "think-low        57.5%   63.2%   49.1%   70.0%   54.3%          138.3               80.5",
    "think-medium     68.1%   65.8%   54.4%   86.7%   77.1%          161.2              103.5",
    "think-high       71.9%   71.1%   56.1%   86.7%   85.7%          194.1              136.4",
]


def test_solve_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert_lines(run(capsys, "solve"), SOLVE)


def test_solve_derived_numbers() -> None:
    rows = {r.label: r for r in ex.run_solve()}
    cot, med, high, direct = rows["cot"], rows["think-medium"], rows["think-high"], rows["direct"]
    assert 3 < 100 * (high.accuracy - med.accuracy) < 4  # "under 4 points"
    assert round(high.tokens - med.tokens) == 33  # "about 33 more tokens"
    assert round(high.tokens / cot.tokens, 1) == 2.7
    assert round(high.tokens / direct.tokens) == 30
    assert round(100 * (high.by_difficulty[5] - cot.by_difficulty[5]), 1) == 40.0
    assert round(100 * (high.by_difficulty[2] - cot.by_difficulty[2]), 1) == 13.2
    assert round(100 * (1 - cot.accuracy), 1) == 50.6 and round(100 * (1 - high.accuracy), 1) == 28.1
    assert round((1 - cot.accuracy) / (1 - high.accuracy), 1) == 1.8
    # quiz Q2: extra tokens per extra point
    assert round((194.1 - 72.7) / (71.9 - 49.4), 1) == 5.4


def test_solve_r076_walkthrough(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "solve", "--id", "r076")
    assert_lines(
        out,
        [
            "=== direct: 3 tokens (0 thinking), wrong",
            "Answer: 12",
            "=== cot: 85 tokens (0 thinking), wrong",
            "Step 1: HomeMesh devices per room: 1 motion + 1 door + 5 bulbs = 7",
            "Step 3: HomeMesh devices: 6 x 7 = 52",
            "=== think-low: 200 tokens (124 thinking), wrong",
            "Wait, step 3 does not add up. Redoing it.",
            "Answer: 22",
            "=== think-medium: 329 tokens (254 thinking), correct",
            "Re-reading the question: step 1 misreads it. Redoing from step 1.",
            "Answer: 52",
        ],
    )
    cot_block = out.split("=== cot")[1].split("===")[0]
    assert "Answer: 12" in cot_block


# --------------------------------------------------------------------------- #
# Lesson 9.2 — lab scale and lab revise                                       #
# --------------------------------------------------------------------------- #

SCALE = [
    "1           61.8%            61.8%            61.8%            77.3",
    "3           69.1%            82.7%            87.3%           230.9",
    "5           76.4%            86.4%            92.7%           386.2",
    "9           85.5%            86.4%            97.3%           695.9",
    "15          89.1%            91.8%           100.0%         1,165.0",
    "45          89.1%            93.6%           100.0%         3,515.6",
    "At N=45: the most frequent answer is wrong on 12 of 110 problems; 12 of them are trap answers. "
    "Problems never solved: 0.",
]
REVISE = [
    "self-critique           61.8%    65.5%    69.1%    66.4%           155.9           4.92             9",
    "arithmetic re-check     61.8%    81.8%    85.5%    85.5%           101.6           1.38             0",
    "process reward model    61.8%    80.9%    90.0%    86.4%           152.8           2.27             6",
]


def test_scale_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert_lines(run(capsys, "scale"), SCALE)


def test_revise_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert_lines(run(capsys, "revise"), REVISE)


def test_scale_derived_numbers() -> None:
    rows = {r.n: r for r in ex.run_scale().rows}
    assert round(rows[45].tokens / rows[15].tokens) == 3
    revise = {r.feedback: r for r in ex.run_revise()}
    recheck = revise["arithmetic re-check"]
    assert round(recheck.accuracy_by_round[-1], 3) == round(rows[9].majority, 3) == 0.855
    assert 6.5 < rows[9].tokens / recheck.tokens < 7  # "almost seven times fewer"
    samples, _ = ex.numeric_samples()
    counts = Counter(answer_key(a.answer) for a in samples["r076"])
    assert len(samples["r076"]) == 45 and counts["22"] == 24 and counts["52"] == 3


# --------------------------------------------------------------------------- #
# Lesson 9.3 — lab search and lab tot                                         #
# --------------------------------------------------------------------------- #

SEARCH = [
    "CoT, 1 sample                     61.8%           77.3       110.4    1.0",
    "majority vote, N=9                85.5%          695.9       993.9    9.0",
    "re-check filter + vote, N=9       86.4%          695.9       993.9    9.0",
    "best-of-9, ORM                    86.4%          695.9       993.9    9.0",
    "best-of-9, PRM                    95.5%          695.9       993.9    9.0",
    "PRM beam search, 3x3              93.6%          274.3     3,055.8   20.5",
    "pass@9 (needs the answer key)     97.3%          695.9       993.9    9.0",
    "Verifier errors over all 4,950 samples (1,854 wrong, 801 of them traps):",
    "arithmetic re-check                      572               238",
    "ORM (score >= 0.5)                       571               518",
    "PRM (min step >= 0.5)                     64               906",
    "1        61.8%              61.8%                     77.3                     77.3",
    "3        82.7%              73.6%                     74.1                     79.5",
    "5        86.4%              74.5%                     72.6                     81.4",
    "9        86.4%              77.3%                     73.8                     82.6",
    "15       91.8%              79.1%                     76.4                     86.1",
    "45       93.6%              80.9%                     77.5                     88.4",
]
TOT = [
    "CoT, 1 sample                   11/50           62.1        78.4    1.0                   39",
    "sample until verified, max 5    30/50          218.6       263.7    3.4                  138",
    "sample until verified, max 15   41/50          406.6       485.7    6.2                  268",
    "ToT BFS, breadth 1              27/50           84.6       922.7   11.1                    0",
    "ToT BFS, breadth 3              35/50          152.9     1,657.5   19.9                    0",
    "ToT BFS, breadth 5              36/50          218.6     2,381.9   28.6                    0",
    "ToT DFS with pruning            34/50           87.9     1,032.0   12.4                    0",
]


def test_search_tables(capsys: pytest.CaptureFixture[str]) -> None:
    assert_lines(run(capsys, "search"), SEARCH)


def test_tot_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert_lines(run(capsys, "tot"), TOT)


def test_search_derived_numbers() -> None:
    result = ex.run_search()
    methods = {m.method: m for m in result.methods}
    assert 100 * (methods["pass@9 (needs the answer key)"].accuracy - methods["best-of-9, PRM"].accuracy) < 2
    assert len(result.errors.__dict__) and result.errors.samples == 110 * 45
    hack = {h.n: h for h in result.hack}
    assert round(100 * (hack[45].honest - hack[45].hacked), 1) == 12.7
    tot = ex.run_tot()
    tot_calls = [r.calls for r in tot if r.method.startswith("ToT")]
    sample_calls = [r.calls for r in tot if r.method.startswith("sample")]
    assert round(min(tot_calls)) == 11 and round(max(tot_calls)) == 29 and max(sample_calls) < 7


def test_traces(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "tot", "--id", "r113")
    assert_lines(
        out,
        [
            "try [2 4 1] (rated likely)",
            "  prune [2 3] (rated impossible)",
            "try [2 12 11/4] (rated likely)",
            "  try [11/4 24] (rated sure)",
            "    no last move reaches the target: backtrack",
            "try [2 11 3] (rated likely)",
            "  try [11 1] (rated sure)",
            "found: 11 + ((12 / 4) - 2)  ·  26 calls, 201 generated tokens",
        ],
    )
    assert "[2 12 15] impossible" in out
    dfs = out.split("=== DFS")[1]
    assert sum(line.startswith("try") for line in dfs.splitlines()) == 3  # found on the third branch
    assert dfs.split("try [2 12 11/4]")[0].count("prune") == 5  # all 5 children of [2 4 1] pruned
    out = run(capsys, "search", "--id", "r076")
    assert_lines(
        out,
        [
            "depth 1: 2 distinct candidates, keep 2",
            "  keep 0.95  HomeMesh devices per room: 1 motion + 1 door = 2",
            "  keep 0.46  HomeMesh devices per room: 1 motion + 1 door + 5 bulbs = 7",
            "  keep 0.85  Free slots: 64 - 12 = 52",
            "  keep 0.23  Free slots: 64 - 11 = 43",
            "  keep 0.01  Free slots: 64 - 42 = 22",
            "Answer 52 (correct), 39 calls, 468 generated tokens, 6391 prompt tokens",
        ],
    )


# --------------------------------------------------------------------------- #
# Lesson 9.4 — lab star and lab rl                                            #
# --------------------------------------------------------------------------- #

STAR = [
    "0              51.2%   82.4%         38.9%        75.0%   20.0%       "
    "      40           27                5         8,625",
    "1              68.8%  100.0%         55.6%        80.0%   48.0%       "
    "      51           21                1         7,508",
    "2              71.2%  100.0%         61.1%        80.0%   52.0%       "
    "      58           19                3         6,770",
    "3              73.8%  100.0%         66.7%        85.0%   52.0%       "
    "      56           21                2         7,178",
    "4              72.5%  100.0%         61.1%        85.0%   52.0%       "
    "       -            -                -             -",
]
STAR_NO_RATIONALIZE = [
    "4              62.5%  100.0%         55.6%        80.0%   28.0%       "
    "       -            -                -             -",
]
STAR_STRICT = [
    "0              51.2%   82.4%         38.9%        75.0%   20.0%       "
    "      39           23                0         8,726",
    "4              73.8%  100.0%         61.1%        85.0%   56.0%       "
    "       -            -                -             -",
]
RL = [
    "0            0.514          53.8%           77.9           0.10        0.10                     10",
    "1            0.530          55.0%           79.8           0.11        0.19                     17",
    "2            0.503          55.0%           81.2           0.09        0.38                     17",
    "3            0.534          60.0%           90.4           0.38        0.70                     18",
    "4            0.606          66.2%          108.8           0.95        0.77                     18",
    "5            0.670          67.5%          112.5           0.98        0.93                     25",
    "6            0.667          67.5%          112.6           0.99        0.93                     18",
    "7            0.658          68.8%          113.0           0.99        0.94                     20",
    "8                -          68.8%          113.0           0.99        0.94                      -",
]


def test_star_tables(capsys: pytest.CaptureFixture[str]) -> None:
    assert_lines(run(capsys, "star"), STAR)
    assert_lines(run(capsys, "star", "--no-rationalize"), STAR_NO_RATIONALIZE)
    assert_lines(run(capsys, "star", "--strict"), STAR_STRICT)
    assert all(r.unfaithful in (0, None) for r in ex.run_star(strict=True))


def test_rl_table(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "rl")
    assert "groups of 8 samples per training problem" in out
    assert_lines(out, RL)


# --------------------------------------------------------------------------- #
# Quiz arithmetic and module facts                                            #
# --------------------------------------------------------------------------- #


def test_quiz_arithmetic() -> None:
    assert 1 - comb(4, 2) / comb(5, 2) == pytest.approx(0.4)
    adv = group_advantages([1, 0, 0, 0])
    assert math.sqrt((0.75**2 + 3 * 0.25**2) / 4) == pytest.approx(0.433, abs=5e-4)
    assert round(adv[0], 2) == 1.73 and round(adv[1], 2) == -0.58
    assert round(194.1 - 72.7, 1) == 121.4 and round(71.9 - 49.4, 1) == 22.5


def test_quiz_has_thirteen_questions_three_reviews() -> None:
    text = (MODULE / "quiz.md").read_text(encoding="utf-8")
    questions = re.findall(r"^### Q(\d+)(.*)$", text, re.M)
    assert len(questions) == 13
    assert sum("🔁" in rest for _, rest in questions) == 3


def test_target_reference_answers_verify() -> None:
    assert sum(check_answer(p, p.answer) for p in target_problems()) == 50


def test_documented_test_count() -> None:
    """lab/README.md and lesson 9.1 say: 54 passed, 3 skipped (the exercises)."""
    tests_dir = Path(__file__).resolve().parent
    counts = {
        f.name: len(re.findall(r"^def test_", f.read_text(encoding="utf-8"), re.M))
        for f in tests_dir.glob("test_*.py")
    }
    assert counts["test_exercises.py"] == 3
    assert sum(counts.values()) - counts["test_exercises.py"] == 54


def test_break_it_groups_of_one() -> None:
    """lab/README.md, Break it 5: with one sample per group, all 80 groups give no signal."""
    rows = ex.run_rl(steps=1, lr=40.0, group=1)
    assert rows[0].zero_signal_groups == 80
    assert rows[1].check_prob == rows[0].check_prob
