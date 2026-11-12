"""Every number quoted in the module's Markdown comes from these runs.

If a test here fails, the code changed: update the lessons, README, quiz, cheatsheet and
lab README together with the expected values below. Exact values unless a tolerance is stated.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import pytest

from m03_playground import cli
from m03_playground.compare import sweep
from m03_playground.conversation import (
    KEY_FACT,
    SCRIPT,
    SUPPORT_SYSTEM,
    growth,
    message_tokens,
    pin_system,
    run_policy,
    summarise_older,
)
from m03_playground.errors import ErrorKind
from m03_playground.guardrails import check_output, scan_injection, scan_knowledge_base
from m03_playground.metrics import (
    EXAMPLE_PRICES,
    Prices,
    Workload,
    decode_tokens_per_second,
    end_to_end_tokens_per_second,
    estimate_cost,
    request_cost,
)
from m03_playground.models import PROFILES, SimulatedModel, simulate_timing
from m03_playground.playground import Playground, Settings
from m03_playground.providers import Completion, Message, OllamaProvider, OpenAICompatibleProvider
from m03_playground.redact import Pseudonymiser, evaluate_all, evaluate_level, load_tickets
from m03_playground.resilience import (
    FlakyProvider,
    ModelGateway,
    RetryPolicy,
    backoff_delay,
    reliability_table,
)
from m03_playground.tracing import TraceWriter

RETURNS_Q = "How long do I have to return a Glow A19?"
COMPARE_Q = "How many days do I have to return a Glow A19?"
WARRANTY_Q = "How long is the warranty?"


def run_cli(argv: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    if argv[0] == "ask":
        argv = [*argv, "--trace-file", str(tmp_path / "t.jsonl")]
    code = cli.main(argv)
    return code, capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Lesson 3.1: profiles, the returns question, cost                            #
# --------------------------------------------------------------------------- #


def test_simulated_profiles_and_example_prices() -> None:
    small, large = PROFILES["fake-small"], PROFILES["fake-large"]
    assert (small.overhead_ms, small.prefill_ms_per_token, small.decode_ms_per_token) == (120.0, 0.25, 20.0)
    assert (large.overhead_ms, large.prefill_ms_per_token, large.decode_ms_per_token) == (300.0, 0.60, 45.0)
    assert (small.context_window, large.context_window) == (1024, 8192)
    assert EXAMPLE_PRICES["fake-small"] == Prices(0.10, 0.40)
    assert EXAMPLE_PRICES["fake-large"] == Prices(1.00, 4.00)  # "ten times more"


def test_returns_question_lifecycle_and_metrics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run_cli(["ask", RETURNS_Q], tmp_path, capsys)
    assert code == 0
    assert "2 messages, 33 prompt tokens (policy pin_system, budget 768)" in out
    assert "fake/fake-small #1 ok" in out
    assert "(trace_id 04d71610954e)" in out
    assert "tokens   33 prompt + 37 completion" in out
    assert "TTFT 128 ms | total 848 ms | decode 50.0 tokens/s | end-to-end 43.6 tokens/s" in out
    assert "€0.0000181 (example prices: €0.10 in / €0.40 out per million tokens)" in out


def test_returns_question_timing_by_hand() -> None:
    t = simulate_timing(PROFILES["fake-small"], prompt_tokens=33, completion_tokens=37)
    assert (t.ttft_ms, t.total_ms) == (128.25, 848.25)
    assert 0.25 * 33 == 8.25 and 36 * 20 == 720
    assert decode_tokens_per_second(37, t.ttft_ms, t.total_ms) == 50.0
    assert round(end_to_end_tokens_per_second(37, t.total_ms), 1) == 43.6
    assert round(request_cost(33, 37, EXAMPLE_PRICES["fake-small"]), 7) == 1.81e-05
    long = simulate_timing(PROFILES["fake-small"], prompt_tokens=200, completion_tokens=37)
    assert (long.ttft_ms, long.total_ms) == (170.0, 890.0)
    assert round(long.ttft_ms - t.ttft_ms) == 42


def test_stream_chunks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, out = run_cli(["ask", RETURNS_Q, "--stream"], tmp_path, capsys)
    for line in (
        "[  128.2 ms] 'You'",
        "[  148.2 ms] ' can'",
        "[  168.2 ms] ' return'",
        "[  188.2 ms] ' any'",
        "... 32 more chunks ...",
        "[  848.2 ms] '.'",
    ):
        assert line in out


def test_monthly_cost_projection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    w = Workload(60, 25, 21, 700, 180)
    assert w.requests_per_month == 31_500
    small = estimate_cost(w, EXAMPLE_PRICES["fake-small"])
    large = estimate_cost(w, EXAMPLE_PRICES["fake-large"])
    assert round(small.per_request, 6) == 0.000142 and round(small.per_day, 2) == 0.21
    assert round(small.per_month, 2) == 4.47
    assert round(large.per_request, 6) == 0.00142 and round(large.per_day, 2) == 2.13
    assert round(large.per_month, 2) == 44.73
    _, out = run_cli(["cost"], tmp_path, capsys)
    assert "60 users x 25 requests/day x 21 days = 31,500 requests/month" in out
    assert "fake-small        0.10/0.40       0.000142       0.21         4.47" in out
    assert "fake-large        1.00/4.00       0.001420       2.13        44.73" in out


# --------------------------------------------------------------------------- #
# Lesson 3.2: payloads, parameters, streaming, retries                        #
# --------------------------------------------------------------------------- #


def test_adapter_defaults_and_payloads() -> None:
    assert OllamaProvider(model="m").timeout == 120.0  # Sam's "120 seconds"
    assert OllamaProvider(model="m").host.endswith(":11434")
    msgs = [
        Message("system", "You are the Hearthwise support assistant."),
        Message("user", "How many sensors can one Hearth Hub take?"),
    ]
    kw = {"temperature": 0.2, "max_tokens": 256, "seed": 7, "stop": None, "tools": None}
    oa = OpenAICompatibleProvider(model="any-model").build_chat_payload(msgs, **kw)
    assert (oa["temperature"], oa["max_tokens"], oa["seed"]) == (0.2, 256, 7)
    ol = OllamaProvider(model="llama3.2:1b").build_chat_payload(msgs, **kw)
    assert ol["stream"] is False
    assert ol["options"] == {"temperature": 0.2, "num_predict": 256, "seed": 7}


def test_temperature_distinct_answers_without_system_prompt() -> None:
    q = [Message("user", COMPARE_Q)]
    distinct = {
        t: len({SimulatedModel().complete(q, temperature=t, seed=s).text for s in range(10)})
        for t in (0.0, 1.5)
    }
    assert distinct == {0.0: 1, 1.5: 3}


def test_parameter_knobs_in_the_playground(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, out = run_cli(["ask", COMPARE_Q, "--temperature", "1.5", "--seed", "0"], tmp_path, capsys)
    assert "outdated_returns_policy" in out and "within 14 days" in out
    _, out = run_cli(["ask", COMPARE_Q, "--max-tokens", "12"], tmp_path, capsys)
    assert "4 output checks     truncated" in out and "34 prompt + 12 completion" in out


def test_stream_ttft_and_decode() -> None:
    chunks = list(SimulatedModel("fake-small").stream([Message("user", WARRANTY_Q)]))
    ttft = chunks[0].t_ms
    decode = (len(chunks) - 1) / ((chunks[-1].t_ms - ttft) / 1000)
    assert (len(chunks), ttft, round(decode, 1)) == (25, 122.5, 50.0)


def test_backoff_ceilings() -> None:
    policy = RetryPolicy(jitter=False)
    assert [backoff_delay(n, policy, random.Random(0)) for n in range(1, 7)] == [0.5, 1, 2, 4, 8, 8]
    assert (RetryPolicy().max_retries, RetryPolicy().timeout_s, RetryPolicy().deadline_s) == (3, 10.0, 30.0)
    assert (1 + 3) * (1 + 3) == 16  # stacked retries: 4 attempts inside 4 attempts


def test_retry_part_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, out = run_cli(["retry", "--requests", "10"], tmp_path, capsys)
    for line in (
        "2  fake-small    0.16 s  rate_limit",
        "3  fake-small    1.00 s  timeout",
        "4  fake-small    1.30 s  timeout",
        "5  fake-large    0.00 s  ok",
    ):
        assert line in out
    assert "answered by fake-large after 23.85 s (simulated)" in out


def test_scripted_failures_in_ask(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, out = run_cli(["ask", WARRANTY_Q, "--flaky", "server,rate_limit"], tmp_path, capsys)
    assert (
        "fake/fake-small #1 server; fake/fake-small #2 rate_limit after 0.42 s; "
        "fake/fake-small #3 ok after 1.00 s"
    ) in out
    assert "2 messages, 28 prompt tokens" in out
    four = ["ask", WARRANTY_Q, "--flaky", "timeout,timeout,timeout,timeout", "--fallback", "fake-large"]
    code, out = run_cli(four, tmp_path, capsys)
    assert code == 1 and "timeout: deadline exceeded after 3 attempt(s)" in out
    code, out = run_cli([*four, "--timeout", "2"], tmp_path, capsys)
    assert code == 0
    assert (
        "#2 timeout after 0.42 s; fake/fake-small #3 timeout after 0.76 s; "
        "fake/fake-small #4 timeout after 0.84 s; fake/fake-large #1 ok"
    ) in out
    code, out = run_cli([*four, "--timeout", "15"], tmp_path, capsys)
    assert code == 1 and "deadline exceeded after 2 attempt(s)" in out
    code, out = run_cli(["ask", WARRANTY_Q, "--max-tokens", "1000"], tmp_path, capsys)
    assert "context_length: 28 prompt tokens + max_tokens 1000 > context window 1024 of fake-small" in out
    assert "after 1 attempt(s)" in out


def test_short_timeout_costs_about_ten_seconds() -> None:
    gateway = ModelGateway(
        [
            FlakyProvider(SimulatedModel("fake-small"), script=[ErrorKind.TIMEOUT] * 4),
            SimulatedModel("fake-large"),
        ],
        RetryPolicy(max_retries=3, timeout_s=2.0, deadline_s=30.0),
    )
    result = gateway.complete([Message("user", WARRANTY_Q)])
    assert [a.outcome for a in result.attempts] == ["timeout"] * 4 + ["ok"]
    assert result.completion.model == "fake-large"
    failures = 4 * 2.0 + sum(a.waited_s for a in result.attempts)
    assert round(failures) == 10


@pytest.fixture(scope="module")
def reliability() -> dict[str, tuple[float, float, float, float]]:
    return {r.strategy: (r.success_rate, r.mean_attempts, r.p50_s, r.p95_s) for r in reliability_table(1000)}


def test_reliability_table(reliability: dict[str, tuple[float, float, float, float]]) -> None:
    expected = {
        "no retries": ("67.4%", "1.00", "0.60", "10.00"),
        "3 retries": ("98.8%", "1.45", "0.60", "11.22"),
        "3 retries + fallback": ("99.9%", "1.46", "0.60", "11.29"),
        "outage: 3 retries": ("0.0%", "3.97", "12.71", "30.00"),
        "outage: 3 retries + fallback": ("90.5%", "4.92", "14.07", "30.00"),
    }
    for name, (succ, att, p50, p95) in expected.items():
        s, a, m, t = reliability[name]
        assert (f"{s:.1%}", f"{a:.2f}", f"{m:.2f}", f"{t:.2f}") == (succ, att, p50, p95)
    assert f"{1 - 0.3:.0%}" == "70%" and f"{1 - 0.3**4:.1%}" == "99.2%"
    assert f"{1 - 0.2**3:.1%}" == "99.2%"  # quiz Q6


# --------------------------------------------------------------------------- #
# Lesson 3.3: conversation growth and policies                                #
# --------------------------------------------------------------------------- #


def test_chat_growth_and_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = (
        "Hi, my order #HW-640183 has not arrived yet.\nHow much is Hearthwise Plus?\n"
        "How long is the warranty?\n/history\n/reset\nHow long is the warranty?\n/quit\n"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(script))
    cli.main(["chat", "--trace-file", str(tmp_path / "t.jsonl")])
    out = capsys.readouterr().out
    for line in (
        "[34+28 tokens | TTFT 128 ms | €0.0000146 | redacted ORDER x1]",
        "[76+15 tokens | TTFT 139 ms | €0.0000136]",
        "[105+25 tokens | TTFT 146 ms | €0.0000205]",
        "user: Hi, my order <ORDER_1> has not arrived yet.",
        "[28+25 tokens | TTFT 127 ms | €0.0000128]",
    ):
        assert line in out


def test_budget_numbers() -> None:
    assert message_tokens(Message("system", SUPPORT_SYSTEM)) == 77
    assert 1024 - 256 == 768 and 8192 - 1024 == 7168 and 4096 - 512 == 3584 and 3584 - 300 == 3284
    rows = {r.turn: r for r in growth()}
    assert [rows[t].prompt_tokens for t in (1, 2, 3, 5, 10, 13, 15, 20)] == [
        116,
        193,
        266,
        368,
        622,
        776,
        875,
        1079,
    ]
    assert [rows[t].cumulative for t in (1, 5, 10, 13, 20)] == [116, 1256, 3800, 5981, 12657]
    assert [rows[t].stateless for t in (1, 5, 10, 13, 20)] == [116, 92, 97, 88, 99]
    assert min(t for t, r in rows.items() if r.prompt_tokens > 768) == 13
    stateless = sum(r.stateless for r in rows.values())
    assert stateless == 1898 and round(rows[20].cumulative / stateless, 1) == 6.7
    exchanges = [
        message_tokens(Message("user", u)) + message_tokens(Message("assistant", a)) for u, a in SCRIPT
    ]
    assert round(sum(exchanges[:19]) / 19) == 52
    assert sum(range(1, 20)) == 190 and len(SCRIPT) == 20


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (
            768,
            {
                "keep_all": (12657, 1079, 8, 0, True),
                "sliding_window": (3110, 266, 0, 17, False),
                "pin_system": (11046, 766, 0, 0, False),
                "summarise": (11128, 757, 0, 0, True),
            },
        ),
        (
            400,
            {
                "keep_all": (12657, 1079, 15, 0, True),
                "sliding_window": (3110, 266, 0, 17, False),
                "pin_system": (6729, 400, 0, 0, False),
                "summarise": (6833, 393, 0, 0, True),
            },
        ),
    ],
)
def test_policy_tables(budget: int, expected: dict[str, tuple[int, int, int, int, bool]]) -> None:
    for name, values in expected.items():
        st = run_policy(name, budget=budget)
        got = (
            st.total_prompt_tokens,
            st.peak_prompt_tokens,
            st.over_budget_turns,
            st.turns_without_system,
            st.remembers_key_fact,
        )
        assert got == values, name
    assert run_policy("summarise").total_prompt_tokens - run_policy("pin_system").total_prompt_tokens == 82


def test_summary_and_forgetting() -> None:
    system, history = Message("system", SUPPORT_SYSTEM), []
    first_forgotten = None
    for turn, (u, a) in enumerate(SCRIPT, 1):
        sent = pin_system(system, history, Message("user", u), 768)
        if first_forgotten is None and not any(KEY_FACT in m.content for m in sent):
            first_forgotten = turn
        if turn == 20:
            final = summarise_older(system, history, Message("user", u), 768)
        history += [Message("user", u), Message("assistant", a)]
    assert first_forgotten == 13
    assert final[1].content == (
        "Summary of earlier turns: order #HW-640183; products Ember T3, Watcher S1, Hearth Hub. Customer "
        "asked: How long does the battery in the Watcher S1 last? | Can the Hearth Hub run automations if my "
        "internet goes | Should I plug the hub into the router with a"
    )
    window = run_policy("sliding_window")
    assert window.turns_without_system == 17  # from turn 4 to turn 20


# --------------------------------------------------------------------------- #
# Lesson 3.4: redaction, injection, traces, compare                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def redaction() -> dict[str, object]:
    return {r.level: r for r in evaluate_all()}


def test_redaction_table(redaction: dict) -> None:
    assert len(load_tickets()) == 1224
    expected = {  # caught emails, phones, names, false positives, precision, recall, leaking
        "naive": (1224, 82, 0, 200, "0.867", "0.941", 82),
        "regex": (1224, 82, 0, 0, "1.000", "0.941", 82),
        "names": (1224, 82, 75, 0, "1.000", "0.995", 7),
    }
    for level, (em, ph, na, fp, prec, rec, leak) in expected.items():
        r = redaction[level]
        assert (r.gold["EMAIL"], r.gold["PHONE"], r.gold["NAME"]) == (1224, 82, 82)
        assert (r.caught["EMAIL"], r.caught["PHONE"], r.caught["NAME"]) == (em, ph, na)
        assert sum(r.false_positives.values()) == fp
        assert (f"{r.precision:.3f}", f"{r.recall:.3f}", r.leaking_tickets) == (prec, rec, leak)
        assert (r.pseudonymised["ORDER"], r.pseudonymised["SERIAL"]) == (251, 6)
    naive_fp = [t for _, _, t in redaction["naive"].fp_examples]
    assert all(len(t) == 10 and t[4] == "-" and t[7] == "-" for t in naive_fp)  # all dates
    missed = [t for _, _, t in redaction["names"].missed_examples]
    assert len(missed) == 7 and all(t.isupper() for t in missed)
    assert redaction["names"].missed_examples[:3] == [
        ("T00199", "NAME", "INES COSTA"),
        ("T00263", "NAME", "PEDRO ALVES"),
        ("T00392", "NAME", "DIEGO TORRES"),
    ]


def test_redaction_without_stoplist(monkeypatch: pytest.MonkeyPatch) -> None:
    import m03_playground.redact as redact

    monkeypatch.setattr(redact, "NOT_NAMES", set())
    r = evaluate_level(load_tickets(), "names")
    assert sum(r.false_positives.values()) == 519
    assert f"{r.precision:.3f}" == "0.727"


def test_ticket_and_pseudonymiser_examples(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, out = run_cli(["redact", "--ticket", "T00008"], tmp_path, capsys)
    assert "  | From: <EMAIL_1>\n  |\n  | I opened the Spark Mini" in out
    assert "  | Best regards\n  | <NAME_1>\n  | <PHONE_1>" in out
    p = Pseudonymiser(level="names")
    text, _ = p.redact("Order #HW-640183. Best regards, Laura Gómez, +34 612 345 678")
    assert text == "Order <ORDER_1>. Best regards, <NAME_1>, <PHONE_1>"
    assert p.restore("I have updated <ORDER_1> for <NAME_1>.") == "I have updated #HW-640183 for Laura Gómez."


def test_injection_tripwires() -> None:
    kb = scan_knowledge_base()
    flagged = [r for r in kb if r.flags]
    assert len(kb) == 26 and len(flagged) == 1
    assert flagged[0].source == "community/community-tips-wifi.html"
    assert flagged[0].flags == ["ignore_instructions", "addressed_to_ai", "hidden_html"]
    assert sum(bool(scan_injection(t["body"])) for t in load_tickets()) == 0
    assert [scan_injection(p) for p in cli.INJECTION_PROBES] == [
        ["ignore_instructions", "system_prompt_probe"],
        [],
        [],
    ]


def test_trace_record_example(tmp_path: Path) -> None:
    pg = Playground(Settings(provider="fake"), tracer=TraceWriter(tmp_path / "t.jsonl"), session_id="ask")
    pg.ask("Ticket from laura.gomez@mail.example: my Glow A19 is faulty, is it under warranty?")
    rec = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8"))
    assert "laura" not in json.dumps(rec)
    expected = {
        "trace_id": "45bee78b53bf",
        "prompt_hash": "f3af369dd1e0",
        "redactions": {"EMAIL": 1},
        "prompt_tokens": 39,
        "completion_tokens": 25,
        "ttft_ms": 129.75,
        "latency_ms": 609.75,
        "elapsed_s": 0.61,
        "cost_eur": 1.39e-05,
        "external": False,
        "output_flags": [],
    }
    assert {k: rec[k] for k in expected} == expected


def test_output_check_example() -> None:
    answer = Completion("Returns are accepted within 14 days.", "fake-small", "fake", 30, 8, 0.0)
    assert check_output(answer) == ["outdated_returns_policy"]


def test_compare_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, out = run_cli(["compare"], tmp_path, capsys)
    for line in (
        "fake-small   0.0      1/10     0/10    37.0      128       848    0.0182",
        "fake-small   0.7      2/10     0/10    31.8      128       744    0.0161",
        "fake-small   1.5      3/10     2/10    27.8      128       664    0.0145",
        "fake-large   0.0      1/10     0/10    37.0      320      1940    0.1820",
        "fake-large   0.7      2/10     0/10    34.4      320      1823    0.1716",
        "fake-large   1.5      2/10     0/10    31.8      320      1706    0.1612",
    ):
        assert line in out
    rows = sweep(COMPARE_Q, temperatures=(0.0, 1.5), samples=10)
    assert [(r.model, r.temperature, r.distinct, r.flagged) for r in rows] == [
        ("fake-small", 0.0, 1, 0),
        ("fake-small", 1.5, 3, 2),
        ("fake-large", 0.0, 1, 0),
        ("fake-large", 1.5, 2, 0),
    ]


def test_guard_boundary_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, out = run_cli(["guard"], tmp_path, capsys)
    assert "knowledge base: 1 of 26 documents flagged" in out
    assert "tickets: 0 of 1,224 flagged" in out
    assert "openai-compatible, default URL   EXTERNAL" in out
    assert "redaction off + external provider: refused" in out


# --------------------------------------------------------------------------- #
# Lab README, chat session and quiz arithmetic                                #
# --------------------------------------------------------------------------- #


def test_lab_readme_chat_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = (
        "How much is Hearthwise Plus?\n/temp 1.5\nHow many devices can one hub connect?\n"
        "/model fake-large\nMy order #HW-640183 has not arrived. Best regards, Laura Gómez, "
        "+34 612 345 678\n/cost\n/quit\n"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(script))
    cli.main(["chat", "--trace-file", str(tmp_path / "t.jsonl")])
    out = capsys.readouterr().out
    for line in (
        "[28+15 tokens | TTFT 127 ms | €0.0000088]",
        "[59+14 tokens | TTFT 135 ms | €0.0000115]",
        "[100+20 tokens | TTFT 360 ms | €0.0001800 | redacted ORDER x1, NAME x1, PHONE x1]",
        "requests 3 | prompt tokens 187 | completion tokens 49 | cost €0.0002003 (example prices)",
        "mean TTFT 207 ms over 3 request(s)",
    ):
        assert line in out


def test_quiz_arithmetic() -> None:
    t = simulate_timing(PROFILES["fake-small"], prompt_tokens=120, completion_tokens=41)  # Q2
    assert (t.ttft_ms, t.total_ms) == (150.0, 950.0)
    assert decode_tokens_per_second(41, t.ttft_ms, t.total_ms) == 50.0
    assert round(end_to_end_tokens_per_second(41, t.total_ms), 1) == 43.2
    w = Workload(60, 25, 21, 1400, 180)  # Q3
    est = estimate_cost(w, EXAMPLE_PRICES["fake-large"])
    assert round(est.per_request, 5) == 0.00212 and round(est.per_month, 2) == 66.78
    assert round(request_cost(0, 180, EXAMPLE_PRICES["fake-large"]), 5) == 0.00072


def test_test_suite_size(request: pytest.FixtureRequest) -> None:
    """Setup sections quote `105 passed, 2 skipped`. Your own extra tests only make it grow."""
    items = request.session.items
    files = {Path(str(item.path)).name for item in items}
    if len(files) < 10:
        pytest.skip("only meaningful when the whole suite runs")
    live = [item for item in items if "ollama" in item.keywords or "openai" in item.keywords]
    assert len(live) == 2
    assert len(items) - len(live) >= 105
