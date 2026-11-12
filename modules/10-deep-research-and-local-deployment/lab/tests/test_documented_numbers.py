"""Every number quoted in the module's Markdown (lessons, README, quiz, cheatsheet, lab README).

If you change a default and one of these fails, either revert the change or update the docs and
this file together. Expected-output blocks are compared line by line with a real CLI run.
"""

from __future__ import annotations

import math
import re
from datetime import date

import pytest

from m10_deep_research import credibility, planner, reader, synthesise
from m10_deep_research import loop as loop_mod
from m10_deep_research.cli import OUTPUTS, main
from m10_deep_research.corpus import Page
from m10_deep_research.evaluate import ABLATIONS, load_workspace, run_ablations, run_and_grade, summarise
from m10_deep_research.grade import grade_report
from m10_deep_research.loop import ResearchConfig
from m10_deep_research.memory import (
    bits_per_weight,
    decode_ceiling,
    estimate,
    kv_bytes_per_token,
    kv_cache_bytes,
    sweep,
    weight_bytes,
)
from m10_deep_research.providers import OllamaProvider
from m10_deep_research.quantize import compare
from m10_deep_research.serving import StepModel, make_requests, speculative


@pytest.fixture(scope="module")
def ws():
    return load_workspace()


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> list[str]:
    main(argv)
    return capsys.readouterr().out.rstrip("\n").splitlines()


# --------------------------------------------------------------------------- #
# Lesson 10.1                                                                 #
# --------------------------------------------------------------------------- #


def test_rq1_run_counts(ws) -> None:
    result, grade = run_and_grade(ws, "rq1")
    s = result.stats
    counts = (len(result.subqueries), s.iterations, s.searches, s.pages_opened, s.llm_calls)
    assert counts == (5, 1, 5, 20, 22)
    assert s.llm_calls - 2 == 20  # 1 plan + 20 reads + 1 write
    assert round(20 / 22, 2) == 0.91  # quiz Q2: about 91% of calls are reading
    assert (s.prompt_tokens, s.completion_tokens) == (4082, 1603)
    assert (s.tokens_read, s.tokens_noted, s.tokens_kept) == (1028, 202, 202)
    assert s.stop_reason == "all sub-queries covered"
    assert (grade.facts_covered, grade.facts_total, grade.distinct_domains, grade.min_sources) == (7, 7, 4, 4)
    assert len(result.notes) == 5  # "[7] when there are 5 sources"
    assert "2 to 6 web-search sub-queries" in planner.PLANNER_SYSTEM


def test_trace_block(capsys) -> None:
    assert cli(["research", "rq1", "--trace"], capsys) == [
        "plan  s1 [market] 'smart thermostat sales 2025 growth Mexico Colombia Chile'",
        "plan  s2 [adoption] 'smart home adoption households Mexico Colombia Chile'",
        "plan  s3 [energy prices] 'household electricity price kWh Mexico Colombia Chile'",
        "plan  s4 [logistics] 'parcel delivery time days Mexico Colombia Chile'",
        "plan  s5 [competition] 'competitor smart thermostat available Mexico Colombia Chile'",
        "search s1 'smart thermostat sales 2025 growth Mexico Colombia Chile'",
        "  note  https://marketlens.example/reports/smart-thermostats-2025 (score 0.8582)",
        "  note  https://marketlens.example/reports/smart-home-adoption-2025 (score 0.5364)",
        "  note  https://energyprices.example/2025/household-electricity (score 0.3509)",
        "  note  https://logistics-news.example/latam-parcel-delivery-2025 (score 0.2995)",
        "search s2 'smart home adoption households Mexico Colombia Chile'",
        "search s3 'household electricity price kWh Mexico Colombia Chile'",
        "search s4 'parcel delivery time days Mexico Colombia Chile'",
        "search s5 'competitor smart thermostat available Mexico Colombia Chile'",
        "  note  https://corvid-home.example/thermo-x (score 0.1896)",
        "reflect iteration 1: 5 new notes, 0 sub-queries open",
        "rq1: 5 sub-queries, 1 iteration(s), 5 searches, 20 pages opened, 22 LLM calls",
        "     LLM tokens: 4082 prompt + 1603 completion",
        "     tokens read 1028 -> noted 202 -> kept 202; stop: all sub-queries covered",
        "     notes 5, gaps none; facts 7/7",
        "     report written to outputs/rq1.md",
    ]


def test_grade_block(capsys) -> None:
    assert cli(["grade"], capsys) == [
        "question   facts  supported  valid cites  domains  low-cred  traps  PII",
        "rq1         7/7         7/7         5/5      4/4         0      0    0",
        "rq2         4/4         4/4         5/5      4/3         0      0    0",
        "rq3         2/2         2/2         1/1      1/1         0      0    0",
        "rq4         4/4         4/4         6/6      4/3         0      0    0",
    ]


def test_budget_and_stop_reasons(ws, capsys) -> None:
    assert ResearchConfig() == ResearchConfig(
        max_iterations=3, max_searches=20, top_k=4, notes_per_subquery=2,
        min_credibility=0.4, min_relevance=0.4, half_life_days=365.0,
    )
    assert cli(["grade", "rq1", "--max-searches", "2"], capsys)[1] == (
        "rq1         5/7         5/7         4/4      2/4         0      0    0")
    r4, _ = run_and_grade(ws, "rq4")
    assert (r4.stats.iterations, r4.stats.stop_reason) == (2, "no new evidence")
    assert r4.gaps == ["logistics (Portugal)"]
    assert "search s4 'Portugal parcel delivery time days Portugal'" in r4.trace


def test_search_blocks(capsys) -> None:
    assert cli(["search", "how much do smart thermostats save"], capsys) == [
        "   7.21  cred 0.9  https://eheo.example/reports/smart-thermostat-savings-2025",
        "         2025-02-20 · policy prior 0.9",
        "   5.16  cred 0.0  https://gadgetdeals.example/spain-smart-thermostat-boom",
        "         2025-11-02 · unknown domain 0.5; keyword stuffing; shouting; sales language",
        "   5.16  cred 0.0  https://gadgetdeals.example/spain-smart-thermostat-boom"
        "?utm_source=newsletter&utm_medium=email",
        "         2025-11-02 · unknown domain 0.5; keyword stuffing; shouting; sales language",
        "   4.60  cred 0.5  https://techtips.example/heating/heating-schedules-that-save-energy",
        "         2025-06-21 · unknown domain 0.5",
        "   4.58  cred 0.7  https://smarthomeweekly.example/heating/heating-schedules-that-save-energy-guide",
        "         2025-10-21 · policy prior 0.7",
    ]
    assert cli(["search", "Corvid Thermo X standby power", "-k", "4"], capsys) == [
        "  18.61  cred 0.5  https://newsmirror.example/syndicated/reviews-corvid-thermo-x-review-updated",
        "         2025-06-12 · unknown domain 0.5",
        "  18.61  cred 0.7  https://smarthomeweekly.example/reviews/corvid-thermo-x-review-updated",
        "         2025-06-12 · policy prior 0.7",
        "  17.49  cred 0.5  https://newsmirror.example/syndicated/reviews-corvid-thermo-x-review",
        "         2024-02-08 · unknown domain 0.5",
        "  17.49  cred 0.7  https://smarthomeweekly.example/reviews/corvid-thermo-x-review",
        "         2024-02-08 · policy prior 0.7",
    ]


def test_traps_exist_in_the_snapshot(ws) -> None:
    old = ws.snapshot.get("https://smarthomeweekly.example/reviews/corvid-thermo-x-review")
    new = ws.snapshot.get("https://smarthomeweekly.example/reviews/corvid-thermo-x-review-updated")
    spam = ws.snapshot.get("https://gadgetdeals.example/spain-smart-thermostat-boom")
    assert "0.9 W" in old.text and "0.4 W" in new.text and "2 million" in spam.text
    assert round((0.9 + 0.4) / 2, 2) == 0.65  # the average a report must not invent


def test_edited_report_grade(ws, capsys) -> None:
    result, _ = run_and_grade(ws, "rq1")
    g = grade_report(result.report.replace("260,000", "280,000"), ws.questions["rq1"], ws.snapshot, ws.labels)
    assert (g.facts_covered, g.facts_supported, g.valid_cited_lines, g.cited_lines) == (6, 4, 4, 5)
    cli(["research", "rq1"], capsys)
    edited = OUTPUTS / "rq1-edited.md"
    edited.write_text(result.report.replace("260,000", "280,000"), encoding="utf-8")
    assert cli(["grade-file", "rq1", str(edited)], capsys) == [
        "rq1: facts 6/7, supported 4/7, valid cited lines 4/5, domains 4/4, low-cred 0, traps none, PII 0"]


# --------------------------------------------------------------------------- #
# Lesson 10.2                                                                 #
# --------------------------------------------------------------------------- #


def test_rq1_plan(ws) -> None:
    result, _ = run_and_grade(ws, "rq1")
    assert [(s.id, s.aspect, s.query) for s in result.subqueries] == [
        ("s1", "market", "smart thermostat sales 2025 growth Mexico Colombia Chile"),
        ("s2", "adoption", "smart home adoption households Mexico Colombia Chile"),
        ("s3", "energy prices", "household electricity price kWh Mexico Colombia Chile"),
        ("s4", "logistics", "parcel delivery time days Mexico Colombia Chile"),
        ("s5", "competition", "competitor smart thermostat available Mexico Colombia Chile"),
    ]


def page(text: str, domain: str = "unknown.example", title: str = "Plain title") -> Page:
    url = f"https://{domain}/x"
    return Page(url, url, domain, 200, "text/html", title, "", "2025-01-01", "en", (text,))


def test_credibility_constants_and_penalties(ws) -> None:
    pol = credibility.SOURCE_POLICY
    priors = (pol["marketlens.example"], pol["smarthomeweekly.example"], pol["corvid-home.example"])
    assert priors == (0.9, 0.7, 0.6)
    assert credibility.DEFAULT_PRIOR == 0.5
    calm = "a quiet sentence about homes and heating and lights in the evening " * 2
    assert credibility.score_page(page(calm)).score == 0.5
    assert credibility.score_page(page("cheap smart plug " * 6)).score == 0.2  # stuffing: minus 0.3
    assert credibility.score_page(page(calm + " WOW!!!")).score == 0.3  # shouting: minus 0.2
    assert credibility.score_page(page(calm + " Buy now.")).score == 0.3  # sales: minus 0.2
    url = "https://f.example/t"
    posts = ("user a: hello 1", "user b: hi 2")
    forum = Page(url, url, "f.example", 200, "text/html", "t", "", "", "en", posts)
    assert credibility.score_page(forum).score == 0.3
    spam = credibility.score_page(ws.snapshot.get("https://gadgetdeals.example/spain-smart-thermostat-boom"))
    assert spam.score == 0.0 and len(spam.reasons) == 4  # prior + three signals: 0.5 - 0.7, clamped
    thread = ws.snapshot.get("https://forum.homeautomators.example/t/ember-t3-wifi-drops")
    assert credibility.score_page(thread).score == 0.3


def test_recency_numbers(ws) -> None:
    assert ws.snapshot.as_of == date(2025, 11, 9)
    assert (date(2025, 11, 9) - date(2024, 2, 8)).days == 640
    from m10_deep_research.notes import recency_weight
    assert recency_weight("2024-02-08", ws.snapshot.as_of, 365) == 0.2966
    assert recency_weight("2025-06-12", ws.snapshot.as_of, 365) == 0.7521


def test_rq2_notes(ws) -> None:
    result, _ = run_and_grade(ws, "rq2")
    s = result.stats
    assert (s.mirrors_merged, s.superseded, s.tokens_noted, s.tokens_kept) == (5, 1, 204, 151)


def test_seventeen_key_facts(ws) -> None:
    assert sum(len(q["key_facts"]) for q in ws.questions.values()) == 17
    assert len(ws.questions) == 4


def test_ablation_block(capsys) -> None:
    assert len(ABLATIONS) == 6
    assert cli(["ablate"], capsys) == [
        "configuration                  facts  valid cites  low-cred  traps  PII  enough src  tokens kept",
        "full                           17/17       17/17        0      0    0         4/4          594",
        "no-credibility                 17/17       19/19        4      1    0         4/4          654",
        "no-recency                     17/17       18/18        0      1    0         4/4          647",
        "no-sanitiser                   17/17       17/17        0      0    0         4/4          594",
        "no-dedup                       16/17       29/29        1      0    0         3/4         1190",
        "no-credibility+no-sanitiser    17/17       19/19        4      2    2         4/4          701",
    ]
    assert round(1190 / 594) == 2  # "twice the tokens"


def test_ablation_details(ws) -> None:
    by_q = {q: run_and_grade(ws, q, ABLATIONS["no-credibility"])[1] for q in sorted(ws.questions)}
    assert by_q["rq3"].trap_hits == ["spam figure"]
    both = run_and_grade(ws, "rq2", ABLATIONS["no-credibility+no-sanitiser"])[1]
    assert both.trap_hits == ["injected instruction"] and both.pii_leaks == 2
    r_nocred, _ = run_and_grade(ws, "rq2", ABLATIONS["no-credibility"])
    assert (r_nocred.stats.injections_removed, r_nocred.stats.pii_redacted) == (2, 4)


def test_no_dedup_block(ws, capsys) -> None:
    out = cli(["research", "rq1", "--no-dedup"], capsys)
    assert out == [
        "rq1: 5 sub-queries, 1 iteration(s), 5 searches, 20 pages opened, 22 LLM calls",
        "     LLM tokens: 4763 prompt + 2126 completion",
        "     tokens read 1028 -> noted 926 -> kept 485; stop: all sub-queries covered",
        "     notes 10, gaps none; facts 6/7",
        "     report written to outputs/rq1.md",
    ]
    result, _ = run_and_grade(ws, "rq1", ABLATIONS["no-dedup"])
    findings = [ln for ln in result.report.split("## Sources")[0].splitlines() if ln.startswith("- **")]
    assert len(findings) == 10 and sum("(MarketLens," in ln for ln in findings) == 8
    assert "corvid-home.example" not in result.report


# --------------------------------------------------------------------------- #
# Lesson 10.3                                                                 #
# --------------------------------------------------------------------------- #


def test_local_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    p = OllamaProvider()
    assert (p.host, p.model) == ("http://localhost:11434", "llama3.2:1b")


def test_memory_worked_example() -> None:
    assert bits_per_weight(4, 32) == 4.5 and bits_per_weight(8, 32) == 8.5
    assert kv_bytes_per_token(32, 8, 128) == 128 * 1024 == 131_072
    base = estimate(8e9, 4, 8192, 32, 8, 128, group_size=32)
    assert (base.weights_gb, base.kv_cache_gb, base.overhead_gb, base.total_gb) == (4.5, 1.07, 0.56, 6.13)
    assert round(base.weights_gb + kv_cache_bytes(32, 8, 128, 8192) / 1e9, 2) == 5.57
    assert estimate(8e9, 16, 8192, 32, 8, 128).total_gb == 18.78
    no_gqa = estimate(8e9, 4, 8192, 32, 32, 128, group_size=32)
    long_ctx = estimate(8e9, 4, 32768, 32, 8, 128, group_size=32)
    batch8 = estimate(8e9, 4, 8192, 32, 8, 128, batch=8, group_size=32)
    assert (no_gqa.kv_cache_gb, no_gqa.total_gb) == (4.29, 9.67)
    assert (long_ctx.kv_cache_gb, long_ctx.total_gb) == (4.29, 9.67)
    assert (batch8.kv_cache_gb, batch8.total_gb) == (8.59, 14.4)
    assert round(long_ctx.kv_cache_gb - base.kv_cache_gb, 2) == 3.22
    assert estimate(8e9, 8, 8192, 32, 8, 128, group_size=32).weights_gb - base.weights_gb == 4.0


def test_memory_blocks(capsys) -> None:
    assert cli(["memory", "--ram", "16"], capsys) == [
        "model: 8B parameters at 4-bit (4.5 bits per weight with scales)",
        "weights      4.50 GB",
        "KV cache     1.07 GB  (128 KiB per token x 8192 tokens x batch 1)",
        "overhead     0.56 GB  (10% planning margin)",
        "total        6.13 GB",
        "fits in 12 GB (75% of 16 GB RAM)",
    ]
    assert cli(["memory", "--bandwidth", "100"], capsys)[-1] == (
        "decode ceiling 22.2 tokens/s for one user at 100 GB/s memory bandwidth (upper bound)")
    assert cli(["memory", "--sweep", "--ram", "16"], capsys)[1:] == [
        "  params    16-bit     8-bit     4-bit",
        "      1B      3.4       2.3       1.8 ",
        "      3B      7.8       4.7       3.0 ",
        "      8B     18.8✗     10.5       6.1 ",
        "     14B     32.0✗     17.5✗      9.8 ",
        "     32B     71.6✗     38.6✗     21.0✗",
        "     70B    155.2✗     83.0✗     44.5✗",
        "(GB, total; ✗ = does not fit)",
    ]
    assert {(r.params, r.bits): r.total_gb for r in sweep(16)}[(8e9, 4)] == 6.13


def test_speed_ceilings() -> None:
    w3 = estimate(3e9, 4, 8192, 32, 8, 128, group_size=32).weights_gb
    assert (w3, decode_ceiling(100, w3)) == (1.69, 59.2)
    assert decode_ceiling(100, 4.5) == 22.2
    assert decode_ceiling(100, 8.5) == 11.8
    assert decode_ceiling(100, 16.0) == 6.2
    assert estimate(3e9, 4, 8192, 32, 8, 128, group_size=32).total_gb == 3.04


def test_quantize_block(capsys) -> None:
    assert cli(["quantize"], capsys) == [
        "Seeded 256 x 1024 weight matrix with uneven channels and 0.1% outliers",
        "scheme             bits/weight  weight err  output err   zeroed",
        "int8 per-tensor          8.000      0.1299      0.1291    26.4%",
        "int8 per-channel         8.016      0.0219      0.0221     2.5%",
        "int4 per-tensor          4.000      0.9246      0.9209    99.6%",
        "int4 per-channel         4.016      0.3743      0.3781    40.2%",
        "int4 group-128           4.125      0.1743      0.1749    19.4%",
        "int4 group-32            4.500      0.1154      0.1141    14.1%",
    ]
    r = {x.name: x for x in compare()}
    assert round(r["int8 per-tensor"].output_error / r["int8 per-channel"].output_error) == 6
    assert round(r["int4 per-tensor"].output_error, 2) == 0.92
    assert round(r["int4 group-32"].output_error, 2) == 0.11
    assert round(r["int8 per-tensor"].output_error, 2) == 0.13
    assert round(r["int8 per-channel"].output_error, 3) == 0.022
    assert 2 ** (8 - 1) - 1 == 127 and 2 ** (4 - 1) - 1 == 7


def test_weight_quiz_numbers() -> None:
    assert weight_bytes(14e9, 4, 32) == 7.875e9
    assert kv_cache_bytes(32, 8, 128, context=4096, batch=4) == 2_147_483_648
    assert round(2_147_483_648 / 1e9, 2) == 2.15


# --------------------------------------------------------------------------- #
# Lesson 10.4                                                                 #
# --------------------------------------------------------------------------- #


def test_workload_and_step_model() -> None:
    assert StepModel() == StepModel(decode_base_ms=25.0, decode_per_seq_ms=1.0, prefill_per_token_ms=0.1)
    reqs = make_requests()
    assert len(reqs) == 64
    assert min(r.prompt_tokens for r in reqs) >= 100 and max(r.prompt_tokens for r in reqs) <= 1000
    assert min(r.output_tokens for r in reqs) >= 16 and max(r.output_tokens for r in reqs) <= 512


def test_serve_sim_blocks(capsys) -> None:
    assert cli(["serve-sim"], capsys) == [
        "64 requests, one every 1500 ms on average (simulated time, not a benchmark)",
        "scheduler                  TTFT p50   TTFT p95    TPOT  tokens/s",
        "no batching                242552ms   414066ms  26.0ms      38.3",
        "static (batch 16)           10453ms    17260ms  36.8ms     167.4",
        "continuous (batch 16)          98ms      155ms  34.1ms     195.7",
    ]
    assert cli(["serve-sim", "--batch", "8"], capsys)[-1] == (
        "continuous (batch 8)          622ms     6227ms  33.2ms     195.6")
    assert round(242552 / 60000) == 4 and round(10453 / 1000) == 10  # "about four minutes", "about 10 s"


def test_kv_cache_blocks(capsys) -> None:
    assert cli(["kv-cache"], capsys) == [
        "KV budget 8 GB, 128 KiB per token, max length 8192, page size 16 tokens",
        "strategy       reserved     used   waste  fit at once",
        "contiguous      524,288   54,923   89.5%        7/64",
        "paged            55,440   54,923    0.9%       64/64",
    ]
    assert cli(["kv-cache", "--block", "8"], capsys)[-1] == (
        "paged            55,168   54,923    0.4%       64/64")
    assert cli(["kv-cache", "--max-seq-len", "32768"], capsys)[-2:] == [
        "contiguous    2,097,152   54,923   97.4%        1/64",
        "paged            55,440   54,923    0.9%       64/64",
    ]


def test_speculative_blocks(capsys) -> None:
    assert cli(["speculate"], capsys) == [
        "acceptance  draft k  tokens/pass  speed-up",
        "      0.50        4        1.938     1.61x",
        "      0.70        4        2.773     2.31x",
        "      0.90        4        4.095     3.41x",
    ]
    assert speculative(0.8, 4, 0.05) == (3.362, 2.801)  # quiz Q10: 2.80x


def test_cost_blocks(capsys) -> None:
    assert cli(["cost", "--peak-tokens-s", "400"], capsys) == [
        "All prices are inputs you supply; the defaults are placeholders, not real quotes.",
        "API:       1.80 EUR/day",
        "local:     2.54 EUR/day  (hardware 2.31 + energy 0.23)",
        "local becomes cheaper at 1.4x today's tokens",
        "capacity: 4 server(s) for a peak of 400 tokens/s at 150 tokens/s each with 30% headroom",
    ]
    assert cli(["cost", "--hardware", "0"], capsys)[2] == (
        "local:     0.23 EUR/day  (hardware 0.00 + energy 0.23)")
    assert round(2.54 - 1.80, 2) == 0.74
    assert math.ceil(400 * 1.3 / 150) == 4


# --------------------------------------------------------------------------- #
# Lab README: Break it                                                        #
# --------------------------------------------------------------------------- #


def test_break_min_credibility_zero(ws) -> None:
    config = ResearchConfig(min_credibility=0.0)
    row = summarise("x", [run_and_grade(ws, q, config) for q in sorted(ws.questions)])
    assert (row.low_credibility_citations, row.trap_hits) == (2, 1)
    assert run_and_grade(ws, "rq3", ResearchConfig(min_credibility=0.0))[1].trap_hits == ["spam figure"]


def test_break_wrong_citations(ws, monkeypatch: pytest.MonkeyPatch) -> None:
    original = synthesise.fake_writer

    def shifted(messages):
        n = sum(line.startswith('{"n"') for line in messages[-1].content.splitlines())
        body, sources = original(messages).split("## Sources")
        return re.sub(r"\[(\d+)\]", lambda m: f"[{int(m.group(1)) % n + 1}]", body) + "## Sources" + sources

    monkeypatch.setattr(loop_mod, "fake_writer", shifted)
    supported = {q: run_and_grade(ws, q)[1].facts_supported for q in sorted(ws.questions)}
    assert supported == {"rq1": 0, "rq2": 0, "rq3": 2, "rq4": 0}


def test_break_phone_redaction(ws, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reader, "PHONE_RE", re.compile(r"(?!x)x"))
    leaks = {q: run_and_grade(ws, q, ABLATIONS["no-credibility"])[1].pii_leaks for q in sorted(ws.questions)}
    assert leaks == {"rq1": 0, "rq2": 1, "rq3": 0, "rq4": 0}


def test_break_one_note_per_subquery(ws) -> None:
    pairs = [run_and_grade(ws, q, ResearchConfig(notes_per_subquery=1)) for q in sorted(ws.questions)]
    row = summarise("x", pairs)
    assert (row.facts_covered, row.facts_total, row.tokens_kept) == (16, 17, 516)
    rq4_urls = [n.url for n in pairs[3][0].notes]
    assert "https://corvid-home.example/thermo-x" in rq4_urls
    assert "https://tessera.example/cam-4" not in rq4_urls


def test_full_ablation_rows_match_docs(ws) -> None:
    rows = {r.name: r for r in run_ablations(ws, ["full", "no-dedup"])}
    assert rows["full"].tokens_kept == 594 and rows["no-dedup"].questions_with_enough_sources == 3


def test_thresholds_quoted_in_prose() -> None:
    """Constants the lessons quote in words (8%, 15%, ten times, 100 to 1,000 tokens...)."""
    import inspect

    from m10_deep_research import quantize, serving

    score_src = inspect.getsource(credibility.score_page)
    assert "stuffing_ratio(text) > 0.08" in score_src and "shouting_ratio(text) > 0.15" in score_src
    assert "score -= 0.3" in score_src and score_src.count("score -= 0.2") == 2
    assert "w[mask] *= 10.0" in inspect.getsource(quantize.make_weights)
    assert "outlier_share: float = 0.001" in inspect.getsource(quantize.make_weights)
    gen = inspect.getsource(serving.make_requests)
    assert "rng.randint(100, 1000)" in gen and "rng.randint(16, 512)" in gen
    assert f"{7.875:.2f}" == "7.88"  # quiz Q6
