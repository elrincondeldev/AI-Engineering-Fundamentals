"""Every number quoted in the module's Markdown, asserted against a real run.

If you change the lab and one of these tests fails, update the lessons, the lab
README, the quiz and the cheatsheet together with the expected value here.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from m06_support_bot import cli
from m06_support_bot import guardrails as guardrails_module
from m06_support_bot import text as text_module
from m06_support_bot.bot import ABLATIONS, BotConfig, SupportBot
from m06_support_bot.data import datasets_dir, iter_csv, read_jsonl
from m06_support_bot.evaluate import GATES, by_category, evaluate, sweep_thresholds
from m06_support_bot.judge import JudgeCase, build_cases, calibrate, heuristic_judge
from m06_support_bot.kb import load_documents
from m06_support_bot.metrics import KEY_FACTS, faithfulness
from m06_support_bot.prompting import SYSTEM_PROMPT
from m06_support_bot.raft import RaftConfig, build_raft, stats, validate

MODULE = Path(__file__).resolve().parents[2]
RETURN_Q = "What is the return window for Hearthwise products?"
TIPS_Q = "My Glow A19 keeps disconnecting from Wi-Fi. Any tips?"
PII_Q = "Hi, I'm ana@example.com, phone +34 612 345 678. Why is my Lookout C2 blinking orange?"


@pytest.fixture(autouse=True, scope="module")
def _fake_provider() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LLM_PROVIDER", "fake")
        yield


def run_cli(*args: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(list(args))
    return code, buf.getvalue()


@pytest.fixture(scope="module")
def full():
    return evaluate()


@pytest.fixture(scope="module")
def ablations():
    return {name: evaluate(cfg) for name, cfg in ABLATIONS.items()}


@pytest.fixture(scope="module")
def judge_cases():
    return build_cases()


# --------------------------------------------------------------------------- datasets (quoted facts)
def test_dataset_facts_quoted_in_the_lessons() -> None:
    ticket = next(r for r in iter_csv("tickets/tickets.csv") if r["ticket_id"] == "T00607")
    assert "How do I send back a Glow A19 I bought 13 days ago? It's still in the box." in ticket["body"]
    rows = list(iter_csv("tickets/tickets.csv"))
    share_es = sum(r["language"] == "es" for r in rows) / len(rows)
    assert 0.25 <= share_es <= 0.35  # "about 30% of support tickets are in Spanish"
    docs = {d.id: d for d in load_documents()}
    assert "14 days" in docs["kb-010"].text and docs["kb-010"].status == "superseded"
    assert "30 days" in docs["kb-011"].text and docs["kb-011"].status == "current"
    readme = (datasets_dir() / "README.md").read_text(encoding="utf-8")
    assert "| 0.917 |" in readme  # BM25 doc recall@5 baseline quoted in lesson 6.3
    golden = read_jsonl("rag_eval/qa.jsonl")
    assert len(golden) == 40 and sum(r["answerable"] for r in golden) == 36
    assert sum(r["category"] == "unanswerable" for r in golden) == 3
    assert sum(r["category"] == "internal" for r in golden) == 1
    spec = docs["kb-007"].text
    assert spec.count("Device: ") == 6  # "a six-row HTML comparison table"


# --------------------------------------------------------------------------- lesson 6.1
def test_lesson_1_numbers() -> None:
    assert "6. Answer in {language}" in SYSTEM_PROMPT and "1. Answer only from the documents" in SYSTEM_PROMPT
    bot = SupportBot()
    out = bot.ask(RETURN_Q)
    assert out.evidence_score == 0.559 and bot.config.threshold == 0.5
    assert out.citations == ["kb-011"]
    student = bot.ask("Does Hearthwise offer a student discount?")
    assert student.evidence_score == 0.073 and student.reason == "no_answer"
    loose = SupportBot(BotConfig(threshold=0.0)).ask("Does Hearthwise offer a student discount?")
    assert loose.answer == "Hearthwise Plus is an optional subscription. [kb-015]"
    old = SupportBot(ABLATIONS["no_status_filter"]).ask(RETURN_Q)
    assert old.answer == "You can return most Hearthwise products within 14 days of delivery. [kb-010]"


def test_lesson_1_try_it_outputs() -> None:
    code, out = run_cli("ask", RETURN_Q)
    assert code == 0
    assert "Evidence     : 0.559 (threshold 0.50)" in out
    assert ("Answer       : From 2025-03-01 you can return any Hearthwise product within 30 days of "
            "delivery, whether it is opened or not. [kb-011]") in out
    _, out = run_cli("sanitize")
    assert "words seen by a naive parser : 100" in out and "words kept by the sanitiser  : 72" in out
    _, out = run_cli("ask", TIPS_Q, "--config", "no_sanitiser")
    assert "output=credential_request,foreign_contact" in out and "Outcome      : blocked" in out
    _, out = run_cli("ask", "¿Cuántos días tengo para devolver un producto?")
    assert "en un plazo de 30 días desde la entrega, esté abierto o no. [kb-030]" in out
    _, out = run_cli("ask", "How long does delivery to Chile take?")
    assert "Answer       : País: Chile; Plazo: 9 a 12 días hábiles. [kb-031]" in out


def test_status_filter_and_injection_blast_radius(ablations) -> None:
    s, _ = ablations["no_status_filter"]
    assert (s.violations, s.violation_ids) == (2, ("q001", "q002"))
    s, results = ablations["no_sanitiser"]
    assert s.guardrail_blocks == 2
    assert sorted(r.id for r in results if r.guardrail) == ["q026", "q040"]


# --------------------------------------------------------------------------- lesson 6.2
def test_lesson_2_failure_buckets(full) -> None:
    _, results = full
    r = {x.id: x for x in results}
    assert r["q017"].context_hit is False and not r["q017"].correct
    assert "kb-023" in r["q032"].context_doc_ids and not r["q032"].correct
    assert r["q035"].reason == "no_answer"
    assert r["q021"].correct and r["q021"].language_match is False


def test_lesson_2_raft_numbers() -> None:
    examples, train, test = build_raft()
    st = stats(examples)
    assert (len(train), len(test)) == (30, 10)
    assert sum(r["answerable"] for r in train) == 27 and sum(not r["answerable"] for r in train) == 3
    assert st == {"examples": 90, "with_oracle": 67, "abstain_targets": 23, "avg_tokens": 730.8,
                  "total_tokens": 65775}
    answerable = [e for e in examples if e["question_id"] not in ("q036", "q037", "q039")]
    assert len(answerable) == 81 and sum(e["has_oracle"] for e in answerable) == 67
    checks = validate(examples, train, test)
    assert all(c.passed for c in checks) and len(checks) == 8
    first = examples[0]
    assert first["context_chunk_ids"] == ["kb-012#1", "kb-012#3", "kb-011#1", "kb-021#1"]
    assert examples[4]["id"] == "q003-2" and examples[4]["oracle_chunk"] is None
    assert RaftConfig().p_oracle == 0.8 and RaftConfig().copies == 3 and RaftConfig().n_docs == 4


def test_lesson_2_raft_variants() -> None:
    assert stats(build_raft(RaftConfig(no_oracle_target="answer"))[0])["abstain_targets"] == 9
    assert stats(build_raft(RaftConfig(p_oracle=1.0))[0])["abstain_targets"] == 9
    assert stats(build_raft(RaftConfig(n_docs=8))[0])["avg_tokens"] == 1185.6
    cfg = RaftConfig(include_restricted=True)
    ex, tr, te = build_raft(cfg)
    failed = {c.name: c.failures for c in validate(ex, tr, te, cfg) if not c.passed}
    assert list(failed) == ["no_restricted_docs"] and len(failed["no_restricted_docs"]) == 19


def test_lesson_2_try_it_outputs() -> None:
    code, out = run_cli("raft", "--show", "0")
    assert code == 0
    assert "Examples     : 90 | with oracle: 67 | NO_ANSWER targets: 23" in out
    assert "Tokens       : 730.8 per example (approx.) | 65775 in total" in out
    assert 'Evidence: [kb-011] "From 2025-03-01 you can return any Hearthwise product within 30 days' in out
    assert "Answer: 30 days from delivery. [kb-011]" in out
    _, out = run_cli("raft", "--show", "4")
    assert "documents: kb-013#2, kb-003#2, kb-040#5, kb-013#1" in out and "Answer: NO_ANSWER" in out
    code, out = run_cli("raft", "--include-restricted")
    assert code == 1 and "FAIL  no_restricted_docs (19 examples, e.g. q001-1, q001-3, q003-2)" in out


# --------------------------------------------------------------------------- lesson 6.3
def test_lesson_3_summary(full) -> None:
    s, results = full
    assert (s.questions, s.answerable, s.unanswerable) == (40, 36, 4)
    assert (s.correct, s.correct_rate, s.fact_recall, s.token_f1) == (32, 0.889, 0.903, 0.542)
    assert (s.faithfulness, s.context_relevance, s.context_recall, s.mrr) == (1.0, 0.611, 0.972, 0.972)
    assert sum(bool(r.context_hit) for r in results if r.answerable) == 35
    assert (s.abstention_accuracy, s.unanswerable_abstained, s.false_abstentions) == (0.975, 4, 1)
    assert [r.id for r in results if r.answerable and r.reason != "answered"] == ["q035"]
    assert (s.violations, s.leaks, s.guardrail_blocks, s.language_match) == (0, 0, 0, 0.943)
    mismatched = [r.id for r in results if r.language_match is False]
    assert mismatched == ["q021", "q022"]
    r17 = next(r for r in results if r.id == "q017")
    assert len(r17.context_doc_ids) == 4 and set(r17.context_doc_ids) <= {"kb-004", "kb-021"}
    assert KEY_FACTS["q001"] == ["30 days"] and KEY_FACTS["q012"] == ["3680 w"]
    assert KEY_FACTS["q023"] == ["30 dias|30 days"]


def test_lesson_3_categories_and_review_list(full) -> None:
    _, results = full
    keys = ("category", "n", "answered", "correct", "abstention_ok", "violations")
    assert [tuple(row[k] for k in keys) for row in by_category(results)] == [
        ("version_trap", 2, 2, 2, 2, 0),
        ("policy", 10, 10, 10, 10, 0),
        ("multi_doc", 3, 2, 1, 2, 0),
        ("fact", 9, 9, 8, 9, 0),
        ("procedure", 3, 3, 3, 3, 0),
        ("long_doc", 2, 2, 2, 2, 0),
        ("table", 2, 2, 1, 2, 0),
        ("cross_lingual", 2, 2, 2, 2, 0),
        ("spanish", 2, 2, 2, 2, 0),
        ("unanswerable", 3, 0, 0, 3, 0),
        ("internal", 1, 0, 0, 1, 0),
        ("injection_exposed", 1, 1, 1, 1, 0),
    ]
    _, out = run_cli("eval")
    review = out.split("To review:\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert [line.split()[0] for line in review] == ["q017", "q021", "q022", "q029", "q032", "q035"]


def test_lesson_3_faithful_but_wrong(ablations) -> None:
    _, results = ablations["no_status_filter"]
    q001 = next(r for r in results if r.id == "q001")
    assert q001.faithfulness == 1.0 and q001.correct is False
    assert q001.context_doc_ids == ["kb-010", "kb-011", "kb-011", "kb-010"]  # quiz Q3
    assert q001.reciprocal_rank == 0.5 and q001.context_relevance == 0.5 and q001.context_hit


def test_lesson_3_ablation_table(ablations) -> None:
    rows = {n: (s.correct, s.abstention_accuracy, s.violations, s.leaks, s.guardrail_blocks)
            for n, (s, _) in ablations.items()}
    assert rows == {
        "full": (32, 0.975, 0, 0, 0),
        "no_status_filter": (30, 0.975, 2, 0, 0),
        "no_audience_filter": (32, 0.95, 1, 1, 0),
        "no_sanitiser": (30, 0.925, 0, 0, 2),
        "no_sanitiser_no_guardrail": (32, 0.975, 1, 2, 0),
        "no_rewrite": (30, 0.925, 0, 0, 0),
    }
    s, results = ablations["no_sanitiser_no_guardrail"]
    assert s.violation_ids == ("q040",) and s.leak_ids == ("q026", "q040")
    q026 = next(r for r in results if r.id == "q026")
    assert q026.faithfulness == 1.0  # quiz Q5: faithful to a poisoned context
    assert ablations["no_audience_filter"][0].leak_ids == ("q039",)


def test_lesson_3_threshold_sweep() -> None:
    sweep = sweep_thresholds([0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    rows = {s.config: (s.answered, s.correct, s.unanswerable_abstained, s.false_abstentions,
                       s.abstention_accuracy) for s in sweep}
    assert rows == {
        "threshold=0.00": (40, 32, 0, 0, 0.9),
        "threshold=0.20": (39, 32, 1, 0, 0.925),
        "threshold=0.30": (38, 32, 2, 0, 0.95),
        "threshold=0.40": (37, 32, 3, 0, 0.975),
        "threshold=0.50": (35, 32, 4, 1, 0.975),
        "threshold=0.60": (31, 28, 4, 5, 0.875),
        "threshold=0.70": (27, 24, 4, 9, 0.775),
    }


def test_lesson_3_judge_calibration(judge_cases) -> None:
    kinds = {k: sum(c.kind == k for c in judge_cases)
             for k in ("original", "paraphrase", "number_swap", "foreign_claim", "negation")}
    assert len(judge_cases) == 137
    assert kinds == {"original": 35, "paraphrase": 35, "number_swap": 21, "foreign_claim": 35, "negation": 11}
    table = {}
    for tau in (0.5, 0.7, 0.9):
        for num in (False, True):
            cal = calibrate("x", heuristic_judge(tau, num), judge_cases)
            caught = tuple(cal.caught[k][0] for k in ("number_swap", "foreign_claim", "negation"))
            table[(tau, num)] = (cal.agreement, cal.false_alarms, cal.faithful_cases, *caught)
    assert table == {
        (0.5, False): (0.759, 1, 70, 0, 35, 0),
        (0.5, True): (0.912, 1, 70, 21, 35, 0),
        (0.7, False): (0.701, 9, 70, 0, 35, 0),
        (0.7, True): (0.854, 9, 70, 21, 35, 0),
        (0.9, False): (0.635, 34, 70, 15, 35, 1),
        (0.9, True): (0.679, 34, 70, 21, 35, 1),
    }


def test_lesson_3_gate() -> None:
    assert GATES == {"correct_rate": 0.85, "faithfulness": 0.95, "abstention_accuracy": 0.95,
                     "violations": 0, "leaks": 0}
    code, out = run_cli("eval", "--gate")
    assert code == 0 and out.rstrip().endswith("Regression gate: PASS")
    code, out = run_cli("eval", "--config", "no_status_filter", "--gate")
    assert code == 1
    expected = "Regression gate: FAIL correct_rate=0.833 (limit 0.85); violations=2 (limit 0)"
    assert out.rstrip().endswith(expected)


# --------------------------------------------------------------------------- lesson 6.4
def test_lesson_4_routes_and_redaction(full) -> None:
    _, results = full
    routes = {r: sum(x.route == r for x in results) for r in ("faq", "rag", "human", "safety")}
    assert routes == {"faq": 2, "rag": 38, "human": 0, "safety": 0}
    assert sorted(r.id for r in results if r.route == "faq") == ["q027", "q028"]
    assert BotConfig().faq_min_jaccard == 0.6
    out = SupportBot().ask(PII_Q)
    assert out.redacted_question == "Hi, I'm [EMAIL], phone [PHONE]. Why is my Lookout C2 blinking orange?"
    assert out.evidence_score == 0.765
    assert out.answer == "Light: Blinking orange; Meaning: No Wi-Fi connection. [kb-024]"


def test_lesson_4_first_version_bugs(monkeypatch: pytest.MonkeyPatch) -> None:
    # The placeholder bug: placeholders and greetings counted as search words.
    monkeypatch.setattr(text_module, "PLACEHOLDER_RE", re.compile(r"(?!x)x"))
    monkeypatch.setattr(text_module, "STOPWORDS", text_module.STOPWORDS - {"hi", "m"})
    out = SupportBot().ask(PII_Q)
    assert out.evidence_score == 0.413 and out.reason == "no_answer"
    assert "kb-014" in [c.doc_id for c in out.context]  # the privacy page
    monkeypatch.undo()
    # The "spark" bug: a safety keyword that matches a product name.
    monkeypatch.setattr(guardrails_module, "SAFETY_RE", re.compile(
        guardrails_module.SAFETY_RE.pattern.replace("sparks|sparking", "sparks?"), re.IGNORECASE))
    s, results = evaluate()
    assert [r.id for r in results if r.route == "safety"] == ["q012"]
    assert (s.correct, s.false_abstentions) == (31, 2)


def test_lesson_4_rewrite_ablation(ablations) -> None:
    s, _ = ablations["no_rewrite"]
    assert (s.correct, s.abstention_accuracy) == (30, 0.925)


def test_lesson_4_budget() -> None:
    code, out = run_cli("budget")
    assert code == 0
    rows = re.findall(r"^\s+(\d)\s+([\d.]+)\s+([\d.]+)\s+(\d+)/36$", out, re.MULTILINE)
    assert rows == [("1", "301.1", "0.972", "27"), ("2", "392.1", "0.972", "31"),
                    ("4", "582.9", "0.972", "32"), ("8", "1027.0", "1.000", "33")]
    assert round(1027.0 / 582.9 - 1, 2) == 0.76  # "about 76% more prompt tokens"
    assert 0.30 < 1 - 392.1 / 582.9 < 0.36  # "saves a third of the tokens"
    assert "Routes (top_k=4): faq 2 | rag 38 | human 0 | safety 0" in out
    assert "Model calls: 38 of 40 questions" in out
    s8, res8 = evaluate(BotConfig(name="k8", top_k=8))
    assert next(r for r in res8 if r.id == "q017").correct  # the extra correct answer at top_k=8


# --------------------------------------------------------------------------- lab README exercises
def test_lab_readme_exercise_results(monkeypatch: pytest.MonkeyPatch, judge_cases) -> None:
    # Step 3 "Your turn": an internal-code canary turns the voucher leak into a block.
    import m06_support_bot.bot as bot_module

    original = guardrails_module.check_output

    def with_canary(answer: str) -> list[str]:
        problems = original(answer)
        if re.search(r"\b[A-Z0-9]+-INTERNAL\b", answer):
            problems.append("internal_code")
        return problems

    monkeypatch.setattr(bot_module, "check_output", with_canary)
    s, _ = evaluate(ABLATIONS["no_audience_filter"])
    assert (s.leaks, s.guardrail_blocks) == (0, 1)
    monkeypatch.undo()
    # Step 7 "Your turn": a unit swap ('days' -> 'weeks').
    originals = [c for c in judge_cases if c.kind == "original" and re.search(r"\bdays\b", c.answer)]
    swapped = [JudgeCase(c.qid, "unit_swap", re.sub(r"\bdays\b", "weeks", c.answer, count=1), c.context,
                         False) for c in originals]
    assert len(swapped) == 5
    assert sum(not heuristic_judge(0.5, True)(c) for c in swapped) == 0
    assert sum(not heuristic_judge(0.9, True)(c) for c in swapped) == 3


def test_lab_readme_retry_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    original_ask = SupportBot.ask

    def ask(self: SupportBot, question: str):
        out = original_ask(self, question)
        if out.route == "rag" and out.evidence_score is not None and out.evidence_score < 0.6 \
                and self.config.top_k < 8:
            return SupportBot(self.config.but(top_k=8)).ask(question)
        return out

    monkeypatch.setattr(SupportBot, "ask", ask)
    s, results = evaluate()
    assert (s.correct, s.unanswerable_abstained) == (33, 3)
    q037 = next(r for r in results if r.id == "q037")
    assert "€89" in q037.answer and "kb-005" in q037.citations


# --------------------------------------------------------------------------- Markdown consistency
def test_markdown_quotes_match_the_runs() -> None:
    lesson3 = (MODULE / "03-evaluating-rag.md").read_text(encoding="utf-8")
    _, out = run_cli("eval")
    summary = out.split("\n\ncategory", 1)[0]
    for line in summary.splitlines()[2:]:
        assert line in lesson3, line
    lab_readme = (MODULE / "lab" / "README.md").read_text(encoding="utf-8")
    for args in (("ablate",), ("judge",), ("budget",)):
        _, out = run_cli(*args)
        for line in out.strip().splitlines():
            assert line in lab_readme, line
    quiz = (MODULE / "quiz.md").read_text(encoding="utf-8")
    assert len(re.findall(r"^### Q\d+", quiz, re.MULTILINE)) == 14
    assert "(14 questions" in (MODULE / "README.md").read_text(encoding="utf-8")


def test_lab_readme_test_counts() -> None:
    tests_dir = Path(__file__).resolve().parent
    sources = [p.read_text(encoding="utf-8") for p in tests_dir.glob("test_*.py")]
    n_tests = sum(len(re.findall(r"^def test_", src, re.MULTILINE)) for src in sources)
    assert n_tests >= 72  # the README shows "70 passed, 2 skipped"
    live = (tests_dir / "test_live.py").read_text(encoding="utf-8")
    assert len(re.findall(r"^@pytest\.mark\.(?:ollama|openai)$", live, re.MULTILINE)) == 2
