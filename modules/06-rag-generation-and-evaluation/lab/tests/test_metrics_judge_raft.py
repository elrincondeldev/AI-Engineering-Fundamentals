"""Evaluation metrics, judge calibration and the RAFT dataset builder."""

from __future__ import annotations

from m06_support_bot.kb import build_chunks, load_documents
from m06_support_bot.metrics import (
    check_claim,
    evidence_units,
    faithfulness,
    key_fact_recall,
    split_claims,
    token_f1,
    violations,
)
from m06_support_bot.judge import negate, paraphrase, swap_number
from m06_support_bot.raft import RaftConfig, build_raft, split_questions, validate
from m06_support_bot.data import read_jsonl

CHUNKS = {c.chunk_id: c for c in build_chunks(load_documents())}
RETURNS = [CHUNKS["kb-011#1"]]


def test_key_facts_match_whole_phrases_and_alternatives() -> None:
    assert key_fact_recall("Refunds take 5 to 7 business days.", ["5 to 7 business days"]) == 1.0
    assert key_fact_recall("It weighs 1234 g.", ["34"]) == 0.0
    assert key_fact_recall("Tienes 30 días.", ["30 dias|30 days"]) == 1.0
    assert key_fact_recall("2.4 GHz only", ["2.4 ghz", "5 ghz"]) == 0.5


def test_token_f1() -> None:
    assert token_f1("30 days from delivery. [kb-011]", "30 days from delivery.") == 1.0
    assert token_f1("Something else entirely.", "30 days from delivery.") == 0.0


def test_claims_are_split_on_sentences_and_citations() -> None:
    assert split_claims("A first claim. [kb-011] A second one. Third here. [kb-012]") == [
        "A first claim.", "A second one.", "Third here."]


def test_faithfulness_is_claim_level_and_checks_numbers() -> None:
    good = "You can return any Hearthwise product within 30 days of delivery. [kb-011]"
    assert faithfulness(good, RETURNS) == 1.0
    assert faithfulness(good.replace("30", "31"), RETURNS) == 0.0
    assert faithfulness(good.replace("30", "31"), RETURNS, check_numbers=False) == 1.0
    mixed = good + " Express delivery costs €6.95. [kb-011]"
    assert faithfulness(mixed, RETURNS) == 0.5
    check = check_claim("There is no restocking fee.", evidence_units(RETURNS))
    assert check.supported and check.evidence == "There is no restocking fee."


def test_violations_are_case_insensitive() -> None:
    assert violations("Use code gw20-internal", ["GW20-INTERNAL"]) == ["GW20-INTERNAL"]


def test_judge_corruptions() -> None:
    assert swap_number("Within 30 days of 2025-03-01. [kb-011]") == "Within 31 days of 2025-03-01. [kb-011]"
    assert swap_number("No numbers here. [kb-011]") is None
    assert negate("The Glow A19 cannot join WPA3.") == "The Glow A19 can join WPA3."
    assert negate("There is no restocking fee.") == "There is a restocking fee."
    assert negate("Desde el 2025-03-01 puedes devolver cualquier producto.") is None
    assert paraphrase("There is no fee.") == "According to our help centre, there is no fee."


def test_split_is_stratified_and_deterministic() -> None:
    rows = read_jsonl("rag_eval/qa.jsonl")
    train, test = split_questions(rows, 0.25, seed=6)
    assert len(train) + len(test) == 40 and not {r["id"] for r in train} & {r["id"] for r in test}
    assert any(not r["answerable"] for r in test) and any(not r["answerable"] for r in train)
    assert split_questions(rows, 0.25, seed=6) == (train, test)


def test_raft_examples_have_the_expected_shape() -> None:
    examples, train, test = build_raft()
    assert all(c.passed for c in validate(examples, train, test))
    ex = next(e for e in examples if e["has_oracle"])
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    target = ex["messages"][-1]["content"]
    assert target.startswith("Evidence: [kb-") and "\nAnswer: " in target
    assert len(ex["context_doc_ids"]) == 4


def test_raft_validation_catches_restricted_documents() -> None:
    cfg = RaftConfig(include_restricted=True)
    examples, train, test = build_raft(cfg)
    failed = {c.name for c in validate(examples, train, test, cfg) if not c.passed}
    assert failed == {"no_restricted_docs"}


def test_raft_answer_mode_keeps_answers_without_oracle() -> None:
    examples, _, _ = build_raft(RaftConfig(no_oracle_target="answer"))
    unanswerable = ("q036", "q037", "q039")
    no_oracle = [e for e in examples if not e["has_oracle"] and e["question_id"] not in unanswerable]
    assert no_oracle and all(not e["messages"][-1]["content"].endswith("NO_ANSWER") for e in no_oracle)
