"""The knowledge-based index answers exact product-fact questions with provenance."""

from __future__ import annotations

import pytest

from m05_retrieval.chunking import prepare_corpus
from m05_retrieval.knowledge import KnowledgeIndex, extract_triples
from m05_retrieval.parsing import load_kb


@pytest.fixture(scope="module")
def kg() -> KnowledgeIndex:
    docs, _ = prepare_corpus(load_kb())
    return KnowledgeIndex(extract_triples(docs))


def test_triples_come_from_tables(kg: KnowledgeIndex) -> None:
    assert len(kg.triples) == 90
    products = ["Ember T3", "Glow A19", "Hearth Hub", "Lookout C2", "Spark Mini", "Watcher S1"]
    assert sorted(kg.by_subject) == products


def test_lookup_answers_table_questions(kg: KnowledgeIndex) -> None:
    (weight,) = kg.lookup("How much does the Watcher S1 weigh?")
    assert (weight.object, weight.doc_id) == ("34", "kb-007")
    (temp,) = kg.lookup("What is the operating temperature range of the Lookout C2?")
    assert temp.object == "0 to 40"


def test_superlatives(kg: KnowledgeIndex) -> None:
    assert kg.lookup("Which Hearthwise device was launched most recently?")[0].subject == "Lookout C2"
    assert kg.lookup("Which device is the cheapest?")[0].subject == "Glow A19"


def test_exact_lookup_ignores_qualifiers_it_does_not_model(kg: KnowledgeIndex) -> None:
    # The question asks for a price in Mexico, which the help centre never states.
    found = kg.lookup("What is the price of the Lookout C2 in Mexico?")
    assert {t.object for t in found} == {"€89", "89"}
    assert kg.lookup("Does the Ember T3 support dual-fuel heating systems?") == []


def test_sku_triples_exist_for_the_build_it_exercise(kg: KnowledgeIndex) -> None:
    # Step 6 "Build it" in the lab README: learners add a hint so that lookup() returns this triple.
    assert any(t.predicate == "SKU" and t.object == "HW-WCH-S1" for t in kg.by_subject["Watcher S1"])
