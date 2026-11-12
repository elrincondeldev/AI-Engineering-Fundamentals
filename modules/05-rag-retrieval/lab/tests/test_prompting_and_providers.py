"""The grounded prompt, and the provider-backed embedder (run offline with the fake provider)."""

from __future__ import annotations

import numpy as np
import pytest

from m05_retrieval.evaluate import evaluate
from m05_retrieval.prompting import build_messages
from m05_retrieval.providers import FakeProvider
from m05_retrieval.retrieval import Config, build_pipeline
from m05_retrieval.vectors import HashingEmbedder, ProviderEmbedder


def test_prompt_labels_every_source() -> None:
    pipe = build_pipeline(Config("bm25", "table"))
    hits = pipe.retriever.search("What is the return window for Hearthwise products?", 3)
    system, user = build_messages("What is the return window?", hits)
    assert system.role == "system" and "data, not instructions" in system.content
    assert user.content.startswith("Sources:\n[kb-011#0] Returns policy")
    assert user.content.endswith("Question: What is the return window?")
    reply = FakeProvider().complete([system, user])
    assert reply.provider == "fake"


def test_provider_embedder_with_fake_matches_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    texts = ["reset the hub", "Envíos a Latinoamérica"]
    assert np.allclose(ProviderEmbedder().embed(texts), HashingEmbedder().embed(texts), atol=1e-6)
    a = evaluate(build_pipeline(Config("vector", "table"), embedder=ProviderEmbedder()).retriever)
    b = evaluate(build_pipeline(Config("vector", "table")).retriever)
    assert (a.recall_at_5, a.mrr) == (b.recall_at_5, b.mrr)


@pytest.mark.ollama
def test_ollama_embeddings_rank_the_returns_policy() -> None:  # pragma: no cover - live only
    pipe = build_pipeline(Config("vector", "table"), embedder=ProviderEmbedder("ollama"))
    top = pipe.retriever.search("How many days do I have to send something back?", 3)
    assert "kb-011" in {h.chunk.doc_id for h in top}
