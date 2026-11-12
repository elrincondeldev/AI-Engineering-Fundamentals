"""Build and validate a RAFT-style fine-tuning set (lesson 6.2). No model is trained here.

RAFT (retrieval-augmented fine-tuning) trains a model on examples that look
like real RAG calls: a question, a handful of documents (one *oracle* that
answers it plus *distractors* that do not) and a target answer that quotes the
oracle before answering. In a share ``1 - p_oracle`` of the examples the oracle
is missing, so the model also learns what to do when retrieval fails.

Our support bot must not guess, so by default a missing oracle teaches
``NO_ANSWER`` (``no_oracle_target="abstain"``). The original recipe keeps the
answer instead (``"answer"``), which pushes the model to memorise domain facts.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .bot import SupportBot
from .data import read_jsonl
from .kb import Chunk, build_chunks, load_documents
from .metrics import GLOBAL_DENYLIST, KEY_FACTS, key_fact_recall, token_f1
from .prompting import NO_ANSWER, build_messages, citations
from .providers import count_tokens
from .retriever import Filters
from .text import content_tokens, jaccard, split_units

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"


@dataclass(frozen=True)
class RaftConfig:
    p_oracle: float = 0.8  # share of examples that contain the oracle document
    n_docs: int = 4  # documents per example (oracle + distractors)
    copies: int = 3  # examples per training question, each with a fresh sample of distractors
    test_share: float = 0.25  # golden questions held out for evaluation after fine-tuning
    seed: int = 6
    no_oracle_target: str = "abstain"  # or "answer" (the original RAFT recipe)
    include_restricted: bool = False  # allow internal and superseded documents (a bad idea; see validate)


def split_questions(rows: list[dict], test_share: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded split, stratified by ``answerable`` so both splits contain refusals."""
    rng = random.Random(seed)
    train, test = [], []
    for answerable in (True, False):
        group = [r for r in rows if r["answerable"] == answerable]
        rng.shuffle(group)
        cut = max(1, round(len(group) * test_share))
        test += group[:cut]
        train += group[cut:]
    key = {r["id"]: i for i, r in enumerate(rows)}
    return sorted(train, key=lambda r: key[r["id"]]), sorted(test, key=lambda r: key[r["id"]])


def best_quote(row: dict, chunk: Chunk) -> tuple[str, float]:
    """The unit of ``chunk`` that best supports the reference answer, and how well."""
    facts = KEY_FACTS.get(row["id"], [])
    scored = [
        ((key_fact_recall(u, facts), token_f1(u, row["reference_answer"])), u)
        for u in split_units(chunk.text)
    ]
    if not scored:
        return "", 0.0
    (recall, _), unit = max(scored, key=lambda x: x[0])
    return unit, recall


def oracle_for(row: dict, pool: list[Chunk]) -> tuple[Chunk, str] | None:
    """The chunk (from a relevant document) whose best unit contains the most key facts."""
    best: tuple[float, Chunk, str] | None = None
    for chunk in pool:
        if chunk.doc_id not in row["relevant_doc_ids"]:
            continue
        quote, recall = best_quote(row, chunk)
        if recall > 0 and (best is None or recall > best[0]):
            best = (recall, chunk, quote)
    return (best[1], best[2]) if best else None


def target_text(row: dict, context: list[Chunk], oracle: tuple[Chunk, str] | None, cfg: RaftConfig) -> str:
    """Chain-of-thought target: quote the evidence, dismiss the distractors, then answer."""
    ids = list(dict.fromkeys(c.doc_id for c in context))
    if oracle is not None:
        chunk, quote = oracle
        others = [i for i in ids if i != chunk.doc_id]
        ignored = ", ".join(f"[{i}]" for i in others) if others else "none"
        return (
            f'Evidence: [{chunk.doc_id}] "{quote}"\n'
            f"Reasoning: [{chunk.doc_id}] answers the question. The other documents ({ignored}) "
            f"are about something else, so I ignore them.\n"
            f"Answer: {row['reference_answer']} [{chunk.doc_id}]"
        )
    listed = ", ".join(f"[{i}]" for i in ids)
    if row["answerable"] and cfg.no_oracle_target == "answer":
        return (
            "Evidence: none of the documents answers this.\n"
            f"Reasoning: The documents ({listed}) do not cover it; "
            "I answer from what I learned in training.\n"
            f"Answer: {row['reference_answer']}"
        )
    return (
        "Evidence: none of the documents answers this.\n"
        f"Reasoning: The documents ({listed}) do not cover it, so I must not guess.\n"
        f"Answer: {NO_ANSWER}"
    )


def build_raft(cfg: RaftConfig | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (train examples, train questions, test questions)."""
    cfg = cfg or RaftConfig()
    rng = random.Random(cfg.seed)
    bot = SupportBot()
    filters = Filters(audiences=None, statuses=None) if cfg.include_restricted else Filters()
    pool = [c for c in bot.chunks if filters.allows(c)]
    rows = read_jsonl("rag_eval/qa.jsonl")
    train, test = split_questions(rows, cfg.test_share, cfg.seed)
    examples: list[dict] = []
    for row in train:
        oracle = oracle_for(row, pool) if row["answerable"] else None
        negatives = [c for c in pool if c.doc_id not in row["relevant_doc_ids"]]
        hard = [h.chunk for h in bot.index.search(content_tokens(row["question"]), k=20, filters=filters)
                if h.chunk.doc_id not in row["relevant_doc_ids"]]
        for copy in range(cfg.copies):
            with_oracle = oracle is not None and rng.random() < cfg.p_oracle
            n_distract = cfg.n_docs - (1 if with_oracle else 0)
            n_hard = min(len(hard), n_distract // 2 + 1)
            chosen = rng.sample(hard[:8], min(n_hard, len(hard[:8])))
            rest = [c for c in negatives if c not in chosen]
            chosen += rng.sample(rest, n_distract - len(chosen))
            context = chosen + ([oracle[0]] if with_oracle else [])
            rng.shuffle(context)  # the oracle must not always sit in the same position
            target = target_text(row, context, oracle if with_oracle else None, cfg)
            messages = build_messages(row["question"], context, row["lang"])
            examples.append({
                "id": f"{row['id']}-{copy + 1}",
                "question_id": row["id"],
                "has_oracle": with_oracle,
                "oracle_chunk": oracle[0].chunk_id if with_oracle and oracle else None,
                "context_doc_ids": [c.doc_id for c in context],
                "context_chunk_ids": [c.chunk_id for c in context],
                "messages": [{"role": m.role, "content": m.content} for m in messages]
                + [{"role": "assistant", "content": target}],
            })
    return examples, train, test


@dataclass(frozen=True)
class Check:
    name: str
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


_EVIDENCE_RE = re.compile(r'^Evidence: \[(kb-\d{3})\] "(.*)"$', re.MULTILINE)


def _flat(text: str) -> str:
    return " ".join(text.split())


def validate(
    examples: list[dict], train: list[dict], test: list[dict], cfg: RaftConfig | None = None
) -> list[Check]:
    """Data-quality checks to run before spending money on fine-tuning."""
    cfg = cfg or RaftConfig()
    by_chunk = {c.chunk_id: c for c in build_chunks(load_documents())}
    rows = {r["id"]: r for r in train + test}
    out: dict[str, list[str]] = {k: [] for k in (
        "context_size", "oracle_flag", "quotes_verbatim", "citations_in_context",
        "no_restricted_docs", "no_forbidden_strings", "distractors_irrelevant", "no_test_leakage")}
    for ex in examples:
        row, ctx = rows[ex["question_id"]], [by_chunk[i] for i in ex["context_chunk_ids"]]
        target = ex["messages"][-1]["content"]
        if len(ctx) != cfg.n_docs:
            out["context_size"].append(ex["id"])
        if ex["has_oracle"] != any(c.doc_id in row["relevant_doc_ids"] for c in ctx):
            out["oracle_flag"].append(ex["id"])
        evidence = _EVIDENCE_RE.search(target)
        if evidence and not any(c.doc_id == evidence.group(1) and _flat(evidence.group(2)) in _flat(c.text)
                                for c in ctx):
            out["quotes_verbatim"].append(ex["id"])
        if not set(citations(target)) <= {c.doc_id for c in ctx}:
            out["citations_in_context"].append(ex["id"])
        if any(c.doc.audience != "public" or c.doc.status != "current" for c in ctx):
            out["no_restricted_docs"].append(ex["id"])
        if any(s.lower() in target.lower() for s in [*row["must_not_contain"], *GLOBAL_DENYLIST]):
            out["no_forbidden_strings"].append(ex["id"])
        if any(c.doc_id in row["relevant_doc_ids"] for c in ctx if c.chunk_id != ex["oracle_chunk"]):
            out["distractors_irrelevant"].append(ex["id"])
    test_q = [set(content_tokens(r["question"])) for r in test]
    for r in train:
        if any(jaccard(set(content_tokens(r["question"])), t) >= 0.8 for t in test_q):
            out["no_test_leakage"].append(r["id"])
    return [Check(name, fails) for name, fails in out.items()]


def stats(examples: list[dict]) -> dict:
    """Counts used in the lesson and the lab README."""
    n = len(examples)
    tokens = [sum(count_tokens(m["content"]) for m in ex["messages"]) for ex in examples]
    return {
        "examples": n,
        "with_oracle": sum(ex["has_oracle"] for ex in examples),
        "abstain_targets": sum(ex["messages"][-1]["content"].endswith(NO_ANSWER) for ex in examples),
        "avg_tokens": round(sum(tokens) / n, 1) if n else 0.0,
        "total_tokens": sum(tokens),
    }


def write_jsonl(rows: list[dict], name: str) -> Path:
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / name
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
