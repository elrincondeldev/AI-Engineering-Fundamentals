"""Evaluating LLMs: reference-based metrics, a contamination detector, arena ratings and a judge.

Everything here is deterministic. The arena votes are *simulated* from hidden strengths with a
fixed seed, so you can check whether a rating method recovers the truth.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .data import datasets_dir, read_jsonl
from .decoding import BENCH_PROMPTS, DecodeConfig, generate
from .ngram import UNK, NgramLM
from .providers import FakeProvider, LLMProvider, Message
from .text import html_text, load_corpus, markdown_text, ngrams, tokenize

# --------------------------------------------------------------------------- #
# Reference-based metrics                                                     #
# --------------------------------------------------------------------------- #

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """Lower-case, drop punctuation and articles, squeeze spaces (the usual QA normalisation)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalised strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall after normalisation."""
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Sentence BLEU: geometric mean of clipped n-gram precisions (n = 1..4) times a brevity penalty.

    Orders above 1 use add-one smoothing so one missing 4-gram does not zero the score.
    """
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p_grams, r_grams = Counter(ngrams(pred, n)), Counter(ngrams(ref, n))
        clipped = sum((p_grams & r_grams).values())
        total = max(sum(p_grams.values()), 0)
        if n == 1:
            if clipped == 0:
                return 0.0
            log_sum += math.log(clipped / total)
        else:
            log_sum += math.log((clipped + 1) / (total + 1))
    brevity = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / len(pred))
    return brevity * math.exp(log_sum / max_n)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1: based on the longest common subsequence of tokens."""
    pred, ref = tokenize(prediction), tokenize(reference)
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


# Candidate answers to rag_eval question q001 ("What is the return window for Hearthwise
# products?"). ``correct`` is what a careful human grader would say.
METRIC_CANDIDATES: list[tuple[str, str, bool]] = [
    ("copy of the reference", "30 days from delivery.", True),
    ("correct paraphrase", "You have 30 days after delivery to return it.", True),
    ("correct, different words", "Thirty days, counted from the day it arrives.", True),
    ("wrong (old policy)", "14 days from delivery.", False),
    ("correct but verbose", "Thanks for asking! Returns are accepted for 30 days from delivery, "
                            "opened or not, and there is no restocking fee.", True),
]


@dataclass(frozen=True)
class MetricRow:
    name: str
    correct: bool
    em: float
    f1: float
    bleu: float
    rouge_l: float


def metric_table(question_id: str = "q001") -> tuple[str, list[MetricRow]]:
    """Score every candidate in :data:`METRIC_CANDIDATES` against the reference answer."""
    ref = next(q["reference_answer"] for q in read_jsonl("rag_eval/qa.jsonl") if q["id"] == question_id)
    rows = [
        MetricRow(name, ok, exact_match(text, ref), token_f1(text, ref), bleu(text, ref), rouge_l(text, ref))
        for name, text, ok in METRIC_CANDIDATES
    ]
    return ref, rows


@dataclass(frozen=True)
class PerplexityRow:
    text: str
    tokens: int
    unk_share: float
    perplexity: float


def perplexity_report() -> list[PerplexityRow]:
    """Perplexity of a trigram model (trained on 9 of every 10 corpus documents) on several texts."""
    docs = [tokenize(d.text) for d in load_corpus()]
    train = [t for i, t in enumerate(docs) if i % 10 != 9]
    held_out = [t for i, t in enumerate(docs) if i % 10 == 9]
    lm = NgramLM().fit(train)
    spanish = [
        tokenize(markdown_text((datasets_dir() / m["path"]).read_text(encoding="utf-8")))
        for m in read_jsonl("kb/manifest.jsonl")
        if m["lang"] == "es" and m["format"] == "markdown"
    ]

    def outputs(cfg: DecodeConfig) -> list[list[str]]:
        return [generate(lm, p, cfg).tokens for p in BENCH_PROMPTS]

    sets = {
        "training documents": train,
        "held-out documents": held_out,
        "greedy outputs": outputs(DecodeConfig(strategy="greedy", max_new_tokens=50)),
        "top-p 0.9 outputs": outputs(DecodeConfig(top_p=0.9, seed=0, max_new_tokens=50)),
        "T=1.5 outputs": outputs(DecodeConfig(temperature=1.5, seed=0, max_new_tokens=50)),
        "Spanish help-centre pages": spanish,
    }
    rows = []
    for name, texts in sets.items():
        flat = [t for doc in texts for t in doc]
        unk = sum(lm.map(t) == UNK for t in flat) / len(flat)
        rows.append(PerplexityRow(name, len(flat), unk, lm.perplexity(texts)))
    return rows


# --------------------------------------------------------------------------- #
# Benchmark contamination                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalItem:
    source: str  # "rag_eval" or "reasoning"
    id: str
    tokens: tuple[str, ...]  # the question (the model's input)


def load_eval_items() -> list[EvalItem]:
    """Every evaluation question from ``rag_eval`` and ``reasoning`` as tokens."""
    items = [
        EvalItem("rag_eval", q["id"], tuple(tokenize(q["question"])))
        for q in read_jsonl("rag_eval/qa.jsonl")
    ]
    items += [
        EvalItem("reasoning", r["id"], tuple(tokenize(r["question"])))
        for r in read_jsonl("reasoning/problems.jsonl")
    ]
    return items


def load_web_texts() -> dict[str, str]:
    """Visible text of every successfully fetched HTML page in ``web/pages.jsonl``."""
    return {
        p["url"]: html_text(p["html"])
        for p in read_jsonl("web/pages.jsonl")
        if p["status"] == 200 and p["content_type"] == "text/html"
    }


def item_in_page(item: Sequence[str], page_tokens: Sequence[str], n: int, min_share: float) -> bool:
    """True if at least ``min_share`` of the item's n-grams occur in the page.

    Items shorter than ``n`` tokens are compared as one n-gram of their own length.
    """
    size = min(n, len(item))
    grams = set(ngrams(list(item), size))
    if not grams:
        return False
    page = set(ngrams(list(page_tokens), size))
    return len(grams & page) / len(grams) >= min_share


def find_contamination(
    pages: dict[str, str], items: Sequence[EvalItem], n: int = 8, min_share: float = 1.0
) -> dict[str, list[str]]:
    """Map each page URL to the evaluation items whose questions it contains."""
    found: dict[str, list[str]] = {}
    for url, text in pages.items():
        toks = tokenize(text)
        hits = [it.id for it in items if item_in_page(it.tokens, toks, n, min_share)]
        if hits:
            found[url] = hits
    return found


def verbatim_hits(pages: dict[str, str], items: Sequence[EvalItem]) -> set[tuple[str, str]]:
    """(url, item id) for every question that appears word for word in a page: the ground truth."""
    out = set()
    for url, text in pages.items():
        page = " " + " ".join(tokenize(text)) + " "
        out |= {(url, it.id) for it in items if " " + " ".join(it.tokens) + " " in page}
    return out


@dataclass(frozen=True)
class ContaminationRow:
    n: int
    min_share: float
    pages_flagged: int
    pages_correct: int
    items_flagged: int
    items_correct: int


def contamination_sweep(
    settings: Sequence[tuple[int, float]] = ((3, 0.5), (8, 0.5), (8, 1.0), (13, 1.0)),
) -> tuple[list[ContaminationRow], int, int]:
    """Run the detector with several (n, min_share) settings.

    Returns the rows, the number of pages labelled as contaminated in ``web/labels.jsonl`` and the
    number of (page, question) pairs that are verbatim copies.
    """
    pages = load_web_texts()
    items = load_eval_items()
    bad_pages = {row["url"] for row in read_jsonl("web/labels.jsonl") if row["contamination"]}
    truth = verbatim_hits(pages, items)
    rows = []
    for n, share in settings:
        found = find_contamination(pages, items, n, share)
        pairs = {(u, i) for u, hits in found.items() for i in hits}
        pages_ok, items_ok = len(set(found) & bad_pages), len(pairs & truth)
        rows.append(ContaminationRow(n, share, len(found), pages_ok, len(pairs), items_ok))
    return rows, len(bad_pages), len(truth)


# --------------------------------------------------------------------------- #
# Leaderboards from pairwise votes: Elo and Bradley–Terry                     #
# --------------------------------------------------------------------------- #

# Hidden strengths (log-odds scale) of four fictional candidate assistants. The votes below are
# simulated from them, so a good rating method should recover this order: A > B > C > D.
ARENA_STRENGTHS = {"model-A": 1.0, "model-B": 0.5, "model-C": 0.3, "model-D": -0.4}


@dataclass(frozen=True)
class Vote:
    left: str
    right: str
    winner: str  # equal to left or right


def simulate_votes(
    strengths: dict[str, float] | None = None, n_votes: int = 600, seed: int = 0, left_bias: float = 0.0
) -> list[Vote]:
    """Simulated side-by-side votes. P(left wins) = sigmoid(s_left - s_right + left_bias)."""
    strengths = strengths or ARENA_STRENGTHS
    names = sorted(strengths)
    rng = np.random.default_rng(seed)
    votes = []
    for _ in range(n_votes):
        i, j = rng.choice(len(names), size=2, replace=False)
        left, right = names[i], names[j]
        p_left = 1 / (1 + math.exp(-(strengths[left] - strengths[right] + left_bias)))
        votes.append(Vote(left, right, left if rng.random() < p_left else right))
    return votes


def elo_ratings(votes: Sequence[Vote], k: float = 32.0, start: float = 1000.0) -> dict[str, float]:
    """Online Elo: after each vote, move both ratings by k * (result - expected)."""
    ratings: dict[str, float] = {}
    for v in votes:
        ra, rb = ratings.setdefault(v.left, start), ratings.setdefault(v.right, start)
        expected_left = 1 / (1 + 10 ** ((rb - ra) / 400))
        score_left = 1.0 if v.winner == v.left else 0.0
        ratings[v.left] = ra + k * (score_left - expected_left)
        ratings[v.right] = rb - k * (score_left - expected_left)
    return ratings


def bradley_terry_ratings(
    votes: Sequence[Vote], steps: int = 2000, lr: float = 0.5, l2: float = 1e-4, start: float = 1000.0
) -> dict[str, float]:
    """Maximum-likelihood Bradley–Terry strengths, reported on the Elo scale (mean = ``start``).

    Unlike online Elo, the result does not depend on the order of the votes.
    """
    names = sorted({v.left for v in votes} | {v.right for v in votes})
    idx = {m: i for i, m in enumerate(names)}
    x = np.zeros((len(votes), len(names)))
    y = np.zeros(len(votes))
    for row, v in enumerate(votes):
        x[row, idx[v.left]], x[row, idx[v.right]] = 1.0, -1.0
        y[row] = 1.0 if v.winner == v.left else 0.0
    theta = np.zeros(len(names))
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ theta)))
        theta -= lr * (x.T @ (p - y) / len(votes) + l2 * theta)
    theta -= theta.mean()
    scale = 400 / math.log(10)  # natural-log odds -> Elo points
    return {m: start + scale * float(theta[idx[m]]) for m in names}


def bootstrap_bt(votes: Sequence[Vote], n_boot: int = 100, seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% bootstrap interval of each model's Bradley–Terry rating (resampling votes)."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(votes), size=len(votes))
        for name, rating in bradley_terry_ratings([votes[i] for i in pick], steps=500).items():
            samples.setdefault(name, []).append(rating)
    return {m: (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))) for m, s in samples.items()}


def ranking(ratings: dict[str, float]) -> list[str]:
    return sorted(ratings, key=lambda m: -ratings[m])


def true_elo_scale(strengths: dict[str, float] | None = None, start: float = 1000.0) -> dict[str, float]:
    """The hidden strengths on the Elo scale, centred like :func:`bradley_terry_ratings`."""
    strengths = strengths or ARENA_STRENGTHS
    mean = sum(strengths.values()) / len(strengths)
    return {m: start + 400 / math.log(10) * (s - mean) for m, s in strengths.items()}


# --------------------------------------------------------------------------- #
# LLM-as-a-judge: prompt, parsing and bias diagnostics                        #
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You are a strict evaluator of customer-support answers for Hearthwise. Compare two answers to "
    "the same question. Judge correctness first, then helpfulness, then tone. Ignore length and the "
    "order in which the answers are shown. Reply with exactly one letter: A or B."
)


def judge_messages(question: str, answer_a: str, answer_b: str) -> list[Message]:
    user = (
        f"Question: {question}\n\n[Answer A]\n{answer_a}\n\n[Answer B]\n{answer_b}\n\n"
        "Which answer is better?"
    )
    return [Message("system", JUDGE_SYSTEM), Message("user", user)]


def parse_verdict(text: str) -> str | None:
    """Return "A" or "B" from a judge reply, or None if it is unclear."""
    match = re.search(r"\b([AB])\b", text.strip())
    return match.group(1) if match else None


def biased_judge_responder(position_bonus: int = 5):
    """A scripted stand-in for an LLM judge with two built-in biases.

    It prefers the answer with more words (verbosity bias) and gives the answer shown first a
    bonus of ``position_bonus`` words (position bias). Use it to test your diagnostics; swap in a
    real model with ``LLM_PROVIDER=ollama``.
    """

    def respond(messages: Sequence[Message]) -> str:
        body = messages[-1].content
        a = body.split("[Answer A]\n", 1)[1].split("\n\n[Answer B]", 1)[0]
        b = body.split("[Answer B]\n", 1)[1].split("\n\nWhich answer", 1)[0]
        return "A" if len(a.split()) + position_bonus >= len(b.split()) else "B"

    return respond


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def strip_padding(pairs: Sequence[dict[str, str]], min_share: float = 0.1) -> list[dict[str, str]]:
    """Remove sentences found in at least ``min_share`` of all responses (the polite filler).

    What is left is the core of each answer, so chosen and rejected have similar lengths.
    """
    counts = Counter(s for p in pairs for side in ("chosen", "rejected") for s in set(_sentences(p[side])))
    filler = {s for s, c in counts.items() if c >= min_share * 2 * len(pairs)}

    def core(text: str) -> str:
        return " ".join(s for s in _sentences(text) if s not in filler)

    return [{**p, "chosen": core(p["chosen"]), "rejected": core(p["rejected"])} for p in pairs]


@dataclass(frozen=True)
class JudgeReport:
    n_pairs: int
    agreement_ab: float  # chosen shown first
    agreement_ba: float  # chosen shown second
    consistency: float  # same answer picked in both orders
    first_position_rate: float  # share of all verdicts that picked the first answer
    longer_rate: float  # share of all verdicts that picked the longer answer


def judge_swap_test(pairs: Sequence[dict[str, str]], provider: LLMProvider | None = None) -> JudgeReport:
    """Ask the judge about every pair twice, once in each order, and measure its biases."""
    provider = provider or FakeProvider(responder=biased_judge_responder())
    agree_ab = agree_ba = consistent = first = longer = 0
    for p in pairs:
        v1 = parse_verdict(provider.complete(judge_messages(p["prompt"], p["chosen"], p["rejected"])).text)
        v2 = parse_verdict(provider.complete(judge_messages(p["prompt"], p["rejected"], p["chosen"])).text)
        pick1 = p["chosen"] if v1 == "A" else p["rejected"]
        pick2 = p["rejected"] if v2 == "A" else p["chosen"]
        agree_ab += pick1 == p["chosen"]
        agree_ba += pick2 == p["chosen"]
        consistent += pick1 == pick2
        first += (v1 == "A") + (v2 == "A")
        long_answer = p["chosen"] if len(p["chosen"].split()) > len(p["rejected"].split()) else p["rejected"]
        longer += (pick1 == long_answer) + (pick2 == long_answer)
    n = len(pairs)
    return JudgeReport(n, agree_ab / n, agree_ba / n, consistent / n, first / (2 * n), longer / (2 * n))
