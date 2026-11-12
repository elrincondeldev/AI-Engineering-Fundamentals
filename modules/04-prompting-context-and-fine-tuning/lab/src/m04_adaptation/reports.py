"""The experiments behind every number quoted in module 04.

The CLI prints these reports and ``tests/test_documented_numbers.py`` asserts them, so the
lessons and the code can never drift apart. Results are cached per process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .context import Assembly, ContextAssembler, PrefixCache, routing_items
from .evaluation import (
    ColumnAudit,
    EvalResult,
    audit_columns,
    evaluate_prompt,
    label_shares,
    majority_baseline,
    make_provider,
)
from .lora import (
    LinearClassifier,
    featurize,
    full_params,
    labels_of,
    lora_params,
    train_full,
    train_lora,
    weight_memory_gb,
)
from .prompts import ExampleSelector, PromptConfig, build_messages
from .providers import Message
from .structured import classify, parse_answer
from .textfeat import stems
from .tickets import (
    INTENTS,
    Ticket,
    clean_text,
    deduplicate,
    exact_duplicate_bodies,
    find_resubmissions,
    four_way_split,
    load_tickets,
    overlap_by_text,
    pii_counts,
    random_split,
    strip_quoted_reply,
    temporal_split,
)

BEST_PROMPT = PromptConfig(descriptions="bilingual", k=8, selection="similar")


@lru_cache(maxsize=1)
def raw_tickets() -> tuple[Ticket, ...]:
    return tuple(load_tickets())


@lru_cache(maxsize=1)
def tickets() -> tuple[Ticket, ...]:
    """All tickets with the 24 resubmitted copies removed: the starting point of every experiment."""
    return tuple(deduplicate(list(raw_tickets())))


def standard_split() -> tuple[list[Ticket], list[Ticket]]:
    """Train before the Lookout C2 launch (2025-09-01), test from the launch onwards."""
    return temporal_split(list(tickets()))


# --------------------------------------------------------------------------- #
# Lesson 1: audit the data and choose an honest split                         #
# --------------------------------------------------------------------------- #


def oov_rate(reference: list[Ticket], rows: list[Ticket]) -> float:
    """Share of word stems in ``rows`` that never appear in ``reference``."""
    vocab: set[str] = set()
    for t in reference:
        vocab |= stems(clean_text(t.body))
    total = unseen = 0
    for t in rows:
        s = stems(clean_text(t.body))
        total += len(s)
        unseen += len(s - vocab)
    return round(unseen / total, 3) if total else 0.0


@dataclass(frozen=True)
class AuditReport:
    rows: int
    resubmissions: int
    duplicate_bodies: int
    after_dedup: int
    train: int
    test: int
    pii: dict[str, int]
    columns: list[ColumnAudit]
    majority: float
    shares_before: dict[str, float]
    shares_after: dict[str, float]
    camera_share_after: float
    oov_august: float
    oov_after_launch: float
    oov_camera: float


@lru_cache(maxsize=1)
def audit_report() -> AuditReport:
    raw = list(raw_tickets())
    clean = list(tickets())
    train, test = standard_split()
    split = four_way_split(clean)
    after = split.new_adapt + split.new_test
    return AuditReport(
        rows=len(raw),
        resubmissions=len(find_resubmissions(raw)),
        duplicate_bodies=exact_duplicate_bodies(raw),
        after_dedup=len(clean),
        train=len(train),
        test=len(test),
        pii=pii_counts(raw),
        columns=audit_columns(train, test),
        majority=majority_baseline(train, test),
        shares_before=label_shares(train),
        shares_after=label_shares(test),
        camera_share_after=round(sum(t.product == "Lookout C2" for t in test) / len(test), 3),
        oov_august=oov_rate(split.old_train, split.old_test),
        oov_after_launch=oov_rate(split.old_train, after),
        oov_camera=oov_rate(split.old_train, [t for t in after if t.product == "Lookout C2"]),
    )


@dataclass(frozen=True)
class SplitRow:
    name: str
    train: int
    test: int
    overlap: int  # test tickets whose exact text is also in train
    accuracy: float


@lru_cache(maxsize=1)
def split_report() -> list[SplitRow]:
    """Same prompt, different splits: how much does the evaluation design change the score?"""
    raw, clean = list(raw_tickets()), list(tickets())
    rows: list[SplitRow] = []
    cases = [
        ("random 80/20, raw rows", random_split(raw), BEST_PROMPT),
        ("random 80/20, deduplicated", random_split(clean), BEST_PROMPT),
        ("temporal, raw rows", temporal_split(raw), BEST_PROMPT),
        ("temporal, deduplicated", temporal_split(clean), BEST_PROMPT),
        ("temporal + resolution_code", temporal_split(clean), BEST_PROMPT.with_(leak_resolution_code=True)),
    ]
    for name, (train, test), cfg in cases:
        result = evaluate_prompt(name, test, train, cfg)
        rows.append(SplitRow(name, len(train), len(test), overlap_by_text(train, test), result.accuracy))
    return rows


# --------------------------------------------------------------------------- #
# Lesson 2: prompt engineering                                                #
# --------------------------------------------------------------------------- #

PROMPT_GRID: list[tuple[str, PromptConfig]] = [
    ("label names only, 0-shot", PromptConfig(descriptions="names")),
    ("English descriptions, 0-shot", PromptConfig(descriptions="english")),
    ("bilingual descriptions, 0-shot", PromptConfig()),
    ("bilingual + 1 random example", PromptConfig(k=1, selection="random")),
    ("bilingual + 4 random examples", PromptConfig(k=4, selection="random")),
    ("bilingual + 8 random examples", PromptConfig(k=8, selection="random")),
    ("bilingual + 8 balanced examples", PromptConfig(k=8, selection="balanced")),
    ("bilingual + 1 similar example", PromptConfig(k=1)),
    ("bilingual + 4 similar examples", PromptConfig(k=4)),
    ("bilingual + 8 similar examples", BEST_PROMPT),
    ("8 similar + user context", BEST_PROMPT.with_(user_context=True)),
]


@lru_cache(maxsize=1)
def prompt_table() -> list[EvalResult]:
    train, test = standard_split()
    return [evaluate_prompt(name, test, train, cfg) for name, cfg in PROMPT_GRID]


@dataclass(frozen=True)
class FormatReport:
    n: int
    strict_valid: int  # first answers that are exactly valid JSON with a known label
    lenient_valid: int  # first answers a tolerant parser can still read
    needed_retry: int
    invalid_after_retry: int
    accuracy_no_retry: float
    accuracy_with_retry: float
    accuracy_reasoning: float
    completion_tokens_plain: float
    completion_tokens_reasoning: float


def _strict(text: str) -> bool:
    try:
        return json.loads(text).get("intent") in INTENTS
    except (json.JSONDecodeError, AttributeError):
        return False


@lru_cache(maxsize=1)
def format_report() -> FormatReport:
    """Structured output: how often the first answer breaks the format, and what retries buy."""
    train, test = standard_split()
    provider = make_provider()
    selector = ExampleSelector(train, BEST_PROMPT)
    strict = lenient = 0
    for t in test:
        text = provider.complete(build_messages(t, BEST_PROMPT, selector)).text
        strict += _strict(text)
        lenient += parse_answer(text, INTENTS).label is not None
    no_retry = evaluate_prompt("no retry", test, train, BEST_PROMPT, max_retries=0)
    with_retry = evaluate_prompt("retry", test, train, BEST_PROMPT, max_retries=1)
    reasoning = evaluate_prompt("reasoning", test, train, BEST_PROMPT.with_(reasoning=True))
    return FormatReport(
        n=len(test),
        strict_valid=strict,
        lenient_valid=lenient,
        needed_retry=len(test) - lenient,
        invalid_after_retry=with_retry.unparseable,
        accuracy_no_retry=no_retry.accuracy,
        accuracy_with_retry=with_retry.accuracy,
        accuracy_reasoning=reasoning.accuracy,
        completion_tokens_plain=with_retry.completion_tokens,
        completion_tokens_reasoning=reasoning.completion_tokens,
    )


# --------------------------------------------------------------------------- #
# Lesson 3: context engineering                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BudgetRow:
    budget: int
    policy: str
    accuracy: float
    examples: float  # mean examples that fit
    tokens: float  # mean prompt tokens actually sent


POLICIES = {"raw": (False, False), "clean": (True, False), "clean+dedupe": (True, True)}


def evaluate_assembler(test: list[Ticket], train: list[Ticket], budget: int, policy: str) -> BudgetRow:
    clean, dedupe = POLICIES[policy]
    cfg = PromptConfig(clean=clean)
    selector = ExampleSelector(train, cfg)
    assembler = ContextAssembler(budget, dedupe=dedupe)
    provider = make_provider()
    hits = examples = tokens = 0
    for t in test:
        assembly = assembler.assemble(routing_items(t, selector, cfg))
        hits += classify(provider, assembly.messages, INTENTS).label == t.intent
        examples += assembly.examples_kept
        tokens += assembly.tokens
    n = len(test)
    return BudgetRow(budget, policy, round(hits / n, 3), round(examples / n, 2), round(tokens / n, 1))


@lru_cache(maxsize=1)
def budget_table(budgets: tuple[int, ...] = (400, 500, 700, 1000)) -> list[BudgetRow]:
    train, test = standard_split()
    return [evaluate_assembler(test, train, b, p) for b in budgets for p in POLICIES]


@lru_cache(maxsize=1)
def k_sweep() -> list[EvalResult]:
    """More examples are not always better: accuracy and cost as k grows."""
    train, test = standard_split()
    return [evaluate_prompt(f"{k} similar examples", test, train, BEST_PROMPT.with_(k=k)) for k in (4, 8, 16, 32)]


@dataclass(frozen=True)
class QuotedSlice:
    n: int
    zero_shot_raw: float
    zero_shot_clean: float
    few_shot_raw: float
    few_shot_clean: float


@lru_cache(maxsize=1)
def quoted_reply_slice() -> QuotedSlice:
    """Tickets that carry a quoted earlier e-mail: the history pollutes the prompt unless removed."""
    train, test = standard_split()
    quoted = [t for t in test if strip_quoted_reply(t.body) != t.body.strip()]

    def acc(cfg: PromptConfig) -> float:
        return evaluate_prompt("q", quoted, train, cfg).accuracy

    return QuotedSlice(len(quoted), acc(PromptConfig(clean=False)), acc(PromptConfig()),
                       acc(BEST_PROMPT.with_(clean=False)), acc(BEST_PROMPT))


def assembly_for(ticket_id: str, budget: int, policy: str = "clean+dedupe") -> tuple[Ticket, Assembly]:
    """The assembled context for one test ticket, with its audit trail."""
    train, test = standard_split()
    ticket = next(t for t in test if t.ticket_id == ticket_id)
    clean, dedupe = POLICIES[policy]
    cfg = PromptConfig(clean=clean)
    assembly = ContextAssembler(budget, dedupe=dedupe).assemble(routing_items(ticket, ExampleSelector(train, cfg), cfg))
    return ticket, assembly


@lru_cache(maxsize=1)
def cache_report() -> tuple[PrefixCache, PrefixCache]:
    """Prefix caching with a stable system prompt versus one that starts with a request id and date."""
    train, test = standard_split()
    selector = ExampleSelector(train, BEST_PROMPT)
    stable, unstable = PrefixCache(), PrefixCache()
    for t in test:
        messages = build_messages(t, BEST_PROMPT, selector)
        stable.observe(messages)
        stamped = Message("system", f"Request {t.ticket_id} received on {t.created_at}.\n" + messages[0].content)
        unstable.observe([stamped, *messages[1:]])
    return stable, unstable


# --------------------------------------------------------------------------- #
# Lesson 4: adapting after the launch (prompting versus fine-tuning)          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AdaptRow:
    name: str
    labelled: int  # labelled tickets the approach learns from
    trainable: int  # numbers changed by training (0 for prompting)
    old: float  # accuracy on August tickets (before the launch)
    new: float  # accuracy on October-December tickets
    camera: float  # accuracy on Lookout C2 tickets from October onwards
    prompt_tokens: float  # mean prompt tokens per request (0 for the classifier)


@lru_cache(maxsize=1)
def prompt_adaptation() -> list[AdaptRow]:
    """Keep the model, change the prompt: new descriptions and/or a fresher example pool."""
    split = four_way_split(list(tickets()))
    old_pool, fresh_pool = split.old_train, split.old_train + split.new_adapt
    zero = PromptConfig()
    cases = [
        ("0-shot, original descriptions", old_pool, zero, 0),
        ("0-shot, descriptions updated", old_pool, zero.with_(descriptions="bilingual_v2"), 0),
        ("8 similar, pool Jan-Jul", old_pool, BEST_PROMPT, len(old_pool)),
        ("8 similar, pool Jan-Jul + Sep", fresh_pool, BEST_PROMPT, len(fresh_pool)),
    ]
    rows = []
    for name, pool, cfg, labelled in cases:
        old = evaluate_prompt(name, split.old_test, pool, cfg)
        new = evaluate_prompt(name, split.new_test, pool, cfg)
        rows.append(AdaptRow(name, labelled, 0, old.accuracy, new.accuracy, new.accuracy_camera, new.prompt_tokens))
    return rows


@dataclass(frozen=True)
class LoRAData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_old: np.ndarray
    y_old: np.ndarray
    X_adapt: np.ndarray
    y_adapt: np.ndarray
    X_new: np.ndarray
    y_new: np.ndarray
    camera: np.ndarray  # mask over the new test set


@lru_cache(maxsize=1)
def lora_data() -> LoRAData:
    s = four_way_split(list(tickets()))
    return LoRAData(
        featurize(s.old_train), labels_of(s.old_train),
        featurize(s.old_test), labels_of(s.old_test),
        featurize(s.new_adapt), labels_of(s.new_adapt),
        featurize(s.new_test), labels_of(s.new_test),
        np.array([t.product == "Lookout C2" for t in s.new_test]),
    )


@lru_cache(maxsize=1)
def base_model() -> LinearClassifier:
    """The 'pre-trained' model: trained on January-July tickets, then frozen."""
    d = lora_data()
    return train_full(d.X_train, d.y_train)


def _scores(model: object) -> tuple[float, float, float]:
    d = lora_data()
    acc = model.accuracy  # type: ignore[attr-defined]
    return acc(d.X_old, d.y_old), acc(d.X_new, d.y_new), acc(d.X_new[d.camera], d.y_new[d.camera])


@dataclass(frozen=True)
class LoRARow:
    name: str
    trainable: int
    share_of_full: float  # trainable / full fine-tuning trainable
    old: float
    new: float
    camera: float
    new_spread: float  # max - min over seeds (0 when there is no randomness)


SEEDS = (0, 1, 2, 3, 4)
FT_STEPS, FT_LR = 100, 2.0


@lru_cache(maxsize=1)
def lora_table() -> list[LoRARow]:
    """Adapt the frozen base to the 97 September tickets: full fine-tuning versus LoRA ranks."""
    d = lora_data()
    base = base_model()
    full = train_full(d.X_adapt, d.y_adapt, start=base, steps=FT_STEPS, lr=FT_LR)
    n_full = full.n_params
    rows = [LoRARow("base (frozen, no update)", 0, 0.0, *_scores(base), 0.0),
            LoRARow("full fine-tuning", n_full, 1.0, *_scores(full), 0.0)]
    for rank in (1, 2, 4, 8):
        runs = [train_lora(base, d.X_adapt, d.y_adapt, rank, steps=FT_STEPS, lr=FT_LR, seed=s) for s in SEEDS]
        scores = np.array([_scores(m) for m in runs])
        mean = scores.mean(axis=0).round(3)
        spread = round(float(scores[:, 1].max() - scores[:, 1].min()), 3)
        rows.append(LoRARow(f"LoRA r={rank}", runs[0].n_params, round(runs[0].n_params / n_full, 3),
                            float(mean[0]), float(mean[1]), float(mean[2]), spread))
    return rows


@dataclass(frozen=True)
class ForgettingRow:
    steps: int
    full_old: float
    full_new: float
    lora_old: float  # rank 4, seed 0
    lora_new: float
    replay_old: float  # full fine-tuning on September + 100 old tickets
    replay_new: float


@lru_cache(maxsize=1)
def forgetting_table(steps_list: tuple[int, ...] = (100, 1000, 4000)) -> list[ForgettingRow]:
    """Train longer on new data only and watch accuracy on old tickets fall; replay prevents it."""
    d = lora_data()
    base = base_model()
    idx = np.random.default_rng(0).choice(len(d.y_train), 100, replace=False)
    X_mix = np.vstack([d.X_adapt, d.X_train[idx]])
    y_mix = np.concatenate([d.y_adapt, d.y_train[idx]])
    rows = []
    for steps in steps_list:
        full = _scores(train_full(d.X_adapt, d.y_adapt, start=base, steps=steps, lr=FT_LR))
        lora = _scores(train_lora(base, d.X_adapt, d.y_adapt, 4, steps=steps, lr=FT_LR, seed=0))
        replay = _scores(train_full(X_mix, y_mix, start=base, steps=steps, lr=FT_LR))
        rows.append(ForgettingRow(steps, full[0], full[1], lora[0], lora[1], replay[0], replay[1]))
    return rows


@dataclass(frozen=True)
class CurveRow:
    labelled: int
    full_new: float
    lora_new: float  # rank 4, mean over seeds


@lru_cache(maxsize=1)
def data_curve(sizes: tuple[int, ...] = (12, 25, 50, 97)) -> list[CurveRow]:
    """How many labelled September tickets does adaptation need? (taken in date order)"""
    d = lora_data()
    base = base_model()
    rows = []
    for n in sizes:
        X, y = d.X_adapt[:n], d.y_adapt[:n]
        full = train_full(X, y, start=base, steps=FT_STEPS, lr=FT_LR).accuracy(d.X_new, d.y_new)
        lora = np.mean([train_lora(base, X, y, 4, steps=FT_STEPS, lr=FT_LR, seed=s).accuracy(d.X_new, d.y_new)
                        for s in SEEDS])
        rows.append(CurveRow(n, full, round(float(lora), 3)))
    return rows


@dataclass(frozen=True)
class ParamRow:
    rank: int
    lora: int
    share: float  # of the full matrix


def param_table(d_in: int = 4096, d_out: int = 4096, ranks: tuple[int, ...] = (1, 2, 4, 8, 16, 64)) -> list[ParamRow]:
    """LoRA size for one weight matrix of a given shape."""
    full = full_params(d_in, d_out)
    return [ParamRow(r, lora_params(d_in, d_out, r), round(lora_params(d_in, d_out, r) / full, 5)) for r in ranks]


def memory_table(n_params: float = 7e9) -> dict[int, float]:
    """Weights-only memory of a model with ``n_params`` parameters at 32, 16, 8 and 4 bits."""
    return {bits: weight_memory_gb(n_params, bits) for bits in (32, 16, 8, 4)}
