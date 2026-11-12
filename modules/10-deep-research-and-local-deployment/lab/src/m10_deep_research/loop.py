"""The deep-research loop: plan -> search -> read -> take notes -> reflect -> write."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from .corpus import Snapshot
from .credibility import score_page
from .notes import Note, NoteStore, recency_weight
from .planner import SubQuery, fake_planner, plan
from .providers import Completion, FakeProvider, LLMProvider, Message, count_tokens
from .reader import fake_reader, mentions, read_page
from .search import BM25Index
from .synthesise import fake_writer, write_report


@dataclass(frozen=True)
class ResearchConfig:
    """Budgets and defences. Each ``use_*`` switch exists so you can measure what it buys."""

    max_iterations: int = 3
    max_searches: int = 20
    top_k: int = 4  # pages opened per search
    notes_per_subquery: int = 2  # kept by compaction
    min_credibility: float = 0.4
    min_relevance: float = 0.4
    half_life_days: float = 365.0
    use_credibility: bool = True
    use_recency: bool = True
    use_sanitiser: bool = True
    use_dedup: bool = True


@dataclass
class Stats:
    iterations: int = 0
    searches: int = 0
    pages_opened: int = 0
    llm_calls: int = 0
    skipped_low_credibility: int = 0
    injections_removed: int = 0
    pii_redacted: int = 0
    quotes_rejected: int = 0
    mirrors_merged: int = 0
    superseded: int = 0
    tokens_read: int = 0  # article text of every page the reader opened
    tokens_noted: int = 0  # evidence passages added to the note store
    tokens_kept: int = 0  # notes handed to the writer after compaction
    bad_citations_removed: int = 0
    prompt_tokens: int = 0  # what an API would bill as input
    completion_tokens: int = 0  # ... and as output
    stop_reason: str = ""


@dataclass
class ResearchResult:
    question_id: str
    report: str
    subqueries: list[SubQuery]
    notes: list[Note]
    gaps: list[str]
    stats: Stats
    trace: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"question_id": self.question_id, "gaps": self.gaps, **asdict(self.stats)}


class Meter:
    """Wrap any provider and count calls and tokens: the numbers you budget and pay for."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner, self.name, self.model = inner, inner.name, inner.model
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def complete(self, messages: Sequence[Message], **kwargs: object) -> Completion:
        out = self.inner.complete(messages, **kwargs)  # type: ignore[arg-type]
        self.calls += 1
        self.prompt_tokens += out.prompt_tokens
        self.completion_tokens += out.completion_tokens
        return out

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.inner.embed(texts)


def default_provider() -> FakeProvider:
    """A FakeProvider that plays all three roles, picked by the system prompt."""
    def respond(messages: Sequence[Message]) -> str:
        system = messages[0].content
        if "planner" in system:
            return fake_planner(messages)
        if "reader" in system:
            return fake_reader(messages)
        return fake_writer(messages)

    return FakeProvider(responder=respond, model="fake-research-1")


def missing_entities(sq: SubQuery, notes: list[Note]) -> list[str]:
    """Entities of ``sq`` that no note mentions yet ("any evidence" if there are no notes)."""
    if not notes:
        return list(sq.entities) or ["any evidence"]
    text = " ".join(f"{n.title} {n.text}" for n in notes)
    return [e for e in sq.entities if not mentions(text, e)]


def follow_up(sq: SubQuery, missing: list[str]) -> str:
    """Reflection turned into a new search: put the missing entities first."""
    named = [m for m in missing if m != "any evidence"]
    return " ".join(named + [sq.query]) if named else f"{sq.query} data"


def run_research(
    question: dict,
    snapshot: Snapshot,
    index: BM25Index,
    provider: LLMProvider | None = None,
    config: ResearchConfig = ResearchConfig(),
) -> ResearchResult:
    """Research one row of ``research_questions.jsonl`` and return the report and its trace."""
    meter = Meter(provider or default_provider())
    provider = meter
    stats, trace = Stats(), []
    store = NoteStore(dedup=config.use_dedup)
    subqueries = plan(question["question"], provider)
    trace += [f"plan  {sq.id} [{sq.aspect}] {sq.query!r}" for sq in subqueries]
    queue: list[tuple[SubQuery, str]] = [(sq, sq.query) for sq in subqueries]
    opened: set[tuple[str, str]] = set()  # (sub-query id, URL) pairs already read
    for iteration in range(1, config.max_iterations + 1):
        stats.iterations = iteration
        new_notes = 0
        for sq, query in queue:
            if stats.searches >= config.max_searches:
                break
            stats.searches += 1
            trace.append(f"search {sq.id} {query!r}")
            for hit in index.search(query, k=config.top_k):
                page = snapshot.get(hit.url)
                if page is None or not page.ok:
                    continue
                url = page.canonical if config.use_dedup else page.url
                if (sq.id, url) in opened:
                    continue
                opened.add((sq.id, url))
                cred = score_page(page).score if config.use_credibility else 1.0
                if cred < config.min_credibility:
                    stats.skipped_low_credibility += 1
                    trace.append(f"  skip  {page.url} (credibility {cred})")
                    continue
                stats.pages_opened += 1
                stats.tokens_read += count_tokens(page.text)
                passage = read_page(page, query, sq.entities, provider, use_sanitiser=config.use_sanitiser)
                if passage is None or passage.relevance < config.min_relevance:
                    continue
                stats.injections_removed += len(passage.injections)
                stats.pii_redacted += passage.pii
                stats.quotes_rejected += passage.rejected_quotes
                recency = (recency_weight(page.published, snapshot.as_of, config.half_life_days)
                           if config.use_recency else 1.0)
                note = Note(url, page.domain, page.publisher, page.title, page.published, passage.text,
                            cred, recency, {sq.id: passage.relevance})
                if store.add(note):
                    new_notes += 1
                    stats.tokens_noted += note.tokens
                    trace.append(f"  note  {url} (score {note.score})")
        if config.use_recency:
            store.drop_superseded([e for sq in subqueries for e in sq.entities])
        store.compact(config.notes_per_subquery)
        # Reflect: which sub-queries still lack evidence for some of their entities?
        queue = []
        for sq in subqueries:
            if missing := missing_entities(sq, store.for_subquery(sq.id)):
                queue.append((sq, follow_up(sq, missing)))
        trace.append(f"reflect iteration {iteration}: {new_notes} new notes, {len(queue)} sub-queries open")
        if not queue:
            stats.stop_reason = "all sub-queries covered"
            break
        if new_notes == 0:
            stats.stop_reason = "no new evidence"
            break
        if stats.searches >= config.max_searches:
            stats.stop_reason = "search budget spent"
            break
    else:
        stats.stop_reason = "iteration budget spent"
    order = {sq.id: i for i, sq in enumerate(subqueries)}
    notes = sorted(store.notes, key=lambda n: (order[n.owner], -n.score, n.url))
    gaps = [f"{sq.aspect} ({', '.join(missing_entities(sq, store.for_subquery(sq.id)))})" for sq, _ in queue]
    stats.mirrors_merged = store.mirrors_merged
    stats.superseded = len(store.superseded)
    stats.tokens_kept = sum(n.tokens for n in notes)
    report, stats.bad_citations_removed = write_report(
        question["question"], subqueries, notes, gaps, provider, as_of=snapshot.as_of.isoformat()
    )
    stats.llm_calls, stats.prompt_tokens, stats.completion_tokens = (
        meter.calls, meter.prompt_tokens, meter.completion_tokens)
    return ResearchResult(question["id"], report, subqueries, notes, gaps, stats, trace)
