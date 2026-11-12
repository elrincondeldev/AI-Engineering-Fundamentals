"""What the scripted model knows: reading a question, extracting facts, combining them.

A real LLM does all of this implicitly. Our fake model does it with readable rules so
that every decision in a trajectory can be traced to a line of code. The price is
coverage: it only understands the 14 attributes in ``ATTRIBUTES`` and the question
shapes in :func:`analyze`. Treat its scores as a property of the *harness* (search,
guards, control flow), not as a measure of how good a real model would be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .search import tokenize

COUNTRIES = ("Spain", "Portugal", "Mexico", "Colombia", "Chile")
COUNTRY_NAMES = {  # names the model recognises, including Spanish spellings
    "Spain": ("spain", "españa", "spanish", "españoles"),
    "Portugal": ("portugal", "portuguese"),
    "Mexico": ("mexico", "méxico", "mexican"),
    "Colombia": ("colombia", "colombian"),
    "Chile": ("chile", "chilean"),
}
QUESTION_WORDS = {
    "What",
    "How",
    "Which",
    "When",
    "Does",
    "On",
    "By",
    "Among",
    "In",
    "According",
    "Round",
    "Is",
}
NOT_ENTITIES = {"Wi-Fi", "EU", "Fi", "Wi"}


@dataclass(frozen=True)
class Attribute:
    """One kind of fact the model can look for in a page."""

    name: str
    keywords: str  # words added to the search query
    context: tuple[str, ...]  # one must appear (in the clause or page, see ``context_in``)
    pattern: str  # regex; capture groups hold the value
    template: str  # how to print the value
    context_in: str = "page"

    @property
    def label(self) -> str:
        return LABELS[self.name]


LABELS = {
    "standby": "standby power",
    "max_standby": "recommended maximum standby power",
    "devices": "device limit",
    "date": "release date",
    "units": "units sold",
    "growth": "sales growth",
    "adoption": "smart home adoption",
    "delivery": "delivery time",
    "electricity": "electricity price",
    "savings": "heating savings",
    "price": "price",
    "price_mxn": "price in pesos",
    "lumens": "brightness in lumens",
    "resolution": "video resolution",
    "countries": "sales countries",
    "advice": "forum advice",
}

ATTRIBUTES = {
    a.name: a
    for a in (
        Attribute("standby", "standby power", ("standby", "idle"), r"(\d+(?:\.\d+)?)\s?W\b", "{} W"),
        Attribute(
            "max_standby",
            "standby power recommend",
            ("no more than", "at most"),
            r"(\d+(?:\.\d+)?)\s?W\b",
            "{} W",
            "clause",
        ),
        Attribute("devices", "devices network", ("device",), r"(\d[\d,]*)\s+devices\b", "{}"),
        Attribute(
            "date", "release launch date", ("releas", "launch"), r"\b(\d{4}-\d{2}-\d{2})\b", "{}", "clause"
        ),
        Attribute(
            "units",
            "sales units sold",
            ("sold", "sales", "vendieron"),
            r"(\d[\d.,]*(?:\s(?:million|thousand))?)\s+(?:units|smart thermostats|termostatos|cameras)\b",
            "{}",
        ),
        Attribute("growth", "sales growth", ("growth", "crece"), r"(\d+(?:\.\d+)?)%\s*growth", "{}%"),
        Attribute(
            "adoption", "smart home adoption households", ("household", "hogares"), r"(\d+(?:\.\d+)?)%", "{}%"
        ),
        Attribute("delivery", "parcel delivery time", ("deliver",), r"(\d+)\s+days?\b", "{} days"),
        Attribute(
            "electricity", "household electricity price", ("electricity",), r"€\s?(\d+\.\d+)", "€{} per kWh"
        ),
        Attribute(
            "savings", "heating savings study", ("saving",), r"between (\d+)% and (\d+)%", "{}% to {}%"
        ),
        Attribute("price", "price", ("€",), r"€\s?(\d+(?:\.\d+)?)", "€{}"),
        Attribute(
            "price_mxn", "price Mexico pesos", ("peso", "mxn"), r"(\d[\d.,]*)\s*(?:MXN|pesos)\b", "{} MXN"
        ),
        Attribute("lumens", "lumens brightness", ("lumen",), r"(\d[\d,]*)\s*(?:lumens|lm)\b", "{}"),
        Attribute("resolution", "video resolution", ("video", "record"), r"\b(\d{3,4}p|\dK)\b", "{}"),
        Attribute("countries", "available countries sold", ("available in", "sold in"), r"", "{}", "clause"),
        Attribute("advice", "fix forum", ("user ",), r"", "{}", "clause"),
    )
}


@dataclass(frozen=True)
class Slot:
    """One fact the answer needs, for example the price of the Corvid Thermo X."""

    attribute: str
    entity: str
    query: str

    @property
    def key(self) -> str:
        return f"{self.attribute}:{self.entity}"

    @property
    def is_country(self) -> bool:
        return self.entity in COUNTRIES

    def directive(self) -> str:
        """The slot as a structured sub-task another agent can parse."""
        return f"[slot: {self.attribute} | {self.entity} | {self.query}]"

    def describe(self) -> str:
        label = LABELS[self.attribute]
        return f"{label} in {self.entity}" if self.is_country else f"{label} of the {self.entity}"


@dataclass(frozen=True)
class Plan:
    """The model's reading of a question: which facts it needs and how to combine them."""

    question: str
    slots: tuple[Slot, ...]
    # lookup | difference | ratio | max | per_unit_max | only_in_first | within_limit | value_and_limit
    op: str
    options: tuple[str, ...] = ()
    wants_recent: bool = False


@dataclass(frozen=True)
class Fact:
    """A value found in a page, with enough provenance to cite and to resolve conflicts."""

    slot: str
    value: str
    url: str
    published: str
    tier: str
    sentence: str
    order: int = 0  # when the page was opened (earlier = smaller)
    items: tuple[str, ...] = field(default=(), compare=False)

    @property
    def number(self) -> float | None:
        return parse_number(self.value)


# --------------------------------------------------------------------------- #
# Reading the question                                                        #
# --------------------------------------------------------------------------- #


def find_entities(question: str) -> list[str]:
    """Named things in the question: runs of capitalised words and codes (``Ember T3``)."""
    tokens = re.findall(r"[\w.€-]+|[^\w\s]", question)
    runs: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        is_name = tok[:1].isupper() or (current and re.fullmatch(r"[A-Z0-9][\w.]*|X", tok) is not None)
        if is_name and tok not in QUESTION_WORDS:
            current.append(tok)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    names = [re.sub(r"\s+\d{4}$", "", " ".join(r).rstrip(".")) for r in runs]  # drop trailing years
    return [n for n in names if n not in NOT_ENTITIES]


def find_countries(text: str) -> list[str]:
    """Countries mentioned in ``text``, in order of first appearance."""
    low = text.lower()
    found = []
    for country, names in COUNTRY_NAMES.items():
        positions = [m.start() for n in names for m in re.finditer(rf"\b{re.escape(n)}\b", low)]
        if positions:
            found.append((min(positions), country))
    return [c for _, c in sorted(found)]


def detect_attribute(q: str) -> str:
    """Pick the attribute a single-fact question asks about (first matching rule wins)."""
    q = q.lower()
    rules = [
        ("advice", ("recommend to fix", "users recommend")),
        ("max_standby", ("recommend",)),
        ("standby", ("standby",)),
        ("lumens", ("lumens",)),
        ("resolution", ("resolution",)),
        ("devices", ("how many devices",)),
        ("date", ("what date", "when did", "when was")),
        ("growth", ("grow", "growth")),
        ("adoption", ("adoption", "households")),
        ("delivery", ("delivery",)),
        ("electricity", ("electricity",)),
        ("savings", ("savings",)),
        ("price_mxn", ("pesos",)),
        ("price", ("price", "expensive", "cost")),
        ("units", ("sold", "sales", "market")),
        ("countries", ("countries",)),
    ]
    for name, cues in rules:
        if any(c in q for c in cues):
            return name
    return "units"


def _slot(attribute: str, entity: str, question: str, wants_recent: bool) -> Slot:
    attr = ATTRIBUTES[attribute]
    words = [entity, attr.keywords]
    if attribute == "advice":
        words = [w for w in tokenize(question) if w not in {"forum", "users", "recommend"}] + ["fix"]
    if "2025" in question and attribute not in ("date", "advice"):
        words.append("2025")
    if wants_recent:
        words.append("latest")
    acronym = "".join(w[0] for w in entity.split() if w[:1].isupper())
    if len(entity.split()) >= 4:  # long names are often written as acronyms (EHEO)
        words.insert(1, acronym)
    return Slot(attribute, entity, " ".join(dict.fromkeys(" ".join(words).split())))


def _shared_slots(attribute: str, countries: tuple[str, ...], question: str) -> tuple[Slot, ...]:
    """One query for several countries: market reports list them all on one page."""
    year = " 2025" if "2025" in question else ""
    query = f"{ATTRIBUTES[attribute].keywords} {' '.join(countries)}{year}"
    return tuple(Slot(attribute, c, query) for c in countries)


def _options(question: str) -> tuple[str, ...]:
    """The choices listed after the colon: 'Mexico or Chile', 'the Glow A19 or the ...'."""
    tail = question.rsplit(":", 1)[-1].rstrip("?").strip()
    parts = re.split(r",\s*|\s+or\s+|\s+and\s+", tail)
    return tuple(re.sub(r"^the\s+", "", p.strip()) for p in parts if p.strip())


def analyze(question: str) -> Plan:
    """Turn a question into the facts to collect and the operation that combines them."""
    q = question.strip()
    subs = re.findall(r"\[slot: (\w+) \| ([^|\]]+) \| ([^\]]+)\]", q)
    if subs:  # structured sub-tasks sent by a supervisor agent
        slots = tuple(Slot(a, e.strip(), qy.strip()) for a, e, qy in subs)
        return Plan(q, slots, "lookup", wants_recent=any("latest" in s.query for s in slots))
    ql = q.lower()
    recent = any(c in ql for c in ("latest", "after its", "current", "now "))
    ents = [e for e in find_entities(q) if e not in COUNTRIES and e not in ("Spanish", "Mexican")]

    m = re.search(r"how much more expensive is (?:the )?(.+?) than (?:the )?(.+?)(?:,|\?)", q, re.I)
    if m:
        a, b = m.group(1), m.group(2)
        return Plan(q, (_slot("price", a, q, recent), _slot("price", b, q, recent)), "difference")

    m = re.search(r"more (\w+) per (\w+)", ql)
    if m and ":" in q:
        options = _options(q)
        slots = []
        for o in options:
            slots += [_slot(detect_attribute(m.group(1)), o, q, recent), _slot("price", o, q, recent)]
        return Plan(q, tuple(slots), "per_unit_max", options)

    m = re.search(r"in which of these countries is (?:the )?(.+?) sold but (?:the )?(.+?) is not", q, re.I)
    if m:
        a, b = m.group(1), m.group(2)
        slots = (_slot("countries", a, q, recent), _slot("countries", b, q, recent))
        return Plan(q, slots, "only_in_first", _options(q))

    if re.search(r"\b(higher|highest|fastest|largest|lower|lowest)\b", ql) and ql.startswith(
        ("which", "among")
    ):
        attr = detect_attribute(q)
        options = tuple(find_countries(q.split(":")[1] if ":" in q else q))
        return Plan(q, _shared_slots(attr, options, q), "max", options)

    if "recommendation" in ql and len(ents) >= 2:
        product, org = ents[0], ents[1]
        slots = (_slot("standby", product, q, recent), _slot("max_standby", org, q, False))
        op = "value_and_limit" if ql.startswith("what") else "within_limit"
        return Plan(q, slots, op, wants_recent=recent)

    if "how many times larger" in ql:
        return Plan(q, _shared_slots("units", tuple(find_countries(q)[:2]), q), "ratio")

    attr = detect_attribute(q)
    countries = find_countries(q)
    if countries and attr in ("units", "growth", "adoption", "delivery", "electricity"):
        entity = countries[0]
    elif attr == "advice":
        entity = ents[0] if ents else "forum"
    else:  # the most specific name: most words, then first mentioned
        entity = max(ents, key=lambda e: (len(e.split()), -ents.index(e))) if ents else ""
    # One fact: search with the question's own keywords, as most agents do.
    query = " ".join(w for w in dict.fromkeys(tokenize(q)) if w not in {"many", "much", "according"})
    return Plan(q, (Slot(attr, entity, query),), "lookup", wants_recent=recent)


# --------------------------------------------------------------------------- #
# Reading a page                                                              #
# --------------------------------------------------------------------------- #


def parse_number(value: str) -> float | None:
    """'410,000' -> 410000, '410.000' -> 410000 (Spanish), '2 million' -> 2e6, '0.4 W' -> 0.4."""
    m = re.search(r"(\d[\d.,]*)\s*(million|thousand)?", value)
    if not m:
        return None
    raw, scale = m.group(1).rstrip(".,"), m.group(2)
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", raw):
        raw = re.sub(r"[.,]", "", raw)
    try:
        number = float(raw.replace(",", ""))
    except ValueError:
        return None
    return number * {"million": 1e6, "thousand": 1e3, None: 1}[scale]


def clauses(text: str) -> list[str]:
    """Split page text into sentences, and table-like sentences into ``;`` clauses."""
    out = []
    for line in text.splitlines():
        if line.startswith("Published "):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z¿¡\[])", line):
            out.extend(c.strip() for c in sentence.split(";") if c.strip())
    return out


def _mentions(text: str, entity: str) -> bool:
    low = text.lower()
    if entity in COUNTRY_NAMES:
        return any(re.search(rf"\b{re.escape(n)}\b", low) for n in COUNTRY_NAMES[entity])
    return all(t in low for t in tokenize(entity))


def extract(slot: Slot, text: str, url: str, published: str, tier: str, order: int = 0) -> Fact | None:
    """Find the value ``slot`` asks for in one page, or None."""
    attr = ATTRIBUTES[slot.attribute]
    url_words = " ".join(re.findall(r"[a-z0-9]+", url.lower()))
    acronym = "".join(w[0] for w in slot.entity.split() if w[:1].isupper()).lower()
    page_ok = _mentions(f"{text} {url_words}", slot.entity) or (len(acronym) >= 3 and acronym in url_words)
    if not page_ok and not slot.is_country:
        return None
    if attr.context_in == "page" and not any(c in text.lower() for c in attr.context):
        return None
    best: tuple[int, str, str, tuple[str, ...]] | None = None
    for clause in clauses(text):
        low = clause.lower()
        if slot.is_country and not _mentions(clause, slot.entity):
            continue
        if attr.context_in == "clause" and not any(c in low for c in attr.context):
            continue
        value, items = _value(slot, attr, clause)
        if value is None:
            continue
        score = len(set(tokenize(slot.query)) & set(tokenize(clause))) + any(c in low for c in attr.context)
        if best is None or score > best[0]:
            best = (score, value, clause, items)
    if best is None:
        return None
    return Fact(slot.key, best[1], url, published, tier, best[2], order, best[3])


def _value(slot: Slot, attr: Attribute, clause: str) -> tuple[str | None, tuple[str, ...]]:
    if attr.name == "countries":
        found = tuple(find_countries(clause))
        return (", ".join(found), found) if found else (None, ())
    if attr.name == "advice":
        body = clause.split(":", 1)[-1].strip()
        imperative = re.match(r"(give|use|set|fix|try|move|change|put)\b", body, re.I)
        return (body, ()) if imperative and not body.endswith("?") else (None, ())
    m = re.search(attr.pattern, clause, re.IGNORECASE if attr.name in ("price_mxn",) else 0)
    if not m:
        return None, ()
    return attr.template.format(*m.groups()), ()


# --------------------------------------------------------------------------- #
# Resolving conflicts and composing the answer                                #
# --------------------------------------------------------------------------- #

TIER_RANK = {"high": 3, "medium": 2, "low": 1, "spam": 0, "": 0}


def same_value(a: Fact, b: Fact) -> bool:
    if a.items or b.items:
        return set(a.items) == set(b.items)
    na, nb = a.number, b.number
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-9
    return a.value.lower() == b.value.lower()


def resolve(facts: list[Fact], prefer_credible: bool, prefer_recent: bool) -> Fact | None:
    """Choose one value when sources disagree.

    Groups facts that agree, then ranks groups by: source tier (if ``prefer_credible``),
    publication date (if ``prefer_recent``), number of agreeing sources, first seen.
    Returns the earliest-opened fact of the winning group (that is the one we cite).
    """
    groups: list[list[Fact]] = []
    for f in sorted(facts, key=lambda f: f.order):
        for g in groups:
            if same_value(g[0], f):
                g.append(f)
                break
        else:
            groups.append([f])
    if not groups:
        return None

    def key(g: list[Fact]) -> tuple:
        tier = max(TIER_RANK[f.tier] for f in g) if prefer_credible else 0
        date = max(f.published for f in g) if prefer_recent else ""
        return (tier, date, len(g), -min(f.order for f in g))

    best = max(groups, key=key)
    if prefer_credible:
        return max(best, key=lambda f: (TIER_RANK[f.tier], -f.order))
    return best[0]


def fmt_number(x: float) -> str:
    return f"{x:,.0f}" if x == int(x) else f"{x:g}"


@dataclass(frozen=True)
class Answer:
    """The final answer: a short form for grading, an explanation and its sources."""

    short: str
    explanation: str
    sources: tuple[str, ...]
    support: tuple[tuple[str, str], ...] = ()  # (url, value it supports)

    def render(self) -> str:
        cites = " ".join(f"[{i}] {u}" for i, u in enumerate(self.sources, 1))
        return f"Final Answer: {self.short}\nExplanation: {self.explanation}\nSources: {cites or 'none'}"


def compose(plan: Plan, chosen: dict[str, Fact | None]) -> Answer:
    """Combine the chosen facts with the plan's operation into a cited answer."""
    facts = [chosen.get(s.key) for s in plan.slots]
    if any(f is None for f in facts) or not facts:
        missing = [s.describe() for s, f in zip(plan.slots, facts) if f is None]
        return Answer("unknown", f"I could not find the {' and the '.join(missing)} in the snapshot.", ())
    fs: list[Fact] = [f for f in facts if f is not None]
    sources = tuple(dict.fromkeys(f.url for f in fs))
    ref = {u: i for i, u in enumerate(sources, 1)}
    cite = [f"{s.describe()}: {f.value.rstrip('.')} [{ref[f.url]}]" for s, f in zip(plan.slots, fs)]
    support = tuple((f.url, f.value) for f in fs)
    nums = [f.number for f in fs]
    short = fs[0].value
    op = plan.op
    if op == "difference" and None not in nums:
        short = fmt_number(nums[0] - nums[1])  # type: ignore[operator]
    elif op == "ratio" and None not in nums:
        short = f"{nums[0] / nums[1]:.1f}"  # type: ignore[operator]
    elif op == "max" and None not in nums:
        short = plan.options[max(range(len(nums)), key=lambda i: nums[i])]  # type: ignore[arg-type]
    elif op == "per_unit_max" and None not in nums:
        per = [nums[i] / nums[i + 1] for i in range(0, len(nums), 2)]  # type: ignore[operator]
        best = max(range(len(per)), key=lambda i: per[i])
        short = plan.options[best]
        cite.append("per unit: " + ", ".join(f"{o} {p:.1f}" for o, p in zip(plan.options, per)))
    elif op == "only_in_first":
        only = [o for o in plan.options if o in fs[0].items and o not in fs[1].items]
        short = ", ".join(only) if only else "none"
    elif op in ("within_limit", "value_and_limit") and None not in nums:
        ok = nums[0] <= nums[1]  # type: ignore[operator]
        verdict = (
            f"yes, {fs[0].value} is within the {fs[1].value} recommendation"
            if ok
            else f"no, it is above the {fs[1].value} recommendation"
        )
        short = ("yes" if ok else "no") if op == "within_limit" else f"{fs[0].value}; {verdict}"
    explanation = "; ".join(cite).rstrip(".") + "."
    return Answer(short, explanation[0].upper() + explanation[1:], sources, support)
