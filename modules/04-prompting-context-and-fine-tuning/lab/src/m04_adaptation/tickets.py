"""Load, clean, deduplicate and split the support tickets (datasets/tickets/tickets.csv)."""

from __future__ import annotations

import html
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

from .data import iter_csv

TICKETS_CSV = "tickets/tickets.csv"
LAUNCH_DATE = "2025-09-01"  # the Lookout C2 camera goes on sale: concept drift starts here
INTENTS = (
    "order_status",
    "return_request",
    "warranty_claim",
    "connectivity_issue",
    "billing_question",
    "account_access",
    "product_question",
    "cancel_subscription",
)

_PHONE_RE = re.compile(r"\+\d{2,3}(?:[ -]?\d{2,4}){2,4}")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SIGNATURE_STARTS = ("--", "thanks,", "best regards", "sent from my phone")


@dataclass(frozen=True)
class Ticket:
    """One support ticket. ``resolution_code`` is only known after the ticket is solved."""

    ticket_id: str
    created_at: str
    channel: str
    language: str
    product: str
    subject: str
    body: str
    customer_email: str
    intent: str
    resolution_code: str


def load_tickets() -> list[Ticket]:
    """Read every ticket, in file order (sorted by date)."""
    return [Ticket(**row) for row in iter_csv(TICKETS_CSV)]


# --------------------------------------------------------------------------- #
# Cleaning and personal data                                                  #
# --------------------------------------------------------------------------- #


def strip_quoted_reply(text: str) -> str:
    """Drop the quoted earlier e-mail ("> On ... wrote:") that some tickets carry."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">")).strip()


def strip_signature(text: str) -> str:
    """Cut the text at the first line that starts a signature block."""
    kept: list[str] = []
    for line in text.splitlines():
        if line.strip().lower().startswith(_SIGNATURE_STARTS):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def redact_pii(text: str) -> str:
    """Replace phone numbers and e-mail addresses with placeholders."""
    return _EMAIL_RE.sub("[EMAIL]", _PHONE_RE.sub("[PHONE]", text))


def clean_text(text: str) -> str:
    """Unescape HTML entities, remove quoted replies and signatures, redact PII, squash spaces."""
    text = html.unescape(text)
    text = strip_signature(strip_quoted_reply(text))
    text = redact_pii(text)
    return " ".join(text.split())


def pii_counts(tickets: list[Ticket]) -> dict[str, int]:
    """How many ticket bodies contain each kind of personal data or noise."""
    return {
        "phone_number": sum(bool(_PHONE_RE.search(t.body)) for t in tickets),
        "email_address": sum(bool(_EMAIL_RE.search(t.body)) for t in tickets),
        "signature": sum(strip_signature(t.body) != t.body.strip() for t in tickets),
        "quoted_reply": sum(strip_quoted_reply(t.body) != t.body.strip() for t in tickets),
        "html_entity": sum(html.unescape(t.body) != t.body for t in tickets),
    }


# --------------------------------------------------------------------------- #
# Duplicates                                                                  #
# --------------------------------------------------------------------------- #


def find_resubmissions(tickets: list[Ticket]) -> list[tuple[Ticket, Ticket]]:
    """Pairs (original, copy): same customer, same text, sent again the next day."""
    first_seen: dict[tuple[str, str], Ticket] = {}
    pairs: list[tuple[Ticket, Ticket]] = []
    for t in tickets:
        key = (t.customer_email, t.body)
        original = first_seen.get(key)
        if original is None:
            first_seen[key] = t
            continue
        next_day = (date.fromisoformat(original.created_at) + timedelta(days=1)).isoformat()
        if t.created_at == next_day:
            pairs.append((original, t))
    return pairs


def deduplicate(tickets: list[Ticket]) -> list[Ticket]:
    """Remove resubmitted copies and keep the first ticket of each pair."""
    copies = {copy.ticket_id for _, copy in find_resubmissions(tickets)}
    return [t for t in tickets if t.ticket_id not in copies]


def exact_duplicate_bodies(tickets: list[Ticket]) -> int:
    """Tickets whose body text already appeared earlier (any customer)."""
    counts = Counter(t.body for t in tickets)
    return sum(n - 1 for n in counts.values() if n > 1)


# --------------------------------------------------------------------------- #
# Splits                                                                      #
# --------------------------------------------------------------------------- #


def temporal_split(tickets: list[Ticket], cutoff: str = LAUNCH_DATE) -> tuple[list[Ticket], list[Ticket]]:
    """Train on the past, test on the future: everything before ``cutoff`` vs the rest."""
    train = [t for t in tickets if t.created_at < cutoff]
    test = [t for t in tickets if t.created_at >= cutoff]
    return train, test


def random_split(tickets: list[Ticket], test_fraction: float = 0.2, seed: int = 0) -> tuple[list[Ticket], list[Ticket]]:
    """The tempting (and misleading) split: shuffle, then cut."""
    shuffled = tickets[:]
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - test_fraction))
    return shuffled[:cut], shuffled[cut:]


@dataclass(frozen=True)
class FourWaySplit:
    """Periods used to study adaptation after the launch.

    old_train: January-July · old_test: August · new_adapt: September · new_test: October onwards.
    """

    old_train: list[Ticket]
    old_test: list[Ticket]
    new_adapt: list[Ticket]
    new_test: list[Ticket]


def four_way_split(tickets: list[Ticket]) -> FourWaySplit:
    """Split by month so that every set is strictly later than the one before it."""
    return FourWaySplit(
        old_train=[t for t in tickets if t.created_at < "2025-08-01"],
        old_test=[t for t in tickets if "2025-08-01" <= t.created_at < LAUNCH_DATE],
        new_adapt=[t for t in tickets if LAUNCH_DATE <= t.created_at < "2025-10-01"],
        new_test=[t for t in tickets if t.created_at >= "2025-10-01"],
    )


def overlap_by_text(train: list[Ticket], test: list[Ticket]) -> int:
    """Test tickets whose exact body also appears in train (a leak across the split)."""
    seen = {t.body for t in train}
    return sum(t.body in seen for t in test)
