"""Your turn: five small functions that close real gaps in this agent.

Each one raises ``NotImplementedError`` until you write it. Their tests in
``tests/test_exercises.py`` are skipped until then, so ``uv run pytest`` stays
green while you work. Solutions are in ``lab/README.md`` (hidden in <details>).
"""

from __future__ import annotations

from .agents import Result, Step
from .skills import Fact


def mask_street_addresses(text: str) -> str:
    """Replace street addresses such as 'Calle Falsa 123' with '[address]'.

    Noor found that the sanitiser masks e-mails and phone numbers but not street
    addresses (see the 'holiday-lights-automation' forum page). Handle at least the
    pattern '<Calle|Avenida|Street|Road> <Name> <number>'.
    """
    raise NotImplementedError


def repeated_action(steps: list[Step]) -> bool:
    """Stop condition: True when the last two steps called the same tool with the same argument.

    An agent that repeats itself is stuck; the loop should stop early instead of
    spending its whole budget.
    """
    raise NotImplementedError


def vote_by_domain(facts: list[Fact]) -> Fact | None:
    """Majority vote where each *domain* gets one vote, so copies cannot outvote originals.

    With ``--naive``, a forum page and its ``?utm_`` copy 'agree' on a wrong number and
    outvote the official specification. Count at most one vote per domain; break ties
    by the earliest-opened fact (smallest ``order``). Return the earliest-opened fact
    of the winning value, or None for an empty list.
    """
    raise NotImplementedError


def citation_recall(result: Result, gold_urls: list[str]) -> float:
    """Share of gold URLs the answer cites (1.0 when there are no gold URLs).

    Precision asks 'is every citation good?'; recall asks 'did we cite the sources
    the answer depends on?'. Compare URLs after removing ``utm_`` parameters.
    """
    raise NotImplementedError


def cost_per_success(results: list[Result], successes: list[bool]) -> float:
    """Total tokens spent divided by the number of successful tasks (inf if none succeeded).

    Sam's favourite number: it charges an agent for its failures too.
    """
    raise NotImplementedError
