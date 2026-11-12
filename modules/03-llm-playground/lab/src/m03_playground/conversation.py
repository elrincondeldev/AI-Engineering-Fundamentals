"""Conversation state, token budgets and truncation policies.

A chat model is stateless: every request must carry the whole conversation the
model should "remember". This file keeps that state and decides what to send
when it no longer fits.

Token counts use ``count_tokens`` from ``providers.py`` (words and punctuation)
plus 4 tokens of per-message overhead, exactly like ``FakeProvider``. Real
tokenizers give different numbers; the *shape* of the growth is what matters.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .providers import Message, count_tokens

MESSAGE_OVERHEAD = 4  # role markers and separators added by the chat template

SUPPORT_SYSTEM = (
    "You are the Hearthwise support assistant. Answer in the customer's language, in at most three "
    "sentences. Use only current Hearthwise policies: returns within 30 days with no restocking fee and a "
    "2-year warranty (3 years with Hearthwise Plus). Never ask for passwords or card numbers. If you are "
    "not sure, offer to hand the conversation to a human agent."
)

KEY_FACT = "#HW-640183"  # mentioned once, in the first user turn; needed again in the last one

# A recorded conversation from a playground session (user turn, assistant turn). Replaying a
# recording means the only thing that changes between policies is the prompt we build.
SCRIPT: list[tuple[str, str]] = [
    (
        "Hi, I ordered an Ember T3 and a Hearth Hub last week, order #HW-640183. The tracking page has not "
        "changed for three days. Can you check it?",
        "Thanks for the order number. Standard delivery in Spain takes 2 to 4 business days, so a pause of "
        "three days usually means the parcel is waiting at a depot. I have asked the courier for an update "
        "and you will get an email as soon as it moves.",
    ),
    (
        "OK. While I wait, does the Ember T3 need a C-wire?",
        "The Ember T3 is designed for a 24 V wired system with a C-wire, and it has two AA batteries as a "
        "backup. If you are unsure about your wiring, a human agent can check photos before you install it.",
    ),
    (
        "I think I have a C-wire. Which app do I use to set it up?",
        "Use the Hearthwise app. Add the thermostat under Devices, follow the wiring steps, and the Ember T3 "
        "will learn your schedule in about 7 days.",
    ),
    (
        "Can I also connect two Watcher S1 motion sensors?",
        "Yes. The Watcher S1 talks HomeMesh, not Wi-Fi, so it needs the Hearth Hub. One hub connects up to "
        "64 HomeMesh devices, so two sensors are no problem.",
    ),
    (
        "How long does the battery in the Watcher S1 last?",
        "The Watcher S1 uses a CR2450 coin battery that lasts about 18 months with the default settings.",
    ),
    (
        "Good. Can the Hearth Hub run automations if my internet goes down?",
        "Yes. The Hearth Hub runs automations locally, so your sensors and lights keep working together "
        "even without internet.",
    ),
    (
        "Should I plug the hub into the router with a cable or use Wi-Fi?",
        "Both work, but the Ethernet port gives a more stable connection. If you use Wi-Fi, remember that "
        "the hub only supports 2.4 GHz.",
    ),
    (
        "What does Hearthwise Plus add for these devices?",
        "Hearthwise Plus costs €4.99 per month or €49 per year. It adds one extra year of warranty and 30 "
        "days of cloud video history for cameras.",
    ),
    (
        "I don't have a camera yet. Is the warranty without Plus enough?",
        "Every Hearthwise device has a 2-year warranty from the delivery date. Plus extends it to 3 years, "
        "so it is worth it only if you value the extra year or plan to buy a camera.",
    ),
    (
        "Speaking of cameras, does the Lookout C2 work on 5 GHz Wi-Fi?",
        "Yes. The Lookout C2 is the first Hearthwise device that supports both 2.4 GHz and 5 GHz Wi-Fi. It "
        "records 2K video with night vision up to 8 metres.",
    ),
    (
        "Can it record to a memory card instead of the cloud?",
        "Yes, the Lookout C2 records locally to a microSD card of up to 256 GB. Cloud video history needs "
        "Hearthwise Plus.",
    ),
    (
        "If I buy the camera and don't like it, can I return it?",
        "Yes. You can return any Hearthwise product within 30 days of delivery, opened or not, with no "
        "restocking fee. Start the return in the app under Account > Orders.",
    ),
    (
        "How long does the refund take?",
        "We refund the original payment method within 5 to 7 business days after the item reaches our "
        "warehouse. You get an email at each step.",
    ),
    (
        "Do I need to reset the camera before sending it back?",
        "Yes. Remove it from your account and reset it to factory settings: hold the reset button under the "
        "base for 10 seconds until you hear a chime.",
    ),
    (
        "And how do I reset the Hearth Hub if I ever need to?",
        "Hold the button on the back of the hub for 20 seconds until the ring light turns orange.",
    ),
    (
        "Is there express delivery for the camera?",
        "Yes. Express delivery takes 1 business day and costs €6.95 for orders placed before 14:00. "
        "Standard delivery is free for orders over €49.",
    ),
    ("The camera costs €89, right?", "Yes, the Lookout C2 costs €89, so standard delivery would be free."),
    (
        "Can the Spark Mini handle a 2000 W heater?",
        "Yes. The Spark Mini handles a maximum load of 3680 W (16 A at 230 V), so a 2000 W heater is within "
        "its limit.",
    ),
    ("Great, thanks for all the help today.", "You are welcome. Is there anything else I can help you with?"),
    (
        "Yes, one last thing: can you check the delivery status of my first order again?",
        "Of course. I will check the courier status for that order now.",
    ),
]

PRODUCT_NAMES = [
    "Ember T3",
    "Glow A19",
    "Spark Mini",
    "Watcher S1",
    "Lookout C2",
    "Hearth Hub",
    "Hearthwise Plus",
]
ORDER_RE = re.compile(r"#HW-\d{6}", re.IGNORECASE)


def message_tokens(message: Message) -> int:
    """Approximate tokens for one message, overhead included."""
    return count_tokens(message.content) + MESSAGE_OVERHEAD


def prompt_tokens(messages: Sequence[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


# --------------------------------------------------------------------------- #
# Truncation policies                                                         #
# --------------------------------------------------------------------------- #
# Each policy gets the system message, the earlier turns and the new user message,
# and returns the messages to send. ``budget`` is the input-token budget:
# context window minus the tokens reserved for the answer (max_tokens).

Policy = Callable[[Message, Sequence[Message], Message, int], list[Message]]


def keep_all(system: Message, history: Sequence[Message], user: Message, budget: int) -> list[Message]:
    """Send everything. Simple, until the conversation outgrows the context window."""
    return [system, *history, user]


def sliding_window(
    system: Message, history: Sequence[Message], user: Message, budget: int, last_n: int = 6
) -> list[Message]:
    """Keep the last ``last_n`` messages of the whole list. The system prompt falls out first!"""
    return [system, *history, user][-last_n:]


def _recent_that_fit(history: Sequence[Message], room: int) -> list[Message]:
    """Newest (user, assistant) pairs that fit in ``room`` tokens, oldest first."""
    kept: list[Message] = []
    used = 0
    pairs = [list(history[i : i + 2]) for i in range(0, len(history), 2)]
    for pair in reversed(pairs):
        cost = prompt_tokens(pair)
        if used + cost > room:
            break
        kept[:0] = pair
        used += cost
    return kept


def pin_system(system: Message, history: Sequence[Message], user: Message, budget: int) -> list[Message]:
    """Always keep the system prompt and the new message; fill the rest with the newest turns."""
    room = budget - message_tokens(system) - message_tokens(user)
    return [system, *_recent_that_fit(history, room), user]


def summarise_turns(messages: Sequence[Message], max_tokens: int = 60) -> str:
    """Deterministic extractive summary: hard facts first, then what the customer asked.

    In production a (cheaper) model writes this summary. This stand-in keeps order
    numbers and product names, which are exactly the facts a model tends to need later.
    """
    users = [m.content for m in messages if m.role == "user"]
    everything = " ".join(m.content for m in messages)
    orders = list(dict.fromkeys(o.upper() for o in ORDER_RE.findall(everything)))
    products = [p for p in PRODUCT_NAMES if p.lower() in everything.lower()]
    facts = []
    if orders:
        facts.append("order " + ", ".join(orders))
    if products:
        facts.append("products " + ", ".join(products))
    asks = [" ".join(re.split(r"(?<=[.?!])\s", u)[-1].split()[:10]) for u in users]
    while True:
        text = "Summary of earlier turns: " + "; ".join(facts) + ". Customer asked: " + " | ".join(asks)
        if count_tokens(text) <= max_tokens or not asks:
            return text
        asks.pop(0)  # forget the oldest questions first, never the hard facts


def summarise_older(system: Message, history: Sequence[Message], user: Message, budget: int) -> list[Message]:
    """Pin the system prompt, keep recent turns verbatim and replace older ones with a summary."""
    room = budget - message_tokens(system) - message_tokens(user)
    recent = _recent_that_fit(history, room)
    older = list(history[: len(history) - len(recent)])
    if not older:
        return [system, *recent, user]
    summary = Message("system", summarise_turns(older))
    recent = _recent_that_fit(history, room - message_tokens(summary))
    older = list(history[: len(history) - len(recent)])
    summary = Message("system", summarise_turns(older))
    return [system, summary, *recent, user]


POLICIES: dict[str, Policy] = {
    "keep_all": keep_all,
    "sliding_window": sliding_window,
    "pin_system": pin_system,
    "summarise": summarise_older,
}


# --------------------------------------------------------------------------- #
# Conversation state for the chat REPL                                        #
# --------------------------------------------------------------------------- #


@dataclass
class Conversation:
    """The state one chat session keeps between requests."""

    system: str = SUPPORT_SYSTEM
    history: list[Message] = field(default_factory=list)

    def build(self, user_text: str, *, policy: str = "pin_system", budget: int = 768) -> list[Message]:
        return POLICIES[policy](
            Message("system", self.system), self.history, Message("user", user_text), budget
        )

    def record(self, user_text: str, assistant_text: str) -> None:
        self.history += [Message("user", user_text), Message("assistant", assistant_text)]

    def reset(self) -> None:
        self.history.clear()


# --------------------------------------------------------------------------- #
# Budget experiment (``uv run lab budget``)                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GrowthRow:
    turn: int
    prompt_tokens: int  # what keep_all sends at this turn
    cumulative: int  # sum of prompt tokens sent so far
    stateless: int  # system + this user message only (no memory at all)


def growth(script: Sequence[tuple[str, str]] = SCRIPT, system: str = SUPPORT_SYSTEM) -> list[GrowthRow]:
    """Prompt tokens per turn when the full history is resent every time."""
    rows: list[GrowthRow] = []
    history: list[Message] = []
    total = 0
    sys_msg = Message("system", system)
    for turn, (user_text, assistant_text) in enumerate(script, 1):
        user = Message("user", user_text)
        sent = prompt_tokens(keep_all(sys_msg, history, user, 0))
        total += sent
        rows.append(GrowthRow(turn, sent, total, prompt_tokens([sys_msg, user])))
        history += [user, Message("assistant", assistant_text)]
    return rows


@dataclass(frozen=True)
class PolicyStats:
    policy: str
    total_prompt_tokens: int
    peak_prompt_tokens: int
    over_budget_turns: int  # requests that would fail with a context-length error
    turns_without_system: int
    remembers_key_fact: bool  # is KEY_FACT still in the prompt of the last turn?


def run_policy(
    name: str, budget: int = 768, script: Sequence[tuple[str, str]] = SCRIPT, system: str = SUPPORT_SYSTEM
) -> PolicyStats:
    """Replay the script with one policy and measure what it sends."""
    policy = POLICIES[name]
    sys_msg = Message("system", system)
    history: list[Message] = []
    sizes: list[int] = []
    over = missing_system = 0
    last: list[Message] = []
    for user_text, assistant_text in script:
        user = Message("user", user_text)
        last = policy(sys_msg, history, user, budget)
        size = prompt_tokens(last)
        sizes.append(size)
        over += size > budget
        missing_system += not any(m.role == "system" and m.content == system for m in last)
        history += [user, Message("assistant", assistant_text)]
    return PolicyStats(
        policy=name,
        total_prompt_tokens=sum(sizes),
        peak_prompt_tokens=max(sizes),
        over_budget_turns=over,
        turns_without_system=missing_system,
        remembers_key_fact=any(KEY_FACT in m.content for m in last),
    )
