"""A seeded, *simulated* reasoning model for the Hearthwise problems (a FakeProvider responder).

Read this first: this is NOT a language model and it does not "understand" anything. It is a small
program, built for this lab, that writes step-by-step solutions in the same format as the reference
steps of ``datasets/reasoning/problems.jsonl`` and makes three kinds of mistakes on purpose:

- **slips**: an arithmetic or comparison error in one step. The chance grows with the problem's
  difficulty (number of steps) and shrinks with the model's skill for that problem family. Later
  steps build on the wrong number, exactly like a real chain of thought.
- **traps**: a systematic misreading (counting Wi-Fi bulbs as HomeMesh devices, paying 15% instead of
  85%, dividing by kWh instead of euros...). Each problem has its own trap probability, so the same
  wrong answer can come back sample after sample.
- **lucky answers**: after a slip, the final answer is sometimes right anyway, so the written
  reasoning does not support the answer (an unfaithful chain of thought).

Because every call goes through ``FakeProvider``, the algorithms in this lab (self-consistency,
verifiers, search, STaR, RL) talk to it through the same chat interface they would use with a real
model. Calling it again with the same prompt behaves like sampling again at temperature > 0: the k-th
call with a given prompt uses its own seeded random stream, so every run prints the same numbers.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from . import prompts
from .problems import Problem, fmt, load_problems
from .providers import Message
from .puzzle import Item, apply, combine, exact, find_move, legal_moves, reachable, strip_outer

# Base model: probability of writing a step correctly, per family, at difficulty 2.
BASE_SKILL = {"order_total": 0.94, "energy": 0.95, "hub_capacity": 0.95, "target": 0.95}
# Base model: average trap probability per family (for ``target``: chance per step of a random move).
BASE_TRAP = {"order_total": 0.12, "energy": 0.12, "hub_capacity": 0.25, "target": 0.50}
FILLERS = (
    "Let me re-read that part of the question.",
    "I will keep two decimals for euro amounts.",
    "That number looks plausible, moving on.",
    "Next I need the following quantity.",
    "I should be careful with the units here.",
    "Let me write this down clearly.",
)
PLAN = "First I will work out exactly what the question asks, then compute step by step and check each step."


def stable_uniform(key: str) -> float:
    """A number in [0, 1) that depends only on ``key`` (same on every machine)."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def perturb(value: Fraction, rng: random.Random) -> Fraction:
    """A plausible slip: the result is off by a little."""
    if value.denominator == 1:
        delta = Fraction(rng.choice([-10, -2, -1, 1, 2, 10]))
    else:
        delta = Fraction(rng.choice([-100, -10, 10, 100]), 100)
    out = value + delta
    if value > 0 >= out:
        out = value + abs(delta)
    return out


@dataclass(frozen=True)
class SimConfig:
    """The knobs of the simulated model. STaR and RL produce new configs ("new weights")."""

    name: str = "base"
    skill: dict[str, float] = field(default_factory=lambda: dict(BASE_SKILL))
    trap: dict[str, float] = field(default_factory=lambda: dict(BASE_TRAP))
    growth: float = 0.5  # extra slip risk per step of difficulty above 2
    direct_factor: float = 3.0  # slips are this much likelier without written steps
    lucky: float = 0.15  # after a slip, chance that the final answer is right anyway
    hint_lucky: float = 0.5  # same, when the answer was given as a hint (rationalisation)
    hint_trap_factor: float = 0.2  # a hint makes the trap this much less likely
    filler_max: int = 3  # extra "thinking aloud" lines in a CoT sample
    check_prob: float = 0.0  # CoT mode: chance of re-checking each step (the RL toy learns this)
    reread_prob: float = 0.0  # CoT mode: chance of re-reading the question at the end (RL learns this too)
    catch_slip: float = 0.75  # chance that one check catches a slip
    catch_trap: float = 0.6  # chance that re-reading the question catches a trap
    fudge: float = 0.6  # target puzzles: chance of claiming the target when the last step misses it
    value_accuracy: float = 0.85  # ToT: chance that a judgement on 3 numbers is right
    intuition: float = 3.0  # ToT: proposals that keep the puzzle solvable are this much likelier
    self_detect_slip: float = 0.25  # self-critique: chance of spotting its own slip
    self_detect_trap: float = 0.10  # self-critique: chance of spotting its own trap
    self_false_alarm: float = 0.08  # self-critique: chance of flagging a correct step

    def slip_probability(self, p: Problem) -> float:
        return min(0.9, (1 - self.skill[p.type]) * (1 + self.growth * (p.difficulty - 2)))

    def trap_probability(self, p: Problem) -> float:
        u = stable_uniform(f"trap|{p.question}")
        # Numeric families: most problems are rarely misread, a few are misread most of the time.
        # Target puzzles: every puzzle needs some search, so the spread is flatter.
        shape = 0.5 + u if p.type == "target" else 4 * u**3
        return min(0.95, self.trap[p.type] * shape)


@dataclass(frozen=True)
class Policy:
    """How one call writes its chain: risk factors and self-checking behaviour."""

    slip_factor: float = 1.0
    trap_factor: float = 1.0
    check_prob: float = 0.0
    check_passes: int = 1
    reread: float = 0.0


THINK_POLICIES = {
    "low": Policy(check_prob=0.5),
    "medium": Policy(check_prob=1.0, reread=0.5),
    "high": Policy(check_prob=1.0, check_passes=2, reread=1.0),
}


@dataclass
class Trace:
    """Hidden decisions of the last call. Only the RL toy reads it (a real trainer uses log-probs)."""

    trap_decisions: list[tuple[str, float, bool]] = field(default_factory=list)
    check_decisions: list[tuple[float, bool]] = field(default_factory=list)
    reread_decisions: list[tuple[float, bool]] = field(default_factory=list)


@dataclass
class Chain:
    """What the simulator knows about a (partial) solution it wrote."""

    phase: int = 0
    values: dict[str, Fraction] = field(default_factory=dict)
    items: tuple[Item, ...] = ()
    kinds: tuple[str, ...] = ()  # one per written step: "ok", "slip" or "trap"

    def copy(self) -> Chain:
        return Chain(self.phase, dict(self.values), self.items, self.kinds)

    @property
    def trapped(self) -> bool:
        return "trap" in self.kinds

    @property
    def slipped(self) -> bool:
        return "slip" in self.kinds


# --------------------------------------------------------------------------- #
# Problem families: how a correct, slipped or trapped step is written          #
# --------------------------------------------------------------------------- #


class Family:
    phases: list[str]
    trap_phase: str | None = None
    no_slip: frozenset[str] = frozenset()

    def start(self) -> Chain:
        return Chain()

    def done(self, c: Chain) -> bool:
        return c.phase >= len(self.phases)

    def can_trap(self, c: Chain) -> bool:
        return self.phases[c.phase] == self.trap_phase

    def can_slip(self, c: Chain) -> bool:
        return self.phases[c.phase] not in self.no_slip

    def write(self, c: Chain, kind: str, rng: random.Random) -> tuple[str, str]:
        raise NotImplementedError

    def answer(self, c: Chain) -> str:
        raise NotImplementedError


class OrderTotal(Family):
    def __init__(self, p: Problem) -> None:
        q = p.question
        self.items = [(int(n), int(price)) for n, price in re.findall(r"(\d+) x [\w ]+? at €(\d+) each", q)]
        m = re.search(r"A (\d+)% discount", q)
        self.discount = int(m.group(1)) if m else 0
        self.threshold = int(re.search(r"more than €(\d+)", q).group(1))  # type: ignore[union-attr]
        self.fee = Fraction(re.search(r"costs €(\d+(?:\.\d+)?)", q).group(1))  # type: ignore[union-attr]
        self.phases = ["subtotal", *(["discount"] if self.discount else []), "shipping", "total"]
        self.trap_phase = "discount" if self.discount else "shipping"

    def write(self, c: Chain, kind: str, rng: random.Random) -> tuple[str, str]:
        v, phase = c.values, self.phases[c.phase]
        if phase == "subtotal":
            value = Fraction(sum(n * p for n, p in self.items))
            value = perturb(value, rng) if kind == "slip" else value
            v["subtotal"] = v["after"] = value
            return "Subtotal: " + " + ".join(f"{n} x {p}" for n, p in self.items) + f" = {fmt(value)}", kind
        if phase == "discount":  # trap: pays the discount rate instead of the rest (15% instead of 85%)
            rate = Fraction(self.discount if kind == "trap" else 100 - self.discount, 100)
            value = v["subtotal"] * rate
            value = perturb(value, rng) if kind == "slip" else value
            v["after"] = value
            return (
                f"Apply {self.discount}% discount: {fmt(v['subtotal'])} x {float(rate):.2f} = {fmt(value)}",
                kind,
            )
        if phase == "shipping":  # slip: wrong comparison; trap: rule applied the wrong way round
            over = v["after"] > self.threshold
            said_over = (not over) if kind == "slip" else over
            free = (not said_over) if kind == "trap" else said_over
            v["ship"] = Fraction(0) if free else self.fee
            word = "over" if said_over else "not over"
            return (
                f"Shipping: {fmt(v['after'])} is {word} €{self.threshold}, so shipping is {fmt(v['ship'])}",
                kind,
            )
        value = v["after"] + v["ship"]
        value = perturb(value, rng) if kind == "slip" else value
        v["total"] = value
        return f"Total: {fmt(v['after'])} + {fmt(v['ship'])} = {fmt(value)}", kind

    def answer(self, c: Chain) -> str:
        return fmt(c.values["total"])


class Energy(Family):
    def __init__(self, p: Problem) -> None:
        q = p.question
        self.kwh = int(re.search(r"uses (\d+) kWh", q).group(1))  # type: ignore[union-attr]
        self.pct = int(re.search(r"by (\d+)%", q).group(1))  # type: ignore[union-attr]
        price = re.search(r"costs €(\d+\.\d+) per kWh", q)
        assert price, p.id
        self.price = Fraction(price.group(1))
        self.payback = "pay for itself" in q
        cost = re.search(r"\(€(\d+)\)", q)
        self.cost = int(cost.group(1)) if self.payback and cost else 0
        self.phases = ["kwh", "eur", *(["payback"] if self.payback else [])]
        self.trap_phase = "payback" if self.payback else "kwh"

    def write(self, c: Chain, kind: str, rng: random.Random) -> tuple[str, str]:
        v, phase = c.values, self.phases[c.phase]
        if phase == "kwh":  # trap: "cuts by 12%" read as "keeps 88%"
            pct = 100 - self.pct if kind == "trap" else self.pct
            value = Fraction(self.kwh * pct, 100)
            v["kwh"] = perturb(value, rng) if kind == "slip" else value
            return f"Energy saved: {self.kwh} x {pct}% = {fmt(v['kwh'])} kWh", kind
        if phase == "eur":
            value = v["kwh"] * self.price
            v["eur"] = perturb(value, rng) if kind == "slip" else value
            return (
                f"Money saved: {fmt(v['kwh'])} x {float(self.price):.2f} = {fmt(v['eur'])} euros per year",
                kind,
            )
        divisor = v["kwh"] if kind == "trap" else v["eur"]  # trap: divides by kWh instead of euros
        years = Fraction(self.cost) / divisor
        years = perturb(years, rng) if kind == "slip" else years
        rounded = round(float(years), 1)
        v["years"] = Fraction(str(rounded))
        return f"Payback: {self.cost} / {fmt(divisor)} = {float(years):.3f} years, rounded to {rounded}", kind

    def answer(self, c: Chain) -> str:
        return f"{float(c.values['years']):.1f}" if self.payback else fmt(c.values["eur"])


class HubCapacity(Family):
    def __init__(self, p: Problem) -> None:
        q = p.question
        nums = re.search(
            r"has (\d+) floors with (\d+) rooms each\. Every room has (\d+) motion sensor\(s\), (\d+) door "
            r"sensor\(s\) and (\d+) bulbs, and the house has (\d+) cameras",
            q,
        )
        assert nums, p.id
        self.floors, self.rooms, self.motion, self.doors, self.bulbs, self.cameras = map(int, nums.groups())
        self.capacity = int(re.search(r"up to (\d+) HomeMesh", q).group(1))  # type: ignore[union-attr]
        self.phases = ["per_room", "rooms", "used", "wifi", "free"]
        self.trap_phase = "per_room"
        self.no_slip = frozenset({"wifi"})

    def write(self, c: Chain, kind: str, rng: random.Random) -> tuple[str, str]:
        v, phase = c.values, self.phases[c.phase]
        if phase == "per_room":  # trap: the bulbs (a distractor) are counted as HomeMesh devices
            extra = self.bulbs if kind == "trap" else 0
            value = Fraction(self.motion + self.doors + extra)
            v["per_room"] = perturb(value, rng) if kind == "slip" else value
            bulbs = f" + {self.bulbs} bulbs" if kind == "trap" else ""
            return (
                f"HomeMesh devices per room: {self.motion} motion + {self.doors} door{bulbs} "
                f"= {fmt(v['per_room'])}",
                kind,
            )
        if phase == "rooms":
            value = Fraction(self.floors * self.rooms)
            v["rooms"] = perturb(value, rng) if kind == "slip" else value
            return f"Rooms: {self.floors} floors x {self.rooms} rooms = {fmt(v['rooms'])}", kind
        if phase == "used":
            value = v["rooms"] * v["per_room"]
            v["used"] = perturb(value, rng) if kind == "slip" else value
            return f"HomeMesh devices: {fmt(v['rooms'])} x {fmt(v['per_room'])} = {fmt(v['used'])}", kind
        if phase == "wifi":
            if c.trapped:
                return "Cameras use Wi-Fi, so they do not count", kind
            return "Bulbs and cameras use Wi-Fi, so they do not count", kind
        value = self.capacity - v["used"]
        v["free"] = perturb(value, rng) if kind == "slip" else value
        return f"Free slots: {self.capacity} - {fmt(v['used'])} = {fmt(v['free'])}", kind

    def answer(self, c: Chain) -> str:
        return fmt(c.values["free"])


class TargetPuzzle(Family):
    """Reach the target with + - * /. 'trap' here means: wander off to a random move."""

    def __init__(self, p: Problem, fudge: float) -> None:
        assert p.target is not None
        self.numbers = list(p.numbers)
        self.target = Fraction(p.target)
        self.fudge = fudge
        self.phases = ["move"] * (len(self.numbers) - 1)
        self.reference: list[tuple[Fraction, str, Fraction]] = []
        for step in p.steps:
            a, op, b = step.split(" = ")[0].split(" ")
            self.reference.append((Fraction(a), op, Fraction(b)))
        states = [sorted(Fraction(n) for n in self.numbers)]
        for a, op, b in self.reference:
            vals = list(states[-1])
            vals.remove(a)
            vals.remove(b)
            out = apply(op, a, b)
            assert out is not None
            states.append(sorted([*vals, out]))
        self.ref_states = states

    def start(self) -> Chain:
        return Chain(items=tuple((Fraction(n), str(n)) for n in self.numbers))

    def can_trap(self, c: Chain) -> bool:
        return True

    def reference_move(self, items: list[Item]) -> tuple[int, str, int] | None:
        """The next move of the reference solution, if ``items`` are still on that path."""
        depth = len(self.numbers) - len(items)
        if depth >= len(self.reference) or sorted(v for v, _ in items) != self.ref_states[depth]:
            return None
        a, op, b = self.reference[depth]
        return find_move(items, a, op, b)

    def write(self, c: Chain, kind: str, rng: random.Random) -> tuple[str, str]:
        items = list(c.items)
        move = None if kind == "trap" else self.reference_move(items)
        if move is None:
            move = rng.choice(legal_moves([v for v, _ in items]))
        i, op, j = move
        a, b = items[i][0], items[j][0]
        result = apply(op, a, b)
        assert result is not None
        written = perturb(result, rng) if kind == "slip" else result
        if len(items) == 2 and written != self.target and rng.random() < self.fudge:
            written, kind = self.target, "slip"  # claims the target although the arithmetic says otherwise
        c.items = tuple(combine(items, move, written))
        return f"{fmt(a)} {op} {fmt(b)} = {fmt(written)}", kind

    def answer(self, c: Chain) -> str:
        return strip_outer(c.items[0][1])


def make_family(p: Problem, fudge: float) -> Family:
    if p.type == "order_total":
        return OrderTotal(p)
    if p.type == "energy":
        return Energy(p)
    if p.type == "hub_capacity":
        return HubCapacity(p)
    return TargetPuzzle(p, fudge)


# --------------------------------------------------------------------------- #
# Message parsing                                                             #
# --------------------------------------------------------------------------- #

_STEP_RE = re.compile(r"^Step (\d+): (.*)$", re.M)


def _task(system: str) -> str:
    table = {
        prompts.SOLVE_DIRECT: "direct",
        prompts.SOLVE_COT: "cot",
        prompts.NEXT_STEP: "next",
        prompts.REVISE: "revise",
        prompts.CRITIQUE: "critique",
        prompts.PROPOSE: "propose",
        prompts.VALUE: "value",
    }
    if system in table:
        return table[system]
    if system.startswith(prompts.SOLVE_THINK.split("{")[0]):
        return "think"
    raise ValueError("The simulator does not recognise this system prompt (see prompts.py).")


def _steps_in(text: str) -> list[str]:
    return [m.group(2) for m in _STEP_RE.finditer(text)]


def _numbers_left(user: str) -> list[Fraction]:
    m = re.search(r"^Numbers left: (.*)$", user, re.M)
    if not m:
        raise ValueError("expected a 'Numbers left:' line")
    return [Fraction(tok) for tok in m.group(1).split()]


# --------------------------------------------------------------------------- #
# The responder                                                               #
# --------------------------------------------------------------------------- #


class Draws:
    """Named random streams for one call.

    Each decision has its own stream (step 3's slip draw, the "lucky" draw...). Two modes asked about
    the same problem therefore face the same luck, step by step ("common random numbers"), so their
    accuracy differences come from the method and not from noise.
    """

    def __init__(self, base: str) -> None:
        self.base = base

    def __call__(self, name: str) -> random.Random:
        return random.Random(f"{self.base}|{name}")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class SimulatedReasoner:
    """Pass an instance as ``FakeProvider(responder=...)``. See the module docstring."""

    def __init__(
        self, config: SimConfig | None = None, *, seed: int = 0, problems: Sequence[Problem] | None = None
    ) -> None:
        self.config = config or SimConfig()
        self.seed = seed
        self._by_question = {p.question: p for p in (problems or load_problems())}
        # Keyed by question text: two dataset rows with the same question are the same prompt to a model.
        self._families: dict[str, Family] = {}
        self._chains: dict[tuple[str, tuple[str, ...]], Chain] = {}
        self._seen: Counter[str] = Counter()
        self.last_trace = Trace()

    def __call__(self, messages: Sequence[Message]) -> str:
        system = messages[0].content if messages[0].role == "system" else ""
        user = messages[-1].content
        task = _task(system)
        prompt_key = _digest(f"{system}\x00{user}")
        index = self._seen[prompt_key]  # the k-th call with this exact prompt is the k-th sample
        self._seen[prompt_key] += 1
        group = "solve" if task in ("direct", "cot", "think") else task
        draws = Draws(f"{self.seed}|{group}|{_digest(user)}|{index}")
        self.last_trace = Trace()
        match = re.match(r"Problem: (.*)", user)
        if not match or match.group(1) not in self._by_question:
            raise ValueError("The simulator only knows the problems in datasets/reasoning/problems.jsonl.")
        p = self._by_question[match.group(1)]
        if task == "direct":
            return self._direct(p, draws)
        if task == "cot":
            return self._cot(p, draws, hint="\nHint:" in user)
        if task == "think":
            effort = re.search(r"Reasoning effort: (\w+)", system).group(1)  # type: ignore[union-attr]
            return self._think(p, draws, effort)
        if task == "next":
            return self._next(p, _steps_in(user), draws)
        if task == "revise":
            feedback = re.search(r"^Feedback: (.*)$", user, re.M).group(1)  # type: ignore[union-attr]
            return self._revise(p, _steps_in(user), feedback, draws)
        if task == "critique":
            return self._critique(p, _steps_in(user), draws("critique"))
        if task == "propose":
            wanted = re.search(r"^Moves wanted: (\d+)$", user, re.M)
            return self._propose(
                p, _numbers_left(user), int(wanted.group(1)) if wanted else 5, draws("propose")
            )
        return self._value(p, _numbers_left(user), draws("value"))

    # -- bookkeeping ----------------------------------------------------------

    def family(self, p: Problem) -> Family:
        if p.question not in self._families:
            self._families[p.question] = make_family(p, self.config.fudge)
        return self._families[p.question]

    def _recall(self, p: Problem, steps: Sequence[str]) -> Chain:
        if not steps:
            return self.family(p).start()
        try:
            return self._chains[(p.question, tuple(steps))]
        except KeyError:
            raise ValueError("The simulator can only continue solutions that it wrote itself.") from None

    def chain_of(self, p: Problem, steps: Sequence[str]) -> Chain:
        """Read-only access to what the simulator knows about a solution it wrote (tests, demos)."""
        return self._recall(p, steps)

    # -- writing --------------------------------------------------------------

    def _step(
        self, p: Problem, chain: Chain, steps: Sequence[str], rng: random.Random, slip_p: float, trap_p: float
    ) -> tuple[str, Chain]:
        """Write one step. Both draws are always made so that streams stay aligned across modes."""
        fam = self.family(p)
        u_trap, u_slip = rng.random(), rng.random()
        kind = "ok"
        if fam.can_trap(chain):
            took = u_trap < trap_p
            self.last_trace.trap_decisions.append((p.type, trap_p, took))
            kind = "trap" if took else kind
        if kind == "ok" and fam.can_slip(chain) and u_slip < slip_p:
            kind = "slip"
        new = chain.copy()
        text, kind = fam.write(new, kind, rng)
        new.phase += 1
        new.kinds = (*new.kinds, kind)
        self._chains[(p.question, (*steps, text))] = new
        return text, new

    def _write(
        self,
        p: Problem,
        draws: Draws,
        policy: Policy,
        *,
        start: Sequence[str] = (),
        first_slip_factor: float = 1.0,
        first_trap_factor: float = 1.0,
    ) -> tuple[list[str], list[str], Chain]:
        """Write a chain (with optional self-checks) from ``start`` to the end."""
        cfg, fam = self.config, self.family(p)
        slip_p = min(0.95, cfg.slip_probability(p) * policy.slip_factor)
        trap_p = min(0.95, cfg.trap_probability(p) * policy.trap_factor)
        steps = list(start)
        chains = [self._recall(p, steps[:i]) for i in range(len(steps) + 1)]
        lines: list[str] = []
        written: Counter[int] = Counter()
        first = [True]

        def step_rng(k: int) -> random.Random:
            written[k] += 1
            return draws(f"s{k}|{written[k] - 1}")

        def run(trap_override: float | None = None) -> None:
            while not fam.done(chains[-1]):
                k = len(steps) + 1
                s_p = slip_p * (first_slip_factor if first[0] else 1.0)
                t_p = (
                    trap_p * (first_trap_factor if first[0] else 1.0)
                    if trap_override is None
                    else trap_override
                )
                first[0] = False
                text, chain = self._step(p, chains[-1], steps, step_rng(k), s_p, t_p)
                lines.append(f"Step {k}: {text}")
                for n in range(policy.check_passes if policy.check_prob > 0 else 0):
                    check = draws(f"c{k}|{n}|{written[k]}")
                    did, caught = check.random() < policy.check_prob, check.random() < cfg.catch_slip
                    self.last_trace.check_decisions.append((policy.check_prob, did))
                    if not did:
                        continue
                    if chain.kinds[-1] == "slip" and caught:
                        lines.append(f"Wait, step {k} does not add up. Redoing it.")
                        text, chain = self._step(p, chains[-1], steps, step_rng(k), slip_p, 0.0)
                        lines.append(f"Step {k}: {text}")
                    else:
                        lines.append(f"Check: step {k} is consistent.")
                steps.append(text)
                chains.append(chain)

        run()
        reread = draws("reread")
        did_reread = reread.random() < policy.reread
        if policy.reread > 0 and p.is_numeric:
            self.last_trace.reread_decisions.append((policy.reread, did_reread))
        if did_reread and p.is_numeric:
            final = chains[-1]
            if final.trapped and reread.random() < cfg.catch_trap:
                j = final.kinds.index("trap")
                lines.append(f"Re-reading the question: step {j + 1} misreads it. Redoing from step {j + 1}.")
                del steps[j:]
                del chains[j + 1 :]
                run(trap_override=0.0)
            else:
                lines.append("Re-reading the question: the steps match what it asks.")
        return lines, steps, chains[-1]

    def _answer(self, p: Problem, chain: Chain, draws: Draws, lucky: float) -> str:
        if p.is_numeric and chain.slipped and not chain.trapped and draws("lucky").random() < lucky:
            return p.answer  # unfaithful: the steps do not support this answer
        return self.family(p).answer(chain)

    # -- tasks ----------------------------------------------------------------

    def _direct(self, p: Problem, draws: Draws) -> str:
        # Without written steps, slips are likelier and (for puzzles) so are bad moves.
        wander = self.config.direct_factor / 2 if p.type == "target" else 1.0
        _, _, chain = self._write(p, draws, Policy(slip_factor=self.config.direct_factor, trap_factor=wander))
        return f"Answer: {self._answer(p, chain, draws, self.config.lucky)}"

    def _cot(self, p: Problem, draws: Draws, hint: bool) -> str:
        cfg = self.config
        policy = Policy(
            check_prob=cfg.check_prob,
            reread=cfg.reread_prob,
            trap_factor=cfg.hint_trap_factor if hint else 1.0,
        )
        lines, _, chain = self._write(p, draws, policy)
        answer = self._answer(p, chain, draws, cfg.hint_lucky if hint else cfg.lucky)
        filler = draws("filler")
        unsure = 2 if chain.slipped or chain.trapped else 0  # a chain that went wrong tends to ramble
        for _ in range(filler.randint(0, cfg.filler_max) + unsure):
            lines.insert(filler.randint(0, len(lines)), filler.choice(FILLERS))
        return "\n".join([*lines, f"Answer: {answer}"])

    def _think(self, p: Problem, draws: Draws, effort: str) -> str:
        lines, steps, chain = self._write(p, draws, THINK_POLICIES[effort])
        answer = self._answer(p, chain, draws, self.config.lucky)
        visible = "\n".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))
        return "\n".join(["<think>", PLAN, *lines, "</think>", visible, f"Answer: {answer}"])

    def _next(self, p: Problem, steps: list[str], draws: Draws) -> str:
        chain = self._recall(p, steps)
        if self.family(p).done(chain):
            return f"Answer: {self._answer(p, chain, draws, self.config.lucky)}"
        cfg, k = self.config, len(steps) + 1
        text, _ = self._step(
            p, chain, steps, draws(f"s{k}|0"), cfg.slip_probability(p), cfg.trap_probability(p)
        )
        return f"Step {k}: {text}"

    def _revise(self, p: Problem, steps: list[str], feedback: str, draws: Draws) -> str:
        m = re.search(r"Step (\d+) is wrong", feedback)
        if m:  # feedback names a step: keep what came before, redo from there with extra care
            k = max(1, min(int(m.group(1)), len(steps)))
            _, new_steps, chain = self._write(
                p, draws, Policy(), start=steps[: k - 1], first_slip_factor=0.3, first_trap_factor=0.2
            )
        else:  # outcome-only feedback: start again, a bit more wary of misreading
            _, new_steps, chain = self._write(p, draws, Policy(trap_factor=0.6))
        answer = self._answer(p, chain, draws, self.config.lucky)
        return prompts.solution_block(new_steps, answer)

    def _critique(self, p: Problem, steps: list[str], rng: random.Random) -> str:
        cfg = self.config
        chain = self._recall(p, steps)
        for k, kind in enumerate(chain.kinds, 1):
            prob = {"slip": cfg.self_detect_slip, "trap": cfg.self_detect_trap}.get(
                kind, cfg.self_false_alarm
            )
            if rng.random() < prob:
                return f"Step {k} is wrong: it does not look right to me."
        return "All steps are correct."

    def _propose(self, p: Problem, values: list[Fraction], k: int, rng: random.Random) -> str:
        fam = self.family(p)
        assert isinstance(fam, TargetPuzzle)
        cfg = self.config
        items: list[Item] = [(v, exact(v)) for v in values]
        unique: dict[tuple[Fraction, ...], tuple[int, str, int]] = {}
        for move in legal_moves(values):
            after = tuple(sorted(v for v, _ in combine(items, move)))
            unique.setdefault(after, move)
        pool = list(unique.items())
        chosen: list[tuple[int, str, int]] = []
        ref = fam.reference_move(items)
        if ref is not None and rng.random() >= cfg.trap_probability(p):
            chosen.append(ref)
            ref_after = tuple(sorted(v for v, _ in combine(items, ref)))
            pool = [(after, m) for after, m in pool if after != ref_after]
        weights = [cfg.intuition if reachable(list(after), fam.target) else 1.0 for after, _ in pool]
        while len(chosen) < k and pool:
            pick = rng.random() * sum(weights)
            idx = 0
            while idx < len(weights) - 1 and pick >= weights[idx]:
                pick -= weights[idx]
                idx += 1
            chosen.append(pool.pop(idx)[1])
            weights.pop(idx)
        if ref in chosen:  # put the "good idea" somewhere in the list, not always first
            chosen.remove(ref)
            chosen.insert(rng.randint(0, len(chosen)), ref)  # type: ignore[arg-type]
        slip_p = cfg.slip_probability(p)
        lines = []
        for i, op, j in chosen:
            a, b = values[i], values[j]
            out = apply(op, a, b)
            assert out is not None
            written = perturb(out, rng) if rng.random() < slip_p else out
            lines.append(f"{exact(a)} {op} {exact(b)} = {exact(written)}")
        return "\n".join(lines)

    def _value(self, p: Problem, values: list[Fraction], rng: random.Random) -> str:
        target = Fraction(p.target or 0)
        if len(values) == 1:
            return "sure" if values[0] == target else "impossible"
        solvable = reachable(values, target)
        accuracy = 0.97 if len(values) == 2 else self.config.value_accuracy
        verdict = solvable if rng.random() < accuracy else not solvable
        if not verdict:
            return "impossible"
        return "sure" if len(values) == 2 else "likely"
