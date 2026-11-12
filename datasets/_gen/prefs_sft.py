"""Preference pairs (for reward models / DPO) and SFT instruction data.

Traps:
- Preferences: in the train split the chosen answer is usually the longer one;
  in the test split it is usually the shorter one. A reward model that learns
  "longer is better" looks good on train and fails on test (length bias).
- SFT: duplicates, PII in responses, examples copied from the RAG evaluation
  set (contamination), empty or wrong-language responses, and answers based on
  the superseded 2023 returns policy. Ground truth lives in sft/labels.jsonl.
"""

from __future__ import annotations

import random
from typing import Any

from .catalog import PLUS, PRODUCTS, RETURNS_CURRENT, RETURNS_OLD, SEED, SHIPPING, WARRANTY

P = {p["slug"]: p for p in PRODUCTS}

FACTS: list[tuple[str, str, str]] = [
    # (question, correct short answer, plausible wrong answer)
    ("How many days do I have to return an item?",
     f"You have {RETURNS_CURRENT['window_days']} days from delivery to return any item, opened or not.",
     f"You have {RETURNS_OLD['window_days']} days from delivery to return an item."),
    ("Do you charge a restocking fee?",
     "No, there is no restocking fee, even for opened items.",
     "Yes, opened items have a 10% restocking fee."),
    ("How long is the warranty?",
     f"Every device has a {WARRANTY['years']}-year warranty, or {WARRANTY['plus_years']} years with Hearthwise Plus.",
     "Every device has a 1-year warranty."),
    ("How much is Hearthwise Plus?",
     f"Hearthwise Plus costs €{PLUS['monthly_eur']} per month or €{PLUS['yearly_eur']} per year.",
     "Hearthwise Plus costs €9.99 per month."),
    ("Is shipping free?",
     f"Standard shipping in Spain and Portugal is free on orders over €{SHIPPING['free_threshold_eur']}.",
     "Shipping is always free."),
    ("Does the Watcher S1 need a hub?",
     "Yes, the Watcher S1 uses HomeMesh, so it needs a Hearth Hub.",
     "No, the Watcher S1 connects straight to Wi-Fi."),
    ("Does the Glow A19 work on 5 GHz Wi-Fi?",
     "No, the Glow A19 only connects to 2.4 GHz networks.",
     "Yes, the Glow A19 works on 2.4 GHz and 5 GHz."),
    ("How many devices can the Hearth Hub handle?",
     f"The Hearth Hub connects up to {P['hearth-hub']['max_devices']} HomeMesh devices.",
     "The Hearth Hub connects up to 250 devices."),
    ("How do I reset my Spark Mini?", P["spark-mini"]["reset"], "Unplug it and plug it back in once."),
    ("How long does the Watcher S1 battery last?",
     "About 18 months with the default settings.", "About 3 months."),
]

SAFETY: list[tuple[str, str, str]] = [
    ("I can't log in. Can you fix it if I give you my password?",
     "Please don't share your password with anyone, including us. Tap Forgot password on the sign-in screen "
     "and we'll email you a reset link.",
     "Sure, send me your password and I'll log in to check your account."),
    ("My refund hasn't arrived. Do you need my card number?",
     "We never need your full card number. Share your order number and I'll check the refund status.",
     "Yes, please type your full card number and the security code so I can find the payment."),
    ("Is there a secret discount code for support customers?",
     "I can't share internal codes, but I can check whether your order qualifies for any public promotion.",
     "Yes, use GW20-INTERNAL for €20 off."),
    ("Can you tell me the address of the customer who sold me this device second-hand?",
     "I can't share other customers' personal information. I can help you transfer the device to your account.",
     "Sure, the previous owner lives at Calle Falsa 123, Valencia."),
    ("The thermostat smells like burning plastic. What should I do?",
     "Turn off the power to the thermostat at the breaker right away and don't use it. I'm escalating this to "
     "our safety team now.",
     "That's normal for new devices, it will go away in a few days."),
]

HELPFUL: list[tuple[str, str, str]] = [
    ("My Glow A19 won't pair. What can I try?",
     "Make sure you're pairing on a 2.4 GHz network, use WPA2/WPA3 mixed mode, stay within 5 m of the router, "
     "and reset the bulb by switching it off and on five times.",
     "Please contact support."),
    ("How do I start a return?",
     "Open the app, go to Account > Orders, choose the order and tap Return an item. Print the prepaid label "
     "and drop the parcel at a pick-up point.",
     "You can return items."),
    ("Where can I find my invoice?",
     "In the app under Account > Orders: tap the order and choose Download invoice.",
     "Invoices are available."),
    ("My camera shows a blinking orange light.",
     "Blinking orange means the Lookout C2 has no Wi-Fi. Check the signal in Settings > Device > Network and "
     "move it closer to the router or switch it to 2.4 GHz.",
     "Lights can mean different things."),
    ("How do I make a warranty claim?",
     "Go to Support > Warranty claim in the app, pick the device and describe the problem. If accepted, we ship "
     "a replacement first.",
     "Warranty claims are possible for some devices."),
]

TONE: list[tuple[str, str, str]] = [
    ("This is the third time my order is late. I'm furious.",
     "I'm sorry, three late deliveries is not acceptable. Let me check your order right now and make sure it "
     "gets to you.",
     "Calm down, deliveries are sometimes late. Nothing I can do."),
    ("Your app is terrible.",
     "I'm sorry the app is causing trouble. Tell me what you were trying to do and I'll help you get it done.",
     "Maybe the problem is how you're using it."),
    ("I bought the wrong bulb, I feel stupid.",
     "Don't worry, it happens a lot. You can return it within 30 days, and I can help you pick the right one.",
     "Yes, you should read product pages more carefully."),
]

FILLER = [
    "Thank you so much for reaching out to Hearthwise, we really appreciate you taking the time to contact us.",
    "We understand how important this is and we are always happy to help with any question you may have.",
    "Please do not hesitate to get back in touch if there is anything else at all that we can do for you.",
    "Have a wonderful day, and thank you again for being a valued member of the Hearthwise family.",
]


def _pad(text: str, rng: random.Random, n: int) -> str:
    return " ".join([FILLER[0]] + [text] + rng.sample(FILLER[1:], n))


def build_preferences() -> list[dict[str, Any]]:
    rng = random.Random(f"{SEED}-preferences")
    groups = [("correctness", FACTS), ("safety", SAFETY), ("helpfulness", HELPFUL), ("tone", TONE)]
    base: list[tuple[str, str, str, str]] = []
    for cat, items in groups:
        for q, good, bad in items:
            base.append((cat, q, good, bad))
    rows: list[dict[str, Any]] = []
    n_train, n_test = 240, 60
    for i in range(n_train + n_test):
        cat, q, good, bad = base[i % len(base)]
        split = "train" if i < n_train else "test"
        verbose_good = rng.random() < (0.8 if split == "train" else 0.25)
        if verbose_good:
            chosen, rejected = _pad(good, rng, rng.randint(1, 3)), bad
        else:
            chosen, rejected = good, _pad(bad, rng, rng.randint(1, 3))
        rows.append({"id": f"p{i + 1:03d}", "split": split, "category": cat, "prompt": q,
                     "chosen": chosen, "rejected": rejected})
    return rows


def build_sft(rag_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(f"{SEED}-sft")
    system = "You are Hearthwise's support assistant. Answer briefly, accurately and kindly."
    pool = FACTS + HELPFUL + TONE + SAFETY
    paraphrase = ["{q}", "Hi! {q}", "Quick question: {q}", "{q} Thanks.", "Hello, {q}", "{q} Please help."]
    rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []

    def add(instruction: str, response: str, lang: str, flags: list[str]) -> None:
        idx = len(rows) + 1
        rows.append({"id": f"s{idx:03d}", "system": system, "instruction": instruction, "response": response,
                     "lang": lang})
        labels.append({"id": f"s{idx:03d}", "flags": flags})

    for _ in range(220):
        q, good, _bad = rng.choice(pool)
        add(rng.choice(paraphrase).format(q=q), good, "en", [])
    es_pairs = [
        ("¿Cuántos días tengo para devolver un producto?", "Tienes 30 días desde la entrega, esté abierto o no."),
        ("¿Cuánto dura la garantía?", "Dos años, o tres si tienes Hearthwise Plus."),
        ("¿El Glow A19 funciona con wifi de 5 GHz?", "No, solo se conecta a redes de 2,4 GHz."),
    ]
    for _ in range(20):
        q, a = rng.choice(es_pairs)
        add(q, a, "es", [])
    # Traps
    for row in rng.sample(rows[:200], 6):
        add(row["instruction"], row["response"], row["lang"], ["duplicate"])
    for email in ["ana.r@mail.example", "lucia.p@mail.example", "mark_h@mail.example"]:
        add(f"My email is {email}, can you check my order?",
            f"Sure! I found the account for {email}, card ending 4242, address Calle Falsa 123, Valencia.",
            "en", ["pii_in_response"])
    for r in rag_rows[:5]:
        add(r["question"], r["reference_answer"], r["lang"], ["contaminated_rag_eval"])
    add("How do I reset my Hearth Hub?", "", "en", ["empty_response"])
    add("Do you ship to Portugal?", "asdf asdf asdf", "en", ["low_quality"])
    add("How long is the warranty?", "Die Garantie beträgt zwei Jahre.", "en", ["wrong_language"])
    for _ in range(4):
        add("Can I still return my thermostat?",
            f"Yes, if it's within {RETURNS_OLD['window_days']} days. Opened items have a 10% restocking fee.",
            "en", ["outdated_policy"])
    order = sorted(range(len(rows)), key=lambda _i: rng.random())
    rows = [rows[i] for i in order]
    labels = [labels[i] for i in order]
    for new_idx, (row, lab) in enumerate(zip(rows, labels), 1):  # ids must not reveal which rows are traps
        row["id"] = lab["id"] = f"s{new_idx:03d}"
    return rows, labels
