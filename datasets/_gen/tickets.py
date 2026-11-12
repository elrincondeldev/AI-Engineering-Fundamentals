"""Synthetic customer-support tickets with deliberate traps.

Traps (documented in datasets/README.md):
- ``resolution_code`` is written after the ticket is solved and maps 1:1 to the
  label: a textbook target leak.
- Concept drift: the Lookout C2 camera launches on 2025-09-01. Afterwards the
  mix of intents changes and camera tickets use vocabulary never seen before.
- About 4% label noise, about 2% duplicated tickets, quoted replies,
  signatures and personal data (PII) inside the text.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any

from .catalog import INTENTS, SEED

PRODUCTS_BEFORE = [("Ember T3", 30), ("Glow A19", 25), ("Spark Mini", 20), ("Watcher S1", 10), ("Hearth Hub", 15)]
PRODUCTS_AFTER = [("Lookout C2", 40), ("Ember T3", 18), ("Glow A19", 15), ("Spark Mini", 12), ("Watcher S1", 7),
                  ("Hearth Hub", 8)]
PRIOR_BEFORE = [18, 14, 10, 15, 12, 10, 14, 7]
PRIOR_AFTER = [15, 12, 8, 28, 12, 8, 12, 5]
RESOLUTION = {
    "order_status": "RC-SHIP", "return_request": "RC-RMA", "warranty_claim": "RC-WTY",
    "connectivity_issue": "RC-NET", "billing_question": "RC-BILL", "account_access": "RC-ACC",
    "product_question": "RC-INFO", "cancel_subscription": "RC-CXL",
}

EN: dict[str, list[str]] = {
    "order_status": [
        "Hi, I ordered a {product} on {date} and it still hasn't arrived. Order {order}. Where is it?",
        "Can you tell me when order {order} will be delivered? The tracking page has not changed in {n} days.",
        "My {product} order says shipped but nothing has moved since {date}.",
        "Is order {order} lost? It was supposed to arrive this week.",
        "The courier says delivered but I never got my {product}. Order {order}.",
        "How long does shipping usually take? I ordered a {product} {n} days ago.",
    ],
    "return_request": [
        "I want to return my {product}. It doesn't fit my needs. Order {order}.",
        "How do I send back a {product} I bought {n} days ago? It's still in the box.",
        "Please start a return for order {order}, I changed my mind about the {product}.",
        "I opened the {product} but I'd like to return it. Is that possible?",
        "Can I get a refund for the {product}? I bought the wrong model.",
        "I need a return label for my {product}, order {order}.",
    ],
    "warranty_claim": [
        "My {product} stopped working after {n} months. It won't turn on at all.",
        "The {product} is faulty, the screen flickers constantly. Is this covered by warranty?",
        "I think my {product} is broken, it gets very hot. Can you replace it?",
        "The battery in my {product} died after a few weeks of use. Warranty claim please.",
        "My {product} has a hardware fault, the button is stuck. How do I make a claim?",
        "Dead {product} after {n} months, serial number {serial}. What are my options?",
    ],
    "connectivity_issue": [
        "My {product} keeps disconnecting from Wi-Fi every evening.",
        "I can't pair the {product} with the app, it fails at the Wi-Fi step.",
        "The {product} shows offline in the app even though my internet works.",
        "After changing my router the {product} won't connect anymore.",
        "The app says device unreachable for my {product}. I've restarted everything.",
        "Pairing my {product} times out every time. Router is 2 metres away.",
    ],
    "billing_question": [
        "I was charged twice for my order {order}. Can you check?",
        "Why did my card get charged {amount} this month?",
        "Can I get an invoice with my company's tax number for order {order}?",
        "I don't recognise a charge of {amount} from Hearthwise on my statement.",
        "Can I pay with PayPal instead of my card next time?",
        "My payment for Hearthwise Plus failed, what happens now?",
    ],
    "account_access": [
        "I can't log in to my account, it says my password is wrong.",
        "I never received the password reset email.",
        "My account is locked after too many attempts. Please help.",
        "I changed phones and lost my two-factor codes, how do I get back in?",
        "How do I change the email address on my account?",
        "Someone else may have access to my account, I see devices I don't own.",
    ],
    "product_question": [
        "Does the {product} work without a hub?",
        "What's the difference between the {product} and the older model?",
        "Is the {product} compatible with my home's wiring?",
        "How many devices can I connect to one hub?",
        "Can the {product} be used outdoors?",
        "Does the {product} support 5 GHz Wi-Fi?",
    ],
    "cancel_subscription": [
        "Please cancel my Hearthwise Plus subscription.",
        "How do I stop my Plus plan from renewing?",
        "I want to cancel Plus and get a refund for this year.",
        "Cancel my subscription please, I don't use the cloud recordings.",
        "I no longer need Hearthwise Plus. How do I end it?",
    ],
}

ES: dict[str, list[str]] = {
    "order_status": [
        "Hola, pedí un {product} el {date} y todavía no ha llegado. Pedido {order}.",
        "¿Cuándo llega el pedido {order}? El seguimiento no cambia desde hace {n} días.",
        "Mi pedido del {product} pone enviado pero no se mueve.",
    ],
    "return_request": [
        "Quiero devolver el {product}, no es lo que necesito. Pedido {order}.",
        "¿Cómo devuelvo un {product} que compré hace {n} días?",
        "Abrí el {product} pero quiero devolverlo. ¿Se puede?",
    ],
    "warranty_claim": [
        "Mi {product} dejó de funcionar a los {n} meses. No enciende.",
        "El {product} está defectuoso, se calienta mucho. ¿Lo cubre la garantía?",
        "La batería del {product} murió en pocas semanas. Quiero reclamar la garantía.",
    ],
    "connectivity_issue": [
        "Mi {product} se desconecta del wifi todas las noches.",
        "No consigo vincular el {product} con la app, falla en el paso del wifi.",
        "El {product} aparece sin conexión en la app aunque tengo internet.",
    ],
    "billing_question": [
        "Me habéis cobrado dos veces el pedido {order}.",
        "¿Por qué me han cobrado {amount} este mes?",
        "Necesito una factura con el NIF de mi empresa para el pedido {order}.",
    ],
    "account_access": [
        "No puedo entrar en mi cuenta, dice que la contraseña es incorrecta.",
        "No me llega el correo para restablecer la contraseña.",
        "Mi cuenta está bloqueada tras varios intentos.",
    ],
    "product_question": [
        "¿El {product} funciona sin hub?",
        "¿El {product} es compatible con el wifi de 5 GHz?",
        "¿Se puede usar el {product} en exteriores?",
    ],
    "cancel_subscription": [
        "Quiero cancelar mi suscripción a Hearthwise Plus.",
        "¿Cómo evito que se renueve el plan Plus?",
        "Cancelad mi suscripción, por favor, no uso las grabaciones.",
    ],
}

# Camera-specific wording that only appears after the launch (drift). Keyed by intent.
CAMERA_BY_INTENT = {
    "warranty_claim": ["The night vision on my Lookout C2 stopped working, the IR LEDs stay dark.",
                       "My Lookout C2 lens is fogged inside after two weeks."],
    "product_question": ["Can the Lookout C2 record to a NAS instead of a microSD card?",
                         "Does the Lookout C2 detect people and pets separately?",
                         "¿La Lookout C2 graba audio además de vídeo?"],
    "return_request": ["The Lookout C2 field of view is too wide for my hallway, I'd like to send it back.",
                       "La Lookout C2 no me convence, quiero mandarla de vuelta."],
    "cancel_subscription": ["I only got Plus for the camera clips, please stop it.",
                            "Ya no necesito el historial de clips en la nube, quitadme Plus."],
    "billing_question": ["Why does my Plus renewal show a camera fee I didn't expect?"],
}
CAMERA_EN = [
    "The Lookout C2 light is blinking orange and live view won't load.",
    "My new camera is stuck on 'connecting' since I switched it to 5 GHz.",
    "Live view on the Lookout C2 freezes after a few seconds, night vision clips are black.",
    "Camera keeps going offline at night, the status LED turns orange.",
    "The Lookout C2 says no signal after the latest firmware update.",
]
CAMERA_ES = [
    "La luz de la Lookout C2 parpadea en naranja y no carga la vista en directo.",
    "La cámara se queda en 'conectando' desde que la pasé a 5 GHz.",
]

GREETINGS = ["", "", "Hi, ", "Hello, ", "Hey, ", "Good morning, ", "Hola, ", "Buenas, "]
CLOSINGS = ["", "", " Thanks.", " Thank you!", " Please help.", " Gracias.", " Urgent!", " Cheers."]
# Tickets that mention two intents; the label is the action the customer wants.
AMBIGUOUS = [
    ("return_request", "My {product} keeps disconnecting from Wi-Fi, I give up, I want my money back."),
    ("warranty_claim", "I want to return my {product} because it stopped working after {n} months."),
    ("cancel_subscription", "I was charged {amount} for Plus again, please cancel it."),
    ("billing_question", "I cancelled Plus last month but I was still charged {amount}. Why?"),
    ("account_access", "I can't log in so I can't see where my order {order} is."),
    ("order_status", "I want to return the {product} but it hasn't even arrived yet. Where is order {order}?"),
    ("connectivity_issue", "Is the {product} broken? It won't connect to Wi-Fi after the update."),
]

SIGNATURES = ["\n\n--\nSent from my phone", "\n\nThanks,\nA very patient customer",
              "\n\nBest regards\n{name}\n{phone}"]
QUOTED = "\n\n> On {date}, Hearthwise Support wrote:\n> Thanks for contacting us. Could you share your order number?"
NAMES = ["Laura Gómez", "Pedro Alves", "Sam Carter", "Maria Rossi", "Diego Torres", "Ines Costa"]
PHONES = ["+34 612 345 678", "+351 913 456 789", "+52 55 8765 4321"]


def _weighted(rng: random.Random, pairs: list[tuple[str, int]]) -> str:
    names = [p for p, _ in pairs]
    weights = [w for _, w in pairs]
    return rng.choices(names, weights=weights, k=1)[0]


def _fill(text: str, rng: random.Random, product: str, date: str) -> str:
    repl = {
        "{product}": product,
        "{order}": f"#HW-{rng.randint(100000, 999999)}",
        "{date}": date,
        "{n}": str(rng.randint(2, 14)),
        "{amount}": f"€{rng.choice([4.99, 49, 12, 19, 89, 149])}",
        "{serial}": f"SN{rng.randint(10**7, 10**8 - 1)}",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def build_tickets() -> list[dict[str, Any]]:
    rng = random.Random(f"{SEED}-tickets")
    launch = dt.date(2025, 9, 1)
    start = dt.date(2025, 1, 1)
    rows: list[dict[str, Any]] = []
    n_before, n_after = 800, 400
    dates = sorted(
        [start + dt.timedelta(days=rng.randrange((launch - start).days)) for _ in range(n_before)]
        + [launch + dt.timedelta(days=rng.randrange(122)) for _ in range(n_after)]
    )
    for i, day in enumerate(dates):
        after = day >= launch
        intent = rng.choices(INTENTS, weights=PRIOR_AFTER if after else PRIOR_BEFORE, k=1)[0]
        lang = "es" if rng.random() < 0.3 else "en"
        product = _weighted(rng, PRODUCTS_AFTER if after else PRODUCTS_BEFORE)
        date_txt = (day - dt.timedelta(days=rng.randint(3, 12))).isoformat()
        if after and product == "Lookout C2" and intent == "connectivity_issue" and rng.random() < 0.8:
            body = rng.choice(CAMERA_ES if lang == "es" else CAMERA_EN)
        elif after and product == "Lookout C2" and intent in CAMERA_BY_INTENT and rng.random() < 0.6:
            body = rng.choice(CAMERA_BY_INTENT[intent])
        elif rng.random() < 0.08:
            intent, template = rng.choice(AMBIGUOUS)
            body = _fill(template, rng, product, date_txt)
        else:
            body = _fill(rng.choice((ES if lang == "es" else EN)[intent]), rng, product, date_txt)
        body = rng.choice(GREETINGS) + body + rng.choice(CLOSINGS)
        if rng.random() < 0.10:
            body += _fill(QUOTED, rng, product, date_txt)
        if rng.random() < 0.15:
            body += rng.choice(SIGNATURES).replace("{name}", rng.choice(NAMES)).replace("{phone}", rng.choice(PHONES))
        if rng.random() < 0.04:
            body = body.upper()
        if rng.random() < 0.05:
            body = body.replace("&", "&amp;").replace("'", "&#39;")
        label = intent
        if rng.random() < 0.04:  # label noise
            label = rng.choice([x for x in INTENTS if x != intent])
        subject = body.split(".")[0].split("?")[0][:60]
        rows.append({
            "ticket_id": f"T{i + 1:05d}",
            "created_at": day.isoformat(),
            "channel": rng.choice(["email", "chat", "web_form"]),
            "language": lang,
            "product": product,
            "subject": subject,
            "body": body,
            "customer_email": f"customer{rng.randint(1000, 9999)}@mail.example",
            "intent": label,
            "resolution_code": RESOLUTION[label],
        })
    # Duplicates: the same customer resubmits the same text a day later.
    dups = []
    for row in rng.sample(rows, 24):
        copy = dict(row)
        copy["created_at"] = (dt.date.fromisoformat(row["created_at"]) + dt.timedelta(days=1)).isoformat()
        dups.append(copy)
    rows.extend(dups)
    rows.sort(key=lambda r: (r["created_at"], r["ticket_id"]))
    for i, row in enumerate(rows, 1):
        row["ticket_id"] = f"T{i:05d}"
    return rows
