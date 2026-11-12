"""Input and output guardrails around the model (lesson 6.4).

Input:  redact personal data before it reaches the prompt or the logs; spot
        safety emergencies and explicit requests for a person.
Output: block answers that ask for credentials or point customers to contact
        details outside Hearthwise's own domain, whatever the model was told.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ALLOWED_DOMAINS = ("hearthwise.example",)

EMAIL_RE = re.compile(r"\b[\w.+-]+@([\w-]+(?:\.[\w-]+)+)\b")
URL_RE = re.compile(r"\bhttps?://([\w.-]+)", re.IGNORECASE)
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
PHONE_RE = re.compile(r"(?<![\w-])\+?\d(?:[ -]?\d){8,12}(?![\w-])")

SAFETY_RE = re.compile(
    r"\b(smoke|smoking|burning|burnt|fire|flames?|sparks|sparking|overheat\w*|melting"
    r"|humo|quemad\w*|fuego|chispas)\b",
    re.IGNORECASE,
)
HUMAN_RE = re.compile(
    r"\b(?:talk|speak|chat)\s+(?:to|with)\s+(?:a\s+|an\s+|someone|somebody|a real\s+)?"
    r"(?:human|person|agent|someone|somebody)\b|\bhuman agent\b|\breal person\b"
    r"|\bhablar con (?:una persona|un agente|alguien)\b",
    re.IGNORECASE,
)
CREDENTIAL_RE = re.compile(
    r"\b(?:email|e-mail|send|share|give|provide|enter|type|reply with|confirm|envía|envia|comparte)\b"
    r"[^.]{0,30}\b(?:your|their|tu|su)\b[^.]{0,20}"
    r"\b(?:password|passcode|pin|cvv|card number|contraseña|contrasena)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Redaction:
    text: str
    found: tuple[str, ...]


def redact_pii(text: str) -> Redaction:
    """Replace e-mail addresses, card numbers and phone numbers with placeholders."""
    found: list[str] = []
    for label, rx in (("EMAIL", EMAIL_RE), ("CARD", CARD_RE), ("PHONE", PHONE_RE)):
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[{label}]", text)
    return Redaction(text, tuple(found))


def is_safety_risk(text: str) -> bool:
    """Smoke, burning smells, sparks, overheating: a person must take over now."""
    return bool(SAFETY_RE.search(text))


def wants_human(text: str) -> bool:
    return bool(HUMAN_RE.search(text))


def _domain_allowed(domain: str) -> bool:
    domain = domain.lower().rstrip(".")
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS)


def check_output(answer: str) -> list[str]:
    """Return the reasons to block an answer (empty list = safe to send)."""
    problems: list[str] = []
    if CREDENTIAL_RE.search(answer):
        problems.append("credential_request")
    domains = [m.group(1) for m in EMAIL_RE.finditer(answer)] + [m.group(1) for m in URL_RE.finditer(answer)]
    if any(not _domain_allowed(d) for d in domains):
        problems.append("foreign_contact")
    return problems
