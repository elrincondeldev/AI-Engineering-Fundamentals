"""Unit tests for each cleaning stage. They double as the specification when you rebuild a function."""

from __future__ import annotations

from m01_foundations.cleaning import (
    QualityConfig,
    build_eval_index,
    contamination_hits,
    detect_language,
    html_to_text,
    looks_binary,
    normalise_url,
    quality_failures,
    redact_pii,
    url_is_blocked,
)

PAGE = """<!doctype html><html lang="en"><head><title>T | Brand</title></head><body>
<header><div class="logo">Brand</div><nav><a href="/">Home</a></nav></header>
<div class="cookie-banner">We use cookies. <button>Accept all</button></div>
<article><h1>Heating schedules</h1><p class="byline">Published 2025-01-01</p>
<p>Lower the setpoint by one degree &amp; see if anyone notices.</p><p>Draught-proofing helps.</p></article>
<footer>&copy; 2025 Brand</footer></body></html>"""


def test_normalise_url_drops_tracking_fragment_and_case() -> None:
    a = normalise_url("https://Forum.Example/t/x/?utm_source=newsletter&utm_medium=email#top")
    assert a == "https://forum.example/t/x"
    assert normalise_url("https://a.example/p?b=2&a=1") == "https://a.example/p?a=1&b=2"
    assert normalise_url("https://a.example:443/") == "https://a.example/"


def test_url_blocklist() -> None:
    assert url_is_blocked("https://free-casino.example/win")
    assert not url_is_blocked("https://eheo.example/reports/x")


def test_html_to_text_keeps_article_and_drops_boilerplate() -> None:
    text = html_to_text(PAGE)
    assert text.splitlines() == [
        "Heating schedules",
        "Lower the setpoint by one degree & see if anyone notices.",
        "Draught-proofing helps.",
    ]
    for junk in ("cookies", "Home", "Brand", "Published"):
        assert junk not in text


def test_html_to_text_without_article_uses_body() -> None:
    assert html_to_text("<html><body><h1>Not Found</h1></body></html>") == "Not Found"


def test_looks_binary() -> None:
    assert looks_binary("%PDF-1.7 \x00\x01 binary")
    assert not looks_binary("A normal sentence.")


def test_detect_language_on_four_languages() -> None:
    assert detect_language("Put the plug on a timer so it switches off at night.") == "en"
    assert detect_language("Programa el enchufe para que apague la tele por la noche.") == "es"
    german = "Schalten Sie die Geräte nachts aus, um Strom zu sparen und das Netz zu entlasten."
    assert detect_language(german) == "de"
    assert detect_language("Programmez le chauffage pour qu'il baisse quand la maison est vide.") == "fr"
    assert detect_language("12345 67890") == "unknown"


def test_quality_rules() -> None:
    spam = "cheap smart plug cheap smart plug cheap smart plug! Click here for the cheap smart plug."
    assert "repetitive" in quality_failures(spam)
    assert quality_failures("No posts found.") == ["too_short"]
    assert "shouting" in quality_failures("BUY NOW THE BEST DEAL OF THE YEAR FOR EVERY HOME TODAY.")
    good = "Warm white works best in the bedroom, while cooler light suits a home office."
    assert quality_failures(good) == []
    assert quality_failures(good, QualityConfig(min_words=50)) == ["too_short"]


def test_redact_pii() -> None:
    text = "Email me at lucia.p@mail.example or call +34 611 222 333. I live at Calle Falsa 123, Valencia."
    redacted, found = redact_pii(text)
    assert redacted == "Email me at <EMAIL> or call <PHONE>. I live at <ADDRESS>."
    assert dict(found) == {"EMAIL": 1, "PHONE": 1, "ADDRESS": 1}
    assert redact_pii("Released on 2025-09-01 for €89.")[1] == {}


def test_contamination_needs_a_shared_long_ngram() -> None:
    index = build_eval_index(["What is the return window for Hearthwise products? 30 days from delivery."])
    leaked = "Q: What is the return window for Hearthwise products? A: 30 days from delivery."
    assert contamination_hits(leaked, index) > 0
    assert contamination_hits("Hearthwise products ship from Valencia within two days.", index) == 0
