"""Snapshot loading, HTML extraction, credibility signals, dedup and BM25 search."""

from __future__ import annotations

from m08_web_agent.credibility import recency_score, source_tier, spam_signals
from m08_web_agent.search import BM25, Guards, SearchEngine, duplicate_groups, tokenize
from m08_web_agent.web import canonical_url, html_to_text, load_labels, load_snapshot


def test_snapshot_loads_every_page() -> None:
    snap = load_snapshot()
    assert len(snap.pages) == 134
    assert len(snap.readable) == 125


def test_extraction_drops_navigation_banners_and_ads() -> None:
    html = (
        '<html lang="en"><head><title>T | Site</title><meta name="date" content="2025-01-02"></head><body>'
        '<nav>Home</nav><div class="cookie-banner">cookies</div><div class="ads">BUY</div>'
        '<article><h1>Head</h1><p>Body text.</p><p style="display:none">hidden</p></article>'
        "<footer>f</footer></body></html>"
    )
    title, text, published, lang = html_to_text(html)
    assert (title, published, lang) == ("T", "2025-01-02", "en")
    assert text.splitlines() == ["Head", "Body text."]


def test_canonical_url_removes_tracking() -> None:
    url = "https://forum.homeautomators.example/t/ember-t3-wifi-drops?utm_source=newsletter&utm_medium=email"
    assert canonical_url(url) == "https://forum.homeautomators.example/t/ember-t3-wifi-drops"


def test_spam_detector_matches_labels_exactly() -> None:
    labels = load_labels()
    pages = load_snapshot().readable
    flagged = {p.url for p in pages if spam_signals(p).is_spam}
    labelled = {p.url for p in pages if labels[p.url]["is_spam"]}
    assert flagged == labelled and len(flagged) == 10


def test_source_tiers() -> None:
    snap = load_snapshot()
    assert source_tier(snap.get("https://marketlens.example/reports/smart-thermostats-2025")) == "high"
    assert source_tier(snap.get("https://forum.homeautomators.example/t/homemesh-device-limits")) == "low"
    assert source_tier(snap.get("https://gadgetdeals.example/spain-smart-thermostat-boom")) == "spam"
    assert source_tier(snap.get("https://corvid-home.example/thermo-x")) == "medium"


def test_recency_score_is_bounded() -> None:
    assert recency_score("2025-11-11") == 1.0
    assert recency_score("2020-01-01") == 0.0
    assert recency_score("") == 0.0


def test_dedup_groups_every_labelled_duplicate() -> None:
    labels = load_labels()
    pages = load_snapshot().readable
    rep = duplicate_groups(pages)
    for p in pages:
        target = labels[p.url]["duplicate_of"] or labels[p.url]["near_duplicate_of"]
        if target:
            assert rep[p.url] == rep[target], p.url
    assert len(set(rep.values())) == 100
    # the original, not the mirror, represents a syndicated group
    mirror = "https://newsmirror.example/syndicated/reviews-corvid-thermo-x-review-updated"
    assert rep[mirror] == "https://smarthomeweekly.example/reviews/corvid-thermo-x-review-updated"


def test_bm25_prefers_documents_with_rare_query_terms() -> None:
    bm = BM25([tokenize("smart plug energy"), tokenize("homemesh devices limit"), tokenize("smart bulb")])
    q = tokenize("homemesh limit")
    assert bm.score(q, 1) > bm.score(q, 0) == 0


def test_credibility_buries_spam_in_search() -> None:
    query = "smart thermostats sold spain 2025"
    naive = SearchEngine(load_snapshot(), Guards.none()).search(query)
    guarded = SearchEngine(load_snapshot(), Guards()).search(query)
    assert naive[0].url.startswith("https://gadgetdeals.example/")
    assert guarded[0].url == "https://marketlens.example/reports/smart-thermostats-2025"
    assert all(h.tier for h in guarded) and not any(h.tier for h in naive)
