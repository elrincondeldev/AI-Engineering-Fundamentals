"""End-to-end tests: the research loop, the writer, the grader and the ablations."""

from __future__ import annotations

import pytest

from m10_deep_research.evaluate import ABLATIONS, load_workspace, run_ablations, run_and_grade
from m10_deep_research.grade import anchors, grade_report, line_covers, numbers
from m10_deep_research.loop import ResearchConfig
from m10_deep_research.synthesise import drop_bad_citations


@pytest.fixture(scope="module")
def ws():
    return load_workspace()


def test_same_command_same_report(ws) -> None:
    a, _ = run_and_grade(ws, "rq1")
    b, _ = run_and_grade(ws, "rq1")
    assert a.report == b.report and a.trace == b.trace


@pytest.mark.parametrize("qid", ["rq1", "rq2", "rq3", "rq4"])
def test_full_pipeline_covers_every_key_fact_with_valid_citations(ws, qid) -> None:
    result, grade = run_and_grade(ws, qid)
    assert grade.facts_supported == grade.facts_total
    assert grade.valid_cited_lines == grade.cited_lines
    assert grade.enough_sources and grade.low_credibility_citations == 0
    assert not grade.trap_hits and grade.pii_leaks == 0
    assert "## Sources" in result.report


def test_report_marks_gaps_and_proxies(ws) -> None:
    result, _ = run_and_grade(ws, "rq4")
    assert result.gaps == ["logistics (Portugal)"]
    assert "No usable evidence found for: logistics (Portugal)." in result.report
    assert "proxy for demand" in result.report


def test_writer_infers_where_a_competitor_is_not_sold(ws) -> None:
    result, _ = run_and_grade(ws, "rq1")
    assert "Available in Spain, Portugal, Mexico and Chile. Not listed: Colombia." in result.report


def test_grader_catches_a_changed_number(ws) -> None:
    result, honest = run_and_grade(ws, "rq1")
    edited = result.report.replace("260,000", "280,000")
    g = grade_report(edited, ws.questions["rq1"], ws.snapshot, ws.labels)
    assert honest.facts_supported == 7
    assert (g.facts_covered, g.facts_supported, g.valid_cited_lines) == (6, 4, 4)


def test_grader_rejects_citations_to_missing_sources(ws) -> None:
    result, _ = run_and_grade(ws, "rq3")
    g = grade_report(result.report.replace("[1]\n", "[7]\n", 1), ws.questions["rq3"], ws.snapshot, ws.labels)
    assert g.valid_cited_lines == 0


def test_drop_bad_citations() -> None:
    text, removed = drop_bad_citations("A [1]. B [9]. C [2].", n_sources=2)
    assert text == "A [1]. B . C [2]." and removed == 1


def test_anchors_and_number_normalisation() -> None:
    found = numbers("Mexico: 260,000 units (18% growth), €0.09, 1080p, Ember T3")
    assert found == {"260000", "18", "0.09", "1080"}
    a = anchors("Corvid Thermo X is sold in Mexico and Chile but not in Colombia")
    assert not a.numbers and len(a.entities) == 4
    assert line_covers("Corvid Thermo X: Available in Mexico and Chile. Not listed: Colombia.", a)


def test_small_search_budget_stops_early(ws) -> None:
    result, grade = run_and_grade(ws, "rq1", ResearchConfig(max_searches=2))
    assert result.stats.stop_reason == "search budget spent"
    assert grade.facts_covered == 5 and not grade.enough_sources


def test_each_defence_has_a_measurable_effect(ws) -> None:
    rows = {r.name: r for r in run_ablations(ws)}
    assert set(rows) == set(ABLATIONS)
    full = rows["full"]
    assert rows["no-credibility"].low_credibility_citations > full.low_credibility_citations
    assert rows["no-recency"].trap_hits > full.trap_hits
    assert rows["no-dedup"].tokens_kept > full.tokens_kept
    assert rows["no-sanitiser"] == rows["no-sanitiser"].__class__(**{**full.__dict__, "name": "no-sanitiser"})
    assert rows["no-credibility+no-sanitiser"].pii_leaks > rows["no-credibility"].pii_leaks
