"""Every number quoted in module 04's Markdown comes from these reports.

If a test here fails, a lesson, the README, the quiz, the cheatsheet or the lab README now
shows a wrong number: rerun the command, update the Markdown, then update this file.

Prompting and context results are pure Python and asserted exactly. The numpy LoRA results
are asserted with a tolerance of 0.011 (about one ticket out of 96), because a different
BLAS library can flip a borderline prediction.
"""

from __future__ import annotations

import pytest

from m04_adaptation import reports
from m04_adaptation.tickets import four_way_split

TOL = 0.011


def _by_name(rows: list, name: str):  # noqa: ANN202 - small helper
    return next(r for r in rows if r.name == name)


# --------------------------------------------------------------------------- #
# Lesson 1 · data audit and splits                                            #
# --------------------------------------------------------------------------- #


def test_audit_counts() -> None:
    a = reports.audit_report()
    assert (a.rows, a.resubmissions, a.duplicate_bodies, a.after_dedup) == (1224, 24, 51, 1200)
    assert (a.train, a.test) == (800, 400)
    assert a.pii == {"phone_number": 82, "email_address": 0, "signature": 205, "quoted_reply": 142,
                     "html_entity": 23}
    assert a.majority == 0.120


def test_leak_audit() -> None:
    cols = {c.column: c for c in reports.audit_report().columns}
    assert cols["resolution_code"].accuracy == 1.000
    assert cols["resolution_code"].distinct_values == 8
    assert not cols["resolution_code"].known_at_prediction_time
    assert (cols["channel"].accuracy, cols["language"].accuracy, cols["product"].accuracy) == (0.120, 0.125, 0.125)
    assert cols["customer_email"].accuracy == 0.113


def test_drift_numbers() -> None:
    a = reports.audit_report()
    assert (a.shares_before["connectivity_issue"], a.shares_after["connectivity_issue"]) == (0.139, 0.278)
    assert (a.shares_before["order_status"], a.shares_after["order_status"]) == (0.181, 0.120)
    assert (a.shares_before["return_request"], a.shares_after["return_request"]) == (0.155, 0.095)
    assert a.camera_share_after == 0.395
    assert (a.oov_august, a.oov_after_launch, a.oov_camera) == (0.000, 0.186, 0.465)


def test_four_way_split_sizes() -> None:
    s = four_way_split(list(reports.tickets()))
    assert [len(s.old_train), len(s.old_test), len(s.new_adapt), len(s.new_test)] == [704, 96, 97, 303]


def test_split_report() -> None:
    rows = {r.name: (r.train, r.test, r.overlap, r.accuracy) for r in reports.split_report()}
    assert rows["random 80/20, raw rows"] == (979, 245, 16, 0.959)
    assert rows["random 80/20, deduplicated"] == (960, 240, 6, 0.921)
    assert rows["temporal, raw rows"] == (814, 410, 13, 0.820)
    assert rows["temporal, deduplicated"] == (800, 400, 11, 0.828)
    assert rows["temporal + resolution_code"][3] == 0.920


# --------------------------------------------------------------------------- #
# Lesson 2 · prompt engineering                                               #
# --------------------------------------------------------------------------- #

PROMPT_ROWS = {
    # name: (accuracy, camera, spanish, prompt tokens)
    "label names only, 0-shot": (0.270, 0.228, 0.125, 130.5),
    "English descriptions, 0-shot": (0.618, 0.443, 0.333, 276.1),
    "bilingual descriptions, 0-shot": (0.723, 0.519, 0.683, 350.9),
    "bilingual + 1 random example": (0.728, 0.519, 0.692, 389.0),
    "bilingual + 4 random examples": (0.735, 0.525, 0.708, 504.3),
    "bilingual + 8 random examples": (0.728, 0.532, 0.692, 656.8),
    "bilingual + 8 balanced examples": (0.740, 0.557, 0.708, 654.3),
    "bilingual + 1 similar example": (0.782, 0.595, 0.750, 388.8),
    "bilingual + 4 similar examples": (0.828, 0.633, 0.775, 502.6),
    "bilingual + 8 similar examples": (0.828, 0.633, 0.775, 654.0),
    "8 similar + user context": (0.845, 0.677, 0.833, 790.5),
}


@pytest.mark.parametrize("name", list(PROMPT_ROWS))
def test_prompt_table(name: str) -> None:
    r = _by_name(reports.prompt_table(), name)
    assert r.n == 400
    assert (r.accuracy, r.accuracy_camera, r.accuracy_spanish, r.prompt_tokens) == PROMPT_ROWS[name]


def test_format_report() -> None:
    f = reports.format_report()
    assert (f.n, f.strict_valid, f.lenient_valid, f.needed_retry, f.invalid_after_retry) == (400, 379, 390, 10, 0)
    assert (f.accuracy_no_retry, f.accuracy_with_retry, f.accuracy_reasoning) == (0.810, 0.828, 0.828)
    assert (f.completion_tokens_plain, f.completion_tokens_reasoning) == (9.4, 35.7)


def test_show_prompt_example(capsys: pytest.CaptureFixture[str]) -> None:
    from m04_adaptation.cli import main

    main(["show-prompt"])
    assert "-> predicted: product_question · true: connectivity_issue · attempts: 1 · prompt tokens: 415" in (
        capsys.readouterr().out)
    main(["show-prompt", "--k", "0", "--descriptions", "bilingual_v2"])
    assert "-> predicted: connectivity_issue · true: connectivity_issue" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Lesson 3 · context engineering                                              #
# --------------------------------------------------------------------------- #

BUDGET_ROWS = {
    # (budget, policy): (accuracy, examples kept, tokens sent)
    (400, "raw"): (0.790, 0.71, 388.4),
    (400, "clean"): (0.797, 0.89, 390.5),
    (400, "clean+dedupe"): (0.797, 0.86, 389.6),
    (500, "raw"): (0.820, 3.21, 485.9),
    (500, "clean"): (0.828, 3.50, 486.0),
    (500, "clean+dedupe"): (0.830, 3.26, 478.3),
    (700, "raw"): (0.815, 8.17, 684.4),
    (700, "clean"): (0.843, 8.96, 685.1),
    (700, "clean+dedupe"): (0.830, 6.07, 583.0),
    (1000, "raw"): (0.807, 14.64, 957.4),
    (1000, "clean"): (0.843, 15.67, 933.2),
    (1000, "clean+dedupe"): (0.830, 6.28, 592.0),
}


def test_budget_table() -> None:
    rows = {(r.budget, r.policy): (r.accuracy, r.examples, r.tokens) for r in reports.budget_table()}
    assert rows == BUDGET_ROWS


def test_k_sweep() -> None:
    rows = {r.name: (r.accuracy, r.prompt_tokens) for r in reports.k_sweep()}
    assert rows == {
        "4 similar examples": (0.828, 502.6),
        "8 similar examples": (0.828, 654.0),
        "16 similar examples": (0.840, 955.0),
        "32 similar examples": (0.825, 1555.6),
    }


def test_quoted_reply_slice() -> None:
    q = reports.quoted_reply_slice()
    assert (q.n, q.zero_shot_raw, q.zero_shot_clean, q.few_shot_raw, q.few_shot_clean) == (
        47, 0.681, 0.809, 0.596, 0.830)


def test_context_example_for_one_ticket() -> None:
    _, compact = reports.assembly_for("T00822", 500, "clean+dedupe")
    assert (compact.tokens_offered, compact.tokens, compact.examples_kept) == (884, 487, 4)
    assert sum(why == "near-duplicate" for _, why in compact.dropped) == 12
    assert compact.kept[0].kind == "system" and compact.kept[0].tokens == 316
    _, raw = reports.assembly_for("T00822", 500, "raw")
    assert (raw.tokens_offered, raw.tokens, raw.examples_kept) == (1351, 447, 1)


def test_cache_report() -> None:
    stable, unstable = reports.cache_report()
    assert (stable.hits, stable.misses, stable.hit_rate, stable.cached_share) == (399, 1, 0.998, 0.496)
    assert (unstable.hits, unstable.misses, unstable.hit_rate, unstable.cached_share) == (0, 400, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Lesson 4 · adapting after the launch                                        #
# --------------------------------------------------------------------------- #


def test_prompt_adaptation() -> None:
    rows = {r.name: (r.labelled, r.old, r.new, r.camera, r.prompt_tokens) for r in reports.prompt_adaptation()}
    assert rows == {
        "0-shot, original descriptions": (0, 0.760, 0.710, 0.496, 352.7),
        "0-shot, descriptions updated": (0, 0.760, 0.851, 0.851, 430.9),
        "8 similar, pool Jan-Jul": (704, 0.917, 0.825, 0.620, 656.0),
        "8 similar, pool Jan-Jul + Sep": (801, 0.917, 0.901, 0.810, 658.7),
    }


LORA_ROWS = {
    # name: (trainable, share of full, old, new, camera)
    "base (frozen, no update)": (0, 0.0, 0.917, 0.828, 0.628),
    "full fine-tuning": (8200, 1.0, 0.906, 0.927, 0.893),
    "LoRA r=1": (1032, 0.126, 0.906, 0.870, 0.770),
    "LoRA r=2": (2064, 0.252, 0.904, 0.893, 0.825),
    "LoRA r=4": (4128, 0.503, 0.904, 0.905, 0.868),
    "LoRA r=8": (8256, 1.007, 0.904, 0.906, 0.874),
}


@pytest.mark.parametrize("name", list(LORA_ROWS))
def test_lora_table(name: str) -> None:
    r = _by_name(reports.lora_table(), name)
    trainable, share, old, new, camera = LORA_ROWS[name]
    assert (r.trainable, r.share_of_full) == (trainable, share)
    assert r.old == pytest.approx(old, abs=TOL)
    assert r.new == pytest.approx(new, abs=TOL)
    assert r.camera == pytest.approx(camera, abs=TOL)


def test_lora_seed_spread() -> None:
    spreads = {r.name: r.new_spread for r in reports.lora_table()}
    assert spreads["LoRA r=1"] == pytest.approx(0.042, abs=TOL)
    assert spreads["LoRA r=8"] == pytest.approx(0.013, abs=TOL)


FORGETTING = {
    # steps: (full old, full new, r=4 old, r=4 new, replay old, replay new)
    100: (0.906, 0.927, 0.906, 0.901, 0.906, 0.927),
    1000: (0.875, 0.904, 0.885, 0.881, 0.906, 0.914),
    4000: (0.792, 0.855, 0.812, 0.848, 0.865, 0.888),
}


def test_forgetting_table() -> None:
    for row in reports.forgetting_table():
        got = (row.full_old, row.full_new, row.lora_old, row.lora_new, row.replay_old, row.replay_new)
        assert got == pytest.approx(FORGETTING[row.steps], abs=TOL)


def test_data_curve() -> None:
    expected = {12: (0.865, 0.867), 25: (0.871, 0.881), 50: (0.911, 0.908), 97: (0.927, 0.905)}
    for row in reports.data_curve():
        assert (row.full_new, row.lora_new) == pytest.approx(expected[row.labelled], abs=TOL)


def test_parameter_and_memory_tables() -> None:
    big = {r.rank: (r.lora, r.share) for r in reports.param_table()}
    assert big[1] == (8_192, 0.00049)
    assert big[8] == (65_536, 0.00391)
    assert big[64] == (524_288, 0.03125)
    small = {r.rank: r.lora for r in reports.param_table(1024, 8)}
    assert small == {1: 1032, 2: 2064, 4: 4128, 8: 8256, 16: 16_512, 64: 66_048}
    assert reports.memory_table() == {32: 28.0, 16: 14.0, 8: 7.0, 4: 3.5}


# --------------------------------------------------------------------------- #
# Lessons 2-3 · single-ticket examples · lab README "Break it" observations   #
# --------------------------------------------------------------------------- #


def test_quoted_reply_ticket_t00974() -> None:
    from m04_adaptation.evaluation import make_provider
    from m04_adaptation.prompts import ExampleSelector, build_messages
    from m04_adaptation.structured import classify
    from m04_adaptation.tickets import INTENTS

    train, test = reports.standard_split()
    ticket = next(t for t in test if t.ticket_id == "T00974")
    provider = make_provider()
    labels = []
    for clean in (False, True):
        cfg = reports.BEST_PROMPT.with_(clean=clean)
        labels.append(classify(provider, build_messages(ticket, cfg, ExampleSelector(train, cfg)), INTENTS).label)
    assert labels == ["product_question", "account_access"]
    assert ticket.intent == "account_access"


def test_derived_differences_quoted_in_lessons() -> None:
    f = reports.format_report()
    assert f.lenient_valid - f.strict_valid == 11
    assert round(f.completion_tokens_reasoning / f.completion_tokens_plain, 1) == 3.8
    rows = {r.name: r for r in reports.prompt_table()}
    assert round(rows["8 similar + user context"].prompt_tokens - rows["bilingual + 8 similar examples"].prompt_tokens,
                 1) == 136.5
    k = {r.name: r.prompt_tokens for r in reports.k_sweep()}
    assert round(k["32 similar examples"] - k["16 similar examples"], 1) == 600.6
    b = {(r.budget, r.policy): r.tokens for r in reports.budget_table()}
    assert round(b[(700, "clean")] - b[(700, "clean+dedupe")], 1) == 102.1


def test_break_it_leaky_example_pool() -> None:
    from m04_adaptation.evaluation import evaluate_prompt

    _, test = reports.standard_split()
    assert evaluate_prompt("leaky pool", test, test, reports.BEST_PROMPT).accuracy == 0.943


def test_break_it_reversed_label_order(monkeypatch: pytest.MonkeyPatch) -> None:
    from m04_adaptation import prompts
    from m04_adaptation.evaluation import evaluate_prompt

    train, test = reports.standard_split()
    original = prompts.descriptions
    monkeypatch.setattr(prompts, "descriptions", lambda kind: dict(reversed(original(kind).items())))
    assert evaluate_prompt("reversed", test, train, prompts.PromptConfig()).accuracy == 0.755


def test_break_it_lora_scale_too_large() -> None:
    import numpy as np

    from m04_adaptation.lora import train_lora

    d = reports.lora_data()
    with np.errstate(all="ignore"):
        model = train_lora(reports.base_model(), d.X_adapt, d.y_adapt, rank=1, alpha=8)
        assert model.accuracy(d.X_new, d.y_new) == pytest.approx(0.122, abs=TOL)


# --------------------------------------------------------------------------- #
# quiz.md calculations                                                        #
# --------------------------------------------------------------------------- #


def test_quiz_calculations() -> None:
    from m04_adaptation.lora import full_params, lora_params, weight_memory_gb

    assert lora_params(1024, 4096, 16) == 81_920
    assert full_params(1024, 4096) == 4_194_304
    assert {r.rank: r.share for r in reports.param_table(1024, 4096)}[16] == 0.01953
    assert (weight_memory_gb(13e9, 16), weight_memory_gb(13e9, 4)) == (26.0, 6.5)
    b = {(r.budget, r.policy): r for r in reports.budget_table()}
    saved = b[(700, "clean")].tokens - b[(700, "clean+dedupe")].tokens
    assert round(saved / b[(700, "clean")].tokens, 2) == 0.15
    assert round(b[(700, "clean")].accuracy - b[(700, "clean+dedupe")].accuracy, 3) == 0.013
