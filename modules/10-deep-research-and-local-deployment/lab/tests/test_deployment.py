"""Tests for the local-deployment maths: memory, quantisation, serving, cost."""

from __future__ import annotations

import numpy as np
import pytest

from m10_deep_research.memory import bits_per_weight, kv_bytes_per_token, kv_cache_bytes, sweep, weight_bytes
from m10_deep_research.quantize import compare, make_weights, quantize
from m10_deep_research.serving import (
    CostInputs,
    daily_cost,
    kv_plans,
    make_requests,
    servers_needed,
    simulate_continuous,
    simulate_static,
    speculative,
)


def test_weight_memory_formula() -> None:
    assert weight_bytes(8e9, 16) == 16e9
    assert weight_bytes(8e9, 8) == 8e9
    assert bits_per_weight(4, 32) == 4.5  # one 16-bit scale per 32 weights


def test_kv_cache_grows_linearly_with_context_and_batch() -> None:
    one = kv_cache_bytes(32, 8, 128, context=4096)
    assert kv_cache_bytes(32, 8, 128, context=8192) == 2 * one
    assert kv_cache_bytes(32, 8, 128, context=4096, batch=4) == 4 * one
    assert kv_bytes_per_token(32, 32, 128) == 4 * kv_bytes_per_token(32, 8, 128)


def test_sweep_marks_what_fits() -> None:
    rows = {(r.params, r.bits): r.fits for r in sweep(16)}
    assert rows[(8e9, 4)] and not rows[(8e9, 16)] and not rows[(70e9, 4)]


def test_quantize_round_trip_and_granularity() -> None:
    w = make_weights(rows=8, cols=64, seed=3)
    w8, n8 = quantize(w, 8, "channel")
    assert n8 == 8 and np.max(np.abs(w - w8)) <= np.abs(w).max(axis=1).max() / 127
    _, n_groups = quantize(w, 4, "group", group_size=32)
    assert n_groups == 16
    with pytest.raises(ValueError):
        quantize(w, 4, "group", group_size=48)


def test_finer_scales_mean_smaller_error() -> None:
    r = {x.name: x for x in compare()}
    assert r["int8 per-channel"].weight_error < r["int8 per-tensor"].weight_error
    assert (r["int4 group-32"].weight_error < r["int4 group-128"].weight_error
            < r["int4 per-channel"].weight_error < r["int4 per-tensor"].weight_error)


def test_continuous_batching_beats_static_batching() -> None:
    reqs = make_requests()
    static, cont = simulate_static(reqs, 16), simulate_continuous(reqs, 16)
    assert cont.ttft_p50_ms < static.ttft_p50_ms and cont.throughput_tok_s > static.throughput_tok_s


def test_paging_wastes_less_memory() -> None:
    contiguous, paged = kv_plans(make_requests(), 8, kv_bytes_per_token(32, 8, 128))
    assert paged.waste < contiguous.waste and paged.concurrent_fit > contiguous.concurrent_fit


def test_speculative_decoding_formula() -> None:
    assert speculative(0.0, 4, 0.05) == (1.0, 0.833)  # every draft rejected: pure overhead
    tokens, speedup = speculative(0.8, 4, 0.05)
    assert tokens == pytest.approx((1 - 0.8**5) / 0.2, abs=1e-3) and speedup > 1


def test_cost_and_capacity() -> None:
    r = daily_cost(CostInputs(1e6, 2e5, 1.0, 4.0, 2500, 36, 150, 8, 0.19))
    assert r.api_eur_per_day == 1.8 and r.local_eur_per_day == 2.54
    assert servers_needed(400, 150, 0.3) == 4
