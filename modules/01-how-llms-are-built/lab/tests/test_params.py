from __future__ import annotations

from dataclasses import replace

from m01_foundations.params import (
    CLASSIC_SMALL,
    LAB_TINY,
    MODERN_1B,
    MOE_8X,
    MOE_FINE,
    count_params,
    kaplan_estimate,
    kv_cache_bytes,
    training_flops,
    weight_memory_gb,
)
from m01_foundations.transformer import TinyConfig, init_params


def test_counter_matches_the_numpy_model_exactly() -> None:
    p = init_params(TinyConfig())
    assert count_params(LAB_TINY).total == sum(v.size for v in p.values())


def test_classic_small_and_kaplan_rule() -> None:
    c = count_params(CLASSIC_SMALL)
    assert c.total == 124_439_808
    assert abs(c.non_embedding - kaplan_estimate(CLASSIC_SMALL)) / c.non_embedding < 0.01


def test_dense_models_use_every_parameter() -> None:
    c = count_params(MODERN_1B)
    assert c.total == c.active


def test_moe_total_vs_active() -> None:
    dense, moe = count_params(MODERN_1B), count_params(MOE_8X)
    expert = moe.breakdown["feed_forward"] // (8 * MODERN_1B.n_layers)
    assert moe.total - moe.active == MODERN_1B.n_layers * 6 * expert  # 6 of 8 experts sit idle per token
    assert moe.active - dense.active == MODERN_1B.n_layers * (expert + MODERN_1B.d_model * 8)


def test_fine_grained_moe_costs_the_same_compute_as_dense() -> None:
    dense, fine = count_params(MODERN_1B), count_params(MOE_FINE)
    router = MODERN_1B.n_layers * MODERN_1B.d_model * 64
    assert fine.active == dense.active + router
    assert fine.total > 5 * dense.total


def test_untied_embeddings_add_a_vocab_by_d_matrix() -> None:
    tied = count_params(replace(MODERN_1B, tied_embeddings=True))
    assert count_params(MODERN_1B).total - tied.total == 32000 * 2048


def test_memory_compute_and_kv_cache() -> None:
    assert weight_memory_gb(1_000_000_000, "bf16") == 2.0
    assert training_flops(10, 100) == 6000
    full = kv_cache_bytes(MODERN_1B, 8192)
    assert full == 16 * 8192 * 2 * 4 * 128 * 2
    gqa_off = kv_cache_bytes(replace(MODERN_1B, n_kv_heads=None), 8192)
    assert gqa_off == 4 * full  # 16 query heads share 4 K/V heads
    local = kv_cache_bytes(MODERN_1B, 8192, local_window=1024, local_per_global=5)
    assert local == (2 * 8192 + 14 * 1024) * 2 * 4 * 128 * 2
    short = kv_cache_bytes(MODERN_1B, 512, local_window=1024, local_per_global=5)
    assert short == kv_cache_bytes(MODERN_1B, 512)  # shorter than the window: nothing to drop
