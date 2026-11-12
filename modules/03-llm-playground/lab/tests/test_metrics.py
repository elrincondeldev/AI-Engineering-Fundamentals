"""Cost and latency arithmetic."""

from __future__ import annotations

import pytest

from m03_playground.metrics import (
    Prices,
    Workload,
    decode_tokens_per_second,
    end_to_end_tokens_per_second,
    estimate_cost,
    percentile,
    request_cost,
)


def test_request_cost_prices_input_and_output_separately() -> None:
    prices = Prices(input_per_mtok=2.0, output_per_mtok=10.0)
    assert request_cost(1_000_000, 0, prices) == pytest.approx(2.0)
    assert request_cost(0, 1_000_000, prices) == pytest.approx(10.0)
    assert request_cost(500, 100, prices) == pytest.approx((500 * 2 + 100 * 10) / 1e6)


def test_tokens_per_second_definitions() -> None:
    # 11 tokens: first at 200 ms, last at 700 ms -> 10 tokens in 0.5 s of decoding
    assert decode_tokens_per_second(11, 200, 700) == pytest.approx(20.0)
    assert end_to_end_tokens_per_second(11, 700) == pytest.approx(11 / 0.7)
    assert decode_tokens_per_second(1, 200, 200) == 0.0


def test_percentile_nearest_rank() -> None:
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([3.0], 99) == 3.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_estimate_cost_scales_linearly() -> None:
    w = Workload(10, 4, 20, 1000, 100)
    est = estimate_cost(w, Prices(1.0, 4.0))
    assert w.requests_per_month == 800
    assert est.per_request == pytest.approx(0.0014)
    assert est.per_day == pytest.approx(0.0014 * 40)
    assert est.per_month == pytest.approx(0.0014 * 800)
