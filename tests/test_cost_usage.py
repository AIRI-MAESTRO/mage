"""T14 — actual cost from actual token usage (shared MAGE/CARE formula)."""

from __future__ import annotations

from mmar_mage.cost import cost_usd_for_usage


def test_matches_per_1k_pricing() -> None:
    # 1000 in @ $0.50/1k + 2000 out @ $1.50/1k = 0.50 + 3.00
    assert cost_usd_for_usage(1000, 2000, 0.5, 1.5) == 3.5


def test_zero_when_price_unknown() -> None:
    # run-731 usage with no known model price → 0.0 (not a crash, not null)
    assert cost_usd_for_usage(44013, 8029, 0.0, 0.0) == 0.0


def test_input_and_output_priced_separately() -> None:
    assert cost_usd_for_usage(1000, 0, 2.0, 99.0) == 2.0  # only input tokens cost
    assert cost_usd_for_usage(0, 1000, 99.0, 3.0) == 3.0  # only output tokens cost


def test_negative_price_is_clamped() -> None:
    assert cost_usd_for_usage(1000, 1000, -5.0, -5.0) == 0.0  # never negative cost
