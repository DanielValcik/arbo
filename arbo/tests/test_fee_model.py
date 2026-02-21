"""Tests for Polymarket fee model.

PM-104 acceptance test: Unit tests for fee curve at 10 different prices,
match with Polymarket docs. Fee-free markets return 0.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arbo.core.fee_model import (
    calculate_edge_after_fee,
    calculate_taker_fee,
    estimate_maker_rebate,
    is_fee_favorable,
)


class TestTakerFee:
    """Fee curve tests at 10 price points."""

    @pytest.mark.parametrize(
        "price,expected_approx",
        [
            (Decimal("0.05"), Decimal("0.0015")),  # 0.05 * 0.95 * 0.0315
            (Decimal("0.10"), Decimal("0.0028")),  # 0.10 * 0.90 * 0.0315
            (Decimal("0.20"), Decimal("0.0050")),  # 0.20 * 0.80 * 0.0315
            (Decimal("0.30"), Decimal("0.0066")),  # 0.30 * 0.70 * 0.0315
            (Decimal("0.40"), Decimal("0.0076")),  # 0.40 * 0.60 * 0.0315
            (Decimal("0.50"), Decimal("0.0079")),  # 0.50 * 0.50 * 0.0315 = max
            (Decimal("0.60"), Decimal("0.0076")),  # symmetric with 0.40
            (Decimal("0.70"), Decimal("0.0066")),  # symmetric with 0.30
            (Decimal("0.80"), Decimal("0.0050")),  # symmetric with 0.20
            (Decimal("0.95"), Decimal("0.0015")),  # symmetric with 0.05
        ],
    )
    def test_fee_curve_at_price(self, price: Decimal, expected_approx: Decimal) -> None:
        """Fee at various prices should match Polymarket docs (within rounding)."""
        fee = calculate_taker_fee(price, fee_enabled=True)
        # Allow 0.001 tolerance for rounding
        assert abs(fee - expected_approx) < Decimal(
            "0.001"
        ), f"At price {price}: expected ~{expected_approx}, got {fee}"

    def test_fee_symmetric(self) -> None:
        """Fee at p and (1-p) should be equal."""
        fee_30 = calculate_taker_fee(Decimal("0.30"), fee_enabled=True)
        fee_70 = calculate_taker_fee(Decimal("0.70"), fee_enabled=True)
        assert fee_30 == fee_70

    def test_max_fee_at_midpoint(self) -> None:
        """Maximum fee is at p=0.50."""
        fee_50 = calculate_taker_fee(Decimal("0.50"), fee_enabled=True)
        fee_30 = calculate_taker_fee(Decimal("0.30"), fee_enabled=True)
        fee_80 = calculate_taker_fee(Decimal("0.80"), fee_enabled=True)
        assert fee_50 > fee_30
        assert fee_50 > fee_80

    def test_fee_free_market_returns_zero(self) -> None:
        """Fee-free markets should always return 0."""
        for price in [Decimal("0.10"), Decimal("0.50"), Decimal("0.90")]:
            assert calculate_taker_fee(price, fee_enabled=False) == Decimal("0")

    def test_edge_prices_return_zero(self) -> None:
        """Prices at 0 or 1 should return 0 fee."""
        assert calculate_taker_fee(Decimal("0"), fee_enabled=True) == Decimal("0")
        assert calculate_taker_fee(Decimal("1"), fee_enabled=True) == Decimal("0")


class TestEdgeAfterFee:
    """Edge calculation accounting for fees."""

    def test_positive_edge_after_fee(self) -> None:
        """Model with 10% edge minus small fee should still be positive."""
        edge = calculate_edge_after_fee(
            model_prob=Decimal("0.60"),
            market_price=Decimal("0.50"),
            fee_enabled=True,
        )
        assert edge > Decimal("0")

    def test_small_edge_eaten_by_fee(self) -> None:
        """Model with 0.5% edge should be eaten by fee at midpoint."""
        edge = calculate_edge_after_fee(
            model_prob=Decimal("0.505"),
            market_price=Decimal("0.50"),
            fee_enabled=True,
        )
        # Fee at 0.50 is ~0.79%, edge is only 0.5%
        assert edge < Decimal("0")

    def test_no_fee_market_preserves_edge(self) -> None:
        """Fee-free market should preserve full edge."""
        edge = calculate_edge_after_fee(
            model_prob=Decimal("0.55"),
            market_price=Decimal("0.50"),
            fee_enabled=False,
        )
        assert edge == Decimal("0.05")


class TestMakerRebate:
    """Maker rebate estimation."""

    def test_rebate_positive(self) -> None:
        """Maker rebate should be positive for valid inputs."""
        rebate = estimate_maker_rebate(
            price=Decimal("0.50"), size=Decimal("100"), market_type="sports"
        )
        assert rebate > Decimal("0")

    def test_crypto_rebate_different_from_sports(self) -> None:
        """Crypto and sports should have different rebate rates."""
        sports = estimate_maker_rebate(
            price=Decimal("0.50"), size=Decimal("100"), market_type="sports"
        )
        crypto = estimate_maker_rebate(
            price=Decimal("0.50"), size=Decimal("100"), market_type="crypto"
        )
        assert sports != crypto


class TestFeeFavorable:
    """Fee favorability check for latency arb."""

    def test_extreme_price_is_favorable(self) -> None:
        """At p=0.96, fee should be <0.3% → favorable."""
        assert is_fee_favorable(Decimal("0.96"), fee_enabled=True) is True

    def test_midpoint_is_not_favorable(self) -> None:
        """At p=0.50, fee is ~0.79% → not favorable."""
        assert is_fee_favorable(Decimal("0.50"), fee_enabled=True) is False

    def test_fee_free_always_favorable(self) -> None:
        """Fee-free markets are always favorable."""
        assert is_fee_favorable(Decimal("0.50"), fee_enabled=False) is True
