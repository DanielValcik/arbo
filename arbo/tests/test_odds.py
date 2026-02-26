"""Tests for odds utility functions (RDH-308).

Tests verify:
1. quarter_kelly() computes correct fraction
2. quarter_kelly() respects max 5% cap
3. half_kelly() remains unchanged
4. Edge cases: zero edge, boundary prices
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from arbo.utils.odds import half_kelly, quarter_kelly


class TestQuarterKelly:
    """Quarter-Kelly position sizing."""

    def test_basic_calculation(self) -> None:
        """quarter_kelly returns ~0.25x of full Kelly."""
        # model_prob=0.60, market_price=0.50 (decimal_odds=2.0)
        # full_kelly = (0.60 * 2.0 - 1) / (2.0 - 1) = 0.20 / 1.0 = 0.20
        # quarter_kelly = 0.20 * 0.25 = 0.05
        result = quarter_kelly(Decimal("0.60"), Decimal("0.50"))
        assert result == Decimal("0.05")

    def test_small_edge(self) -> None:
        """Small edge produces small position."""
        # model_prob=0.55, market_price=0.50
        # full_kelly = (0.55 * 2.0 - 1) / (2.0 - 1) = 0.10
        # quarter = 0.10 * 0.25 = 0.025
        result = quarter_kelly(Decimal("0.55"), Decimal("0.50"))
        assert result == Decimal("0.025")

    def test_max_position_cap(self) -> None:
        """Position capped at max_position_pct."""
        # Large edge → quarter Kelly exceeds 5% cap
        # model_prob=0.90, market_price=0.50
        # full_kelly = (0.90 * 2.0 - 1) / (2.0 - 1) = 0.80
        # quarter = 0.80 * 0.25 = 0.20 → capped at 0.05
        result = quarter_kelly(Decimal("0.90"), Decimal("0.50"))
        assert result == Decimal("0.05")

    def test_custom_max_position(self) -> None:
        """Custom max_position_pct is respected."""
        result = quarter_kelly(
            Decimal("0.90"),
            Decimal("0.50"),
            max_position_pct=Decimal("0.10"),
        )
        assert result == Decimal("0.10")

    def test_no_edge_returns_zero(self) -> None:
        """No edge (model = market) returns zero."""
        result = quarter_kelly(Decimal("0.50"), Decimal("0.50"))
        assert result == Decimal(0)

    def test_negative_edge_returns_zero(self) -> None:
        """Negative edge (market overpriced in our favor) returns zero."""
        result = quarter_kelly(Decimal("0.40"), Decimal("0.50"))
        assert result == Decimal(0)

    def test_boundary_price_zero(self) -> None:
        """Price <= 0 returns zero."""
        assert quarter_kelly(Decimal("0.50"), Decimal("0")) == Decimal(0)
        assert quarter_kelly(Decimal("0.50"), Decimal("-0.1")) == Decimal(0)

    def test_boundary_price_one(self) -> None:
        """Price >= 1 returns zero."""
        assert quarter_kelly(Decimal("0.50"), Decimal("1.0")) == Decimal(0)
        assert quarter_kelly(Decimal("0.50"), Decimal("1.5")) == Decimal(0)

    def test_boundary_prob_zero(self) -> None:
        """Model prob <= 0 returns zero."""
        assert quarter_kelly(Decimal("0"), Decimal("0.50")) == Decimal(0)
        assert quarter_kelly(Decimal("-0.1"), Decimal("0.50")) == Decimal(0)

    def test_boundary_prob_one(self) -> None:
        """Model prob >= 1 returns zero."""
        assert quarter_kelly(Decimal("1.0"), Decimal("0.50")) == Decimal(0)

    def test_half_vs_quarter(self) -> None:
        """Quarter-Kelly is exactly half of half-Kelly (both with same edge)."""
        prob = Decimal("0.60")
        price = Decimal("0.50")
        h = half_kelly(prob, price, max_position_pct=Decimal("1.0"))
        q = quarter_kelly(prob, price, max_position_pct=Decimal("1.0"))
        assert q == h / 2


class TestHalfKellyUnchanged:
    """Verify half_kelly() still works correctly (backward compat)."""

    def test_basic_half_kelly(self) -> None:
        """half_kelly returns 0.5x of full Kelly (capped at 5%)."""
        # full_kelly = 0.20, half = 0.10, but default cap = 0.05
        result = half_kelly(Decimal("0.60"), Decimal("0.50"))
        assert result == Decimal("0.05")
        # Without cap: returns 0.10
        uncapped = half_kelly(Decimal("0.60"), Decimal("0.50"), max_position_pct=Decimal("1.0"))
        assert uncapped == Decimal("0.10")

    def test_half_kelly_cap(self) -> None:
        """half_kelly respects 5% cap."""
        result = half_kelly(Decimal("0.90"), Decimal("0.50"))
        assert result == Decimal("0.05")

    def test_half_kelly_no_edge(self) -> None:
        """half_kelly returns zero when no edge."""
        result = half_kelly(Decimal("0.50"), Decimal("0.50"))
        assert result == Decimal(0)
