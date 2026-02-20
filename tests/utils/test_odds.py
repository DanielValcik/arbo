"""Tests for odds conversion and arbitrage math utilities."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.utils.odds import (
    american_to_decimal,
    arb_margin,
    commission_adjusted_odds,
    decimal_to_implied,
    fractional_to_decimal,
    half_kelly,
    implied_to_decimal,
    overround,
    remove_vig,
)


class TestDecimalToImplied:
    def test_even_odds(self) -> None:
        assert decimal_to_implied(Decimal("2.00")) == Decimal("0.5")

    def test_heavy_favourite(self) -> None:
        result = decimal_to_implied(Decimal("1.25"))
        assert result == Decimal("0.8")

    def test_longshot(self) -> None:
        result = decimal_to_implied(Decimal("10.0"))
        assert result == Decimal("0.1")

    def test_invalid_zero(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            decimal_to_implied(Decimal("0"))

    def test_invalid_negative(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            decimal_to_implied(Decimal("-1.5"))


class TestImpliedToDecimal:
    def test_fifty_percent(self) -> None:
        assert implied_to_decimal(Decimal("0.5")) == Decimal("2")

    def test_certainty(self) -> None:
        assert implied_to_decimal(Decimal("1")) == Decimal("1")

    def test_invalid_zero(self) -> None:
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            implied_to_decimal(Decimal("0"))

    def test_invalid_over_one(self) -> None:
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            implied_to_decimal(Decimal("1.5"))


class TestAmericanToDecimal:
    def test_positive_odds(self) -> None:
        assert american_to_decimal(150) == Decimal("2.5")

    def test_negative_odds(self) -> None:
        assert american_to_decimal(-200) == Decimal("1.5")

    def test_plus_100(self) -> None:
        assert american_to_decimal(100) == Decimal("2")

    def test_minus_100(self) -> None:
        assert american_to_decimal(-100) == Decimal("2")

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            american_to_decimal(0)


class TestFractionalToDecimal:
    def test_five_to_two(self) -> None:
        assert fractional_to_decimal(5, 2) == Decimal("3.5")

    def test_evens(self) -> None:
        assert fractional_to_decimal(1, 1) == Decimal("2")

    def test_odds_on(self) -> None:
        assert fractional_to_decimal(1, 2) == Decimal("1.5")

    def test_zero_denominator(self) -> None:
        with pytest.raises(ValueError, match="Denominator"):
            fractional_to_decimal(5, 0)

    def test_negative_numerator(self) -> None:
        with pytest.raises(ValueError, match="Numerator"):
            fractional_to_decimal(-1, 2)


class TestOverround:
    def test_fair_book(self) -> None:
        probs = [Decimal("0.5"), Decimal("0.5")]
        assert overround(probs) == Decimal("1.0")

    def test_with_margin(self) -> None:
        # Typical bookmaker: 5% overround
        probs = [Decimal("0.525"), Decimal("0.525")]
        assert overround(probs) == Decimal("1.050")

    def test_three_way(self) -> None:
        probs = [Decimal("0.4"), Decimal("0.3"), Decimal("0.35")]
        assert overround(probs) == Decimal("1.05")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            overround([])


class TestRemoveVig:
    def test_already_fair(self) -> None:
        probs = [Decimal("0.5"), Decimal("0.5")]
        result = remove_vig(probs)
        assert result == [Decimal("0.5"), Decimal("0.5")]

    def test_removes_margin(self) -> None:
        # 5% overround: each is 0.525, normalized to 0.5
        probs = [Decimal("0.525"), Decimal("0.525")]
        result = remove_vig(probs)
        assert result[0] == Decimal("0.5")
        assert result[1] == Decimal("0.5")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            remove_vig([])


class TestCommissionAdjustedOdds:
    def test_four_percent(self) -> None:
        # odds=2.10, comm=0.04 → 1 + 1.10 * 0.96 = 2.056
        result = commission_adjusted_odds(Decimal("2.10"), Decimal("0.04"))
        assert result == Decimal("2.056")

    def test_zero_commission(self) -> None:
        result = commission_adjusted_odds(Decimal("3.50"), Decimal("0"))
        assert result == Decimal("3.50")

    def test_high_odds(self) -> None:
        # odds=10.0, comm=0.04 → 1 + 9.0 * 0.96 = 9.64
        result = commission_adjusted_odds(Decimal("10.0"), Decimal("0.04"))
        assert result == Decimal("9.64")


class TestArbMargin:
    def test_arb_exists(self) -> None:
        # back=2.20 at bookmaker, lay=2.05 at Matchbook (4% commission)
        # adj_lay = 1 + 1.05 * 0.96 = 2.008
        # margin = 1/2.20 + 1/2.008 - 1 = 0.4545 + 0.4980 - 1 = -0.0475
        margin = arb_margin(Decimal("2.20"), Decimal("2.05"), Decimal("0.04"))
        assert margin < 0  # Negative = arb exists

    def test_no_arb(self) -> None:
        # back=1.80 at bookmaker, lay=2.00 at exchange
        # adj_lay = 1 + 1.00 * 0.96 = 1.96
        # margin = 1/1.80 + 1/1.96 - 1 = 0.5556 + 0.5102 - 1 = +0.0658
        margin = arb_margin(Decimal("1.80"), Decimal("2.00"), Decimal("0.04"))
        assert margin > 0  # Positive = no arb

    def test_borderline(self) -> None:
        # Same odds with commission should give positive margin (no arb)
        margin = arb_margin(Decimal("2.00"), Decimal("2.00"), Decimal("0.04"))
        assert margin > 0


class TestHalfKelly:
    def test_basic(self) -> None:
        # edge=0.05, odds=2.50 → 0.5 * 0.05 / 1.50 = 0.01667
        result = half_kelly(Decimal("0.05"), Decimal("2.50"))
        expected = Decimal("0.5") * Decimal("0.05") / Decimal("1.50")
        assert result == expected

    def test_high_edge(self) -> None:
        # edge=0.10, odds=2.00 → 0.5 * 0.10 / 1.00 = 0.05
        result = half_kelly(Decimal("0.10"), Decimal("2.00"))
        assert result == Decimal("0.05")

    def test_invalid_odds(self) -> None:
        with pytest.raises(ValueError, match="Odds must be > 1"):
            half_kelly(Decimal("0.05"), Decimal("1.0"))
