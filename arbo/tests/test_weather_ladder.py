"""Tests for temperature laddering logic (Strategy C)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pytest

from arbo.connectors.weather_models import City
from arbo.core.risk_manager import KELLY_FRACTION
from arbo.strategies.weather_ladder import (
    LadderPosition,
    TemperatureLadder,
    build_ladders_by_city,
    build_temperature_ladder,
    calculate_kelly_size,
)
from arbo.strategies.weather_scanner import (
    TemperatureBucket,
    WeatherMarketInfo,
    WeatherSignal,
)


def _make_signal(
    city: City = City.NYC,
    target_date: date | None = None,
    edge: float = 0.10,
    market_price: float = 0.40,
    condition_id: str = "mkt_1",
) -> WeatherSignal:
    """Create a mock WeatherSignal for testing."""
    d = target_date or date(2026, 3, 15)
    return WeatherSignal(
        market=WeatherMarketInfo(
            condition_id=condition_id,
            question=f"Temperature in {city.value} on {d}",
            city=city,
            target_date=d,
            is_high_temp=True,
            bucket=TemperatureBucket(low_c=20.0, high_c=25.0, bucket_type="range"),
            market_price=market_price,
            token_id_yes="tok_yes",
            token_id_no="tok_no",
            neg_risk=False,
            fee_enabled=False,
            volume_24h=50000.0,
            liquidity=25000.0,
        ),
        forecast_temp_c=22.5,
        forecast_probability=market_price + edge,
        edge=edge,
        direction="BUY_YES",
        confidence=0.7,
    )


class TestCalculateKellySize:
    """Test Quarter-Kelly position sizing."""

    def test_positive_edge_returns_positive_size(self) -> None:
        size = calculate_kelly_size(
            edge=0.10, price=0.40, available_capital=Decimal("1000")
        )
        assert size > Decimal("0")

    def test_zero_edge_returns_zero(self) -> None:
        size = calculate_kelly_size(
            edge=0.0, price=0.40, available_capital=Decimal("1000")
        )
        assert size == Decimal("0")

    def test_negative_edge_returns_zero(self) -> None:
        size = calculate_kelly_size(
            edge=-0.05, price=0.40, available_capital=Decimal("1000")
        )
        assert size == Decimal("0")

    def test_quarter_kelly_is_conservative(self) -> None:
        """Quarter-Kelly should give smaller size than full Kelly."""
        quarter = calculate_kelly_size(
            edge=0.10, price=0.40, available_capital=Decimal("1000"),
            kelly_fraction=Decimal("0.25"),
        )
        full = calculate_kelly_size(
            edge=0.10, price=0.40, available_capital=Decimal("1000"),
            kelly_fraction=Decimal("1.0"),
        )
        assert quarter < full
        assert abs(quarter * 4 - full) < Decimal("0.10")  # ~4x ratio

    def test_larger_edge_gives_larger_size(self) -> None:
        small_edge = calculate_kelly_size(
            edge=0.05, price=0.40, available_capital=Decimal("1000")
        )
        large_edge = calculate_kelly_size(
            edge=0.20, price=0.40, available_capital=Decimal("1000")
        )
        assert large_edge > small_edge

    def test_size_proportional_to_capital(self) -> None:
        small = calculate_kelly_size(
            edge=0.10, price=0.40, available_capital=Decimal("500")
        )
        large = calculate_kelly_size(
            edge=0.10, price=0.40, available_capital=Decimal("1000")
        )
        assert abs(large - small * 2) < Decimal("0.10")

    def test_extreme_price_returns_zero(self) -> None:
        assert calculate_kelly_size(0.10, 0.0, Decimal("1000")) == Decimal("0")
        assert calculate_kelly_size(0.10, 1.0, Decimal("1000")) == Decimal("0")

    def test_uses_quarter_kelly_by_default(self) -> None:
        """Verify default kelly_fraction is KELLY_FRACTION (0.25)."""
        assert KELLY_FRACTION == Decimal("0.25")


class TestBuildTemperatureLadder:
    """Test ladder construction from signals."""

    def test_single_signal_ladder(self) -> None:
        signals = [_make_signal(edge=0.10)]
        ladder = build_temperature_ladder(signals, Decimal("1000"))
        assert ladder is not None
        assert ladder.num_positions == 1
        assert ladder.total_size_usdc > Decimal("0")

    def test_multiple_signals_top_3(self) -> None:
        signals = [
            _make_signal(edge=0.05, condition_id="m1"),
            _make_signal(edge=0.15, condition_id="m2"),
            _make_signal(edge=0.10, condition_id="m3"),
            _make_signal(edge=0.20, condition_id="m4"),
        ]
        ladder = build_temperature_ladder(signals, Decimal("1000"), max_positions=3)
        assert ladder is not None
        assert ladder.num_positions == 3
        # Should be ordered by priority (best edge first)
        assert ladder.positions[0].priority == 1
        assert ladder.positions[0].signal.edge == 0.20

    def test_empty_signals_returns_none(self) -> None:
        assert build_temperature_ladder([], Decimal("1000")) is None

    def test_below_min_edge_filtered(self) -> None:
        signals = [_make_signal(edge=0.02)]  # Below default MIN_LADDER_EDGE (0.03)
        ladder = build_temperature_ladder(signals, Decimal("1000"))
        assert ladder is None

    def test_total_size_matches_sum(self) -> None:
        signals = [
            _make_signal(edge=0.10, condition_id="m1"),
            _make_signal(edge=0.15, condition_id="m2"),
        ]
        ladder = build_temperature_ladder(signals, Decimal("1000"))
        assert ladder is not None
        expected_total = sum(p.size_usdc for p in ladder.positions)
        assert ladder.total_size_usdc == expected_total

    def test_remaining_capital_decreases(self) -> None:
        """Each position should be sized from remaining capital after previous."""
        signals = [
            _make_signal(edge=0.15, condition_id="m1"),
            _make_signal(edge=0.10, condition_id="m2"),
        ]
        ladder = build_temperature_ladder(signals, Decimal("1000"))
        assert ladder is not None
        # First position uses full capital, second uses reduced
        assert ladder.positions[0].size_usdc > ladder.positions[1].size_usdc


class TestBuildLaddersByCity:
    """Test building ladders for multiple cities."""

    def test_separate_cities_get_separate_ladders(self) -> None:
        signals = [
            _make_signal(city=City.NYC, edge=0.10, condition_id="nyc_1"),
            _make_signal(city=City.CHICAGO, edge=0.12, condition_id="chi_1"),
        ]
        ladders = build_ladders_by_city(signals, Decimal("1000"))
        assert len(ladders) == 2
        cities = {l.city for l in ladders}
        assert "nyc" in cities
        assert "chicago" in cities

    def test_capital_distributed_proportionally(self) -> None:
        signals = [
            _make_signal(city=City.NYC, edge=0.10, condition_id="nyc_1"),
            _make_signal(city=City.CHICAGO, edge=0.10, condition_id="chi_1"),
        ]
        ladders = build_ladders_by_city(signals, Decimal("1000"))
        # Each city should get ~$500 (1000 / 2 cities)
        for ladder in ladders:
            assert ladder.total_size_usdc < Decimal("500")

    def test_empty_signals_returns_empty(self) -> None:
        assert build_ladders_by_city([], Decimal("1000")) == []

    def test_same_city_different_dates(self) -> None:
        signals = [
            _make_signal(city=City.NYC, target_date=date(2026, 3, 15), edge=0.10, condition_id="m1"),
            _make_signal(city=City.NYC, target_date=date(2026, 3, 16), edge=0.12, condition_id="m2"),
        ]
        ladders = build_ladders_by_city(signals, Decimal("1000"))
        assert len(ladders) == 2
