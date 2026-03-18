"""Tests for Strategy C quality gate (C1f/V4a configuration)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from arbo.connectors.weather_models import City
from arbo.strategies.weather_quality_gate import (
    CITY_OVERRIDES,
    QualityDecision,
    _get_threshold,
    check_signal_quality,
    filter_signals,
)
from arbo.strategies.weather_scanner import (
    TemperatureBucket,
    WeatherMarketInfo,
    WeatherSignal,
)


def _make_signal(
    edge: float = 0.28,
    market_price: float = 0.35,
    volume_24h: float = 50000.0,
    liquidity: float = 25000.0,
    fee_enabled: bool = False,
    confidence: float = 0.7,
    forecast_probability: float | None = None,
    city: City = City.NYC,
) -> WeatherSignal:
    fp = forecast_probability if forecast_probability is not None else market_price + edge
    return WeatherSignal(
        market=WeatherMarketInfo(
            condition_id="mkt_1",
            question=f"Temperature in {city.value}",
            city=city,
            target_date=date(2026, 3, 15),
            is_high_temp=True,
            bucket=TemperatureBucket(low_c=20.0, high_c=25.0, bucket_type="range"),
            market_price=market_price,
            token_id_yes="tok_yes",
            token_id_no="tok_no",
            neg_risk=False,
            fee_enabled=fee_enabled,
            volume_24h=volume_24h,
            liquidity=liquidity,
        ),
        forecast_temp_c=22.5,
        forecast_probability=fp,
        edge=edge,
        direction="BUY_YES",
        confidence=confidence,
    )


def _recent_time() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=1)


def _old_time() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=8)


class TestQualityGatePassing:
    def test_good_signal_passes(self) -> None:
        signal = _make_signal()
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed
        assert decision.signal is signal

    def test_passing_reason(self) -> None:
        signal = _make_signal()
        decision = check_signal_quality(signal, _recent_time())
        assert "passed" in decision.reason.lower()


class TestEdgeFilter:
    """Test minimum edge threshold (C1f: MIN_EDGE=0.10)."""

    def test_below_min_edge_rejected(self) -> None:
        signal = _make_signal(edge=0.05)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "edge" in decision.reason.lower()

    def test_at_min_edge_passes(self) -> None:
        signal = _make_signal(edge=0.28)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed

    def test_max_edge_rejected(self) -> None:
        """C1f: MAX_EDGE=0.90, edge=0.95 should fail."""
        signal = _make_signal(edge=0.95)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "anomaly" in decision.reason.lower()


class TestVolumeFilter:
    """C1f: MIN_VOLUME=0 (no volume filter)."""

    def test_low_volume_passes(self) -> None:
        """V4a/C1f has min_volume=0, so low volume should pass."""
        signal = _make_signal(volume_24h=50.0)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed

    def test_above_volume_passes(self) -> None:
        signal = _make_signal(volume_24h=3000.0)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed


class TestLiquidityFilter:
    def test_low_liquidity_rejected(self) -> None:
        signal = _make_signal(liquidity=100.0)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "liquidity" in decision.reason.lower()


class TestFeeFilter:
    def test_fee_enabled_rejected(self) -> None:
        signal = _make_signal(fee_enabled=True)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "fee" in decision.reason.lower()


class TestConfidenceFilter:
    def test_low_confidence_rejected(self) -> None:
        signal = _make_signal(confidence=0.3)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "confidence" in decision.reason.lower()


class TestFreshnessFilter:
    def test_stale_forecast_rejected(self) -> None:
        signal = _make_signal()
        decision = check_signal_quality(signal, _old_time())
        assert not decision.passed

    def test_fresh_forecast_passes(self) -> None:
        signal = _make_signal()
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed


class TestPriceFilter:
    """Test price range checks (C1f: MIN_PRICE=0.08, MAX_PRICE=0.56)."""

    def test_low_price_rejected(self) -> None:
        signal = _make_signal(market_price=0.06)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "price" in decision.reason.lower() and "below" in decision.reason.lower()

    def test_high_price_rejected(self) -> None:
        # NYC has max_price override of 0.70, so use Paris (no override, global 0.56)
        signal = _make_signal(market_price=0.60, edge=0.15, city=City.PARIS)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "price" in decision.reason.lower() and "above" in decision.reason.lower()

    def test_valid_price_range_passes(self) -> None:
        for price in [0.10, 0.35, 0.50]:
            signal = _make_signal(market_price=price, edge=0.15)
            decision = check_signal_quality(signal, _recent_time())
            assert decision.passed, f"price={price} should pass"


class TestForecastProbFilter:
    """Test minimum forecast probability (C1f: MIN_FORECAST_PROB=0.07)."""

    def test_low_forecast_prob_rejected(self) -> None:
        signal = _make_signal(edge=0.10, market_price=0.35, forecast_probability=0.04)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "forecast prob" in decision.reason.lower()

    def test_high_forecast_prob_passes(self) -> None:
        signal = _make_signal(edge=0.28, market_price=0.35, forecast_probability=0.70)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed


class TestPerCityOverrides:
    """Test per-city threshold overrides (C1f/V4a config)."""

    def test_excluded_city_wellington_rejected(self) -> None:
        """Wellington has min_edge=0.99, so all signals should be rejected."""
        signal = _make_signal(city=City.WELLINGTON, edge=0.50)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "edge" in decision.reason.lower()

    def test_chicago_not_excluded(self) -> None:
        """C1f: Chicago is NOT excluded (unlike AR-0134)."""
        signal = _make_signal(city=City.CHICAGO, edge=0.15, market_price=0.35)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed

    def test_seoul_low_edge_passes(self) -> None:
        """Seoul has min_edge=0.05, so edge=0.06 should pass."""
        signal = _make_signal(city=City.SEOUL, edge=0.06, market_price=0.35)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed

    def test_nyc_per_city_override(self) -> None:
        """NYC has min_edge=0.08, max_price=0.70."""
        signal = _make_signal(city=City.NYC, edge=0.09, market_price=0.35)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed

    def test_toronto_high_price_rejected(self) -> None:
        """Toronto has max_price=0.60, so price=0.65 should fail."""
        signal = _make_signal(city=City.TORONTO, market_price=0.65, edge=0.15)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "above" in decision.reason.lower()

    def test_get_threshold_city_override(self) -> None:
        assert _get_threshold("min_edge", "seoul") == 0.05
        assert _get_threshold("max_price", "munich") == 0.80
        assert _get_threshold("min_edge", "atlanta") == 0.08

    def test_get_threshold_global_default(self) -> None:
        assert _get_threshold("min_edge", "chicago") == 0.10  # No override → global
        assert _get_threshold("max_price", "chicago") == 0.56  # No override → global

    def test_get_threshold_unknown_city(self) -> None:
        assert _get_threshold("min_edge", "unknown_city") == 0.10


class TestFilterSignals:
    def test_filters_mixed_signals(self) -> None:
        signals = [
            _make_signal(edge=0.28),  # Should pass (NYC)
            _make_signal(edge=0.02),  # Should fail (low edge)
            _make_signal(edge=0.30),  # Should pass
        ]
        passed = filter_signals(signals, _recent_time())
        assert len(passed) == 2

    def test_empty_input(self) -> None:
        passed = filter_signals([], _recent_time())
        assert passed == []

    def test_all_pass(self) -> None:
        signals = [_make_signal(edge=0.28), _make_signal(edge=0.30)]
        passed = filter_signals(signals, _recent_time())
        assert len(passed) == 2

    def test_all_fail(self) -> None:
        signals = [_make_signal(edge=0.01), _make_signal(edge=0.02)]
        passed = filter_signals(signals, _recent_time())
        assert len(passed) == 0

    def test_excluded_cities_filtered_out(self) -> None:
        """Signals from excluded cities (Wellington) should not pass."""
        signals = [
            _make_signal(city=City.WELLINGTON, edge=0.28),  # Excluded
            _make_signal(city=City.NYC, edge=0.28),          # Should pass
            _make_signal(city=City.CHICAGO, edge=0.28),      # Should pass (not excluded in C1f)
            _make_signal(city=City.PARIS, edge=0.28),        # Should pass
        ]
        passed = filter_signals(signals, _recent_time())
        assert len(passed) == 3
        passed_cities = {s.market.city for s in passed}
        assert City.WELLINGTON not in passed_cities
