"""Tests for Strategy C quality gate (AR-0134 configuration)."""

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
    """Return a timestamp from 1 hour ago (fresh forecast)."""
    return datetime.now(timezone.utc) - timedelta(hours=1)


def _old_time() -> datetime:
    """Return a timestamp from 8 hours ago (stale forecast)."""
    return datetime.now(timezone.utc) - timedelta(hours=8)


class TestQualityGatePassing:
    """Test signals that should pass quality checks."""

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
    """Test minimum edge threshold (AR-0134: MIN_EDGE=0.10)."""

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
        """AR-0134: MAX_EDGE=0.70, edge=0.75 should fail."""
        signal = _make_signal(edge=0.75)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "anomaly" in decision.reason.lower()


class TestVolumeFilter:
    """Test minimum volume threshold (AR-0134: MIN_VOLUME=510)."""

    def test_low_volume_rejected(self) -> None:
        signal = _make_signal(volume_24h=400.0)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "volume" in decision.reason.lower()

    def test_above_volume_passes(self) -> None:
        signal = _make_signal(volume_24h=3000.0)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed


class TestLiquidityFilter:
    """Test minimum liquidity threshold."""

    def test_low_liquidity_rejected(self) -> None:
        signal = _make_signal(liquidity=100.0)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "liquidity" in decision.reason.lower()


class TestFeeFilter:
    """Test fee-free requirement."""

    def test_fee_enabled_rejected(self) -> None:
        signal = _make_signal(fee_enabled=True)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "fee" in decision.reason.lower()


class TestConfidenceFilter:
    """Test minimum confidence threshold."""

    def test_low_confidence_rejected(self) -> None:
        signal = _make_signal(confidence=0.3)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "confidence" in decision.reason.lower()


class TestFreshnessFilter:
    """Test forecast freshness requirement."""

    def test_stale_forecast_rejected(self) -> None:
        signal = _make_signal()
        decision = check_signal_quality(signal, _old_time())
        assert not decision.passed
        assert "old" in decision.reason.lower() or "forecast" in decision.reason.lower()

    def test_fresh_forecast_passes(self) -> None:
        signal = _make_signal()
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed


class TestPriceFilter:
    """Test price range checks (AR-0134: MIN_PRICE=0.05, MAX_PRICE=0.70)."""

    def test_low_price_rejected(self) -> None:
        signal = _make_signal(market_price=0.03)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "price" in decision.reason.lower() and "below" in decision.reason.lower()

    def test_high_price_rejected(self) -> None:
        signal = _make_signal(market_price=0.75, edge=0.15)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "price" in decision.reason.lower() and "above" in decision.reason.lower()

    def test_wide_price_range_passes(self) -> None:
        """AR-0134 allows much wider price range than before."""
        for price in [0.06, 0.35, 0.65]:
            signal = _make_signal(market_price=price, edge=0.15)
            decision = check_signal_quality(signal, _recent_time())
            assert decision.passed, f"price={price} should pass"


class TestForecastProbFilter:
    """Test minimum forecast probability (AR-0134: MIN_FORECAST_PROB=0.06)."""

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
    """Test per-city threshold overrides (AR-0134 config)."""

    def test_excluded_city_chicago_rejected(self) -> None:
        """Chicago has min_edge=0.99, so all signals should be rejected."""
        signal = _make_signal(city=City.CHICAGO, edge=0.50)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "edge" in decision.reason.lower()
        assert "chicago" in decision.reason.lower()

    def test_excluded_city_seoul_rejected(self) -> None:
        signal = _make_signal(city=City.SEOUL, edge=0.50)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed

    def test_nyc_not_excluded(self) -> None:
        """AR-0134: NYC is NOT excluded (no override)."""
        signal = _make_signal(city=City.NYC, edge=0.15)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed

    def test_toronto_per_city_override(self) -> None:
        """Toronto has min_edge=0.005, max_price=0.40 — edge=0.15 should pass."""
        signal = _make_signal(city=City.TORONTO, edge=0.15, market_price=0.35)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed

    def test_toronto_high_price_rejected(self) -> None:
        """Toronto has max_price=0.40, so price=0.45 should fail."""
        signal = _make_signal(city=City.TORONTO, market_price=0.45, edge=0.15)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "above" in decision.reason.lower()

    def test_dallas_wide_price_range(self) -> None:
        """Dallas has max_price=0.80 — price=0.75 should pass."""
        signal = _make_signal(city=City.DALLAS, market_price=0.75, edge=0.15)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed

    def test_miami_strict_price(self) -> None:
        """Miami has max_price=0.40, min_edge=0.05 — price=0.45 should fail."""
        signal = _make_signal(city=City.MIAMI, market_price=0.45, edge=0.15)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed

    def test_get_threshold_city_override(self) -> None:
        assert _get_threshold("min_edge", "chicago") == 0.99
        assert _get_threshold("max_price", "dallas") == 0.80
        assert _get_threshold("min_edge", "ankara") == 0.005

    def test_get_threshold_global_default(self) -> None:
        assert _get_threshold("min_edge", "nyc") == 0.10
        assert _get_threshold("max_price", "nyc") == 0.70

    def test_get_threshold_unknown_city(self) -> None:
        assert _get_threshold("min_edge", "unknown_city") == 0.10


class TestFilterSignals:
    """Test batch filtering."""

    def test_filters_mixed_signals(self) -> None:
        signals = [
            _make_signal(edge=0.28),  # Should pass (NYC)
            _make_signal(edge=0.02),  # Should fail (low edge)
            _make_signal(volume_24h=400),  # Should fail (low volume)
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

    def test_custom_thresholds(self) -> None:
        signal = _make_signal(edge=0.28)
        # With default threshold (0.10) should pass
        passed_default = filter_signals([signal], _recent_time())
        assert len(passed_default) == 1

        # With higher threshold (0.30) should fail
        passed_strict = filter_signals([signal], _recent_time(), min_edge=0.30)
        assert len(passed_strict) == 0

    def test_excluded_cities_filtered_out(self) -> None:
        """Signals from excluded cities (Chicago, Seoul) should not pass."""
        signals = [
            _make_signal(city=City.CHICAGO, edge=0.28),   # Excluded (min_edge=0.99)
            _make_signal(city=City.NYC, edge=0.28),        # Should pass
            _make_signal(city=City.SEOUL, edge=0.28),      # Excluded (min_edge=0.99)
            _make_signal(city=City.PARIS, edge=0.28),      # Should pass
        ]
        passed = filter_signals(signals, _recent_time())
        assert len(passed) == 2
        passed_cities = {s.market.city for s in passed}
        assert City.CHICAGO not in passed_cities
        assert City.SEOUL not in passed_cities
