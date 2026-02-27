"""Tests for Strategy C quality gate."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from arbo.connectors.weather_models import City
from arbo.strategies.weather_quality_gate import (
    QualityDecision,
    check_signal_quality,
    filter_signals,
)
from arbo.strategies.weather_scanner import (
    TemperatureBucket,
    WeatherMarketInfo,
    WeatherSignal,
)


def _make_signal(
    edge: float = 0.10,
    market_price: float = 0.40,
    volume_24h: float = 50000.0,
    liquidity: float = 25000.0,
    fee_enabled: bool = False,
    confidence: float = 0.7,
    city: City = City.NYC,
) -> WeatherSignal:
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
        forecast_probability=market_price + edge,
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
    """Test minimum edge threshold."""

    def test_below_min_edge_rejected(self) -> None:
        signal = _make_signal(edge=0.03)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "edge" in decision.reason.lower()

    def test_at_min_edge_passes(self) -> None:
        signal = _make_signal(edge=0.05)
        decision = check_signal_quality(signal, _recent_time())
        assert decision.passed


class TestVolumeFilter:
    """Test minimum volume threshold."""

    def test_low_volume_rejected(self) -> None:
        signal = _make_signal(volume_24h=1500.0)
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
        signal = _make_signal(liquidity=500.0)
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
    """Test extreme price sanity check."""

    def test_extreme_low_price_rejected(self) -> None:
        signal = _make_signal(market_price=0.01)
        decision = check_signal_quality(signal, _recent_time())
        assert not decision.passed
        assert "extreme" in decision.reason.lower()

    def test_extreme_high_price_rejected(self) -> None:
        signal = _make_signal(market_price=0.99, edge=0.005)
        decision = check_signal_quality(signal, _recent_time(), min_edge=0.001)
        assert not decision.passed
        assert "extreme" in decision.reason.lower()


class TestFilterSignals:
    """Test batch filtering."""

    def test_filters_mixed_signals(self) -> None:
        signals = [
            _make_signal(edge=0.10),  # Should pass
            _make_signal(edge=0.02),  # Should fail (low edge)
            _make_signal(volume_24h=1000),  # Should fail (low volume)
            _make_signal(edge=0.15),  # Should pass
        ]
        passed = filter_signals(signals, _recent_time())
        assert len(passed) == 2

    def test_empty_input(self) -> None:
        passed = filter_signals([], _recent_time())
        assert passed == []

    def test_all_pass(self) -> None:
        signals = [_make_signal(edge=0.10), _make_signal(edge=0.15)]
        passed = filter_signals(signals, _recent_time())
        assert len(passed) == 2

    def test_all_fail(self) -> None:
        signals = [_make_signal(edge=0.01), _make_signal(edge=0.02)]
        passed = filter_signals(signals, _recent_time())
        assert len(passed) == 0

    def test_custom_thresholds(self) -> None:
        signal = _make_signal(edge=0.08)
        # With default threshold (0.05) should pass
        passed_default = filter_signals([signal], _recent_time())
        assert len(passed_default) == 1

        # With higher threshold (0.10) should fail
        passed_strict = filter_signals([signal], _recent_time(), min_edge=0.10)
        assert len(passed_strict) == 0
