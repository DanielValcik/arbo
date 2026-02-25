"""Tests for weather market scanner (Strategy C)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from arbo.connectors.weather_models import (
    City,
    DailyForecast,
    WeatherForecast,
    WeatherSource,
    fahrenheit_to_celsius,
)
from arbo.strategies.weather_scanner import (
    TemperatureBucket,
    WeatherSignal,
    _normal_cdf,
    estimate_bucket_probability,
    is_high_temp_market,
    parse_city,
    parse_target_date,
    parse_temperature_bucket,
    scan_weather_market,
    scan_weather_markets,
)


# ================================================================
# Mock market objects
# ================================================================
@dataclass
class MockMarket:
    """Minimal market mock matching GammaMarket interface."""

    condition_id: str = "0xabc123"
    question: str = ""
    category: str = "weather"
    price_yes: Decimal | None = Decimal("0.50")
    price_no: Decimal | None = Decimal("0.50")
    token_id_yes: str = "tok_yes_1"
    token_id_no: str = "tok_no_1"
    neg_risk: bool = False
    fee_enabled: bool = False
    volume_24h: Decimal = Decimal("50000")
    liquidity: Decimal = Decimal("25000")
    slug: str = "weather-nyc"


def _make_forecast(
    city: City,
    high_c: float = 20.0,
    low_c: float = 10.0,
    target_date: date | None = None,
) -> WeatherForecast:
    """Create a mock WeatherForecast for testing."""
    d = target_date or date(2026, 3, 15)
    return WeatherForecast(
        city=city,
        source=WeatherSource.NOAA,
        fetched_at=datetime.now(timezone.utc),
        daily_forecasts=[
            DailyForecast(
                date=d,
                temp_high_c=high_c,
                temp_low_c=low_c,
                condition="Sunny",
                precip_probability=10.0,
            ),
        ],
    )


# ================================================================
# Tests: City Parsing
# ================================================================
class TestParseCity:
    def test_nyc(self) -> None:
        assert parse_city("Will the high temperature in NYC be above 75°F?") == City.NYC

    def test_new_york(self) -> None:
        assert parse_city("New York high temperature on March 15") == City.NYC

    def test_chicago(self) -> None:
        assert parse_city("Chicago daily high above 50°F?") == City.CHICAGO

    def test_london(self) -> None:
        assert parse_city("London temperature on March 10") == City.LONDON

    def test_seoul(self) -> None:
        assert parse_city("Seoul high temperature forecast") == City.SEOUL

    def test_buenos_aires(self) -> None:
        assert parse_city("Buenos Aires temperature above 30°C?") == City.BUENOS_AIRES

    def test_unknown_city(self) -> None:
        assert parse_city("Will it rain tomorrow?") is None


# ================================================================
# Tests: Date Parsing
# ================================================================
class TestParseDate:
    def test_full_month(self) -> None:
        assert parse_target_date("March 15, 2026") == date(2026, 3, 15)

    def test_month_day_no_year(self) -> None:
        result = parse_target_date("March 15", default_year=2026)
        assert result == date(2026, 3, 15)

    def test_abbreviated_month(self) -> None:
        assert parse_target_date("Mar 15, 2026") == date(2026, 3, 15)

    def test_no_date(self) -> None:
        assert parse_target_date("Will it be hot?") is None


# ================================================================
# Tests: Temperature Bucket Parsing
# ================================================================
class TestParseBucket:
    def test_range_fahrenheit(self) -> None:
        bucket = parse_temperature_bucket("70-74°F")
        assert bucket is not None
        assert bucket.bucket_type == "range"
        assert abs(bucket.low_c - fahrenheit_to_celsius(70)) < 0.1
        assert abs(bucket.high_c - fahrenheit_to_celsius(74)) < 0.1

    def test_above_fahrenheit(self) -> None:
        bucket = parse_temperature_bucket("above 75°F")
        assert bucket is not None
        assert bucket.bucket_type == "above"
        assert abs(bucket.low_c - fahrenheit_to_celsius(75)) < 0.1
        assert bucket.high_c is None

    def test_below_fahrenheit(self) -> None:
        bucket = parse_temperature_bucket("below 45°F")
        assert bucket is not None
        assert bucket.bucket_type == "below"
        assert bucket.low_c is None
        assert abs(bucket.high_c - fahrenheit_to_celsius(45)) < 0.1

    def test_celsius_range(self) -> None:
        bucket = parse_temperature_bucket("20-25°C")
        assert bucket is not None
        assert bucket.bucket_type == "range"
        assert bucket.low_c == 20.0
        assert bucket.high_c == 25.0

    def test_above_with_or(self) -> None:
        bucket = parse_temperature_bucket("75°F or above")
        assert bucket is not None
        assert bucket.bucket_type == "above"

    def test_no_bucket(self) -> None:
        assert parse_temperature_bucket("Will it rain?") is None


class TestBucketContains:
    def test_range_contains(self) -> None:
        bucket = TemperatureBucket(low_c=20.0, high_c=25.0, bucket_type="range")
        assert bucket.contains_temp(22.0) is True
        assert bucket.contains_temp(19.9) is False
        assert bucket.contains_temp(25.0) is False  # upper bound exclusive
        assert bucket.contains_temp(20.0) is True  # lower bound inclusive

    def test_above_contains(self) -> None:
        bucket = TemperatureBucket(low_c=25.0, high_c=None, bucket_type="above")
        assert bucket.contains_temp(30.0) is True
        assert bucket.contains_temp(25.0) is True  # inclusive
        assert bucket.contains_temp(24.9) is False

    def test_below_contains(self) -> None:
        bucket = TemperatureBucket(low_c=None, high_c=20.0, bucket_type="below")
        assert bucket.contains_temp(15.0) is True
        assert bucket.contains_temp(20.0) is False  # exclusive
        assert bucket.contains_temp(19.9) is True


# ================================================================
# Tests: High/Low Temperature
# ================================================================
class TestIsHighTemp:
    def test_high_temp(self) -> None:
        assert is_high_temp_market("High temperature in NYC") is True

    def test_low_temp(self) -> None:
        assert is_high_temp_market("Low temperature in NYC") is False

    def test_overnight_low(self) -> None:
        assert is_high_temp_market("Overnight low in Chicago") is False

    def test_default_high(self) -> None:
        assert is_high_temp_market("NYC temperature above 75°F") is True


# ================================================================
# Tests: Probability Estimation
# ================================================================
class TestEstimateProbability:
    def test_forecast_equals_bucket_center(self) -> None:
        """When forecast is at bucket center, probability should be moderate."""
        forecast = DailyForecast(
            date=date(2026, 3, 15), temp_high_c=22.5, temp_low_c=10.0
        )
        bucket = TemperatureBucket(low_c=20.0, high_c=25.0, bucket_type="range")
        prob = estimate_bucket_probability(forecast, bucket, is_high=True)
        # Centered in 5°C range with σ=2.5 → ~68% should be in bucket
        assert 0.5 < prob < 0.9

    def test_forecast_well_above_bucket(self) -> None:
        """When forecast is well above bucket, probability should be very low."""
        forecast = DailyForecast(
            date=date(2026, 3, 15), temp_high_c=30.0, temp_low_c=10.0
        )
        bucket = TemperatureBucket(low_c=20.0, high_c=22.0, bucket_type="range")
        prob = estimate_bucket_probability(forecast, bucket, is_high=True)
        assert prob < 0.05

    def test_above_bucket_high_probability(self) -> None:
        """Forecast 30°C, bucket 'above 25°C' → high probability."""
        forecast = DailyForecast(
            date=date(2026, 3, 15), temp_high_c=30.0, temp_low_c=20.0
        )
        bucket = TemperatureBucket(low_c=25.0, high_c=None, bucket_type="above")
        prob = estimate_bucket_probability(forecast, bucket, is_high=True)
        assert prob > 0.9

    def test_below_bucket_low_probability(self) -> None:
        """Forecast 30°C, bucket 'below 25°C' → low probability."""
        forecast = DailyForecast(
            date=date(2026, 3, 15), temp_high_c=30.0, temp_low_c=20.0
        )
        bucket = TemperatureBucket(low_c=None, high_c=25.0, bucket_type="below")
        prob = estimate_bucket_probability(forecast, bucket, is_high=True)
        assert prob < 0.1

    def test_low_temperature_mode(self) -> None:
        """Test that low temp mode uses temp_low_c from forecast."""
        forecast = DailyForecast(
            date=date(2026, 3, 15), temp_high_c=25.0, temp_low_c=10.0
        )
        bucket = TemperatureBucket(low_c=8.0, high_c=12.0, bucket_type="range")
        prob = estimate_bucket_probability(forecast, bucket, is_high=False)
        assert prob > 0.4


class TestNormalCDF:
    def test_zero(self) -> None:
        assert abs(_normal_cdf(0) - 0.5) < 0.001

    def test_positive(self) -> None:
        assert _normal_cdf(2.0) > 0.97

    def test_negative(self) -> None:
        assert _normal_cdf(-2.0) < 0.03

    def test_symmetry(self) -> None:
        assert abs(_normal_cdf(1.0) + _normal_cdf(-1.0) - 1.0) < 0.001


# ================================================================
# Tests: Single Market Scanning
# ================================================================
class TestScanWeatherMarket:
    def test_detects_mispriced_market(self) -> None:
        """Market priced at 50% but forecast says 90% → strong BUY_YES signal."""
        market = MockMarket(
            question="Will the high temperature in NYC be above 75°F on March 15?",
            price_yes=Decimal("0.30"),
            volume_24h=Decimal("50000"),
        )
        # Forecast: 28°C ≈ 82°F → well above 75°F threshold
        forecasts = {City.NYC: _make_forecast(City.NYC, high_c=28.0)}

        signal = scan_weather_market(market, forecasts, min_edge=0.05)
        assert signal is not None
        assert signal.direction == "BUY_YES"
        assert signal.edge > 0.05
        assert signal.market.city == City.NYC

    def test_no_signal_when_fairly_priced(self) -> None:
        """Market priced correctly → no signal."""
        # Forecast high = 24°C = 75.2°F, threshold = 75°F → ~50% probability
        market = MockMarket(
            question="Will the high temperature in NYC be above 75°F on March 15?",
            price_yes=Decimal("0.50"),
            volume_24h=Decimal("50000"),
        )
        forecasts = {City.NYC: _make_forecast(City.NYC, high_c=fahrenheit_to_celsius(75))}

        signal = scan_weather_market(market, forecasts, min_edge=0.10)
        assert signal is None

    def test_no_signal_low_volume(self) -> None:
        market = MockMarket(
            question="Will the high temperature in NYC be above 75°F on March 15?",
            price_yes=Decimal("0.30"),
            volume_24h=Decimal("1000"),
        )
        forecasts = {City.NYC: _make_forecast(City.NYC, high_c=28.0)}

        signal = scan_weather_market(market, forecasts, min_volume=10000)
        assert signal is None

    def test_no_signal_missing_city(self) -> None:
        market = MockMarket(
            question="Will it rain tomorrow?",
            price_yes=Decimal("0.50"),
        )
        signal = scan_weather_market(market, {})
        assert signal is None

    def test_no_signal_missing_forecast(self) -> None:
        market = MockMarket(
            question="Will the high temperature in NYC be above 75°F on March 15?",
            price_yes=Decimal("0.30"),
        )
        # No NYC forecast available
        signal = scan_weather_market(market, {})
        assert signal is None

    def test_buy_no_signal(self) -> None:
        """Market overpriced — forecast says low probability."""
        market = MockMarket(
            question="Will the high temperature in NYC be above 90°F on March 15?",
            price_yes=Decimal("0.70"),
            volume_24h=Decimal("50000"),
        )
        # Forecast: 20°C = 68°F → well below 90°F → YES is overpriced
        forecasts = {City.NYC: _make_forecast(City.NYC, high_c=20.0)}

        signal = scan_weather_market(market, forecasts, min_edge=0.05)
        assert signal is not None
        assert signal.direction == "BUY_NO"
        assert signal.edge > 0.05


# ================================================================
# Tests: Batch Market Scanning
# ================================================================
class TestScanWeatherMarkets:
    def test_scan_multiple_markets(self) -> None:
        markets = [
            MockMarket(
                condition_id="market1",
                question="Will the high temperature in NYC be above 75°F on March 15?",
                price_yes=Decimal("0.30"),
                volume_24h=Decimal("50000"),
            ),
            MockMarket(
                condition_id="market2",
                question="Will the high temperature in Chicago be above 60°F on March 15?",
                price_yes=Decimal("0.30"),
                volume_24h=Decimal("50000"),
            ),
        ]
        forecasts = {
            City.NYC: _make_forecast(City.NYC, high_c=28.0),
            City.CHICAGO: _make_forecast(City.CHICAGO, high_c=20.0),
        }

        signals = scan_weather_markets(markets, forecasts, min_edge=0.05)
        assert len(signals) >= 1
        # Should be sorted by edge descending
        if len(signals) > 1:
            assert signals[0].edge >= signals[1].edge

    def test_only_weather_category(self) -> None:
        """Non-weather markets should be skipped."""
        markets = [
            MockMarket(
                question="Will the high temperature in NYC be above 75°F on March 15?",
                category="politics",
                price_yes=Decimal("0.30"),
                volume_24h=Decimal("50000"),
            ),
        ]
        forecasts = {City.NYC: _make_forecast(City.NYC, high_c=28.0)}

        signals = scan_weather_markets(markets, forecasts)
        assert len(signals) == 0

    def test_empty_markets(self) -> None:
        signals = scan_weather_markets([], {})
        assert signals == []
