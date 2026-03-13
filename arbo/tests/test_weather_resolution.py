"""Tests for WeatherResolutionChecker — METAR-based weather market resolution."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from arbo.connectors.weather_iem import DailyObservation
from arbo.connectors.weather_models import City
from arbo.strategies.weather_resolution import (
    WeatherResolutionChecker,
    _bucket_contains_resolution_temp,
)
from arbo.strategies.weather_scanner import TemperatureBucket


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_market(
    question: str,
    token_id_yes: str = "yes_token",
    token_id_no: str = "no_token",
    closed: bool = True,
) -> MagicMock:
    """Create a mock GammaMarket."""
    m = MagicMock()
    m.question = question
    m.token_id_yes = token_id_yes
    m.token_id_no = token_id_no
    m.closed = closed
    return m


def _make_position(
    token_id: str = "yes_token",
    market_condition_id: str = "cond_123",
) -> MagicMock:
    """Create a mock PaperPosition."""
    p = MagicMock()
    p.token_id = token_id
    p.market_condition_id = market_condition_id
    return p


def _make_observation(
    city: City,
    target_date: date,
    resolution_temp: float,
    resolution_unit: str = "F",
) -> DailyObservation:
    """Create a DailyObservation for testing."""
    if resolution_unit == "F":
        max_f = resolution_temp
        max_c = (max_f - 32) * 5 / 9
    else:
        max_c = resolution_temp
        max_f = max_c * 9 / 5 + 32
    return DailyObservation(
        city=city,
        station="KTEST",
        date=target_date,
        max_temp_c=round(max_c, 2),
        min_temp_c=round(max_c - 10, 2),
        max_temp_f=round(max_f, 2),
        min_temp_f=round(max_f - 18, 2),
        obs_count=24,
        resolution_temp=resolution_temp,
        resolution_unit=resolution_unit,
    )


# ---------------------------------------------------------------------------
# _bucket_contains_resolution_temp tests
# ---------------------------------------------------------------------------


class TestBucketContains:
    """Tests for bucket temperature matching."""

    def test_range_bucket_hit_fahrenheit(self) -> None:
        """Fahrenheit temp falling in a range bucket."""
        # "between 34-35°F" → low=1.11°C, high=2.22°C
        bucket = TemperatureBucket(low_c=1.11, high_c=2.22, bucket_type="range")
        # 35°F = 1.67°C → inside [1.11, 2.22)
        assert _bucket_contains_resolution_temp(bucket, 35, "F") is True

    def test_range_bucket_miss_fahrenheit(self) -> None:
        """Fahrenheit temp outside a range bucket."""
        bucket = TemperatureBucket(low_c=1.11, high_c=2.22, bucket_type="range")
        # 40°F = 4.44°C → outside
        assert _bucket_contains_resolution_temp(bucket, 40, "F") is False

    def test_below_bucket_hit(self) -> None:
        """Temp below threshold in a 'below' bucket."""
        # "33°F or below" → high_c ≈ 1.11°C (threshold)
        bucket = TemperatureBucket(low_c=None, high_c=1.11, bucket_type="below")
        # 30°F = -1.11°C → below 1.11
        assert _bucket_contains_resolution_temp(bucket, 30, "F") is True

    def test_below_bucket_miss(self) -> None:
        """Temp above threshold in a 'below' bucket."""
        bucket = TemperatureBucket(low_c=None, high_c=1.11, bucket_type="below")
        # 40°F = 4.44°C → above 1.11
        assert _bucket_contains_resolution_temp(bucket, 40, "F") is False

    def test_above_bucket_hit(self) -> None:
        """Temp above threshold in an 'above' bucket."""
        bucket = TemperatureBucket(low_c=23.89, high_c=None, bucket_type="above")
        # 80°F = 26.67°C → above 23.89
        assert _bucket_contains_resolution_temp(bucket, 80, "F") is True

    def test_celsius_city(self) -> None:
        """Celsius temp in a range bucket."""
        bucket = TemperatureBucket(low_c=6, high_c=7, bucket_type="range")
        # 6°C → inside [6, 7)
        assert _bucket_contains_resolution_temp(bucket, 6, "C") is True
        # 7°C → outside [6, 7)
        assert _bucket_contains_resolution_temp(bucket, 7, "C") is False


# ---------------------------------------------------------------------------
# WeatherResolutionChecker tests
# ---------------------------------------------------------------------------


class TestWeatherResolutionChecker:
    """Tests for WeatherResolutionChecker.check_resolution()."""

    @pytest.mark.asyncio
    async def test_yes_wins_when_temp_in_bucket(self) -> None:
        """YES wins when actual temperature falls in the market's bucket."""
        yesterday = date.today() - timedelta(days=1)
        question = (
            f"Will the highest temperature in New York City be between "
            f"34-35°F on {yesterday.strftime('%B')} {yesterday.day}?"
        )
        market = _make_market(question)
        position = _make_position(token_id="yes_token")

        iem = AsyncMock()
        obs = _make_observation(City.NYC, yesterday, resolution_temp=35, resolution_unit="F")
        iem.get_daily_observation = AsyncMock(return_value=obs)

        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is not None
        is_resolved, token_won = result
        assert is_resolved is True
        assert token_won is True

    @pytest.mark.asyncio
    async def test_no_wins_when_temp_outside_bucket(self) -> None:
        """NO wins when actual temperature is outside the bucket."""
        yesterday = date.today() - timedelta(days=1)
        question = (
            f"Will the highest temperature in New York City be between "
            f"34-35°F on {yesterday.strftime('%B')} {yesterday.day}?"
        )
        market = _make_market(question)
        position = _make_position(token_id="no_token")

        iem = AsyncMock()
        obs = _make_observation(City.NYC, yesterday, resolution_temp=50, resolution_unit="F")
        iem.get_daily_observation = AsyncMock(return_value=obs)

        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is not None
        is_resolved, token_won = result
        assert is_resolved is True
        assert token_won is True  # NO token wins when temp outside bucket

    @pytest.mark.asyncio
    async def test_future_date_returns_none(self) -> None:
        """Future date returns None (not resolved yet)."""
        tomorrow = date.today() + timedelta(days=1)
        question = (
            f"Will the highest temperature in New York City be between "
            f"34-35°F on {tomorrow.strftime('%B')} {tomorrow.day}?"
        )
        market = _make_market(question)
        position = _make_position()

        iem = AsyncMock()
        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is None
        iem.get_daily_observation.assert_not_called()

    @pytest.mark.asyncio
    async def test_today_returns_none(self) -> None:
        """Same-day returns None (wait for full day of data)."""
        today = date.today()
        question = (
            f"Will the highest temperature in Chicago be between "
            f"50-54°F on {today.strftime('%B')} {today.day}?"
        )
        market = _make_market(question)
        position = _make_position()

        iem = AsyncMock()
        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is None

    @pytest.mark.asyncio
    async def test_iem_unavailable_returns_none(self) -> None:
        """IEM error returns None (falls back to price-based)."""
        yesterday = date.today() - timedelta(days=1)
        question = (
            f"Will the highest temperature in New York City be between "
            f"34-35°F on {yesterday.strftime('%B')} {yesterday.day}?"
        )
        market = _make_market(question)
        position = _make_position()

        iem = AsyncMock()
        iem.get_daily_observation = AsyncMock(side_effect=Exception("IEM down"))

        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is None

    @pytest.mark.asyncio
    async def test_no_observations_returns_none(self) -> None:
        """No METAR data returns None."""
        yesterday = date.today() - timedelta(days=1)
        question = (
            f"Will the highest temperature in London be "
            f"8°C or below on {yesterday.strftime('%B')} {yesterday.day}?"
        )
        market = _make_market(question)
        position = _make_position()

        iem = AsyncMock()
        iem.get_daily_observation = AsyncMock(return_value=None)

        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is None

    @pytest.mark.asyncio
    async def test_celsius_city_resolution(self) -> None:
        """Non-US city resolves correctly in Celsius."""
        yesterday = date.today() - timedelta(days=1)
        question = (
            f"Will the highest temperature in Seoul be "
            f"6°C on {yesterday.strftime('%B')} {yesterday.day}?"
        )
        market = _make_market(question)
        position = _make_position(token_id="yes_token")

        iem = AsyncMock()
        # 6°C observed — should match "be 6°C" bucket (6 ≤ temp < 7)
        obs = _make_observation(City.SEOUL, yesterday, resolution_temp=6, resolution_unit="C")
        iem.get_daily_observation = AsyncMock(return_value=obs)

        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is not None
        is_resolved, token_won = result
        assert is_resolved is True
        assert token_won is True

    @pytest.mark.asyncio
    async def test_unparseable_question_returns_none(self) -> None:
        """Unparseable question returns None."""
        market = _make_market("Some random non-weather question?")
        position = _make_position()

        iem = AsyncMock()
        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is None

    @pytest.mark.asyncio
    async def test_token_mismatch_returns_none(self) -> None:
        """Token not matching yes/no returns None."""
        yesterday = date.today() - timedelta(days=1)
        question = (
            f"Will the highest temperature in New York City be between "
            f"34-35°F on {yesterday.strftime('%B')} {yesterday.day}?"
        )
        market = _make_market(question, token_id_yes="yes_t", token_id_no="no_t")
        position = _make_position(token_id="unknown_token")

        iem = AsyncMock()
        obs = _make_observation(City.NYC, yesterday, resolution_temp=35, resolution_unit="F")
        iem.get_daily_observation = AsyncMock(return_value=obs)

        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is None

    @pytest.mark.asyncio
    async def test_below_bucket_resolution(self) -> None:
        """'33°F or below' bucket resolves correctly."""
        yesterday = date.today() - timedelta(days=1)
        question = (
            f"Will the highest temperature in New York City be "
            f"33°F or below on {yesterday.strftime('%B')} {yesterday.day}?"
        )
        market = _make_market(question)
        position = _make_position(token_id="yes_token")

        iem = AsyncMock()
        # 30°F observed → below 33°F → YES wins
        obs = _make_observation(City.NYC, yesterday, resolution_temp=30, resolution_unit="F")
        iem.get_daily_observation = AsyncMock(return_value=obs)

        checker = WeatherResolutionChecker(iem)
        result = await checker.check_resolution(position, market)

        assert result is not None
        _, token_won = result
        assert token_won is True
