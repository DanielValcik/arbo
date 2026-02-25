"""Tests for NOAA Weather API connector."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

import pytest
from aioresponses import aioresponses

from arbo.connectors.weather_models import City, WeatherSource, fahrenheit_to_celsius
from arbo.connectors.weather_noaa import NOAAWeatherClient, NOAAWeatherError

# Sample NOAA forecast response
SAMPLE_FORECAST_RESPONSE = {
    "type": "Feature",
    "properties": {
        "updated": "2026-02-25T20:00:00+00:00",
        "periods": [
            {
                "number": 1,
                "name": "Today",
                "startTime": "2026-02-25T14:00:00-05:00",
                "endTime": "2026-02-25T18:00:00-05:00",
                "isDaytime": True,
                "temperature": 42,
                "temperatureUnit": "F",
                "windSpeed": "10 mph",
                "windDirection": "SW",
                "shortForecast": "Partly Sunny",
                "detailedForecast": "Partly sunny with a high near 42.",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 10},
            },
            {
                "number": 2,
                "name": "Tonight",
                "startTime": "2026-02-25T18:00:00-05:00",
                "endTime": "2026-02-26T06:00:00-05:00",
                "isDaytime": False,
                "temperature": 28,
                "temperatureUnit": "F",
                "windSpeed": "8 mph",
                "windDirection": "NW",
                "shortForecast": "Partly Cloudy",
                "detailedForecast": "Partly cloudy with a low near 28.",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 5},
            },
            {
                "number": 3,
                "name": "Wednesday",
                "startTime": "2026-02-26T06:00:00-05:00",
                "endTime": "2026-02-26T18:00:00-05:00",
                "isDaytime": True,
                "temperature": 50,
                "temperatureUnit": "F",
                "windSpeed": "12 mph",
                "windDirection": "S",
                "shortForecast": "Sunny",
                "detailedForecast": "Sunny with a high near 50.",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 0},
            },
            {
                "number": 4,
                "name": "Wednesday Night",
                "startTime": "2026-02-26T18:00:00-05:00",
                "endTime": "2026-02-27T06:00:00-05:00",
                "isDaytime": False,
                "temperature": 35,
                "temperatureUnit": "F",
                "windSpeed": "7 mph",
                "windDirection": "S",
                "shortForecast": "Mostly Cloudy",
                "detailedForecast": "Mostly cloudy with a low near 35.",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 20},
            },
        ],
    },
}

NYC_FORECAST_URL = re.compile(r"https://api\.weather\.gov/gridpoints/OKX/33,37/forecast.*")
CHICAGO_FORECAST_URL = re.compile(r"https://api\.weather\.gov/gridpoints/LOT/76,73/forecast.*")


class TestNOAAForecast:
    """Test NOAA forecast fetching and parsing."""

    async def test_fetch_nyc_forecast(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(NYC_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            forecast = await client.get_forecast(City.NYC)
            await client.close()

        assert forecast.city == City.NYC
        assert forecast.source == WeatherSource.NOAA
        assert len(forecast.daily_forecasts) >= 2
        assert forecast.fetched_at.tzinfo is not None

    async def test_fetch_chicago_forecast(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(CHICAGO_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            forecast = await client.get_forecast(City.CHICAGO)
            await client.close()

        assert forecast.city == City.CHICAGO
        assert forecast.source == WeatherSource.NOAA

    async def test_invalid_city_raises(self) -> None:
        client = NOAAWeatherClient()
        with pytest.raises(ValueError, match="not covered by NOAA"):
            await client.get_forecast(City.LONDON)
        await client.close()

    async def test_forecast_parsing_temperatures(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(NYC_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            forecast = await client.get_forecast(City.NYC)
            await client.close()

        # Feb 25 — high 42°F = 5.6°C, low 28°F = -2.2°C
        day1 = forecast.daily_forecasts[0]
        assert day1.date == date(2026, 2, 25)
        assert abs(day1.temp_high_c - fahrenheit_to_celsius(42)) < 0.1
        assert abs(day1.temp_low_c - fahrenheit_to_celsius(28)) < 0.1
        assert day1.condition == "Partly Sunny"
        assert day1.precip_probability == 10.0

    async def test_forecast_multiple_days(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(NYC_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            forecast = await client.get_forecast(City.NYC)
            await client.close()

        assert len(forecast.daily_forecasts) == 2
        assert forecast.daily_forecasts[0].date == date(2026, 2, 25)
        assert forecast.daily_forecasts[1].date == date(2026, 2, 26)

    async def test_get_forecast_for_date(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(NYC_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            forecast = await client.get_forecast(City.NYC)
            await client.close()

        result = forecast.get_forecast_for_date(date(2026, 2, 26))
        assert result is not None
        assert abs(result.temp_high_c - fahrenheit_to_celsius(50)) < 0.1

        # Non-existent date
        assert forecast.get_forecast_for_date(date(2026, 3, 1)) is None


class TestNOAACache:
    """Test NOAA response caching."""

    async def test_cache_hit_avoids_second_request(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(NYC_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            f1 = await client.get_forecast(City.NYC)
            # Second call should use cache (no mock registered for 2nd call)
            f2 = await client.get_forecast(City.NYC)
            await client.close()

        assert f1.city == f2.city
        assert f1.daily_forecasts == f2.daily_forecasts


class TestNOAAErrorHandling:
    """Test NOAA error handling and retry logic."""

    async def test_server_error_retries(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(NYC_FORECAST_URL, status=500, body="Internal Server Error")
            m.get(NYC_FORECAST_URL, status=500, body="Internal Server Error")
            m.get(NYC_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            forecast = await client.get_forecast(City.NYC)
            await client.close()

        assert forecast.city == City.NYC

    async def test_persistent_error_raises(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            for _ in range(3):
                m.get(NYC_FORECAST_URL, status=500, body="Error")
            with pytest.raises(NOAAWeatherError, match="server error"):
                await client.get_forecast(City.NYC)
            await client.close()

    async def test_client_error_raises(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(NYC_FORECAST_URL, status=404, body="Not Found")
            with pytest.raises(NOAAWeatherError, match="404"):
                await client.get_forecast(City.NYC)
            await client.close()


class TestNOAAGetAll:
    """Test fetching all NOAA cities at once."""

    async def test_get_all_forecasts(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(NYC_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            m.get(CHICAGO_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            forecasts = await client.get_all_forecasts()
            await client.close()

        assert len(forecasts) == 2
        cities = {f.city for f in forecasts}
        assert City.NYC in cities
        assert City.CHICAGO in cities

    async def test_get_all_partial_failure(self) -> None:
        client = NOAAWeatherClient()
        with aioresponses() as m:
            m.get(NYC_FORECAST_URL, payload=SAMPLE_FORECAST_RESPONSE)
            for _ in range(3):
                m.get(CHICAGO_FORECAST_URL, status=500, body="Error")
            forecasts = await client.get_all_forecasts()
            await client.close()

        # Only NYC should succeed
        assert len(forecasts) == 1
        assert forecasts[0].city == City.NYC


class TestFahrenheitConversion:
    """Test temperature conversion utility."""

    def test_freezing(self) -> None:
        assert fahrenheit_to_celsius(32) == 0.0

    def test_boiling(self) -> None:
        assert fahrenheit_to_celsius(212) == 100.0

    def test_negative(self) -> None:
        assert fahrenheit_to_celsius(0) == pytest.approx(-17.8, abs=0.1)

    def test_body_temp(self) -> None:
        assert fahrenheit_to_celsius(98.6) == pytest.approx(37.0, abs=0.1)
