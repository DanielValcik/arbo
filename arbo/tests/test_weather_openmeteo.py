"""Tests for Open-Meteo weather API connector."""

from __future__ import annotations

import re
from datetime import date

import pytest
from aioresponses import aioresponses

from arbo.connectors.weather_models import City, WeatherSource
from arbo.connectors.weather_openmeteo import OpenMeteoWeatherClient, OpenMeteoWeatherError

# Sample Open-Meteo response
SAMPLE_OPENMETEO_RESPONSE = {
    "latitude": 37.57,
    "longitude": 127.0,
    "generationtime_ms": 0.5,
    "utc_offset_seconds": 32400,
    "timezone": "Asia/Seoul",
    "timezone_abbreviation": "KST",
    "daily_units": {
        "time": "iso8601",
        "temperature_2m_max": "°C",
        "temperature_2m_min": "°C",
        "precipitation_probability_max": "%",
        "weather_code": "wmo code",
    },
    "daily": {
        "time": [
            "2026-02-25",
            "2026-02-26",
            "2026-02-27",
            "2026-02-28",
            "2026-03-01",
            "2026-03-02",
            "2026-03-03",
        ],
        "temperature_2m_max": [5.2, 8.1, 10.3, 7.5, 3.8, 6.0, 9.2],
        "temperature_2m_min": [-2.1, 0.5, 3.2, 1.0, -3.5, -1.0, 2.0],
        "precipitation_probability_max": [10, 45, 70, 30, 5, 15, 55],
        "weather_code": [1, 3, 61, 2, 0, 1, 63],
    },
}

OPENMETEO_URL = re.compile(r"https://api\.open-meteo\.com/v1/forecast.*")


class TestOpenMeteoForecast:
    """Test Open-Meteo forecast fetching and parsing."""

    async def test_fetch_seoul_forecast(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            forecast = await client.get_forecast(City.SEOUL)
            await client.close()

        assert forecast.city == City.SEOUL
        assert forecast.source == WeatherSource.OPEN_METEO
        assert len(forecast.daily_forecasts) == 7
        assert forecast.fetched_at.tzinfo is not None

    async def test_fetch_buenos_aires_forecast(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            forecast = await client.get_forecast(City.BUENOS_AIRES)
            await client.close()

        assert forecast.city == City.BUENOS_AIRES
        assert forecast.source == WeatherSource.OPEN_METEO

    async def test_invalid_city_raises(self) -> None:
        client = OpenMeteoWeatherClient()
        with pytest.raises(ValueError, match="not covered by Open-Meteo"):
            await client.get_forecast(City.NYC)
        await client.close()

    async def test_temperature_parsing(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            forecast = await client.get_forecast(City.SEOUL)
            await client.close()

        day1 = forecast.daily_forecasts[0]
        assert day1.date == date(2026, 2, 25)
        assert day1.temp_high_c == 5.2
        assert day1.temp_low_c == -2.1
        assert day1.precip_probability == 10.0

    async def test_seven_day_forecast(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            forecast = await client.get_forecast(City.SEOUL)
            await client.close()

        assert len(forecast.daily_forecasts) == 7
        dates = [f.date for f in forecast.daily_forecasts]
        assert dates[0] == date(2026, 2, 25)
        assert dates[-1] == date(2026, 3, 3)

    async def test_weather_code_parsing(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            forecast = await client.get_forecast(City.SEOUL)
            await client.close()

        # Code 1 = "Mainly clear"
        assert forecast.daily_forecasts[0].condition == "Mainly clear"
        # Code 61 = "Slight rain"
        assert forecast.daily_forecasts[2].condition == "Slight rain"
        # Code 0 = "Clear sky"
        assert forecast.daily_forecasts[4].condition == "Clear sky"

    async def test_get_forecast_for_date(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            forecast = await client.get_forecast(City.SEOUL)
            await client.close()

        result = forecast.get_forecast_for_date(date(2026, 2, 27))
        assert result is not None
        assert result.temp_high_c == 10.3
        assert result.temp_low_c == 3.2


class TestOpenMeteoCache:
    """Test Open-Meteo response caching."""

    async def test_cache_hit(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            f1 = await client.get_forecast(City.SEOUL)
            f2 = await client.get_forecast(City.SEOUL)
            await client.close()

        assert f1.daily_forecasts == f2.daily_forecasts

    async def test_different_cities_not_cached(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            f1 = await client.get_forecast(City.SEOUL)
            f2 = await client.get_forecast(City.BUENOS_AIRES)
            await client.close()

        assert f1.city == City.SEOUL
        assert f2.city == City.BUENOS_AIRES


class TestOpenMeteoErrorHandling:
    """Test Open-Meteo error handling."""

    async def test_server_error_retries(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, status=500, body="Error")
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            forecast = await client.get_forecast(City.SEOUL)
            await client.close()

        assert forecast.city == City.SEOUL

    async def test_persistent_error_raises(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            for _ in range(3):
                m.get(OPENMETEO_URL, status=500, body="Error")
            with pytest.raises(OpenMeteoWeatherError, match="server error"):
                await client.get_forecast(City.SEOUL)
            await client.close()


class TestOpenMeteoGetAll:
    """Test fetching all Open-Meteo cities."""

    async def test_get_all_forecasts(self) -> None:
        client = OpenMeteoWeatherClient()
        with aioresponses() as m:
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            m.get(OPENMETEO_URL, payload=SAMPLE_OPENMETEO_RESPONSE)
            forecasts = await client.get_all_forecasts()
            await client.close()

        assert len(forecasts) == 2
        cities = {f.city for f in forecasts}
        assert City.SEOUL in cities
        assert City.BUENOS_AIRES in cities
