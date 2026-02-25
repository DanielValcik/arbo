"""Tests for Met Office DataHub API connector."""

from __future__ import annotations

import re
from datetime import date

import pytest
from aioresponses import aioresponses

from arbo.connectors.weather_models import City, WeatherSource
from arbo.connectors.weather_metoffice import MetOfficeWeatherClient, MetOfficeWeatherError

# Sample Met Office daily forecast response
SAMPLE_DAILY_RESPONSE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-0.1278, 51.5074]},
            "properties": {
                "requestPointDistance": 0.0,
                "modelRunDate": "2026-02-25T12:00Z",
                "timeSeries": [
                    {
                        "time": "2026-02-25T00:00Z",
                        "dayMaxScreenTemperature": 8.5,
                        "nightMinScreenTemperature": 3.2,
                        "significantWeatherCode": 3,
                        "dayProbabilityOfPrecipitation": 15,
                    },
                    {
                        "time": "2026-02-26T00:00Z",
                        "dayMaxScreenTemperature": 10.1,
                        "nightMinScreenTemperature": 5.0,
                        "significantWeatherCode": 12,
                        "dayProbabilityOfPrecipitation": 65,
                    },
                    {
                        "time": "2026-02-27T00:00Z",
                        "dayMaxScreenTemperature": 7.3,
                        "nightMinScreenTemperature": 2.1,
                        "significantWeatherCode": 1,
                        "dayProbabilityOfPrecipitation": 5,
                    },
                ],
            },
        }
    ],
}

METOFFICE_URL = re.compile(
    r"https://data\.hub\.api\.metoffice\.gov\.uk/sitespecific/v0/point/daily.*"
)


class TestMetOfficeForecast:
    """Test Met Office forecast fetching and parsing."""

    async def test_fetch_london_forecast(self) -> None:
        client = MetOfficeWeatherClient(api_key="test-key-123")
        with aioresponses() as m:
            m.get(METOFFICE_URL, payload=SAMPLE_DAILY_RESPONSE)
            forecast = await client.get_forecast()
            await client.close()

        assert forecast.city == City.LONDON
        assert forecast.source == WeatherSource.MET_OFFICE
        assert len(forecast.daily_forecasts) == 3
        assert forecast.fetched_at.tzinfo is not None

    async def test_temperature_parsing(self) -> None:
        client = MetOfficeWeatherClient(api_key="test-key-123")
        with aioresponses() as m:
            m.get(METOFFICE_URL, payload=SAMPLE_DAILY_RESPONSE)
            forecast = await client.get_forecast()
            await client.close()

        day1 = forecast.daily_forecasts[0]
        assert day1.date == date(2026, 2, 25)
        assert day1.temp_high_c == 8.5
        assert day1.temp_low_c == 3.2
        assert day1.precip_probability == 15.0

    async def test_condition_parsing(self) -> None:
        client = MetOfficeWeatherClient(api_key="test-key-123")
        with aioresponses() as m:
            m.get(METOFFICE_URL, payload=SAMPLE_DAILY_RESPONSE)
            forecast = await client.get_forecast()
            await client.close()

        # Code 3 = "Partly cloudy (day)"
        assert forecast.daily_forecasts[0].condition == "Partly cloudy (day)"
        # Code 12 = "Light rain"
        assert forecast.daily_forecasts[1].condition == "Light rain"

    async def test_multiple_days(self) -> None:
        client = MetOfficeWeatherClient(api_key="test-key-123")
        with aioresponses() as m:
            m.get(METOFFICE_URL, payload=SAMPLE_DAILY_RESPONSE)
            forecast = await client.get_forecast()
            await client.close()

        assert len(forecast.daily_forecasts) == 3
        dates = [f.date for f in forecast.daily_forecasts]
        assert dates == [date(2026, 2, 25), date(2026, 2, 26), date(2026, 2, 27)]


class TestMetOfficeCache:
    """Test Met Office response caching."""

    async def test_cache_hit(self) -> None:
        client = MetOfficeWeatherClient(api_key="test-key-123")
        with aioresponses() as m:
            m.get(METOFFICE_URL, payload=SAMPLE_DAILY_RESPONSE)
            f1 = await client.get_forecast()
            f2 = await client.get_forecast()
            await client.close()

        assert f1.daily_forecasts == f2.daily_forecasts


class TestMetOfficeErrorHandling:
    """Test Met Office error handling."""

    async def test_auth_error_raises(self) -> None:
        client = MetOfficeWeatherClient(api_key="invalid-key")
        with aioresponses() as m:
            m.get(METOFFICE_URL, status=403, body="Forbidden")
            with pytest.raises(MetOfficeWeatherError, match="invalid or expired"):
                await client.get_forecast()
            await client.close()

    async def test_server_error_retries(self) -> None:
        client = MetOfficeWeatherClient(api_key="test-key-123")
        with aioresponses() as m:
            m.get(METOFFICE_URL, status=500, body="Error")
            m.get(METOFFICE_URL, payload=SAMPLE_DAILY_RESPONSE)
            forecast = await client.get_forecast()
            await client.close()

        assert forecast.city == City.LONDON

    async def test_persistent_error_raises(self) -> None:
        client = MetOfficeWeatherClient(api_key="test-key-123")
        with aioresponses() as m:
            for _ in range(3):
                m.get(METOFFICE_URL, status=500, body="Error")
            with pytest.raises(MetOfficeWeatherError, match="server error"):
                await client.get_forecast()
            await client.close()
