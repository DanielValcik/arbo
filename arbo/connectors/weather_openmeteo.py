"""Open-Meteo API connector for Seoul and Buenos Aires weather forecasts.

Uses api.open-meteo.com — free, no API key, no authentication.
Provides daily temperature forecasts via simple REST API.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timezone
from typing import Any

import aiohttp

from arbo.connectors.weather_models import (
    CITY_COORDS,
    City,
    DailyForecast,
    WeatherForecast,
    WeatherSource,
)
from arbo.utils.logger import get_logger

logger = get_logger("weather_openmeteo")

BASE_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_S = 3600  # 1 hour

# Cities covered by Open-Meteo (free, global coverage, no API key)
_COVERED_CITIES = {
    City.SEOUL,
    City.BUENOS_AIRES,
    City.ATLANTA,
    City.TORONTO,
    City.ANKARA,
    City.SAO_PAULO,
    City.MIAMI,
    City.PARIS,
    City.DALLAS,
    City.SEATTLE,
    City.WELLINGTON,
}

# Exponential backoff config
_MAX_RETRIES = 3
_BASE_DELAY_S = 1.0


class OpenMeteoWeatherError(Exception):
    """Error fetching weather data from Open-Meteo."""


class OpenMeteoWeatherClient:
    """Async client for Open-Meteo weather API."""

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None
        self._cache: dict[str, tuple[float, WeatherForecast]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_with_retry(self, params: dict[str, str]) -> dict[str, Any]:
        """Fetch forecast with exponential backoff on errors."""
        session = await self._get_session()
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                async with session.get(BASE_URL, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()  # type: ignore[no-any-return]
                    if resp.status >= 500:
                        body = await resp.text()
                        last_error = OpenMeteoWeatherError(
                            f"Open-Meteo server error {resp.status}: {body[:200]}"
                        )
                        delay = _BASE_DELAY_S * (2**attempt)
                        logger.warning(
                            "openmeteo_server_error",
                            status=resp.status,
                            attempt=attempt + 1,
                            retry_in=delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    body = await resp.text()
                    raise OpenMeteoWeatherError(
                        f"Open-Meteo API error {resp.status}: {body[:200]}"
                    )
            except aiohttp.ClientError as e:
                last_error = OpenMeteoWeatherError(f"Open-Meteo connection error: {e}")
                delay = _BASE_DELAY_S * (2**attempt)
                logger.warning(
                    "openmeteo_connection_error",
                    error=str(e),
                    attempt=attempt + 1,
                    retry_in=delay,
                )
                await asyncio.sleep(delay)

        raise last_error or OpenMeteoWeatherError("Open-Meteo fetch failed after retries")

    def _parse_forecast(self, city: City, data: dict[str, Any]) -> WeatherForecast:
        """Parse Open-Meteo response into WeatherForecast model."""
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        precip_probs = daily.get("precipitation_probability_max", [])
        weather_codes = daily.get("weather_code", [])

        forecasts = []
        for i, date_str in enumerate(dates):
            d = date.fromisoformat(date_str)
            high = highs[i] if i < len(highs) and highs[i] is not None else None
            low = lows[i] if i < len(lows) and lows[i] is not None else None

            if high is None and low is None:
                continue

            precip = None
            if i < len(precip_probs) and precip_probs[i] is not None:
                precip = float(precip_probs[i])

            condition = ""
            if i < len(weather_codes) and weather_codes[i] is not None:
                condition = _wmo_code_to_string(int(weather_codes[i]))

            forecasts.append(
                DailyForecast(
                    date=d,
                    temp_high_c=float(high) if high is not None else float(low) + 5,  # type: ignore[arg-type]
                    temp_low_c=float(low) if low is not None else float(high) - 5,  # type: ignore[arg-type]
                    condition=condition,
                    precip_probability=precip,
                )
            )

        return WeatherForecast(
            city=city,
            source=WeatherSource.OPEN_METEO,
            fetched_at=datetime.now(timezone.utc),
            daily_forecasts=forecasts,
        )

    async def get_forecast(self, city: City) -> WeatherForecast:
        """Get weather forecast for an Open-Meteo-covered city.

        Results are cached for 1 hour.

        Raises:
            ValueError: If city is not covered by Open-Meteo.
            OpenMeteoWeatherError: If API request fails after retries.
        """
        if city not in _COVERED_CITIES:
            raise ValueError(f"City {city} is not covered by Open-Meteo. Use Seoul or Buenos Aires.")

        cache_key = city.value
        if cache_key in self._cache:
            cached_time, cached_forecast = self._cache[cache_key]
            if time.monotonic() - cached_time < CACHE_TTL_S:
                logger.debug("openmeteo_cache_hit", city=city.value)
                return cached_forecast

        lat, lon = CITY_COORDS[city]
        params = {
            "latitude": str(lat),
            "longitude": str(lon),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "timezone": "auto",
            "forecast_days": "7",
        }

        logger.info("openmeteo_fetch_forecast", city=city.value, lat=lat, lon=lon)
        data = await self._fetch_with_retry(params)
        forecast = self._parse_forecast(city, data)

        self._cache[cache_key] = (time.monotonic(), forecast)

        logger.info(
            "openmeteo_forecast_fetched",
            city=city.value,
            days=len(forecast.daily_forecasts),
        )
        return forecast

    async def get_all_forecasts(self) -> list[WeatherForecast]:
        """Fetch forecasts for all Open-Meteo cities (Seoul, Buenos Aires)."""
        tasks = [self.get_forecast(city) for city in _COVERED_CITIES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        forecasts = []
        for city, result in zip(_COVERED_CITIES, results):
            if isinstance(result, Exception):
                logger.error("openmeteo_forecast_failed", city=city.value, error=str(result))
            else:
                forecasts.append(result)
        return forecasts


def _wmo_code_to_string(code: int) -> str:
    """Convert WMO weather interpretation code to human-readable string."""
    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return codes.get(code, f"Unknown ({code})")
