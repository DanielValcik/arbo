"""NOAA Weather API connector for NYC and Chicago forecasts.

Uses api.weather.gov — free, no API key required, User-Agent header mandatory.
Two-step process: /points → grid coordinates → /gridpoints forecast.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timezone
from typing import Any

import aiohttp

from arbo.connectors.weather_models import (
    City,
    DailyForecast,
    WeatherForecast,
    WeatherSource,
    fahrenheit_to_celsius,
)
from arbo.utils.logger import get_logger

logger = get_logger("weather_noaa")

BASE_URL = "https://api.weather.gov"
USER_AGENT = "(arbo-trading-system, arbo@arbo.click)"
CACHE_TTL_S = 3600  # 1 hour

# Pre-resolved grid points (avoid /points call every time)
_GRID_POINTS: dict[City, tuple[str, int, int]] = {
    City.NYC: ("OKX", 33, 37),
    City.CHICAGO: ("LOT", 76, 73),
}

# Exponential backoff config
_MAX_RETRIES = 3
_BASE_DELAY_S = 1.0


class NOAAWeatherError(Exception):
    """Error fetching weather data from NOAA."""


class NOAAWeatherClient:
    """Async client for NOAA weather API (api.weather.gov)."""

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None
        self._cache: dict[str, tuple[float, WeatherForecast]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
                timeout=aiohttp.ClientTimeout(total=15),
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_with_retry(self, url: str) -> dict[str, Any]:
        """Fetch URL with exponential backoff on 5xx errors."""
        session = await self._get_session()
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json()  # type: ignore[no-any-return]
                    if resp.status >= 500:
                        body = await resp.text()
                        last_error = NOAAWeatherError(
                            f"NOAA server error {resp.status}: {body[:200]}"
                        )
                        delay = _BASE_DELAY_S * (2**attempt)
                        logger.warning(
                            "noaa_server_error",
                            status=resp.status,
                            attempt=attempt + 1,
                            retry_in=delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    body = await resp.text()
                    raise NOAAWeatherError(f"NOAA API error {resp.status}: {body[:200]}")
            except aiohttp.ClientError as e:
                last_error = NOAAWeatherError(f"NOAA connection error: {e}")
                delay = _BASE_DELAY_S * (2**attempt)
                logger.warning(
                    "noaa_connection_error",
                    error=str(e),
                    attempt=attempt + 1,
                    retry_in=delay,
                )
                await asyncio.sleep(delay)

        raise last_error or NOAAWeatherError("NOAA fetch failed after retries")

    def _parse_forecast(self, city: City, data: dict[str, Any]) -> WeatherForecast:
        """Parse NOAA forecast JSON into WeatherForecast model."""
        periods = data.get("properties", {}).get("periods", [])

        # Group periods by date to get daily high/low
        daily: dict[date, dict[str, Any]] = {}
        for period in periods:
            start_str = period.get("startTime", "")
            if not start_str:
                continue
            dt = datetime.fromisoformat(start_str)
            d = dt.date()

            if d not in daily:
                daily[d] = {"high": None, "low": None, "condition": "", "precip": None}

            temp = period.get("temperature")
            unit = period.get("temperatureUnit", "F")
            if temp is None:
                continue

            temp_c = fahrenheit_to_celsius(float(temp)) if unit == "F" else float(temp)
            is_day = period.get("isDaytime", True)

            if is_day:
                daily[d]["high"] = temp_c
                daily[d]["condition"] = period.get("shortForecast", "")
            else:
                daily[d]["low"] = temp_c

            precip = period.get("probabilityOfPrecipitation", {})
            if isinstance(precip, dict) and precip.get("value") is not None:
                new_precip = float(precip["value"])
                if daily[d]["precip"] is None or new_precip > daily[d]["precip"]:
                    daily[d]["precip"] = new_precip

        forecasts = []
        for d in sorted(daily.keys()):
            info = daily[d]
            # Skip days with incomplete data
            if info["high"] is None and info["low"] is None:
                continue
            forecasts.append(
                DailyForecast(
                    date=d,
                    temp_high_c=info["high"] if info["high"] is not None else info["low"] + 5,
                    temp_low_c=info["low"] if info["low"] is not None else info["high"] - 5,
                    condition=info["condition"],
                    precip_probability=info["precip"],
                )
            )

        return WeatherForecast(
            city=city,
            source=WeatherSource.NOAA,
            fetched_at=datetime.now(timezone.utc),
            daily_forecasts=forecasts,
        )

    async def get_forecast(self, city: City) -> WeatherForecast:
        """Get weather forecast for a NOAA-covered city (NYC or Chicago).

        Results are cached for 1 hour.

        Raises:
            ValueError: If city is not covered by NOAA.
            NOAAWeatherError: If API request fails after retries.
        """
        if city not in _GRID_POINTS:
            raise ValueError(f"City {city} is not covered by NOAA. Use NYC or Chicago.")

        # Check cache
        cache_key = city.value
        if cache_key in self._cache:
            cached_time, cached_forecast = self._cache[cache_key]
            if time.monotonic() - cached_time < CACHE_TTL_S:
                logger.debug("noaa_cache_hit", city=city.value)
                return cached_forecast

        grid_id, grid_x, grid_y = _GRID_POINTS[city]
        url = f"{BASE_URL}/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast"

        logger.info("noaa_fetch_forecast", city=city.value, url=url)
        data = await self._fetch_with_retry(url)
        forecast = self._parse_forecast(city, data)

        # Update cache
        self._cache[cache_key] = (time.monotonic(), forecast)

        logger.info(
            "noaa_forecast_fetched",
            city=city.value,
            days=len(forecast.daily_forecasts),
        )
        return forecast

    async def get_all_forecasts(self) -> list[WeatherForecast]:
        """Fetch forecasts for all NOAA cities (NYC, Chicago)."""
        tasks = [self.get_forecast(city) for city in _GRID_POINTS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        forecasts = []
        for city, result in zip(_GRID_POINTS.keys(), results):
            if isinstance(result, Exception):
                logger.error("noaa_forecast_failed", city=city.value, error=str(result))
            else:
                forecasts.append(result)
        return forecasts
