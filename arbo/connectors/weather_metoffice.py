"""Met Office DataHub API connector for London weather forecasts.

Uses datahub.metoffice.gov.uk — free tier with API key required.
Provides site-specific daily forecasts.
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
)
from arbo.utils.logger import get_logger

logger = get_logger("weather_metoffice")

BASE_URL = "https://data.hub.api.metoffice.gov.uk/sitespecific/v0"
CACHE_TTL_S = 3600  # 1 hour

# London coordinates
LONDON_LAT = 51.5074
LONDON_LON = -0.1278

# Exponential backoff config
_MAX_RETRIES = 3
_BASE_DELAY_S = 1.0


class MetOfficeWeatherError(Exception):
    """Error fetching weather data from Met Office."""


class MetOfficeWeatherClient:
    """Async client for Met Office DataHub API."""

    def __init__(
        self,
        api_key: str,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._cache: dict[str, tuple[float, WeatherForecast]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "apikey": self._api_key,
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=15),
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_with_retry(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        """Fetch URL with exponential backoff on 5xx errors."""
        session = await self._get_session()
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()  # type: ignore[no-any-return]
                    if resp.status == 403:
                        raise MetOfficeWeatherError("Met Office API key invalid or expired")
                    if resp.status >= 500:
                        body = await resp.text()
                        last_error = MetOfficeWeatherError(
                            f"Met Office server error {resp.status}: {body[:200]}"
                        )
                        delay = _BASE_DELAY_S * (2**attempt)
                        logger.warning(
                            "metoffice_server_error",
                            status=resp.status,
                            attempt=attempt + 1,
                            retry_in=delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    body = await resp.text()
                    raise MetOfficeWeatherError(
                        f"Met Office API error {resp.status}: {body[:200]}"
                    )
            except aiohttp.ClientError as e:
                last_error = MetOfficeWeatherError(f"Met Office connection error: {e}")
                delay = _BASE_DELAY_S * (2**attempt)
                logger.warning(
                    "metoffice_connection_error",
                    error=str(e),
                    attempt=attempt + 1,
                    retry_in=delay,
                )
                await asyncio.sleep(delay)

        raise last_error or MetOfficeWeatherError("Met Office fetch failed after retries")

    def _parse_forecast(self, data: dict[str, Any]) -> WeatherForecast:
        """Parse Met Office daily forecast JSON into WeatherForecast model."""
        features = data.get("features", [])

        daily: dict[date, dict[str, Any]] = {}

        for feature in features:
            props = feature.get("properties", {})
            time_series = props.get("timeSeries", [])

            for entry in time_series:
                time_str = entry.get("time", "")
                if not time_str:
                    continue
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                d = dt.date()

                if d not in daily:
                    daily[d] = {"high": None, "low": None, "condition": "", "precip": None}

                # Met Office provides max/min screen temperature in Celsius
                max_temp = entry.get("dayMaxScreenTemperature") or entry.get(
                    "maxScreenAirTemperature"
                )
                min_temp = entry.get("nightMinScreenTemperature") or entry.get(
                    "minScreenAirTemperature"
                )

                if max_temp is not None:
                    daily[d]["high"] = float(max_temp)
                if min_temp is not None:
                    daily[d]["low"] = float(min_temp)

                # Weather type code → condition string
                weather_type = entry.get("significantWeatherCode")
                if weather_type is not None:
                    daily[d]["condition"] = _weather_code_to_string(int(weather_type))

                precip = entry.get("dayProbabilityOfPrecipitation") or entry.get(
                    "nightProbabilityOfPrecipitation"
                )
                if precip is not None:
                    daily[d]["precip"] = float(precip)

        forecasts = []
        for d in sorted(daily.keys()):
            info = daily[d]
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
            city=City.LONDON,
            source=WeatherSource.MET_OFFICE,
            fetched_at=datetime.now(timezone.utc),
            daily_forecasts=forecasts,
        )

    async def get_forecast(self) -> WeatherForecast:
        """Get weather forecast for London.

        Results are cached for 1 hour.

        Raises:
            MetOfficeWeatherError: If API request fails after retries.
        """
        cache_key = "london"
        if cache_key in self._cache:
            cached_time, cached_forecast = self._cache[cache_key]
            if time.monotonic() - cached_time < CACHE_TTL_S:
                logger.debug("metoffice_cache_hit")
                return cached_forecast

        url = f"{BASE_URL}/point/daily"
        params = {
            "latitude": str(LONDON_LAT),
            "longitude": str(LONDON_LON),
        }

        logger.info("metoffice_fetch_forecast", url=url)
        data = await self._fetch_with_retry(url, params)
        forecast = self._parse_forecast(data)

        self._cache[cache_key] = (time.monotonic(), forecast)

        logger.info(
            "metoffice_forecast_fetched",
            city="london",
            days=len(forecast.daily_forecasts),
        )
        return forecast


def _weather_code_to_string(code: int) -> str:
    """Convert Met Office significant weather code to human-readable string."""
    codes = {
        0: "Clear night",
        1: "Sunny day",
        2: "Partly cloudy (night)",
        3: "Partly cloudy (day)",
        5: "Mist",
        6: "Fog",
        7: "Cloudy",
        8: "Overcast",
        9: "Light rain shower (night)",
        10: "Light rain shower (day)",
        11: "Drizzle",
        12: "Light rain",
        13: "Heavy rain shower (night)",
        14: "Heavy rain shower (day)",
        15: "Heavy rain",
        16: "Sleet shower (night)",
        17: "Sleet shower (day)",
        18: "Sleet",
        19: "Hail shower (night)",
        20: "Hail shower (day)",
        21: "Hail",
        22: "Light snow shower (night)",
        23: "Light snow shower (day)",
        24: "Light snow",
        25: "Heavy snow shower (night)",
        26: "Heavy snow shower (day)",
        27: "Heavy snow",
        28: "Thunder shower (night)",
        29: "Thunder shower (day)",
        30: "Thunder",
    }
    return codes.get(code, f"Unknown ({code})")
