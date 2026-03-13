"""IEM METAR connector — airport weather observations from Iowa Environmental Mesonet.

Polymarket resolves weather markets using Weather Underground, which displays
METAR airport data. IEM provides the same METAR data for free, no API key needed.

Data source: mesonet.agron.iastate.edu/cgi-bin/request/asos.py
Format: CSV (not JSON)

This connector fetches daily max/min temperatures from ASOS/AWOS stations
for the cities Polymarket covers. Used for:
1. Resolution: ground-truth temperature after market date passes
2. Calibration: measuring forecast bias vs. actual observations
"""

from __future__ import annotations

import csv
import io
import ssl
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import aiohttp
import certifi

from arbo.connectors.weather_models import City
from arbo.utils.logger import get_logger

logger = get_logger("weather_iem")

# IEM ASOS endpoint
_IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# ICAO station codes for Polymarket cities
# These match Weather Underground stations used for resolution
CITY_STATIONS: dict[City, dict[str, Any]] = {
    City.NYC: {"station": "KLGA", "network": "NY_ASOS", "tz": "America/New_York"},
    City.CHICAGO: {"station": "KORD", "network": "IL_ASOS", "tz": "America/Chicago"},
    City.LONDON: {"station": "EGLC", "network": "GB__ASOS", "tz": "Europe/London"},
    City.SEOUL: {"station": "RKSI", "network": "KR__ASOS", "tz": "Asia/Seoul"},
    City.BUENOS_AIRES: {"station": "SAEZ", "network": "AR__ASOS", "tz": "America/Argentina/Buenos_Aires"},
    City.ATLANTA: {"station": "KATL", "network": "GA_ASOS", "tz": "America/New_York"},
    City.TORONTO: {"station": "CYYZ", "network": "CA_ON_ASOS", "tz": "America/Toronto"},
    City.ANKARA: {"station": "LTAC", "network": "TR__ASOS", "tz": "Europe/Istanbul"},
    City.SAO_PAULO: {"station": "SBGR", "network": "BR__ASOS", "tz": "America/Sao_Paulo"},
    City.MIAMI: {"station": "KMIA", "network": "FL_ASOS", "tz": "America/New_York"},
    City.PARIS: {"station": "LFPG", "network": "FR__ASOS", "tz": "Europe/Paris"},
    City.DALLAS: {"station": "KDFW", "network": "TX_ASOS", "tz": "America/Chicago"},
    City.SEATTLE: {"station": "KSEA", "network": "WA_ASOS", "tz": "America/Los_Angeles"},
    City.WELLINGTON: {"station": "NZWN", "network": "NZ__ASOS", "tz": "Pacific/Auckland"},
}

# US cities use Fahrenheit for Polymarket resolution
_US_CITIES = {City.NYC, City.CHICAGO, City.ATLANTA, City.MIAMI, City.DALLAS, City.SEATTLE}
# Toronto uses Celsius despite being in North America
_FAHRENHEIT_CITIES = _US_CITIES


@dataclass
class DailyObservation:
    """Daily weather observation from METAR data."""

    city: City
    station: str
    date: date
    max_temp_c: float
    min_temp_c: float
    max_temp_f: float
    min_temp_f: float
    obs_count: int  # Number of METAR reports in the day
    resolution_temp: float  # The temperature used for resolution
    resolution_unit: str  # "F" or "C"


class IEMClient:
    """Iowa Environmental Mesonet ASOS client.

    Fetches METAR airport observations for weather market resolution.
    Free, no API key required, global coverage.
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        """Create HTTP session."""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(ssl=ssl_ctx),
        )

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_daily_observation(
        self,
        city: City,
        target_date: date,
    ) -> DailyObservation | None:
        """Fetch daily max/min temperature observation for a city.

        Args:
            city: City to fetch observation for.
            target_date: Date to query.

        Returns:
            DailyObservation or None if data unavailable.
        """
        station_info = CITY_STATIONS.get(city)
        if station_info is None:
            logger.warning("iem_unknown_city", city=city.value)
            return None

        if self._session is None:
            await self.initialize()

        station = station_info["station"]
        network = station_info["network"]

        # IEM expects date range (we query single day)
        start = target_date
        end = target_date + timedelta(days=1)

        params = {
            "station": station,
            "data": "tmpf",  # Temperature in Fahrenheit (raw METAR)
            "tz": station_info["tz"],
            "format": "onlycomma",  # CSV output
            "latlon": "no",
            "elev": "no",
            "missing": "empty",
            "trace": "empty",
            "direct": "no",
            "report_type": "3",  # METAR + SPECI
            "year1": str(start.year),
            "month1": str(start.month),
            "day1": str(start.day),
            "year2": str(end.year),
            "month2": str(end.month),
            "day2": str(end.day),
        }

        try:
            async with self._session.get(_IEM_BASE, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "iem_http_error",
                        station=station,
                        status=resp.status,
                    )
                    return None

                text = await resp.text()
                return self._parse_csv_response(city, station, target_date, text)

        except Exception as e:
            logger.error("iem_fetch_error", station=station, error=str(e))
            return None

    def _parse_csv_response(
        self,
        city: City,
        station: str,
        target_date: date,
        csv_text: str,
    ) -> DailyObservation | None:
        """Parse IEM CSV response into DailyObservation.

        IEM returns CSV with columns: station, valid, tmpf
        We extract all temperature readings for the day and compute max/min.
        """
        reader = csv.DictReader(io.StringIO(csv_text))
        temps_f: list[float] = []

        for row in reader:
            tmpf_str = row.get("tmpf", "").strip()
            if not tmpf_str or tmpf_str == "M":
                continue
            try:
                temps_f.append(float(tmpf_str))
            except ValueError:
                continue

        if not temps_f:
            logger.info(
                "iem_no_observations",
                station=station,
                date=str(target_date),
            )
            return None

        max_f = max(temps_f)
        min_f = min(temps_f)
        max_c = round((max_f - 32) * 5 / 9, 2)
        min_c = round((min_f - 32) * 5 / 9, 2)

        # Resolution temp depends on whether PM uses F or C for this city
        uses_f = city in _FAHRENHEIT_CITIES
        resolution_temp = round(max_f) if uses_f else round(max_c)
        resolution_unit = "F" if uses_f else "C"

        return DailyObservation(
            city=city,
            station=station,
            date=target_date,
            max_temp_c=max_c,
            min_temp_c=min_c,
            max_temp_f=round(max_f, 2),
            min_temp_f=round(min_f, 2),
            obs_count=len(temps_f),
            resolution_temp=resolution_temp,
            resolution_unit=resolution_unit,
        )
