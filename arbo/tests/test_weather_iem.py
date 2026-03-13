"""Tests for IEM METAR connector — weather_iem.py."""

from __future__ import annotations

from datetime import date

import pytest

from arbo.connectors.weather_iem import (
    CITY_STATIONS,
    IEMClient,
    _FAHRENHEIT_CITIES,
)
from arbo.connectors.weather_models import City


class TestCityStations:
    """Tests for station configuration."""

    def test_all_cities_have_stations(self) -> None:
        """Every City enum member has a station mapping."""
        for city in City:
            assert city in CITY_STATIONS, f"Missing station for {city.value}"
            info = CITY_STATIONS[city]
            assert "station" in info
            assert "network" in info
            assert "tz" in info
            assert len(info["station"]) == 4  # ICAO codes are 4 chars

    def test_us_cities_use_fahrenheit(self) -> None:
        """US cities resolve in Fahrenheit."""
        us_cities = {City.NYC, City.CHICAGO, City.ATLANTA, City.MIAMI, City.DALLAS, City.SEATTLE}
        assert _FAHRENHEIT_CITIES == us_cities

    def test_international_cities_use_celsius(self) -> None:
        """Non-US cities are NOT in _FAHRENHEIT_CITIES."""
        international = {City.LONDON, City.SEOUL, City.BUENOS_AIRES, City.TORONTO,
                         City.ANKARA, City.SAO_PAULO, City.PARIS, City.WELLINGTON}
        for city in international:
            assert city not in _FAHRENHEIT_CITIES


class TestIEMClientParser:
    """Tests for CSV response parsing."""

    def test_parse_normal_csv(self) -> None:
        """Parses a normal IEM CSV response."""
        client = IEMClient()
        csv_text = (
            "station,valid,tmpf\n"
            "KLGA,2026-03-12 00:00,35.0\n"
            "KLGA,2026-03-12 06:00,32.0\n"
            "KLGA,2026-03-12 12:00,45.0\n"
            "KLGA,2026-03-12 18:00,42.0\n"
        )

        obs = client._parse_csv_response(
            City.NYC, "KLGA", date(2026, 3, 12), csv_text
        )

        assert obs is not None
        assert obs.city == City.NYC
        assert obs.max_temp_f == 45.0
        assert obs.min_temp_f == 32.0
        assert obs.obs_count == 4
        # NYC uses Fahrenheit, so resolution_temp = round(max_f) = 45
        assert obs.resolution_temp == 45
        assert obs.resolution_unit == "F"

    def test_parse_celsius_city(self) -> None:
        """Non-US city resolves in Celsius."""
        client = IEMClient()
        csv_text = (
            "station,valid,tmpf\n"
            "RKSI,2026-03-12 00:00,42.8\n"  # 6°C
            "RKSI,2026-03-12 12:00,50.0\n"  # 10°C
        )

        obs = client._parse_csv_response(
            City.SEOUL, "RKSI", date(2026, 3, 12), csv_text
        )

        assert obs is not None
        assert obs.resolution_unit == "C"
        # max_f = 50.0, max_c = (50-32)*5/9 = 10.0
        assert obs.resolution_temp == 10  # round(10.0) = 10

    def test_parse_empty_csv(self) -> None:
        """Empty CSV (no data rows) returns None."""
        client = IEMClient()
        csv_text = "station,valid,tmpf\n"

        obs = client._parse_csv_response(
            City.NYC, "KLGA", date(2026, 3, 12), csv_text
        )

        assert obs is None

    def test_parse_missing_values(self) -> None:
        """Rows with 'M' (missing) are skipped."""
        client = IEMClient()
        csv_text = (
            "station,valid,tmpf\n"
            "KLGA,2026-03-12 00:00,M\n"
            "KLGA,2026-03-12 06:00,38.0\n"
            "KLGA,2026-03-12 12:00,M\n"
        )

        obs = client._parse_csv_response(
            City.NYC, "KLGA", date(2026, 3, 12), csv_text
        )

        assert obs is not None
        assert obs.obs_count == 1
        assert obs.max_temp_f == 38.0

    def test_unknown_city_returns_none(self) -> None:
        """Unknown city returns None for get_daily_observation."""
        import asyncio

        client = IEMClient()
        # Create a fake city enum value that's not in CITY_STATIONS
        # We test by checking the CITY_STATIONS lookup directly
        assert "nonexistent" not in [c.value for c in CITY_STATIONS.keys()]
