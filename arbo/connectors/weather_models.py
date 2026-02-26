"""Shared Pydantic models for weather data across all weather API connectors.

Used by NOAA, Met Office, and Open-Meteo connectors.
"""

from __future__ import annotations

from datetime import date as date_type  # noqa: TC003
from datetime import datetime  # noqa: TC003
from enum import Enum

from pydantic import BaseModel, Field


class WeatherSource(str, Enum):
    """Weather data source identifiers."""

    NOAA = "noaa"
    MET_OFFICE = "met_office"
    OPEN_METEO = "open_meteo"


class City(str, Enum):
    """Supported cities for weather strategy."""

    NYC = "nyc"
    CHICAGO = "chicago"
    LONDON = "london"
    SEOUL = "seoul"
    BUENOS_AIRES = "buenos_aires"
    ATLANTA = "atlanta"
    TORONTO = "toronto"
    ANKARA = "ankara"
    SAO_PAULO = "sao_paulo"
    MIAMI = "miami"
    PARIS = "paris"
    DALLAS = "dallas"
    SEATTLE = "seattle"
    WELLINGTON = "wellington"


# Coordinates for each city (latitude, longitude)
CITY_COORDS: dict[City, tuple[float, float]] = {
    City.NYC: (40.7128, -74.0060),
    City.CHICAGO: (41.8781, -87.6298),
    City.LONDON: (51.5074, -0.1278),
    City.SEOUL: (37.5665, 126.9780),
    City.BUENOS_AIRES: (-34.6037, -58.3816),
    City.ATLANTA: (33.749, -84.388),
    City.TORONTO: (43.6532, -79.3832),
    City.ANKARA: (39.9334, 32.8597),
    City.SAO_PAULO: (-23.5505, -46.6333),
    City.MIAMI: (25.7617, -80.1918),
    City.PARIS: (48.8566, 2.3522),
    City.DALLAS: (32.7767, -96.7970),
    City.SEATTLE: (47.6062, -122.3321),
    City.WELLINGTON: (-41.2865, 174.7762),
}

# Which source covers which cities
CITY_SOURCE_MAP: dict[City, WeatherSource] = {
    City.NYC: WeatherSource.NOAA,
    City.CHICAGO: WeatherSource.NOAA,
    City.LONDON: WeatherSource.MET_OFFICE,
    City.SEOUL: WeatherSource.OPEN_METEO,
    City.BUENOS_AIRES: WeatherSource.OPEN_METEO,
    City.ATLANTA: WeatherSource.OPEN_METEO,
    City.TORONTO: WeatherSource.OPEN_METEO,
    City.ANKARA: WeatherSource.OPEN_METEO,
    City.SAO_PAULO: WeatherSource.OPEN_METEO,
    City.MIAMI: WeatherSource.OPEN_METEO,
    City.PARIS: WeatherSource.OPEN_METEO,
    City.DALLAS: WeatherSource.OPEN_METEO,
    City.SEATTLE: WeatherSource.OPEN_METEO,
    City.WELLINGTON: WeatherSource.OPEN_METEO,
}


class DailyForecast(BaseModel):
    """Single day's temperature forecast."""

    date: date_type
    temp_high_c: float = Field(description="Forecast high temperature in Celsius")
    temp_low_c: float = Field(description="Forecast low temperature in Celsius")
    condition: str = Field(default="", description="Short forecast condition (e.g. 'Partly Cloudy')")
    precip_probability: float | None = Field(
        default=None, description="Precipitation probability 0-100"
    )


class WeatherForecast(BaseModel):
    """Complete weather forecast for a city from a single source."""

    city: City
    source: WeatherSource
    fetched_at: datetime
    daily_forecasts: list[DailyForecast]

    def get_forecast_for_date(self, target_date: date_type) -> DailyForecast | None:
        """Get forecast for a specific date, or None if not available."""
        for f in self.daily_forecasts:
            if f.date == target_date:
                return f
        return None


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius, rounded to 1 decimal."""
    return round((f - 32) * 5 / 9, 1)
