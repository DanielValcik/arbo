"""Weather market scanner for Strategy C.

Scans Polymarket weather markets via Gamma API, parses temperature bucket ranges
from market titles, and calculates edge by comparing forecast probabilities
against market prices.

Polymarket weather markets typically have titles like:
- "Will the high temperature in NYC be above 75°F on March 15?"
- "NYC high temperature on March 15: 70-74°F?"
- "High temperature in Chicago on March 10?"
  with outcomes like "Above 50°F", "45-50°F", "Below 45°F"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from arbo.connectors.weather_models import (
    City,
    DailyForecast,
    WeatherForecast,
    fahrenheit_to_celsius,
)
from arbo.utils.logger import get_logger

logger = get_logger("weather_scanner")

# City name variations in market titles
_CITY_PATTERNS: dict[City, list[str]] = {
    City.NYC: ["nyc", "new york", "manhattan"],
    City.CHICAGO: ["chicago"],
    City.LONDON: ["london"],
    City.SEOUL: ["seoul"],
    City.BUENOS_AIRES: ["buenos aires"],
}

# Temperature bucket patterns in market questions
# Matches: "70-74°F", "above 75°F", "below 45°F", "70°F to 74°F"
_BUCKET_PATTERNS = [
    # Range: "70-74°F" or "70 to 74°F" or "70°F - 74°F" or "between 70 and 74°F"
    re.compile(
        r"(\d+)\s*°?\s*[fFcC]?\s*(?:-|to|–)\s*(\d+)\s*°\s*([fFcC])",
        re.IGNORECASE,
    ),
    re.compile(
        r"between\s+(\d+)\s*(?:°\s*[fFcC])?\s+and\s+(\d+)\s*°\s*([fFcC])",
        re.IGNORECASE,
    ),
    # Above/below: "above 75°F", "75°F or above", ">= 75°F"
    re.compile(
        r"(?:above|over|>=?|more than|higher than)\s+(\d+)\s*°\s*([fFcC])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+)\s*°\s*([fFcC])\s+(?:or above|or more|or higher|\+)",
        re.IGNORECASE,
    ),
    # Below: "below 45°F", "under 45°F"
    re.compile(
        r"(?:below|under|<=?|less than|lower than)\s+(\d+)\s*°\s*([fFcC])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+)\s*°\s*([fFcC])\s+(?:or below|or less|or lower|-)",
        re.IGNORECASE,
    ),
]

# Date patterns in weather market questions
_DATE_PATTERNS = [
    re.compile(
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?",
        re.IGNORECASE,
    ),
]

_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

# High vs low temperature indicator
_HIGH_KEYWORDS = ["high temperature", "high temp", "daily high"]
_LOW_KEYWORDS = ["low temperature", "low temp", "daily low", "overnight low"]


@dataclass
class TemperatureBucket:
    """A parsed temperature range from a market question."""

    low_c: float | None  # None for "above X" buckets
    high_c: float | None  # None for "below X" buckets
    bucket_type: str  # "range", "above", "below"
    original_text: str = ""

    def contains_temp(self, temp_c: float) -> bool:
        """Check if a temperature falls within this bucket."""
        if self.bucket_type == "above":
            return temp_c >= (self.low_c or 0)
        if self.bucket_type == "below":
            return temp_c < (self.high_c or 0)
        # range
        return (self.low_c or float("-inf")) <= temp_c < (self.high_c or float("inf"))


@dataclass
class WeatherMarketInfo:
    """Parsed weather market metadata."""

    condition_id: str
    question: str
    city: City
    target_date: date
    is_high_temp: bool  # True = high temp market, False = low temp
    bucket: TemperatureBucket
    market_price: float  # Current YES price
    token_id_yes: str
    token_id_no: str
    neg_risk: bool
    fee_enabled: bool
    volume_24h: float
    liquidity: float


@dataclass
class WeatherSignal:
    """A trading signal from weather market analysis."""

    market: WeatherMarketInfo
    forecast_temp_c: float
    forecast_probability: float  # Our estimated probability for this bucket
    edge: float  # forecast_probability - market_price
    direction: str  # "BUY_YES" or "BUY_NO"
    confidence: float  # Based on forecast source reliability


def parse_city(text: str) -> City | None:
    """Extract city from market question text."""
    text_lower = text.lower()
    for city, patterns in _CITY_PATTERNS.items():
        if any(p in text_lower for p in patterns):
            return city
    return None


def parse_target_date(text: str, default_year: int = 2026) -> date | None:
    """Extract target date from market question text."""
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            month_str = match.group(1).lower()
            day = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else default_year
            month = _MONTH_MAP.get(month_str)
            if month:
                try:
                    return date(year, month, day)
                except ValueError:
                    continue
    return None


def parse_temperature_bucket(text: str) -> TemperatureBucket | None:
    """Parse a temperature bucket from market question or outcome text."""
    text_clean = text.strip()

    # Try range patterns first: "70-74°F"
    for pattern in _BUCKET_PATTERNS[:2]:
        match = pattern.search(text_clean)
        if match:
            groups = match.groups()
            low = float(groups[0])
            high = float(groups[1])
            unit = groups[2].upper()
            if unit == "F":
                low = fahrenheit_to_celsius(low)
                high = fahrenheit_to_celsius(high)
            return TemperatureBucket(
                low_c=low, high_c=high, bucket_type="range", original_text=text_clean
            )

    # Try "above" patterns
    for pattern in _BUCKET_PATTERNS[2:4]:
        match = pattern.search(text_clean)
        if match:
            groups = match.groups()
            threshold = float(groups[0])
            unit = groups[1].upper()
            if unit == "F":
                threshold = fahrenheit_to_celsius(threshold)
            return TemperatureBucket(
                low_c=threshold, high_c=None, bucket_type="above", original_text=text_clean
            )

    # Try "below" patterns
    for pattern in _BUCKET_PATTERNS[4:6]:
        match = pattern.search(text_clean)
        if match:
            groups = match.groups()
            threshold = float(groups[0])
            unit = groups[1].upper()
            if unit == "F":
                threshold = fahrenheit_to_celsius(threshold)
            return TemperatureBucket(
                low_c=None, high_c=threshold, bucket_type="below", original_text=text_clean
            )

    return None


def is_high_temp_market(text: str) -> bool:
    """Determine if market is about high temperature (vs low temperature)."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in _LOW_KEYWORDS):
        return False
    # Default to high temperature
    return True


def estimate_bucket_probability(
    forecast: DailyForecast,
    bucket: TemperatureBucket,
    is_high: bool,
) -> float:
    """Estimate probability of forecast temperature falling in bucket.

    Uses a simple normal distribution approximation around the forecast value.
    Standard deviation assumed to be ~2-3°C for daily forecasts.

    Args:
        forecast: The daily forecast with high/low temperatures.
        bucket: The temperature bucket to evaluate.
        is_high: True if evaluating high temperature, False for low.

    Returns:
        Estimated probability (0-1) that actual temperature falls in bucket.
    """
    import math

    forecast_temp = forecast.temp_high_c if is_high else forecast.temp_low_c
    # Forecast uncertainty: ~2.5°C standard deviation for 1-day forecast
    # Increases for further-out forecasts
    sigma = 2.5

    if bucket.bucket_type == "above":
        threshold = bucket.low_c or 0
        # P(X >= threshold) = 1 - Phi((threshold - mu) / sigma)
        z = (threshold - forecast_temp) / sigma
        return 1 - _normal_cdf(z)

    elif bucket.bucket_type == "below":
        threshold = bucket.high_c or 0
        # P(X < threshold) = Phi((threshold - mu) / sigma)
        z = (threshold - forecast_temp) / sigma
        return _normal_cdf(z)

    else:  # range
        low = bucket.low_c if bucket.low_c is not None else forecast_temp - 20
        high = bucket.high_c if bucket.high_c is not None else forecast_temp + 20
        z_low = (low - forecast_temp) / sigma
        z_high = (high - forecast_temp) / sigma
        return _normal_cdf(z_high) - _normal_cdf(z_low)


def _normal_cdf(z: float) -> float:
    """Approximation of the standard normal CDF using math.erf."""
    import math

    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def scan_weather_market(
    market: Any,
    forecasts: dict[City, WeatherForecast],
    min_edge: float = 0.05,
    min_volume: float = 10000.0,
) -> WeatherSignal | None:
    """Analyze a single weather market for trading opportunities.

    Args:
        market: GammaMarket object from market discovery.
        forecasts: Dict of city → latest WeatherForecast.
        min_edge: Minimum edge threshold to generate signal.
        min_volume: Minimum 24h volume to consider market.

    Returns:
        WeatherSignal if edge detected, None otherwise.
    """
    # Parse city from question
    city = parse_city(market.question)
    if city is None:
        return None

    # Check if we have a forecast for this city
    forecast_data = forecasts.get(city)
    if forecast_data is None:
        return None

    # Parse target date
    target_date = parse_target_date(market.question)
    if target_date is None:
        return None

    # Get forecast for target date
    daily_forecast = forecast_data.get_forecast_for_date(target_date)
    if daily_forecast is None:
        return None

    # Parse temperature bucket
    bucket = parse_temperature_bucket(market.question)
    if bucket is None:
        return None

    # Check volume
    volume = float(market.volume_24h) if hasattr(market, "volume_24h") else 0
    if volume < min_volume:
        return None

    # Get market price
    price_yes = float(market.price_yes) if market.price_yes else None
    if price_yes is None or price_yes < 0.01 or price_yes > 0.99:
        return None

    is_high = is_high_temp_market(market.question)
    forecast_prob = estimate_bucket_probability(daily_forecast, bucket, is_high)

    # Calculate edge
    edge = forecast_prob - price_yes

    # Determine direction
    if abs(edge) < min_edge:
        return None

    if edge > 0:
        direction = "BUY_YES"
    else:
        direction = "BUY_NO"
        edge = -edge  # Make positive for signal

    market_info = WeatherMarketInfo(
        condition_id=market.condition_id,
        question=market.question,
        city=city,
        target_date=target_date,
        is_high_temp=is_high,
        bucket=bucket,
        market_price=price_yes,
        token_id_yes=market.token_id_yes or "",
        token_id_no=market.token_id_no or "",
        neg_risk=market.neg_risk,
        fee_enabled=market.fee_enabled,
        volume_24h=volume,
        liquidity=float(market.liquidity) if hasattr(market, "liquidity") else 0,
    )

    forecast_temp = daily_forecast.temp_high_c if is_high else daily_forecast.temp_low_c

    return WeatherSignal(
        market=market_info,
        forecast_temp_c=forecast_temp,
        forecast_probability=forecast_prob if direction == "BUY_YES" else (1 - forecast_prob),
        edge=edge,
        direction=direction,
        confidence=0.7,  # Default confidence, refined later based on source accuracy
    )


def scan_weather_markets(
    markets: list[Any],
    forecasts: dict[City, WeatherForecast],
    min_edge: float = 0.05,
    min_volume: float = 10000.0,
) -> list[WeatherSignal]:
    """Scan all markets for weather trading opportunities.

    Args:
        markets: List of GammaMarket objects.
        forecasts: Dict of city → latest WeatherForecast.
        min_edge: Minimum edge threshold.
        min_volume: Minimum 24h volume.

    Returns:
        List of WeatherSignal objects sorted by edge descending.
    """
    signals = []
    for market in markets:
        # Only look at weather-categorized or fee-free markets
        category = getattr(market, "category", "")
        if category != "weather":
            continue

        signal = scan_weather_market(market, forecasts, min_edge, min_volume)
        if signal:
            signals.append(signal)
            logger.info(
                "weather_signal_detected",
                city=signal.market.city.value,
                date=str(signal.market.target_date),
                direction=signal.direction,
                edge=round(signal.edge, 4),
                forecast_temp=round(signal.forecast_temp_c, 1),
                market_price=signal.market.market_price,
            )

    # Sort by edge descending
    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals
