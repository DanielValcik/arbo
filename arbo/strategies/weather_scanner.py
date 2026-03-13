"""Weather market scanner for Strategy C.

Scans Polymarket weather markets via Gamma API, parses temperature bucket ranges
from market titles, and calculates edge by comparing forecast probabilities
against market prices.

Polymarket weather markets use NegRisk events with titles like:
- Event: "Highest temperature in NYC on February 26?"
- Child markets (9 per event):
  - "Will the highest temperature in New York City be 33°F or below on February 26?"
  - "Will the highest temperature in New York City be between 34-35°F on February 26?"
  - "Will the highest temperature in Seoul be 6°C on February 26?"
  - "Will the highest temperature in London be 8°C or below on February 26?"
"""

from __future__ import annotations

import math
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
    City.ATLANTA: ["atlanta"],
    City.TORONTO: ["toronto"],
    City.ANKARA: ["ankara"],
    City.SAO_PAULO: ["sao paulo", "são paulo"],
    City.MIAMI: ["miami"],
    City.PARIS: ["paris"],
    City.DALLAS: ["dallas"],
    City.SEATTLE: ["seattle"],
    City.WELLINGTON: ["wellington"],
}

# Temperature bucket patterns in market questions
# Real Polymarket formats:
#   Range: "between 34-35°F", "between 76-77°F"
#   Exact: "be 6°C on", "be 9°C on"
#   Below: "33°F or below", "8°C or below", "-7°C or below"
_BUCKET_PATTERNS = [
    # Range: "between 34-35°F" (Polymarket US cities)
    re.compile(
        r"between\s+(-?\d+)\s*-\s*(-?\d+)\s*°\s*([fFcC])",
        re.IGNORECASE,
    ),
    # Range: "70-74°F" or "70 to 74°F" or "70°F - 74°F"
    re.compile(
        r"(-?\d+)\s*°?\s*[fFcC]?\s*(?:-|to|–)\s*(-?\d+)\s*°\s*([fFcC])",
        re.IGNORECASE,
    ),
    # Range: "between 70 and 74°F"
    re.compile(
        r"between\s+(-?\d+)\s*(?:°\s*[fFcC])?\s+and\s+(-?\d+)\s*°\s*([fFcC])",
        re.IGNORECASE,
    ),
    # Below: "-7°C or below", "33°F or below"
    re.compile(
        r"(-?\d+)\s*°\s*([fFcC])\s+or\s+below",
        re.IGNORECASE,
    ),
    # Below: "below 45°F", "under 45°F"
    re.compile(
        r"(?:below|under|<=?|less than|lower than)\s+(-?\d+)\s*°\s*([fFcC])",
        re.IGNORECASE,
    ),
    # Above: "75°F or above"
    re.compile(
        r"(-?\d+)\s*°\s*([fFcC])\s+(?:or above|or more|or higher|\+)",
        re.IGNORECASE,
    ),
    # Above: "above 75°F", "over 75°F"
    re.compile(
        r"(?:above|over|>=?|more than|higher than)\s+(-?\d+)\s*°\s*([fFcC])",
        re.IGNORECASE,
    ),
    # Exact single temp: "be 6°C on" (Polymarket non-US cities, 1-degree buckets)
    re.compile(
        r"be\s+(-?\d+)\s*°\s*([fFcC])\s+on\b",
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
_HIGH_KEYWORDS = ["highest temperature", "high temperature", "high temp", "daily high"]
_LOW_KEYWORDS = ["lowest temperature", "low temperature", "low temp", "daily low", "overnight low"]

# ═══════════════════════════════════════════════════════════════════════════════
# METAR-CALIBRATED PROBABILITY MODEL
# Sigma and bias calibrated from 60-day IEM METAR vs Open-Meteo archive.
# Run: python3 research/calibrate_bias.py --days 60
# ═══════════════════════════════════════════════════════════════════════════════

# Default sigma by days_out (when no per-city override exists)
_FORECAST_SIGMA: dict[int, float] = {
    0: 1.22,
    1: 3.0,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
    6: 5.0,
}

# Per-city sigma overrides — METAR-calibrated
# Data source quality: NOAA (nyc, chicago) > Met Office (london) > Open-Meteo (rest)
_CITY_SIGMA: dict[str, dict[int, float]] = {
    "paris":         {0: 0.75},
    "seattle":       {0: 0.91},
    "london":        {0: 0.92, 1: 2.8},
    "miami":         {0: 1.00},
    "wellington":    {0: 1.07},
    "nyc":           {0: 1.15},
    "chicago":       {0: 1.15, 1: 3.0},
    "dallas":        {0: 1.24},
    "seoul":         {0: 1.32, 1: 2.5},
    "atlanta":       {0: 1.32},
    "sao_paulo":     {0: 1.36},
    "toronto":       {0: 1.38},
    "buenos_aires":  {0: 1.43, 1: 3.0},
    "ankara":        {0: 1.44},
}

# Per-city bias corrections (°C) — measured forecast error vs METAR actual.
# Positive = forecast reads LOW vs actual (add correction to forecast).
# Negative = forecast reads HIGH (subtract).
_CITY_BIAS: dict[str, float] = {
    "buenos_aires": 2.58,
    "nyc":          1.53,
    "wellington":   1.43,
    "atlanta":      1.30,
    "sao_paulo":    1.24,
    "toronto":      0.98,
    "chicago":      0.92,
    "seoul":        0.87,
    "dallas":       0.73,
    "miami":        0.56,
    "seattle":      0.41,
    "london":       0.24,
    "paris":        0.16,
    "ankara":       -0.78,
}

# Probability sharpening: raise raw prob to this power (>1 = more decisive)
_PROB_SHARPENING = 1.05

# Bayesian shrinkage: blend with uniform prior (reduces overconfidence)
_UNIFORM_PRIOR = 0.125  # 1/8 buckets
_SHRINKAGE_WEIGHT = 0.03  # 3% prior weight


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
    """Parse a temperature bucket from market question or outcome text.

    Handles Polymarket formats:
    - Range: "between 34-35°F" or "70-74°F"
    - Below: "33°F or below", "-7°C or below"
    - Above: "75°F or above"
    - Exact: "be 6°C on" (single-degree bucket)
    """
    text_clean = text.strip()

    # Try range patterns (indices 0-2): "between 34-35°F", "70-74°F", "between 70 and 74°F"
    for pattern in _BUCKET_PATTERNS[:3]:
        match = pattern.search(text_clean)
        if match:
            groups = match.groups()
            low = float(groups[0])
            high = float(groups[1])
            unit = groups[2].upper()
            if unit == "F":
                low = fahrenheit_to_celsius(low)
                high = fahrenheit_to_celsius(high + 1)  # inclusive upper bound
            else:
                high = high + 1  # °C ranges are inclusive
            return TemperatureBucket(
                low_c=low, high_c=high, bucket_type="range", original_text=text_clean
            )

    # Try "below" patterns (indices 3-4): "33°F or below", "below 45°F"
    for pattern in _BUCKET_PATTERNS[3:5]:
        match = pattern.search(text_clean)
        if match:
            groups = match.groups()
            threshold = float(groups[0])
            unit = groups[1].upper()
            if unit == "F":
                threshold = fahrenheit_to_celsius(threshold + 1)  # "33 or below" = < 34
            else:
                threshold = threshold + 1  # "8°C or below" = < 9
            return TemperatureBucket(
                low_c=None, high_c=threshold, bucket_type="below", original_text=text_clean
            )

    # Try "above" patterns (indices 5-6): "75°F or above", "above 75°F"
    for pattern in _BUCKET_PATTERNS[5:7]:
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

    # Try exact single-value pattern (index 7): "be 6°C on"
    match = _BUCKET_PATTERNS[7].search(text_clean)
    if match:
        groups = match.groups()
        value = float(groups[0])
        unit = groups[1].upper()
        if unit == "F":
            low = fahrenheit_to_celsius(value)
            high = fahrenheit_to_celsius(value + 1)
        else:
            low = value
            high = value + 1  # single-degree bucket: [6, 7)
        return TemperatureBucket(
            low_c=low, high_c=high, bucket_type="range", original_text=text_clean
        )

    return None


def is_high_temp_market(text: str) -> bool:
    """Determine if market is about high temperature (vs low temperature)."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in _LOW_KEYWORDS):
        return False
    # Default to high temperature
    return True


def _get_sigma(days_out: int, city: str | None = None) -> float:
    """Get forecast sigma for a given days_out and optional city.

    Resolution order: per-city override → global days_out default → 5.0 fallback.
    """
    if city and city in _CITY_SIGMA:
        city_sigmas = _CITY_SIGMA[city]
        if days_out in city_sigmas:
            return city_sigmas[days_out]
        # Fallback to the highest available day key for this city
        max_key = max(city_sigmas.keys())
        if days_out > max_key:
            return _FORECAST_SIGMA.get(days_out, 5.0)
    return _FORECAST_SIGMA.get(days_out, 5.0)


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal CDF."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def estimate_bucket_probability(
    forecast: DailyForecast,
    bucket: TemperatureBucket,
    is_high: bool,
    city: str | None = None,
    days_out: int = 0,
) -> float:
    """Estimate probability of forecast temperature falling in bucket.

    Uses METAR-calibrated per-city sigma and bias corrections. Matches
    the research probability model (strategy_experiment.py) exactly.

    Args:
        forecast: The daily forecast with high/low temperatures.
        bucket: The temperature bucket to evaluate.
        is_high: True if evaluating high temperature, False for low.
        city: City value string (e.g. "chicago") for per-city sigma + bias.
        days_out: Days until resolution (0 = today, 1 = tomorrow).

    Returns:
        Estimated probability (0-1) that actual temperature falls in bucket.
    """
    forecast_temp = forecast.temp_high_c if is_high else forecast.temp_low_c

    # Apply bias correction: corrected = forecast + bias
    if city and city in _CITY_BIAS:
        forecast_temp = forecast_temp + _CITY_BIAS[city]

    sigma = _get_sigma(days_out, city)

    cdf = lambda x: _normal_cdf(x, forecast_temp, sigma)

    if bucket.bucket_type == "above":
        threshold = bucket.low_c or 0
        raw = 1.0 - cdf(threshold)
    elif bucket.bucket_type == "below":
        threshold = bucket.high_c or 0
        raw = cdf(threshold)
    else:  # range
        low = bucket.low_c if bucket.low_c is not None else forecast_temp - 20
        high = bucket.high_c if bucket.high_c is not None else forecast_temp + 20
        raw = cdf(high) - cdf(low)

    # Bayesian shrinkage: blend with uniform prior (reduces overconfidence)
    raw = raw * (1.0 - _SHRINKAGE_WEIGHT) + _UNIFORM_PRIOR * _SHRINKAGE_WEIGHT

    # Probability sharpening: push probabilities toward extremes
    if _PROB_SHARPENING != 1.0 and raw > 0:
        raw = raw ** _PROB_SHARPENING

    return raw


def scan_weather_market(
    market: Any,
    forecasts: dict[City, WeatherForecast],
    min_edge: float = 0.05,
    min_volume: float = 2000.0,
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
    days_out = max(0, (target_date - date.today()).days)
    city_str = city.value
    forecast_prob = estimate_bucket_probability(
        daily_forecast, bucket, is_high, city=city_str, days_out=days_out,
    )

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
    min_volume: float = 2000.0,
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
    weather_count = 0
    parse_fail_count = 0
    for market in markets:
        # Only look at weather-categorized markets
        category = getattr(market, "category", "")
        if category != "weather":
            continue

        weather_count += 1
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
        else:
            # Debug: log why parsing failed
            q = getattr(market, "question", "")
            city = parse_city(q)
            target_date = parse_target_date(q)
            bucket = parse_temperature_bucket(q)
            if city is None or target_date is None or bucket is None:
                parse_fail_count += 1
                logger.debug(
                    "weather_parse_fail",
                    question=q[:100],
                    city_ok=city is not None,
                    date_ok=target_date is not None,
                    bucket_ok=bucket is not None,
                )

    if weather_count > 0:
        logger.info(
            "weather_scan_summary",
            weather_markets=weather_count,
            signals=len(signals),
            parse_failures=parse_fail_count,
        )

    # Sort by edge descending
    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals
