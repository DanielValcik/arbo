"""METAR-based weather market resolution checker.

Polymarket resolves weather markets using Weather Underground, which displays
METAR airport data. This module uses the same METAR data (via IEM) to determine
resolution BEFORE Polymarket marks the market as closed.

This gives us:
1. Faster resolution detection (we know the answer before PM settles)
2. Ground-truth verification (not relying on price_yes > 0.5 heuristic)
3. Correct handling of edge cases where price doesn't cleanly converge
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from arbo.connectors.weather_iem import IEMClient
from arbo.connectors.weather_models import City
from arbo.strategies.weather_scanner import (
    TemperatureBucket,
    parse_city,
    parse_target_date,
    parse_temperature_bucket,
)
from arbo.utils.logger import get_logger

logger = get_logger("weather_resolution")


def _bucket_contains_resolution_temp(
    bucket: TemperatureBucket,
    resolution_temp: float,
    unit: str,
) -> bool:
    """Check if a resolution temperature falls within a bucket.

    The bucket boundaries are in Celsius. Resolution temp may be in F or C.
    We convert resolution temp to Celsius before comparing.

    Args:
        bucket: Parsed temperature bucket from market question.
        resolution_temp: Observed temperature (whole degrees).
        unit: "F" or "C".

    Returns:
        True if the temperature is in the bucket.
    """
    if unit == "F":
        temp_c = (resolution_temp - 32) * 5 / 9
    else:
        temp_c = float(resolution_temp)

    return bucket.contains_temp(temp_c)


class WeatherResolutionChecker:
    """Checks weather market resolution using METAR observations.

    For each open position, parses the market question to extract:
    - City → METAR station
    - Target date → query IEM for that date
    - Temperature bucket → compare observed max temp

    Resolution logic:
    - If target date is today or future → not resolved yet (return None)
    - If target date is yesterday or earlier → fetch METAR, compare
    - If observed temp in bucket → YES wins
    - If observed temp NOT in bucket → NO wins
    """

    def __init__(self, iem_client: IEMClient) -> None:
        self._iem = iem_client

    async def check_resolution(
        self,
        position: Any,
        market: Any,
    ) -> tuple[bool, bool] | None:
        """Check if a weather market position has resolved.

        Args:
            position: PaperPosition with token_id.
            market: GammaMarket with question, token_id_yes, token_id_no.

        Returns:
            (is_resolved, token_won) — token_won is True if the position's
            token is the winning side. Returns None if not resolvable yet.
        """
        question = getattr(market, "question", None)
        if not question:
            return None

        # Parse city
        city = parse_city(question)
        if city is None:
            return None

        # Parse target date
        target = parse_target_date(question)
        if target is None:
            return None

        # Only resolve if target date has fully passed
        today = date.today()
        if target >= today:
            return None

        # Parse temperature bucket
        bucket = parse_temperature_bucket(question)
        if bucket is None:
            logger.debug(
                "resolution_unparseable_bucket",
                question=question[:80],
            )
            return None

        # Determine which token this position holds
        token_id = getattr(position, "token_id", None)
        token_yes = getattr(market, "token_id_yes", None)
        token_no = getattr(market, "token_id_no", None)

        if token_id is None:
            return None

        is_yes = token_id == token_yes
        is_no = token_id == token_no
        if not is_yes and not is_no:
            return None

        # Fetch METAR observation
        try:
            obs = await self._iem.get_daily_observation(city, target)
        except Exception as e:
            logger.warning(
                "resolution_iem_error",
                city=city.value,
                date=str(target),
                error=str(e),
            )
            return None

        if obs is None:
            return None

        # Compare observed temp against bucket
        temp_in_bucket = _bucket_contains_resolution_temp(
            bucket, obs.resolution_temp, obs.resolution_unit
        )

        logger.info(
            "metar_resolution",
            city=city.value,
            date=str(target),
            resolution_temp=obs.resolution_temp,
            resolution_unit=obs.resolution_unit,
            bucket_type=bucket.bucket_type,
            temp_in_bucket=temp_in_bucket,
            is_yes=is_yes,
            obs_count=obs.obs_count,
        )

        # YES wins if temp is in bucket, NO wins if temp is outside
        yes_wins = temp_in_bucket
        token_won = yes_wins if is_yes else not yes_wins

        return (True, token_won)
