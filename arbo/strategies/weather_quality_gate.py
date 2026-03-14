"""Quality gate for Strategy C (Weather).

Replaces the confluence scorer with per-strategy signal validation.
Each signal must pass all quality checks before being forwarded to execution.

Per-city overrides (CITY_OVERRIDES) allow tuning thresholds per city based on
METAR-calibrated autoresearch results. Cities with min_edge=0.99 are effectively
excluded. Per-city min_edge, min_price, max_price are optimized by autoresearch.

Production model: AR-0134 (score=170.1, 273 trades, WR=43.6%, OOS=$297, WF=$2,218)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from arbo.connectors.weather_models import City
from arbo.strategies.weather_scanner import WeatherSignal
from arbo.utils.logger import get_logger

logger = get_logger("weather_quality_gate")

# ── Global quality gate thresholds (AR-0134 autoresearch-optimized) ──
MIN_EDGE = 0.10  # 10% minimum edge
MAX_EDGE = 0.70  # Maximum edge (anomaly filter)
MIN_PRICE = 0.05  # Allow wide price range
MAX_PRICE = 0.70  # Allow wide price range
MIN_VOLUME_24H = 510.0  # Minimum 24h volume
MIN_CONFIDENCE = 0.5  # Minimum forecast confidence
MAX_FORECAST_AGE_HOURS = 6  # Forecast must be less than 6 hours old
MIN_LIQUIDITY = 200.0  # $200 minimum liquidity
MIN_FORECAST_PROB = 0.06  # Minimum absolute probability to trade

# ── Per-city threshold overrides (AR-0134 autoresearch-optimized) ──
# Excluded: Chicago, Seoul (via min_edge=0.99)
# Per-city: min_edge, min_price, max_price tuned per autoresearch
CITY_OVERRIDES: dict[str, dict[str, float]] = {
    # Excluded cities
    City.CHICAGO.value: {"min_edge": 0.99},
    City.SEOUL.value: {"min_edge": 0.99},
    # Per-city optimized thresholds
    City.ANKARA.value: {"max_price": 0.55, "min_edge": 0.005, "min_price": 0.08},
    City.ATLANTA.value: {"max_price": 0.55, "min_edge": 0.02, "min_price": 0.05},
    City.BUENOS_AIRES.value: {"max_price": 0.70, "min_edge": 0.02, "min_price": 0.05},
    City.DALLAS.value: {"max_price": 0.80, "min_edge": 0.05, "min_price": 0.05},
    City.MIAMI.value: {"max_price": 0.40, "min_edge": 0.05, "min_price": 0.08},
    City.SEATTLE.value: {"max_price": 0.55, "min_edge": 0.005, "min_price": 0.15},
    City.TORONTO.value: {"max_price": 0.40, "min_edge": 0.005, "min_price": 0.05},
    City.WELLINGTON.value: {"max_price": 0.50, "min_edge": 0.02, "min_price": 0.05},
}


@dataclass
class QualityDecision:
    """Result of quality gate check."""

    passed: bool
    reason: str
    signal: WeatherSignal | None = None


def _get_threshold(name: str, city: str | None = None) -> float:
    """Get a quality gate threshold, with optional per-city override.

    Args:
        name: Threshold name (e.g. "min_edge", "max_price").
        city: City value string (e.g. "nyc", "paris").

    Returns:
        Per-city override if available, otherwise the global default.
    """
    if city and city in CITY_OVERRIDES:
        override = CITY_OVERRIDES[city].get(name)
        if override is not None:
            return override
    defaults: dict[str, float] = {
        "min_edge": MIN_EDGE,
        "max_edge": MAX_EDGE,
        "min_price": MIN_PRICE,
        "max_price": MAX_PRICE,
        "min_volume": MIN_VOLUME_24H,
        "min_liquidity": MIN_LIQUIDITY,
        "min_forecast_prob": MIN_FORECAST_PROB,
    }
    return defaults.get(name, 0.0)


def check_signal_quality(
    signal: WeatherSignal,
    forecast_fetched_at: datetime,
    min_edge: float | None = None,
    min_volume: float | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    max_forecast_age_hours: int = MAX_FORECAST_AGE_HOURS,
    min_liquidity: float | None = None,
) -> QualityDecision:
    """Validate a weather signal against quality thresholds.

    Applies per-city overrides from CITY_OVERRIDES when available.

    Args:
        signal: The weather signal to validate.
        forecast_fetched_at: When the forecast was fetched (for freshness check).
        min_edge: Minimum edge threshold (None = use per-city or global default).
        min_volume: Minimum 24h volume (None = use per-city or global default).
        min_confidence: Minimum forecast confidence.
        max_forecast_age_hours: Maximum forecast age in hours.
        min_liquidity: Minimum market liquidity (None = use per-city or global default).

    Returns:
        QualityDecision with pass/fail and reason.
    """
    city = signal.market.city.value

    # Resolve thresholds: explicit param > per-city override > global default
    eff_min_edge = min_edge if min_edge is not None else _get_threshold("min_edge", city)
    eff_max_edge = _get_threshold("max_edge", city)
    eff_min_price = _get_threshold("min_price", city)
    eff_max_price = _get_threshold("max_price", city)
    eff_min_volume = min_volume if min_volume is not None else _get_threshold("min_volume", city)
    eff_min_liquidity = (
        min_liquidity if min_liquidity is not None else _get_threshold("min_liquidity", city)
    )
    eff_min_forecast_prob = _get_threshold("min_forecast_prob", city)

    # 1. Edge threshold
    if signal.edge < eff_min_edge:
        return QualityDecision(
            passed=False,
            reason=f"Edge {signal.edge:.4f} below minimum {eff_min_edge} ({city})",
        )

    # 2. Maximum edge (suspiciously high = pricing anomaly)
    if signal.edge > eff_max_edge:
        return QualityDecision(
            passed=False,
            reason=f"Edge {signal.edge:.4f} above max {eff_max_edge} — likely pricing anomaly",
        )

    # 3. Price range check
    if signal.market.market_price < eff_min_price:
        return QualityDecision(
            passed=False,
            reason=f"Price {signal.market.market_price:.2f} below minimum {eff_min_price}",
        )
    if signal.market.market_price > eff_max_price:
        return QualityDecision(
            passed=False,
            reason=f"Price {signal.market.market_price:.2f} above maximum {eff_max_price} ({city})",
        )

    # 4. Volume check
    if signal.market.volume_24h < eff_min_volume:
        return QualityDecision(
            passed=False,
            reason=f"Volume ${signal.market.volume_24h:,.0f} below minimum ${eff_min_volume:,.0f}",
        )

    # 5. Liquidity check
    if signal.market.liquidity < eff_min_liquidity:
        return QualityDecision(
            passed=False,
            reason=f"Liquidity ${signal.market.liquidity:,.0f} below minimum ${eff_min_liquidity:,.0f}",
        )

    # 6. Fee-free requirement
    if signal.market.fee_enabled:
        return QualityDecision(
            passed=False,
            reason="Market has fees enabled — Strategy C requires fee-free markets",
        )

    # 7. Confidence check
    if signal.confidence < min_confidence:
        return QualityDecision(
            passed=False,
            reason=f"Confidence {signal.confidence:.2f} below minimum {min_confidence}",
        )

    # 8. Minimum forecast probability
    if signal.forecast_probability < eff_min_forecast_prob:
        return QualityDecision(
            passed=False,
            reason=(
                f"Forecast prob {signal.forecast_probability:.2f} below minimum "
                f"{eff_min_forecast_prob}"
            ),
        )

    # 9. Forecast freshness
    now = datetime.now(timezone.utc)
    age_hours = (now - forecast_fetched_at).total_seconds() / 3600
    if age_hours > max_forecast_age_hours:
        return QualityDecision(
            passed=False,
            reason=f"Forecast is {age_hours:.1f}h old (max {max_forecast_age_hours}h)",
        )

    logger.info(
        "quality_gate_passed",
        city=city,
        date=str(signal.market.target_date),
        edge=round(signal.edge, 4),
        direction=signal.direction,
        volume=signal.market.volume_24h,
        price=signal.market.market_price,
    )

    return QualityDecision(
        passed=True,
        reason="All quality checks passed",
        signal=signal,
    )


def filter_signals(
    signals: list[WeatherSignal],
    forecast_fetched_at: datetime,
    **kwargs: float | int,
) -> list[WeatherSignal]:
    """Filter a list of signals through the quality gate.

    Args:
        signals: List of weather signals to filter.
        forecast_fetched_at: When the forecast data was fetched.
        **kwargs: Override default thresholds.

    Returns:
        List of signals that passed all quality checks.
    """
    passed = []
    for signal in signals:
        decision = check_signal_quality(signal, forecast_fetched_at, **kwargs)  # type: ignore[arg-type]
        if decision.passed:
            passed.append(signal)
        else:
            logger.info(
                "quality_gate_rejected",
                city=signal.market.city.value,
                edge=round(signal.edge, 4),
                volume=signal.market.volume_24h,
                liquidity=signal.market.liquidity,
                market_price=signal.market.market_price,
                forecast_prob=round(signal.forecast_probability, 4),
                reason=decision.reason,
            )

    logger.info(
        "quality_gate_summary",
        total=len(signals),
        passed=len(passed),
        rejected=len(signals) - len(passed),
    )
    return passed
