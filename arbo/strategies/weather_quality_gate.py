"""Quality gate for Strategy C (Weather).

Replaces the confluence scorer with per-strategy signal validation.
Each signal must pass all quality checks before being forwarded to execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from arbo.strategies.weather_scanner import WeatherSignal
from arbo.utils.logger import get_logger

logger = get_logger("weather_quality_gate")

# Quality gate thresholds
MIN_EDGE = 0.05  # 5% minimum edge
MIN_VOLUME_24H = 10_000.0  # $10K minimum volume
MIN_CONFIDENCE = 0.5  # Minimum forecast confidence
MAX_FORECAST_AGE_HOURS = 6  # Forecast must be less than 6 hours old
MIN_LIQUIDITY = 5_000.0  # $5K minimum liquidity


@dataclass
class QualityDecision:
    """Result of quality gate check."""

    passed: bool
    reason: str
    signal: WeatherSignal | None = None


def check_signal_quality(
    signal: WeatherSignal,
    forecast_fetched_at: datetime,
    min_edge: float = MIN_EDGE,
    min_volume: float = MIN_VOLUME_24H,
    min_confidence: float = MIN_CONFIDENCE,
    max_forecast_age_hours: int = MAX_FORECAST_AGE_HOURS,
    min_liquidity: float = MIN_LIQUIDITY,
) -> QualityDecision:
    """Validate a weather signal against quality thresholds.

    Args:
        signal: The weather signal to validate.
        forecast_fetched_at: When the forecast was fetched (for freshness check).
        min_edge: Minimum edge threshold.
        min_volume: Minimum 24h volume.
        min_confidence: Minimum forecast confidence.
        max_forecast_age_hours: Maximum forecast age in hours.
        min_liquidity: Minimum market liquidity.

    Returns:
        QualityDecision with pass/fail and reason.
    """
    # 1. Edge threshold
    if signal.edge < min_edge:
        return QualityDecision(
            passed=False,
            reason=f"Edge {signal.edge:.4f} below minimum {min_edge}",
        )

    # 2. Volume check
    if signal.market.volume_24h < min_volume:
        return QualityDecision(
            passed=False,
            reason=f"Volume ${signal.market.volume_24h:,.0f} below minimum ${min_volume:,.0f}",
        )

    # 3. Liquidity check
    if signal.market.liquidity < min_liquidity:
        return QualityDecision(
            passed=False,
            reason=f"Liquidity ${signal.market.liquidity:,.0f} below minimum ${min_liquidity:,.0f}",
        )

    # 4. Fee-free requirement
    if signal.market.fee_enabled:
        return QualityDecision(
            passed=False,
            reason="Market has fees enabled — Strategy C requires fee-free markets",
        )

    # 5. Confidence check
    if signal.confidence < min_confidence:
        return QualityDecision(
            passed=False,
            reason=f"Confidence {signal.confidence:.2f} below minimum {min_confidence}",
        )

    # 6. Forecast freshness
    now = datetime.now(timezone.utc)
    age_hours = (now - forecast_fetched_at).total_seconds() / 3600
    if age_hours > max_forecast_age_hours:
        return QualityDecision(
            passed=False,
            reason=f"Forecast is {age_hours:.1f}h old (max {max_forecast_age_hours}h)",
        )

    # 7. Price sanity check (not too extreme)
    if signal.market.market_price < 0.02 or signal.market.market_price > 0.98:
        return QualityDecision(
            passed=False,
            reason=f"Market price {signal.market.market_price} is too extreme (0.02-0.98 range)",
        )

    logger.info(
        "quality_gate_passed",
        city=signal.market.city.value,
        date=str(signal.market.target_date),
        edge=round(signal.edge, 4),
        direction=signal.direction,
        volume=signal.market.volume_24h,
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
            logger.debug(
                "quality_gate_rejected",
                city=signal.market.city.value,
                reason=decision.reason,
            )

    logger.info(
        "quality_gate_summary",
        total=len(signals),
        passed=len(passed),
        rejected=len(signals) - len(passed),
    )
    return passed
