"""Shadow exit tracker for Strategy C — A/B test without executing exits.

Runs alongside actual paper trading. For each open C position, computes
what a METAR-informed exit decision would be using updated forecasts.
After resolution, compares actual P&L with shadow P&L to evaluate whether
exits would have improved performance.

Usage in main_rdh.py:
    # On trade placement:
    shadow_tracker.register_position(signal)

    # Each poll cycle:
    decisions = shadow_tracker.check_exits(positions, prices, forecasts)

    # On resolution:
    shadow_tracker.resolve(token_id, actual_pnl)

    # Report:
    stats = shadow_tracker.stats
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from arbo.connectors.weather_models import City, WeatherForecast
from arbo.strategies.weather_scanner import (
    TemperatureBucket,
    WeatherSignal,
    estimate_bucket_probability,
)
from arbo.utils.logger import get_logger

logger = get_logger("shadow_exit_tracker")

# Default exit parameter — best from backtest sweep (MIN_HOLD_EDGE=0.15)
DEFAULT_MIN_HOLD_EDGE = 0.15


@dataclass
class PositionMeta:
    """Metadata stored when a C position is opened."""

    token_id: str
    condition_id: str
    city: City
    target_date: date
    is_high_temp: bool
    bucket: TemperatureBucket
    entry_price: float
    entry_prob: float
    entry_edge: float
    direction: str  # BUY_YES or BUY_NO
    opened_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ShadowExit:
    """A hypothetical exit decision (not executed)."""

    token_id: str
    exit_price: float
    exit_reason: str
    updated_prob: float
    updated_edge: float
    shadow_pnl_pct: float  # (exit_price - entry_price) / entry_price
    exited_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Comparison:
    """Resolved comparison: what happened vs what shadow would have done."""

    token_id: str
    city: str
    actual_pnl: Decimal
    shadow_exit_pnl: float  # P&L if exited at shadow price
    saved: bool  # shadow exit P&L > actual P&L (exit would have saved money)
    delta: float  # shadow - actual (positive = saved)
    exit_reason: str
    resolved_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ShadowExitTracker:
    """Tracks hypothetical METAR-informed exit decisions for A/B comparison.

    Does NOT execute any exits — only logs what WOULD happen.
    After resolution, compares actual P&L vs shadow P&L.

    Parameters match the best backtest results (sweep_exits.py):
    - min_hold_edge: Exit when updated edge drops below this threshold.
      0.15 was best in backtest (composite_score +0.502 vs baseline).
    """

    def __init__(self, min_hold_edge: float = DEFAULT_MIN_HOLD_EDGE) -> None:
        self._min_hold_edge = min_hold_edge
        self._metadata: dict[str, PositionMeta] = {}  # token_id → meta
        self._shadow_exits: dict[str, ShadowExit] = {}  # token_id → shadow exit
        self._comparisons: list[Comparison] = []

    def register_position(self, signal: WeatherSignal, token_id: str) -> None:
        """Register metadata when a C position is opened.

        Called from main_rdh.py after strategy_c.poll_cycle() places trades.
        """
        meta = PositionMeta(
            token_id=token_id,
            condition_id=signal.market.condition_id,
            city=signal.market.city,
            target_date=signal.market.target_date,
            is_high_temp=signal.market.is_high_temp,
            bucket=signal.market.bucket,
            entry_price=signal.market.market_price,
            entry_prob=signal.forecast_probability,
            entry_edge=signal.edge,
            direction=signal.direction,
        )
        self._metadata[token_id] = meta
        logger.debug(
            "shadow_position_registered",
            token_id=token_id[:20],
            city=signal.market.city.value,
            entry_edge=round(signal.edge, 4),
        )

    def check_exits(
        self,
        open_positions: list[Any],
        current_prices: dict[str, Decimal],
        forecasts: dict[City, WeatherForecast],
    ) -> list[ShadowExit]:
        """Check open C positions for shadow exit signals.

        Uses updated forecasts + current CLOB prices to compute
        whether an exit would be triggered.

        Args:
            open_positions: Paper engine open positions.
            current_prices: token_id → current CLOB price (Decimal).
            forecasts: City → WeatherForecast from strategy_c.

        Returns:
            List of new shadow exit decisions (for logging/alerting).
        """
        new_exits: list[ShadowExit] = []

        for pos in open_positions:
            if getattr(pos, "strategy", "") != "C":
                continue

            token_id = pos.token_id

            # Skip if already shadow-exited
            if token_id in self._shadow_exits:
                continue

            # Need metadata (registered when trade was placed)
            meta = self._metadata.get(token_id)
            if meta is None:
                continue

            # Need current CLOB price
            current_price = current_prices.get(token_id)
            if current_price is None:
                continue

            # Need forecast for this city
            forecast = forecasts.get(meta.city)
            if forecast is None:
                continue

            # Get daily forecast for the resolution date
            daily = forecast.get_forecast_for_date(meta.target_date)
            if daily is None:
                continue

            # Compute updated probability using current forecast
            updated_prob = estimate_bucket_probability(
                daily, meta.bucket, meta.is_high_temp,
                city=meta.city.value, days_out=0,
            )

            # For BUY_NO direction, we need 1 - prob
            if meta.direction == "BUY_NO":
                updated_prob = 1.0 - updated_prob

            current_price_f = float(current_price)
            updated_edge = updated_prob - current_price_f

            # Check METAR-informed exit condition
            if updated_edge < self._min_hold_edge:
                entry_price = float(pos.avg_price)
                pnl_pct = (current_price_f - entry_price) / entry_price if entry_price > 0 else 0

                shadow = ShadowExit(
                    token_id=token_id,
                    exit_price=current_price_f,
                    exit_reason="edge_lost",
                    updated_prob=round(updated_prob, 4),
                    updated_edge=round(updated_edge, 4),
                    shadow_pnl_pct=round(pnl_pct, 4),
                )
                self._shadow_exits[token_id] = shadow
                new_exits.append(shadow)

                logger.info(
                    "shadow_exit_triggered",
                    token_id=token_id[:20],
                    city=meta.city.value,
                    entry_edge=round(meta.entry_edge, 4),
                    updated_edge=round(updated_edge, 4),
                    updated_prob=round(updated_prob, 4),
                    clob_price=current_price_f,
                    pnl_pct=round(pnl_pct, 4),
                )

        return new_exits

    def resolve(self, token_id: str, actual_pnl: Decimal) -> Comparison | None:
        """Record resolution and compare with shadow exit decision.

        Called from main_rdh.py when a C position resolves via METAR.

        Returns Comparison if this position had a shadow exit, else None.
        """
        meta = self._metadata.pop(token_id, None)
        shadow = self._shadow_exits.pop(token_id, None)

        if shadow is None or meta is None:
            return None

        # Compute shadow exit P&L: what we'd have gotten if we sold at shadow price
        entry_price = meta.entry_price
        if entry_price > 0:
            # Shares * exit_price - invested_amount (approximate)
            shadow_pnl = (shadow.exit_price / entry_price - 1.0)
        else:
            shadow_pnl = 0.0

        actual_pnl_f = float(actual_pnl)
        # Normalize actual P&L to same scale (fraction of invested)
        # actual_pnl from paper_engine is in USDC, need to divide by size
        # Since we don't have size here, compare on direction (saved or not)
        saved = shadow_pnl > actual_pnl_f / max(float(entry_price), 0.01)

        comparison = Comparison(
            token_id=token_id,
            city=meta.city.value,
            actual_pnl=actual_pnl,
            shadow_exit_pnl=round(shadow_pnl, 4),
            saved=saved,
            delta=round(shadow_pnl - actual_pnl_f, 4),
            exit_reason=shadow.exit_reason,
        )
        self._comparisons.append(comparison)

        log_fn = logger.info if saved else logger.warning
        log_fn(
            "shadow_exit_resolved",
            token_id=token_id[:20],
            city=meta.city.value,
            actual_pnl=str(actual_pnl),
            shadow_pnl=round(shadow_pnl, 4),
            saved=saved,
            exit_reason=shadow.exit_reason,
        )

        return comparison

    @property
    def stats(self) -> dict[str, Any]:
        """A/B comparison statistics."""
        saves = sum(1 for c in self._comparisons if c.saved)
        regrets = len(self._comparisons) - saves

        return {
            "min_hold_edge": self._min_hold_edge,
            "tracked_positions": len(self._metadata),
            "pending_shadow_exits": len(self._shadow_exits),
            "resolved_comparisons": len(self._comparisons),
            "saves": saves,
            "regrets": regrets,
            "save_rate_pct": round(saves / len(self._comparisons) * 100, 1)
            if self._comparisons else 0.0,
            "recent_comparisons": [
                {
                    "city": c.city,
                    "actual_pnl": str(c.actual_pnl),
                    "shadow_pnl": c.shadow_exit_pnl,
                    "saved": c.saved,
                    "reason": c.exit_reason,
                }
                for c in self._comparisons[-5:]  # Last 5 comparisons
            ],
        }
