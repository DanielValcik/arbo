"""Social Momentum Divergence calculator v2 (Strategy B2-13).

Detects divergences between on-chain/social momentum (Santiment + CoinGecko)
and Polymarket contract prices. Generates signals when on-chain activity
diverges significantly from the prediction market price.

Algorithm:
1. Load latest 2-4 snapshots (24h window) from social_momentum_v2 table
2. Calculate momentum vectors per coin:
   - social_momentum_score = weighted(delta_daa x 0.30, delta_tx x 0.20,
     delta_dev x 0.10, delta_volume x 0.40)
   - price_momentum = price_change_24h
3. Map coins -> Polymarket contracts (manual lookup table)
4. Divergence = social_momentum_score - (normalized price_momentum)
5. Signal when |divergence| > threshold (default 2-sigma from rolling mean)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from arbo.utils.logger import get_logger

logger = get_logger("social_divergence")

# New weights for Santiment + CoinGecko momentum score
W_DAILY_ACTIVE_ADDRESSES = 0.30
W_TRANSACTIONS = 0.20
W_DEV_ACTIVITY = 0.10
W_VOLUME = 0.40

# Minimum number of snapshots needed for calculation
MIN_SNAPSHOTS = 2

# Default divergence threshold (in standard deviations)
DEFAULT_SIGMA_THRESHOLD = 2.0


# ================================================================
# Data models
# ================================================================


@dataclass(frozen=True)
class MomentumSnapshot:
    """A single social momentum snapshot for a coin (v2: Santiment + CoinGecko)."""

    symbol: str
    daily_active_addresses: float
    transactions_count: float
    dev_activity: float
    volume_24h: float
    price: float
    price_change_24h: float
    captured_at: datetime


@dataclass(frozen=True)
class DivergenceSignal:
    """Signal when social momentum diverges from market price.

    Positive divergence = on-chain activity UP but PM price flat/down (BUY signal)
    Negative divergence = on-chain activity DOWN but PM price flat/up (SELL signal)
    """

    symbol: str
    social_momentum_score: float  # -1 to +1 (normalized)
    price_momentum: float  # price_change_24h from CoinGecko
    divergence: float  # social_momentum - normalized_price_momentum
    z_score: float  # divergence in standard deviations
    direction: str  # "LONG" or "SHORT"
    confidence: float  # 0.0 - 1.0
    polymarket_condition_ids: list[str]  # matched PM contracts
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for DB/logging."""
        return {
            "symbol": self.symbol,
            "social_momentum_score": round(self.social_momentum_score, 4),
            "price_momentum": round(self.price_momentum, 4),
            "divergence": round(self.divergence, 4),
            "z_score": round(self.z_score, 2),
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "polymarket_condition_ids": self.polymarket_condition_ids,
            "detected_at": self.detected_at.isoformat(),
        }


# ================================================================
# Coin → Polymarket contract mapping
# ================================================================

# Manual mapping of coin symbols to known Polymarket condition IDs.
# Start with top 10 most liquid crypto PM markets.
# Format: {SYMBOL: [condition_id_1, condition_id_2, ...]}
COIN_TO_PM_CONTRACTS: dict[str, list[str]] = {
    # Populated at runtime from DB or config.
}


# ================================================================
# Calculator
# ================================================================


class SocialDivergenceCalculator:
    """Calculates divergence between on-chain momentum and market prices.

    Uses Santiment on-chain metrics (DAA, transactions, dev activity) and
    CoinGecko volume data to compute a composite momentum score, then
    compares against price momentum to detect divergences.

    Args:
        sigma_threshold: Number of standard deviations for signal threshold.
        coin_mapping: Override coin → PM contract mapping.
        min_snapshots: Minimum snapshots needed per coin.
    """

    def __init__(
        self,
        sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD,
        coin_mapping: dict[str, list[str]] | None = None,
        min_snapshots: int = MIN_SNAPSHOTS,
    ) -> None:
        self._sigma_threshold = sigma_threshold
        self._coin_mapping = coin_mapping or dict(COIN_TO_PM_CONTRACTS)
        self._min_snapshots = min_snapshots

        # Rolling stats for adaptive threshold
        self._divergence_history: dict[str, list[float]] = {}
        self._max_history = 100  # Keep last 100 divergence values per coin

        # Stats
        self._signals_generated = 0
        self._coins_analyzed = 0

    def set_coin_mapping(self, mapping: dict[str, list[str]]) -> None:
        """Update coin → PM contract mapping."""
        self._coin_mapping = mapping
        logger.info("coin_mapping_updated", coins=len(mapping))

    def add_coin_mapping(self, symbol: str, condition_ids: list[str]) -> None:
        """Add or update mapping for a single coin."""
        self._coin_mapping[symbol] = condition_ids

    def calculate_signals(
        self,
        snapshots_by_coin: dict[str, list[MomentumSnapshot]],
    ) -> list[DivergenceSignal]:
        """Calculate divergence signals from momentum snapshots.

        Args:
            snapshots_by_coin: Dict of {symbol: [snapshots]} sorted by
                captured_at ascending. Should contain 2-4 snapshots per coin
                spanning ~24h.

        Returns:
            List of DivergenceSignal for coins with significant divergence.
        """
        signals = []

        for symbol, snapshots in snapshots_by_coin.items():
            if len(snapshots) < self._min_snapshots:
                continue

            self._coins_analyzed += 1

            # Calculate momentum score
            momentum_score = self._compute_momentum_score(snapshots)
            price_momentum = snapshots[-1].price_change_24h

            # Normalize price momentum to [-1, 1] range
            # Using tanh for smooth clamping (±50% maps to ~±0.46)
            normalized_price = math.tanh(price_momentum / 100.0)

            # Divergence = social momentum - price momentum (both normalized)
            divergence = momentum_score - normalized_price

            # Update rolling history
            if symbol not in self._divergence_history:
                self._divergence_history[symbol] = []
            history = self._divergence_history[symbol]
            history.append(divergence)
            if len(history) > self._max_history:
                history.pop(0)

            # Calculate z-score from rolling stats
            z_score = self._compute_z_score(divergence, history)

            # Check threshold
            if abs(z_score) < self._sigma_threshold:
                continue

            # Determine direction
            direction = "LONG" if divergence > 0 else "SHORT"

            # Confidence based on z-score magnitude (asymptotic to 1.0)
            confidence = min(1.0, abs(z_score) / (abs(z_score) + 1.0))

            # Check if we have PM contracts for this coin
            pm_contracts = self._coin_mapping.get(symbol, [])

            signal = DivergenceSignal(
                symbol=symbol,
                social_momentum_score=momentum_score,
                price_momentum=price_momentum,
                divergence=divergence,
                z_score=z_score,
                direction=direction,
                confidence=confidence,
                polymarket_condition_ids=pm_contracts,
            )

            signals.append(signal)
            self._signals_generated += 1

            logger.info(
                "divergence_signal",
                symbol=symbol,
                direction=direction,
                divergence=round(divergence, 4),
                z_score=round(z_score, 2),
                confidence=round(confidence, 3),
                pm_contracts=len(pm_contracts),
            )

        return signals

    def _compute_momentum_score(self, snapshots: list[MomentumSnapshot]) -> float:
        """Compute weighted on-chain momentum score from snapshots.

        Measures the change in on-chain/volume metrics between the oldest and
        newest snapshot in the window.

        Weights: DAA x 0.30, Transactions x 0.20, Dev x 0.10, Volume x 0.40.

        Returns:
            Normalized score in [-1, 1] range.
        """
        oldest = snapshots[0]
        newest = snapshots[-1]

        # Delta daily active addresses (relative change)
        d_daa = _safe_delta(oldest.daily_active_addresses, newest.daily_active_addresses)

        # Delta transactions (relative change)
        d_tx = _safe_delta(oldest.transactions_count, newest.transactions_count)

        # Delta dev activity (relative change)
        d_dev = _safe_delta(oldest.dev_activity, newest.dev_activity)

        # Delta volume (relative change)
        d_vol = _safe_delta(oldest.volume_24h, newest.volume_24h)

        # Weighted combination
        raw_score = (
            W_DAILY_ACTIVE_ADDRESSES * d_daa
            + W_TRANSACTIONS * d_tx
            + W_DEV_ACTIVITY * d_dev
            + W_VOLUME * d_vol
        )

        # Clamp to [-1, 1] using tanh
        return math.tanh(raw_score)

    @staticmethod
    def _compute_z_score(value: float, history: list[float]) -> float:
        """Compute z-score of a value given its history.

        Uses population standard deviation. Returns 0 if insufficient data
        or zero variance.
        """
        if len(history) < 3:
            # Not enough data for meaningful z-score, use raw divergence
            # Scale by a fixed factor to make it comparable
            return value * 3.0  # Amplify early signals slightly

        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std < 1e-10:
            return 0.0

        return (value - mean) / std

    @property
    def stats(self) -> dict[str, Any]:
        """Calculator statistics."""
        return {
            "signals_generated": self._signals_generated,
            "coins_analyzed": self._coins_analyzed,
            "coins_mapped": len(self._coin_mapping),
            "sigma_threshold": self._sigma_threshold,
            "tracked_coins": len(self._divergence_history),
        }


def _safe_delta(old_val: float, new_val: float) -> float:
    """Compute relative change, handling zero base values."""
    if old_val > 0:
        return (new_val - old_val) / old_val
    return 0.0
