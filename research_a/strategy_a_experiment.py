"""
Strategy A (Theta Decay) — Experiment Parameters v2
====================================================

This is the ONLY file modified by autoresearch.
All tunable parameters, signal computation, quality gate, and sizing live here.

v2 changes from v1:
- Position sizing is %-based (fraction of total capital) instead of nominal $
- Harness enforces 1% floor — can't game with micro-positions
- More tunable parameters for signal logic
- Starting from best signal params from v1 round (13.40 -> 71.40)

Run: python3 research_a/backtest_a_harness.py
"""

import math
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

LONGSHOT_PRICE_MAX = 0.092         # Max YES price (best from v1: 0.15 -> 0.085)
MIN_YES_PRICE = 0.01               # Min YES price (skip dust)
MIN_VOLUME_24H = 13500.0           # Minimum 24h volume (sweep15: 292.09 > 277.27)
RESOLUTION_DAYS_MIN = 2            # Min days to resolution (best from v1: 3 -> 2)
RESOLUTION_DAYS_MAX = 21           # Max days to resolution (best from v1: 30 -> 21)


# ═══════════════════════════════════════════════════════════════════════════════
# SPIKE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

ZSCORE_THRESHOLD = 3.0             # Base z-score for spike (best from v1: 3.0 -> 2.9)
SPIKE_LOOKBACK_TICKS = 20          # Rolling window (best from v1: 30 -> 20)
MIN_HISTORY_TICKS = 11             # Min ticks for z-score (best from v1: 12 -> 11)
SPIKE_COOLDOWN_TICKS = 0           # Ticks before re-checking same market after spike entry


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

DISCOUNT_FACTOR = 0.375            # YES actual prob = price x discount (sweep17: 311.23 > 308.26)
MIN_EDGE = 0.055                   # Minimum edge to enter (267.11 > 244.40)
MAX_EDGE = 0.50                    # Maximum edge (suspicious -> skip)

# Zscore-dependent edge adjustments
DISCOUNT_ZSCORE_BONUS = 0.0        # Reduce discount by this per sigma above threshold
EDGE_ZSCORE_BONUS = 0.0            # Add this x (zscore - threshold) to edge

# Volume-dependent edge adjustment
VOLUME_EDGE_ENABLED = False
VOLUME_EDGE_THRESHOLD = 50000.0    # Volume above which edge is reduced
VOLUME_EDGE_REDUCTION = 0.005      # Edge reduction per 100K volume above threshold


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY GATE
# ═══════════════════════════════════════════════════════════════════════════════

ZSCORE_DAYS_PIVOT = 10             # Days pivot for resolution-adjusted threshold
ZSCORE_PENALTY_POWER = 2           # 1=linear, 2=quadratic penalty for longer dates
ZSCORE_PENALTY_COEFF = 0.0005      # Coefficient for days penalty
ZSCORE_SHORT_DATE_BONUS = 0.0      # Reduce threshold per day below pivot (short-date bonus)


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING (%-based)
# ═══════════════════════════════════════════════════════════════════════════════

KELLY_FRACTION = 0.032             # Ultra-conservative Kelly
KELLY_MULTIPLIER = 1.0             # Additional multiplier
KELLY_CAP = 1.0                    # No cap on raw Kelly fraction
POSITION_PCT_MIN = 0.02            # Min 2% of capital per position
POSITION_PCT_MAX = 0.05            # Max 5% of capital per position
MAX_CONCURRENT = 25                # Max concurrent positions (v1: 10 -> 25)
MAX_CAPITAL_DEPLOYED_PCT = 0.80    # Max 80% of capital deployed

# Resolution-time sizing bonus
RESOLUTION_SIZE_ENABLED = False
RESOLUTION_SIZE_BONUS_PCT = 0.0    # Extra position % per day closer to resolution


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

STOP_LOSS_PCT = 0.20               # -30% -> close all
PARTIAL_EXIT_PROFIT_PCT = 0.50     # +50% unrealized -> partial exit
PARTIAL_EXIT_SELL_PCT = 0.50       # Sell 50% of position

# Trailing stop
TRAILING_STOP_ENABLED = False
TRAILING_STOP_PCT = 0.15           # Drawdown from peak to trigger
TRAILING_STOP_ACTIVATION = 0.05    # Min profit % before trailing stop activates

# Time-based exit
TIME_EXIT_ENABLED = False
TIME_EXIT_DAYS_BEFORE = 1          # Exit N days before resolution


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL STATE (reset between windows)
# ═══════════════════════════════════════════════════════════════════════════════

_flow_history: dict = {}
_cooldown: dict = {}


def reset_state():
    """Reset internal state between walk-forward windows."""
    global _flow_history, _cooldown
    _flow_history = {}
    _cooldown = {}


# ═══════════════════════════════════════════════════════════════════════════════
# SPIKE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_spike(market_id: int, flow_value: float) -> float:
    """
    Track taker flow and compute rolling z-score for spike detection.

    Returns:
        Z-score of latest flow value relative to recent history.
    """
    if market_id not in _flow_history:
        _flow_history[market_id] = []

    _flow_history[market_id].append(flow_value)

    # Trim to 2x lookback
    if len(_flow_history[market_id]) > SPIKE_LOOKBACK_TICKS * 2:
        _flow_history[market_id] = _flow_history[market_id][-SPIKE_LOOKBACK_TICKS:]

    history = _flow_history[market_id]
    if len(history) < MIN_HISTORY_TICKS:
        return 0.0

    window = history[-SPIKE_LOOKBACK_TICKS:] if len(history) >= SPIKE_LOOKBACK_TICKS else history
    if len(window) < 3:
        return 0.0

    # Z-score of latest value vs rolling window (excluding latest)
    past = window[:-1]
    mean = sum(past) / len(past)
    var = sum((x - mean) ** 2 for x in past) / len(past)
    std = math.sqrt(var) if var > 0 else 1.0

    if std < 0.01:
        std = 1.0

    return (flow_value - mean) / std


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_entry(price_yes: float, zscore: float) -> Tuple[float, float]:
    """
    Compute edge and model NO probability.

    Model: YES actual prob = price_yes x effective_discount.
    Discount can be adjusted by zscore magnitude.

    Returns:
        (edge, model_no_prob). Edge = model_no_prob - market_no_price.
    """
    # Price-dependent discount: lower-priced longshots have more optimism bias
    # At price 0.01: discount = DISCOUNT_FACTOR * 0.8 (20% lower = more edge)
    # At price 0.085: discount = DISCOUNT_FACTOR * 1.0
    price_scale = -1.80 + 2.75 * (price_yes / LONGSHOT_PRICE_MAX)
    effective_discount = DISCOUNT_FACTOR * price_scale
    effective_discount -= DISCOUNT_ZSCORE_BONUS * max(0, zscore - ZSCORE_THRESHOLD)
    # No clamping — allow negative effective_discount for very cheap longshots

    model_yes_prob = price_yes * effective_discount
    model_no_prob = 1.0 - model_yes_prob
    no_price = 1.0 - price_yes
    edge = model_no_prob - no_price

    # Zscore-based edge bonus
    edge += EDGE_ZSCORE_BONUS * max(0, zscore - ZSCORE_THRESHOLD)

    return edge, model_no_prob


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY GATE
# ═══════════════════════════════════════════════════════════════════════════════

def should_trade(
    edge: float,
    price_yes: float,
    volume_24h: float,
    days_to_resolution: float,
    zscore: float,
) -> bool:
    """Quality gate: all filters must pass to enter a trade."""
    if edge < MIN_EDGE:
        return False
    if edge > MAX_EDGE:
        return False
    if price_yes < MIN_YES_PRICE:
        return False
    if price_yes > LONGSHOT_PRICE_MAX:
        return False
    if volume_24h < MIN_VOLUME_24H:
        return False
    if days_to_resolution < RESOLUTION_DAYS_MIN:
        return False
    if days_to_resolution > RESOLUTION_DAYS_MAX:
        return False

    # Resolution-adjusted z-score threshold with short-date bonus
    days_excess = max(0, days_to_resolution - ZSCORE_DAYS_PIVOT)
    days_bonus = max(0, ZSCORE_DAYS_PIVOT - days_to_resolution) * ZSCORE_SHORT_DATE_BONUS
    adj_threshold = ZSCORE_THRESHOLD - days_bonus + days_excess ** ZSCORE_PENALTY_POWER * ZSCORE_PENALTY_COEFF
    if zscore < adj_threshold:
        return False

    # Volume-dependent edge filter
    if VOLUME_EDGE_ENABLED and volume_24h > VOLUME_EDGE_THRESHOLD:
        vol_excess = (volume_24h - VOLUME_EDGE_THRESHOLD) / 100000.0
        adjusted_edge = edge - vol_excess * VOLUME_EDGE_REDUCTION
        if adjusted_edge < MIN_EDGE:
            return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING (%-based)
# ═══════════════════════════════════════════════════════════════════════════════

def position_size(
    edge: float,
    no_price: float,
    available_capital: float,
    total_capital: float,
) -> float:
    """
    Kelly criterion sizing for NO position.
    Returns position size in USD (as fraction of total_capital).

    Sizing is %-based: clamped between POSITION_PCT_MIN and POSITION_PCT_MAX
    of total_capital. Harness enforces additional 1% floor.
    """
    if no_price <= 0 or no_price >= 1 or edge <= 0 or total_capital <= 0:
        return 0.0

    # Kelly: f* = edge / (1/no_price - 1)
    odds_minus_1 = (1.0 / no_price) - 1.0
    if odds_minus_1 <= 0:
        return 0.0

    kelly_raw = edge / odds_minus_1
    kelly_raw = max(0.0, min(kelly_raw, KELLY_CAP))

    kelly_adjusted = kelly_raw * KELLY_FRACTION * KELLY_MULTIPLIER

    size = available_capital * kelly_adjusted

    # Clamp to %-based position bounds
    min_size = total_capital * POSITION_PCT_MIN
    max_size = total_capital * POSITION_PCT_MAX
    size = max(min_size, min(max_size, size))

    # Don't exceed available capital
    size = min(size, available_capital * 0.95)

    if size < min_size:
        return 0.0

    return round(size, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def check_exit(
    pnl_pct: float,
    peak_pnl_pct: float,
    days_to_resolution: float,
    partial_exited: bool,
) -> Tuple[Optional[str], float]:
    """
    Check exit conditions for an open position.

    Returns:
        (exit_reason, exit_fraction) or (None, 0) if no exit.
        exit_fraction: 1.0 = close all, <1.0 = partial exit.
    """
    # Stop loss
    if pnl_pct <= -STOP_LOSS_PCT:
        return "stop_loss", 1.0

    # Trailing stop (with activation threshold)
    if TRAILING_STOP_ENABLED and peak_pnl_pct >= TRAILING_STOP_ACTIVATION:
        drawdown = peak_pnl_pct - pnl_pct
        if drawdown >= TRAILING_STOP_PCT:
            return "trailing_stop", 1.0

    # Time exit
    if TIME_EXIT_ENABLED and days_to_resolution <= TIME_EXIT_DAYS_BEFORE:
        return "time_exit", 1.0

    # Partial exit
    if not partial_exited and pnl_pct >= PARTIAL_EXIT_PROFIT_PCT:
        return "partial", PARTIAL_EXIT_SELL_PCT

    return None, 0.0
