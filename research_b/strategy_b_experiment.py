"""
Strategy B Reflexivity Surfer — Experiment Parameters
=====================================================

This is the ONLY file modified by autoresearch.
All tunable parameters, signal computation, quality gate, and sizing live here.

Strategy thesis: On-chain/social activity diverges from price momentum.
When divergence is extreme -> mean reversion creates edge on Polymarket crypto
binary markets.

Phase State Machine:
  START (1) -> BOOM (2): social > price, buy YES (expect price UP)
  START (1) -> PEAK (3): price > social, buy NO (expect price DOWN)
  PEAK  (3) -> BUST (4): divergence drops, hold NO position
  BUST  (4) -> START(1): divergence normalizes, exit

Run: python3 research_b/backtest_b_harness_v2.py
"""

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    divergence: float
    z_score: float
    confidence: float
    momentum_score: float
    price_momentum: float
    vol_acceleration: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DIVERGENCE SIGNAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Activity metric weights (must sum to 1.0)
W_VOLUME = 0.70               # Trading volume momentum weight
W_DAA_PROXY = 0.30             # DAA proxy (volume/price ratio) weight

# Lookback and history
MOMENTUM_LOOKBACK = 8          # Days to compute pct change for momentum
MOMENTUM_LOOKBACK_SHORT = 5    # Short-term lookback for multi-timeframe
MULTI_TF_WEIGHT = 0.50         # Weight for short-term momentum (0.50 for long-term)
DIVERGENCE_HISTORY = 30        # Rolling window for z-score computation
MIN_HISTORY = 5                # Minimum history entries before generating signals

# Z-score threshold
SIGMA_THRESHOLD = 0.60         # |z_score| must exceed this to trigger phase change

# Momentum normalization scales
PRICE_MOMENTUM_SCALE = 200.0   # tanh(price_pct_change / scale)
VOLUME_MOMENTUM_SCALE = 30.0   # tanh(volume_pct_change / scale)

# EMA decay for divergence history (R2-NEW)
EMA_DECAY = 0.0                # 0 = simple mean (current), >0 = exponential weight
                               # e.g., 0.1 = recent data weighted ~10x more than oldest

# Volume acceleration — 2nd derivative (R2-NEW)
VOL_ACCELERATION_WEIGHT = 0.10 # 0 = disabled, >0 = blend vol acceleration into signal
VOL_ACCELERATION_LOOKBACK = 3  # Days for acceleration calculation


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE STATE MACHINE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

BOOM_DIVERGENCE = -0.05        # Divergence < this -> Phase 2 (activity UP, price FLAT)
PEAK_DIVERGENCE = 0.20         # Divergence > this -> Phase 3 (price UP, activity FLAT)
BUST_ENTRY = 0.10              # Phase 3->4: divergence drops below this
BUST_EXIT_LOW = -0.05          # Phase 4->1: divergence in this range -> reset
BUST_EXIT_HIGH = 0.05

PHASE_COOLDOWN = 1             # Days after exit before re-entry for same coin


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDING PERIOD
# ═══════════════════════════════════════════════════════════════════════════════

HOLDING_PERIOD_DAYS = 3        # Resolution horizon for synthetic binary market


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY GATE
# ═══════════════════════════════════════════════════════════════════════════════

MIN_EDGE = 0.08                # Minimum edge to trade (8%)
MAX_EDGE = 0.50                # Maximum edge (suspicious -> skip)
MIN_VOLUME_24H = 100.0         # Minimum 24h volume ($)
MIN_LIQUIDITY = 50.0           # Minimum liquidity ($)
MIN_MARKET_PRICE = 0.15        # Skip extreme longshots
MAX_MARKET_PRICE = 0.85        # Skip near-certainties
MIN_DIVERGENCE_ABS = 0.05      # Minimum |divergence| to consider
MIN_CONFIDENCE = 0.25          # Minimum confidence from z-score


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY REGIME FILTER (R2-NEW)
# ═══════════════════════════════════════════════════════════════════════════════

VOLATILITY_REGIME_ENABLED = True    # Enable/disable volatility filter
VOLATILITY_HIGH_THRESHOLD = 0.08    # Skip when daily vol > 8% (too noisy)
VOLATILITY_LOW_THRESHOLD = 0.005    # Skip when daily vol < 0.5% (no movement)


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

KELLY_FRACTION = 0.22          # Optimal Kelly fraction
KELLY_MULTIPLIER = 0.35        # Additional multiplier (effective = 0.0525)
KELLY_RAW_CAP = 0.35           # Cap on raw Kelly fraction
MAX_POSITION_PCT = 0.05        # Max 5% capital per trade
MIN_POSITION_USD = 5.0         # Minimum $5 position
MAX_CONCURRENT = 15            # Max concurrent open positions

# Phase-specific position caps
PHASE2_MAX_POSITION = 5.0      # $5 max for boom trades
PHASE3_MAX_POSITION = 5.0      # $5 max for peak/bust trades

# Asymmetric phase Kelly multipliers (R2-NEW)
PHASE2_KELLY_MULT = 1.0        # Extra multiplier for BOOM phase sizing
PHASE3_KELLY_MULT = 1.0        # Extra multiplier for PEAK/BUST phase sizing

# Confidence-scaled sizing (R2-NEW)
CONFIDENCE_SIZE_SCALE = 0.0    # 0 = disabled, >0 = size *= confidence^alpha
                               # e.g., 0.5 = sqrt(confidence), 1.0 = linear, 2.0 = quadratic


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT RULES
# ═══════════════════════════════════════════════════════════════════════════════

PHASE2_STOP_LOSS = 0.15        # 15% stop loss on boom trades
PHASE3_STOP_LOSS = 0.12        # 12% stop loss on peak trades
PARTIAL_EXIT_GAIN = 0.01       # +1% unrealized gain -> partial exit
PARTIAL_EXIT_PCT = 0.85        # Sell 85% on partial exit

# Trailing stop (R2-NEW)
TRAILING_STOP_ENABLED = True    # Enable trailing stop (adds to fixed stop, not replace)
TRAILING_STOP_PCT = 0.04        # Exit when unrealized drops this much from peak


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION DAMPENING (R2-NEW)
# ═══════════════════════════════════════════════════════════════════════════════

CORR_DAMPENING_THRESHOLD = 0   # 0 = disabled, N = dampen when N+ coins signal same day
CORR_DAMPENING_FACTOR = 0.5    # Size multiplier when dampening triggers


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL STATE (reset between walk-forward windows)
# ═══════════════════════════════════════════════════════════════════════════════

_divergence_history: dict = {}
_phase_state: dict = {}
_vol_momentum_history: dict = {}


def reset_state():
    """Reset all internal state. Called by harness between windows."""
    global _divergence_history, _phase_state, _vol_momentum_history
    _divergence_history = {}
    _phase_state = {}
    _vol_momentum_history = {}


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signals(
    coin: str,
    price_series: List[float],
    volume_series: List[float],
    daa_series: Optional[List[float]] = None,
) -> SignalResult:
    """
    Compute divergence signals from raw price/volume data.

    The divergence measures the gap between activity momentum (volume + DAA proxy)
    and price momentum. When activity leads price (or vice versa), we have edge.
    """
    if len(price_series) < MOMENTUM_LOOKBACK + 1:
        return SignalResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    price_now = price_series[-1]
    vol_now = volume_series[-1]

    # --- Long-term momentum (MOMENTUM_LOOKBACK) ---
    price_prev_long = price_series[-(MOMENTUM_LOOKBACK + 1)]
    if price_prev_long <= 0:
        return SignalResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    price_pct_long = (price_now - price_prev_long) / price_prev_long * 100.0
    price_mom_long = math.tanh(price_pct_long / PRICE_MOMENTUM_SCALE)

    vol_prev_long = volume_series[-(MOMENTUM_LOOKBACK + 1)]
    if vol_prev_long <= 0:
        vol_delta_long = 0.0
    else:
        vol_pct_long = (vol_now - vol_prev_long) / vol_prev_long * 100.0
        vol_delta_long = math.tanh(vol_pct_long / VOLUME_MOMENTUM_SCALE)

    # --- Short-term momentum (MOMENTUM_LOOKBACK_SHORT) ---
    if len(price_series) >= MOMENTUM_LOOKBACK_SHORT + 1:
        price_prev_short = price_series[-(MOMENTUM_LOOKBACK_SHORT + 1)]
        if price_prev_short > 0:
            price_pct_short = (price_now - price_prev_short) / price_prev_short * 100.0
            price_mom_short = math.tanh(price_pct_short / PRICE_MOMENTUM_SCALE)
        else:
            price_mom_short = price_mom_long
        vol_prev_short = volume_series[-(MOMENTUM_LOOKBACK_SHORT + 1)]
        if vol_prev_short > 0:
            vol_pct_short = (vol_now - vol_prev_short) / vol_prev_short * 100.0
            vol_delta_short = math.tanh(vol_pct_short / VOLUME_MOMENTUM_SCALE)
        else:
            vol_delta_short = vol_delta_long
    else:
        price_mom_short = price_mom_long
        vol_delta_short = vol_delta_long

    # Blend multi-timeframe
    w_short = MULTI_TF_WEIGHT
    w_long = 1.0 - w_short
    price_momentum = w_long * price_mom_long + w_short * price_mom_short
    vol_delta = w_long * vol_delta_long + w_short * vol_delta_short

    # --- Volume acceleration (2nd derivative) ---
    vol_accel = 0.0
    if VOL_ACCELERATION_WEIGHT > 0 and coin in _vol_momentum_history:
        prev_deltas = _vol_momentum_history[coin]
        if len(prev_deltas) >= VOL_ACCELERATION_LOOKBACK:
            recent = prev_deltas[-1]
            older = prev_deltas[-VOL_ACCELERATION_LOOKBACK]
            vol_accel = recent - older

    # Track volume momentum for acceleration
    if coin not in _vol_momentum_history:
        _vol_momentum_history[coin] = []
    _vol_momentum_history[coin].append(vol_delta)
    if len(_vol_momentum_history[coin]) > DIVERGENCE_HISTORY:
        _vol_momentum_history[coin] = _vol_momentum_history[coin][-DIVERGENCE_HISTORY:]

    # DAA proxy: volume/price ratio captures "participation intensity"
    if daa_series is not None and len(daa_series) >= MOMENTUM_LOOKBACK + 1:
        daa_now = daa_series[-1]
        daa_prev = daa_series[-(MOMENTUM_LOOKBACK + 1)]
        if daa_prev > 0:
            daa_pct = (daa_now - daa_prev) / daa_prev * 100.0
            daa_delta = math.tanh(daa_pct / VOLUME_MOMENTUM_SCALE)
        else:
            daa_delta = 0.0
    else:
        # Proxy: volume per unit price
        vpp_now = vol_now / price_now if price_now > 0 else 0
        vol_prev_for_vpp = volume_series[-(MOMENTUM_LOOKBACK + 1)]
        vpp_prev = vol_prev_for_vpp / price_prev_long if price_prev_long > 0 else 0
        if vpp_prev > 0:
            vpp_pct = (vpp_now - vpp_prev) / vpp_prev * 100.0
            daa_delta = math.tanh(vpp_pct / VOLUME_MOMENTUM_SCALE)
        else:
            daa_delta = 0.0

    # Weighted momentum score (with optional acceleration blend)
    base_momentum = W_DAA_PROXY * daa_delta + W_VOLUME * vol_delta
    if VOL_ACCELERATION_WEIGHT > 0:
        momentum_score = math.tanh(base_momentum + VOL_ACCELERATION_WEIGHT * vol_accel)
    else:
        momentum_score = math.tanh(base_momentum)

    # Divergence: activity momentum minus price momentum
    divergence = momentum_score - price_momentum

    # Update rolling history
    if coin not in _divergence_history:
        _divergence_history[coin] = []
    _divergence_history[coin].append(divergence)
    if len(_divergence_history[coin]) > DIVERGENCE_HISTORY:
        _divergence_history[coin] = _divergence_history[coin][-DIVERGENCE_HISTORY:]

    # Z-score from rolling history (with optional EMA weighting)
    history = _divergence_history[coin]
    if len(history) < MIN_HISTORY:
        return SignalResult(divergence, 0.0, 0.0, momentum_score, price_momentum, vol_accel)

    if EMA_DECAY > 0 and len(history) > 1:
        # EMA-weighted mean and variance — recent values weighted higher
        weights = []
        w = 1.0
        for _ in range(len(history)):
            weights.append(w)
            w *= (1.0 - EMA_DECAY)
        weights.reverse()  # oldest = lowest weight, newest = highest
        total_w = sum(weights)
        mean_div = sum(wt * d for wt, d in zip(weights, history)) / total_w
        var_div = sum(wt * (d - mean_div) ** 2 for wt, d in zip(weights, history)) / total_w
    else:
        # Simple mean (original behavior)
        mean_div = sum(history) / len(history)
        var_div = sum((d - mean_div) ** 2 for d in history) / len(history)

    std_div = math.sqrt(var_div) if var_div > 0 else 0.0

    if std_div > 0:
        z_score = (divergence - mean_div) / std_div
    else:
        z_score = 0.0

    # Confidence: asymptotic mapping from z_score to [0, 1]
    confidence = min(1.0, abs(z_score) / (abs(z_score) + 1.0))

    return SignalResult(divergence, z_score, confidence, momentum_score, price_momentum, vol_accel)


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY REGIME FILTER (R2-NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def check_volatility_regime(daily_volatility: float) -> bool:
    """
    Check if current volatility regime is suitable for trading.
    Returns True if we should trade, False if we should skip.
    """
    if not VOLATILITY_REGIME_ENABLED:
        return True
    if daily_volatility > VOLATILITY_HIGH_THRESHOLD:
        return False  # Too volatile — signals are noise
    if daily_volatility < VOLATILITY_LOW_THRESHOLD:
        return False  # Too quiet — no movement to capture
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION DAMPENING (R2-NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def get_correlation_dampening(num_signals_today: int) -> float:
    """
    Returns a size multiplier based on how many coins signal on the same day.
    Market-wide moves (many coins signaling) = less coin-specific edge.
    Returns 1.0 if dampening is disabled or below threshold.
    """
    if CORR_DAMPENING_THRESHOLD <= 0:
        return 1.0
    if num_signals_today >= CORR_DAMPENING_THRESHOLD:
        return CORR_DAMPENING_FACTOR
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_phase_transition(
    coin: str, divergence: float, z_score: float, current_date: date
) -> int:
    """
    Update and return the phase for this coin.

    Phase 1 (START): Monitoring, no position.
    Phase 2 (BOOM): Activity leads price -> buy YES.
    Phase 3 (PEAK): Price leads activity -> buy NO.
    Phase 4 (BUST): Peak unwinding -> hold NO.
    """
    if coin not in _phase_state:
        _phase_state[coin] = {"phase": 1, "cooldown_until": None}
    state = _phase_state[coin]

    # Cooldown check
    if state["cooldown_until"] and current_date < state["cooldown_until"]:
        return state["phase"]

    phase = state["phase"]

    if phase == 1:  # START
        if divergence < BOOM_DIVERGENCE and abs(z_score) >= SIGMA_THRESHOLD:
            state["phase"] = 2
        elif divergence > PEAK_DIVERGENCE and abs(z_score) >= SIGMA_THRESHOLD:
            state["phase"] = 3
    elif phase == 2:  # BOOM — stays until trade resolves (harness manages)
        pass
    elif phase == 3:  # PEAK
        if divergence < BUST_ENTRY:
            state["phase"] = 4
    elif phase == 4:  # BUST
        if BUST_EXIT_LOW <= divergence <= BUST_EXIT_HIGH:
            state["phase"] = 1

    return state["phase"]


def reset_phase(coin: str, current_date: date):
    """Reset coin to START phase with cooldown period."""
    if coin in _phase_state:
        _phase_state[coin]["phase"] = 1
        _phase_state[coin]["cooldown_until"] = current_date + timedelta(days=PHASE_COOLDOWN)


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_probability(divergence: float, z_score: float, phase: int) -> float:
    """
    Estimate probability that our directional bet is correct.

    For BOOM (phase 2): activity > price -> price will rise -> P(above threshold) > 0.50
    For PEAK (phase 3/4): price > activity -> price will fall -> P(above threshold) < 0.50

    Returns probability of "coin above threshold at resolution".
    """
    confidence = min(1.0, abs(z_score) / (abs(z_score) + 1.0))

    if phase == 2:  # BOOM — buy YES (expect price UP)
        # Stronger divergence + higher confidence -> higher probability
        base = 0.50 + abs(divergence) * confidence * 0.3
        return min(0.95, max(0.05, base))
    elif phase in (3, 4):  # PEAK/BUST — buy NO (expect price DOWN)
        # Higher divergence -> price will fall -> P(above) is lower
        base = 0.50 - abs(divergence) * confidence * 0.3
        return min(0.95, max(0.05, base))

    return 0.50  # No signal


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY GATE
# ═══════════════════════════════════════════════════════════════════════════════

def should_trade(
    edge: float,
    market_price: float,
    divergence: float,
    confidence: float,
    volume_24h: float,
    liquidity: float,
    phase: int,
) -> bool:
    """Quality gate: all filters must pass to enter a trade."""
    if edge < MIN_EDGE:
        return False
    if edge > MAX_EDGE:
        return False
    if market_price < MIN_MARKET_PRICE:
        return False
    if market_price > MAX_MARKET_PRICE:
        return False
    if volume_24h < MIN_VOLUME_24H:
        return False
    if liquidity < MIN_LIQUIDITY:
        return False
    if abs(divergence) < MIN_DIVERGENCE_ABS:
        return False
    if confidence < MIN_CONFIDENCE:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def position_size(
    edge: float,
    market_price: float,
    available_capital: float,
    total_capital: float,
    phase: int,
    confidence: float = 1.0,
) -> float:
    """
    Kelly criterion with phase-specific limits, asymmetric phase multipliers,
    and optional confidence scaling.
    """
    if market_price <= 0 or market_price >= 1 or edge <= 0:
        return 0.0

    prob = market_price + edge
    if prob <= 0 or prob >= 1:
        return 0.0

    odds = (1.0 / market_price) - 1.0
    if odds <= 0:
        return 0.0

    kelly_raw = (prob * odds - (1.0 - prob)) / odds
    kelly_raw = max(0.0, min(kelly_raw, KELLY_RAW_CAP))

    kelly_adjusted = kelly_raw * KELLY_FRACTION * KELLY_MULTIPLIER

    # Asymmetric phase Kelly multipliers (R2-NEW)
    if phase == 2:
        kelly_adjusted *= PHASE2_KELLY_MULT
    elif phase in (3, 4):
        kelly_adjusted *= PHASE3_KELLY_MULT

    size = available_capital * kelly_adjusted

    # Confidence-scaled sizing (R2-NEW)
    if CONFIDENCE_SIZE_SCALE > 0 and confidence > 0:
        size *= confidence ** CONFIDENCE_SIZE_SCALE

    # Phase-specific caps
    if phase == 2:
        size = min(size, PHASE2_MAX_POSITION)
    elif phase in (3, 4):
        size = min(size, PHASE3_MAX_POSITION)

    # Global cap
    max_size = total_capital * MAX_POSITION_PCT
    size = min(size, max_size)

    if size < MIN_POSITION_USD:
        return 0.0

    return round(size, 2)
