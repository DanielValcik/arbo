"""Strategy D — Exit-timing feature engineering (pure functions).

Separated from build_exit_timing_set.py so each helper is unit-testable
without touching the DB or parquet pipeline.

Design principles:
  * Every feature at time t uses only prices[0..t] (no lookahead).
  * All helpers take primitive types → easy to test.
  * No pandas / numpy in hot path — stdlib math sufficient for 30K-row
    per-trade feature extraction. Batch upgrade to vector ops possible
    later if needed.

References: docs/STRATEGY_D_ML_DESIGN.md §4.
"""
from __future__ import annotations

import math
from typing import Iterable


# ── Rolling statistics ────────────────────────────────────────────────


def rolling_std(prices: list[float]) -> float:
    """Sample standard deviation (n-1 denominator). 0 if n<2."""
    n = len(prices)
    if n < 2:
        return 0.0
    mean = sum(prices) / n
    sq_sum = sum((x - mean) ** 2 for x in prices)
    return math.sqrt(sq_sum / (n - 1))


def rolling_slope(prices: list[float]) -> float:
    """OLS slope of prices vs time index. 0 if n<2 or degenerate."""
    n = len(prices)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(prices) / n
    num = sum((xs[i] - x_mean) * (prices[i] - y_mean) for i in range(n))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den > 1e-12 else 0.0


def pct_up(prices: list[float]) -> float:
    """Fraction of ticks where price[i] > price[i-1]. 0.5 if n<2."""
    if len(prices) < 2:
        return 0.5
    ups = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i - 1])
    return ups / (len(prices) - 1)


def autocorr_1(returns: list[float]) -> float:
    """Lag-1 autocorrelation of returns. 0 if n<3 or degenerate variance."""
    n = len(returns)
    if n < 3:
        return 0.0
    mean = sum(returns) / n
    num = sum((returns[i] - mean) * (returns[i - 1] - mean) for i in range(1, n))
    den = sum((r - mean) ** 2 for r in returns)
    return num / den if den > 1e-12 else 0.0


# ── First passage time — label helper ────────────────────────────────


def find_first_gb_hit(
    trajectory: list[tuple[int, float]],
    target: float,
    side: str,
) -> int | None:
    """Return index of first timestep where GB target was crossed, or None.

    For YES side: trigger when `price >= target` (target = entry + delta).
    For NO side : trigger when `price <= target` (target = entry - delta).
    """
    if side == "yes":
        for i, (_, p) in enumerate(trajectory):
            if p >= target:
                return i
    elif side == "no":
        for i, (_, p) in enumerate(trajectory):
            if p <= target:
                return i
    return None


# ── Feature extraction at time t ──────────────────────────────────────


def compute_features_at(
    trajectory: list[tuple[int, float]],
    t: int,
    entry_price: float,
    target: float,
    stop_loss_price: float,
    side: str,
    model_prob: float,
    edge_at_entry: float,
    total_len_expected: int | None = None,
) -> dict:
    """All features computable at time t WITHOUT looking beyond index t.

    Args:
        trajectory: list of (ts, price) — full post-entry trajectory
        t: current timestep (0-indexed; t=0 is entry tick)
        entry_price: price at trajectory[0]
        target: GB target (side-adjusted: entry+δ for yes, entry-δ for no)
        stop_loss_price: stop-loss level (side-adjusted)
        side: "yes" or "no"
        model_prob: Elo+Pinnacle ensemble P(yes wins) at entry
        edge_at_entry: model_prob - entry_price (signed)
        total_len_expected: expected trajectory length (for elapsed_frac
            normalization). If None, uses len(trajectory).

    Returns:
        Dict of numeric features. Keys documented in
        docs/STRATEGY_D_ML_DESIGN.md §4.
    """
    if t < 0 or t >= len(trajectory):
        raise ValueError(f"t={t} out of range [0, {len(trajectory)})")

    # Prices available at time t (inclusive) — NO LOOKAHEAD
    prices_so_far = [p for (_, p) in trajectory[: t + 1]]
    price_t = prices_so_far[-1]

    # ── Price-state features ──────────────────────────────────────────
    price_return_since_entry = price_t - entry_price
    price_return_pct = (
        price_return_since_entry / entry_price if entry_price > 1e-9 else 0.0
    )
    max_so_far = max(prices_so_far)
    min_so_far = min(prices_so_far)
    max_minus_now = max_so_far - price_t
    drawdown_from_entry = entry_price - min_so_far

    # Side-aware unrealized edge
    if side == "yes":
        unrealized_edge = price_t - entry_price
        peak_profit = max_so_far - entry_price
    else:
        unrealized_edge = entry_price - price_t
        peak_profit = entry_price - min_so_far

    # ── Volatility (rolling) ──────────────────────────────────────────
    vol_5 = rolling_std(prices_so_far[-5:])
    vol_15 = rolling_std(prices_so_far[-15:])
    vol_60 = rolling_std(prices_so_far[-60:])
    vol_ratio_5_60 = vol_5 / vol_60 if vol_60 > 1e-9 else 1.0

    # Realized range
    realized_range_15 = (max(prices_so_far[-15:]) - min(prices_so_far[-15:])
                         if len(prices_so_far) >= 2 else 0.0)

    # ── Momentum ──────────────────────────────────────────────────────
    slope_5 = rolling_slope(prices_so_far[-5:])
    slope_15 = rolling_slope(prices_so_far[-15:])
    pct_up_10 = pct_up(prices_so_far[-10:])

    # Tick-returns for autocorr
    ticks = prices_so_far[-6:]  # 5 return lags from 6 prices
    rets = [ticks[i] - ticks[i - 1] for i in range(1, len(ticks))]
    ret_autocorr = autocorr_1(rets)

    # ── Time features ─────────────────────────────────────────────────
    total_len = total_len_expected or max(len(trajectory), 1)
    elapsed_frac = t / max(total_len - 1, 1)
    remaining_frac = 1.0 - elapsed_frac
    elapsed_ticks = t

    # ── Distance to barriers (side-aware) ─────────────────────────────
    # For YES: we want price ↑ to hit target (higher). gb_distance = target - price.
    # For NO: we want price ↓ to hit target (lower).  gb_distance = price - target.
    if side == "yes":
        gb_distance = target - price_t
        gb_already_touched = 1 if max_so_far >= target else 0
        # Stop-loss: for YES we get stopped if price drops below SL threshold.
        # sl_distance = price - sl (higher is safer, further from trigger).
        sl_distance = price_t - stop_loss_price if stop_loss_price > 0 else 10.0
    else:
        gb_distance = price_t - target
        gb_already_touched = 1 if min_so_far <= target else 0
        sl_distance = stop_loss_price - price_t if stop_loss_price < 2.0 else 10.0

    # Normalize by volatility — how many σ away is the barrier?
    # Cap extreme values to avoid INF when vol≈0 (happens at low-activity ticks).
    def _safe_div(num: float, den: float, cap: float = 100.0) -> float:
        if abs(den) < 1e-6:
            return math.copysign(cap, num) if num != 0 else 0.0
        val = num / den
        return max(-cap, min(cap, val))

    gb_distance_norm = _safe_div(gb_distance, vol_15)
    sl_distance_norm = _safe_div(sl_distance, vol_15)

    return {
        # Price-state
        "price_return_since_entry": price_return_since_entry,
        "price_return_pct": price_return_pct,
        "max_since_entry": max_so_far,
        "min_since_entry": min_so_far,
        "max_minus_now": max_minus_now,
        "drawdown_from_entry": drawdown_from_entry,
        "unrealized_edge": unrealized_edge,
        "peak_profit": peak_profit,
        # Volatility
        "vol_5": vol_5,
        "vol_15": vol_15,
        "vol_60": vol_60,
        "vol_ratio_5_60": vol_ratio_5_60,
        "realized_range_15": realized_range_15,
        # Momentum
        "slope_5": slope_5,
        "slope_15": slope_15,
        "pct_up_10": pct_up_10,
        "ret_autocorr_5": ret_autocorr,
        # Time
        "elapsed_frac": elapsed_frac,
        "remaining_frac": remaining_frac,
        "elapsed_ticks": float(elapsed_ticks),
        # Barrier distances
        "gb_distance": gb_distance,
        "gb_distance_norm": gb_distance_norm,
        "gb_already_touched": float(gb_already_touched),
        "sl_distance": sl_distance,
        "sl_distance_norm": sl_distance_norm,
        # Static entry features (constant for all t in a trade)
        "model_prob_entry": model_prob,
        "edge_at_entry": edge_at_entry,
        "entry_price_level": entry_price,
    }


# ── Public list of feature column names (order matters for XGBoost) ────

# (All numeric feature names — the label columns {"event", "time_to_event"}
# and identity columns {"trade_id", "game_date", ...} are NOT here.)
FEATURE_COLUMNS: list[str] = [
    "price_return_since_entry",
    "price_return_pct",
    "max_since_entry",
    "min_since_entry",
    "max_minus_now",
    "drawdown_from_entry",
    "unrealized_edge",
    "peak_profit",
    "vol_5",
    "vol_15",
    "vol_60",
    "vol_ratio_5_60",
    "realized_range_15",
    "slope_5",
    "slope_15",
    "pct_up_10",
    "ret_autocorr_5",
    "elapsed_frac",
    "remaining_frac",
    "elapsed_ticks",
    "gb_distance",
    "gb_distance_norm",
    "gb_already_touched",
    "sl_distance",
    "sl_distance_norm",
    "model_prob_entry",
    "edge_at_entry",
    "entry_price_level",
]


# Monotonic constraints: +1 = feature ↑ → target ↑, -1 = feature ↑ → target ↓
# Target here is log(time_to_event) via AFT:
#   larger gb_distance  → harder to reach target → LONGER time_to_event → +1
#   larger vol_15       → faster first-passage → SHORTER time_to_event → -1
#   larger elapsed_frac → closer to censoring → SHORTER time_to_event → -1
# Unlisted features get 0 (no constraint).
MONOTONIC_CONSTRAINTS: dict[str, int] = {
    "gb_distance": +1,
    "gb_distance_norm": +1,
    "vol_15": -1,
    "vol_5": -1,
    "elapsed_frac": -1,
    "gb_already_touched": -1,  # once touched, can re-touch quickly
    "peak_profit": -1,          # already near profit ⇒ quick GB
}


def get_monotone_vector(feature_names: list[str]) -> tuple[int, ...]:
    """Monotone-constraint tuple aligned with feature_names (for XGBoost)."""
    return tuple(MONOTONIC_CONSTRAINTS.get(f, 0) for f in feature_names)
