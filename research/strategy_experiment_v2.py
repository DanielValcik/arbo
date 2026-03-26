"""
Strategy C Weather Experiment V2 — EXIT LOGIC EXTENSION
========================================================

Inherits all parameters from strategy_experiment.py and adds
exit/roll logic for backtest_harness_v2.py.

The backtest harness calls check_exit() at day-0 for positions
entered at days_out > 0, using a more accurate day-0 forecast
to decide whether to hold to resolution or exit early.

Run: python3 research/backtest_harness_v2.py
"""

from strategy_experiment import *  # noqa: F401,F403 — inherit all parameters + functions

# ═══════════════════════════════════════════════════════════════════════════════
# EXIT PARAMETERS — Sweep these
# ═══════════════════════════════════════════════════════════════════════════════

EXIT_ENABLED = True

# ── Edge-based exit (core METAR-informed logic) ──
# Exit when updated edge (day-0 forecast_prob - day-0 market_price) drops below this.
#   0.0  = exit only when edge turns negative (loss of informational advantage)
#  >0    = aggressive: require maintaining minimum edge to hold
#  <0    = patient: tolerate slightly negative edge before exiting
MIN_HOLD_EDGE = 0.0

# ── Stop-loss (price-based safety net) ──
# Exit if mark-to-market loss exceeds this fraction of entry price.
#   1.0  = disabled (only resolution causes full loss)
#   0.40 = exit at -40% drawdown
STOP_LOSS_PCT = 1.0

# ── Probability floor ──
# Exit if updated day-0 probability drops below this absolute level.
# Catches cases where model says "very unlikely to win" regardless of edge.
#   0.0  = disabled
#   0.30 = exit if <30% chance of winning
PROB_EXIT_FLOOR = 0.0

# ── Partial profit-taking ──
PARTIAL_PROFIT_ENABLED = False
PARTIAL_PROFIT_THRESHOLD = 0.30   # +30% unrealized gain triggers partial exit
PARTIAL_PROFIT_FRACTION = 0.50    # Sell 50% of position

# ── Rolling to adjacent bucket ──
# When exiting, check if a better bucket exists for the same city/date.
# If so, enter it instead of just exiting (dynamic position rotation).
ROLL_ENABLED = False
MIN_ROLL_EDGE = 0.08              # Minimum edge for roll target bucket


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def check_exit(entry_price, current_price, updated_prob, updated_edge,
               edge_at_entry, pnl_pct, *, city=None):
    """
    Check whether an open position should be exited early.

    Called on resolution day (day-0) for positions entered at days_out > 0.
    Uses updated probability from the more accurate day-0 forecast to decide.

    This is the METAR-informed exit: at day-0 our forecast sigma is ~1.2°C
    vs ~3.0°C at day-1. This dramatically improves our probability estimate,
    letting us identify losers before resolution.

    Args:
        entry_price: Fill price when position was opened (includes taker slippage).
        current_price: Day-0 market price for this bucket.
        updated_prob: Our probability estimate using day-0 forecast.
        updated_edge: updated_prob - current_price.
        edge_at_entry: Edge when position was opened (days_out > 0).
        pnl_pct: Unrealized P&L as fraction: (current_price - entry_price) / entry_price.
        city: City identifier for per-city logic.

    Returns:
        dict with:
          "action": "hold" | "exit" | "partial_exit" | "roll"
          "reason": str (exit reason for metrics)
          "fraction": float (for partial_exit only)
    """
    if not EXIT_ENABLED:
        return {"action": "hold"}

    # 1. Stop-loss (price-based safety net — catch catastrophic drops)
    if STOP_LOSS_PCT < 1.0 and pnl_pct <= -STOP_LOSS_PCT:
        action = "roll" if ROLL_ENABLED else "exit"
        return {"action": action, "reason": "stop_loss"}

    # 2. Probability floor (exit if model says "very unlikely")
    if PROB_EXIT_FLOOR > 0 and updated_prob < PROB_EXIT_FLOOR:
        action = "roll" if ROLL_ENABLED else "exit"
        return {"action": action, "reason": "prob_floor"}

    # 3. Edge-based exit (core logic: exit when informational advantage is lost)
    if updated_edge < MIN_HOLD_EDGE:
        action = "roll" if ROLL_ENABLED else "exit"
        return {"action": action, "reason": "edge_lost"}

    # 4. Partial profit-taking (lock in gains on big winners)
    if PARTIAL_PROFIT_ENABLED and pnl_pct >= PARTIAL_PROFIT_THRESHOLD:
        return {
            "action": "partial_exit",
            "reason": "profit_take",
            "fraction": PARTIAL_PROFIT_FRACTION,
        }

    return {"action": "hold"}
